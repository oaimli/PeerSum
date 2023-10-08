import pandas as pd
import pdb
import json
import os
import argparse
import torch
import pdb

from transformers import Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, \
    get_cosine_schedule_with_warmup

from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import seed_everything

from tokenization_led_fast import LEDTokenizerFast
from modeling_led import LEDForConditionalGeneration
from dataloader_led import get_dataloader_summ

import sys

sys.path.append("../../../")
from utils.metrics import rouge


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss


class AcceptancePredictor(nn.Module):
    def __init__(self, hidden_size, padding_id):
        super(AcceptancePredictor, self).__init__()
        self.linear_activation_stack = nn.Sequential(
            nn.Linear(hidden_size, 2),
        )
        torch.nn.init.xavier_uniform_(self.linear_activation_stack[0].weight)
        self.masked_id = padding_id

    def forward(self, decoder_last_hidden_state, decoder_input_ids):
        mask = torch.ones_like(decoder_input_ids)
        mask = mask.type_as(decoder_input_ids)
        mask[decoder_input_ids == self.masked_id] = 0
        # element-wise multiplication
        mask = torch.unsqueeze(mask, mask.dim())
        mask = mask.repeat(1, 1, decoder_last_hidden_state.size()[-1])
        masked_hidden_states = decoder_last_hidden_state * mask
        logits = self.linear_activation_stack(torch.mean(masked_hidden_states, dim=1))
        logits = logits.view((logits.size()[0], -1))
        return logits


class ConfidencePredictor(nn.Module):
    def __init__(self, hidden_size):
        super(ConfidencePredictor, self).__init__()
        self.linear_activation_stack = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        torch.nn.init.xavier_uniform_(self.linear_activation_stack[0].weight)

    def forward(self, encoder_last_hidden_state):
        predicted_confidences = self.linear_activation_stack(encoder_last_hidden_state)
        predicted_confidences = torch.squeeze(predicted_confidences, dim=-1)
        return predicted_confidences


class RatingPredictor(nn.Module):
    def __init__(self, hidden_size):
        super(RatingPredictor, self).__init__()
        self.linear_activation_stack = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        torch.nn.init.xavier_uniform_(self.linear_activation_stack[0].weight)

    def forward(self, encoder_last_hidden_state):
        predicted_ratings = self.linear_activation_stack(encoder_last_hidden_state)
        predicted_ratings = torch.squeeze(predicted_ratings, dim=-1)
        return predicted_ratings


class DocumentTypePredictor(nn.Module):
    def __init__(self, hidden_size):
        super(DocumentTypePredictor, self).__init__()
        self.linear_activation_stack = nn.Sequential(
            nn.Linear(hidden_size, 7),
            # seven classes, including paper_abstract, official_reviewe, official response, public comment, public response and author general response, author response
        )
        torch.nn.init.xavier_uniform_(self.linear_activation_stack[0].weight)

    def forward(self, encoder_last_hidden_state):
        logits = self.linear_activation_stack(encoder_last_hidden_state)
        logits = torch.squeeze(logits, dim=-1)
        return logits


class MTSummarizer(pl.LightningModule):
    def __init__(self, args):
        super(MTSummarizer, self).__init__()
        self.args = args

        self.tokenizer = LEDTokenizerFast.from_pretrained(args.pretrained_model)
        self.model = LEDForConditionalGeneration.from_pretrained(args.pretrained_model)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.acceptance_predictor = AcceptancePredictor(self.model.config.d_model, self.pad_token_id)
        self.confidence_predictor = ConfidencePredictor(self.model.config.d_model)
        self.rating_predictor = RatingPredictor(self.model.config.d_model)
        self.document_type_predictor = DocumentTypePredictor(self.model.config.d_model)
        self.use_ddp = self.args.speed_strategy.startswith("ddp")
        self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        # self.threadsep_official_token = "<thread-official-sep>"
        # self.threadsep_public_token = "<thread-public-sep>"
        # self.threadsep_author_token = "<thread-author-sep>"
        # self.tokenizer.add_tokens([ self.threadsep_official_token, self.threadsep_public_token, self.threadsep_author_token], special_tokens=True)
        # self.tokenizer.add_tokens(
        #     [self.threadsep_official_token],
        #     special_tokens=True)

        # self.threadsep_official_id = self.tokenizer.convert_tokens_to_ids(self.threadsep_official_token)
        # self.threadsep_public_id = self.tokenizer.convert_tokens_to_ids(self.threadsep_public_token)
        # self.threadsep_author_id = self.tokenizer.convert_tokens_to_ids(self.threadsep_author_token)
        # self.model.resize_token_embeddings(len(self.tokenizer))

        # self.bertscorer = load("bertscore")

    def forward(self, input_ids, document_startings, document_endings, document_levels, document_relationship_matrices,
                output_ids, ratings, confidences):
        decoder_input_ids = output_ids[:, :-1]
        # get the input ids and attention masks together
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask = global_attention_mask.type_as(input_ids)
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1
        global_attention_mask[input_ids == self.docsep_token_id] = 1
        # global_attention_mask[input_ids == (self.docsep_token_id or self.threadsep_official_id or self.threadsep_public_id or self.threadsep_author_id)] = 1
        # global_attention_mask[input_ids == (
        #             self.docsep_token_id or self.threadsep_official_id)] = 1

        attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.type_as(input_ids)
        attention_mask[input_ids == self.pad_token_id] = 0

        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        decoder_attention_mask = decoder_attention_mask.type_as(decoder_input_ids)
        decoder_attention_mask[decoder_input_ids == self.pad_token_id] = 0

        # print("######", input_ids.size(), attention_mask.size())

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            global_attention_mask=global_attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            document_startings=document_startings,
            document_endings=document_endings,
            decoder_document_levels=document_levels,
            document_relationship_matrices=document_relationship_matrices
        )
        lm_logits = outputs.logits
        assert lm_logits.shape[-1] == self.model.config.vocab_size

        decoder_last_hidden_states = outputs.decoder_hidden_states[-1]
        predicted_acceptances = self.acceptance_predictor(decoder_last_hidden_states, decoder_input_ids)
        # print(input_ids.size(), decoder_last_hidden_states.size(), predicted_acceptances.size())

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        predicted_confidences = self.confidence_predictor(encoder_last_hidden_state)
        predicted_ratings = self.rating_predictor(encoder_last_hidden_state)
        predicted_review_roles = self.document_type_predictor(encoder_last_hidden_state)

        return lm_logits, predicted_acceptances, predicted_confidences, predicted_ratings, predicted_review_roles

    def configure_optimizers(self):
        if self.args.optimizer == "adafactor":
            optimizer = Adafactor(
                self.parameters(),
                lr=self.args.lr,
                scale_parameter=False,
                relative_step=False,
            )
        elif self.args.optimizer == "adamw":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=0.0005)

        if self.args.scheduler_type == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps
            )
        elif self.args.scheduler_type == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                        num_training_steps=self.args.total_steps)
        elif self.args.scheduler_type == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.total_steps,
            )

        if self.args.fix_lr:
            return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def shared_step(self, input_ids, document_startings, document_endings, document_levels,
                    document_relationship_matrices, output_ids, acceptances, confidences, ratings, review_roles):
        # print(input_ids.size())

        # different sizes for different samples in each batch
        # print(document_startings[0].size())
        # print(document_endings[0].size())
        # print(document_levels[0].size())
        # print(document_relationship_matrices[0].size())

        # print(output_ids.dtype)
        # print(acceptances.dtype)
        # print(confidences.dtype)
        # print(ratings.dtype)
        # print(review_roles.dtype)

        lm_logits, predicted_acceptances, predicted_confidences, predicted_ratings, predicted_review_roles = self.forward(
            input_ids, document_startings, document_endings, document_levels, document_relationship_matrices,
            output_ids, ratings, confidences)

        # generation loss
        labels = output_ids[:, 1:].clone()
        if self.args.label_smoothing == 0.0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss_generation = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss_generation, nll_loss = label_smoothed_nll_loss(
                lprobs,
                labels,
                self.args.label_smoothing,
                ignore_index=self.pad_token_id,
            )
        if torch.isnan(loss_generation):
            pdb.set_trace()

        # acceptance prediction loss
        ce_loss_acceptance = nn.CrossEntropyLoss()
        # print(predicted_acceptances.size(), acceptances.size())
        loss_acceptance_prediction = ce_loss_acceptance(predicted_acceptances, acceptances)

        # confidence, rating, document_type prediction loss
        loss_confidence_prediction = 0
        loss_rating_prediction = 0
        loss_document_type_prediction = 0
        mse_loss_confidence = nn.MSELoss(reduction="mean")
        mse_loss_rating = nn.MSELoss(reduction="mean")
        cross_entropy_loss_document_type = nn.CrossEntropyLoss()
        batch_size_current = input_ids.size()[0]
        for i in range(batch_size_current):
            input_ids_sample = input_ids[i]

            predicted_confidences_sample = predicted_confidences[i]
            confidences_sample = confidences[i]
            # print(torch.squeeze(predicted_confidences_sample).size(), confidences_sample.size())
            loss_confidence_prediction += mse_loss_confidence(
                torch.masked_select(predicted_confidences_sample, confidences_sample > -1),
                torch.masked_select(confidences_sample, confidences_sample > -1) / 5)

            predicted_ratings_sample = predicted_ratings[i]
            ratings_sample = ratings[i]
            # print(torch.squeeze(predicted_ratings_sample).size(), ratings_sample.size())
            loss_rating_prediction += mse_loss_rating(
                torch.masked_select(predicted_ratings_sample, ratings_sample > -1),
                torch.masked_select(ratings_sample, ratings_sample > -1) / 10)

            predicted_review_roles_sample = predicted_review_roles[i]
            review_roles_sample = review_roles[i]
            target_indexes = torch.nonzero(review_roles_sample > -1).squeeze()
            loss_document_type_prediction += cross_entropy_loss_document_type(
                torch.index_select(predicted_review_roles_sample, 0, target_indexes),
                torch.index_select(review_roles_sample, 0, target_indexes))

        loss_rating_prediction = loss_rating_prediction / batch_size_current
        loss_document_type_prediction = loss_document_type_prediction / batch_size_current
        loss_confidence_prediction = loss_confidence_prediction / batch_size_current

        # print(loss_generation)
        # print(loss_acceptance_prediction)
        # print(loss_confidence_prediction)
        # print(loss_rating_prediction)
        # print(loss_document_type_prediction)

        return loss_generation, loss_acceptance_prediction, loss_confidence_prediction, loss_rating_prediction, loss_document_type_prediction
        # return loss_generation, loss_acceptance_prediction, torch.tensor(0.0), torch.tensor(0.0), loss_rating_prediction

    def training_step(self, batch, batch_idx):
        # pdb.set_trace()
        input_ids, document_startings, document_endings, document_levels, document_relationship_matrices, output_ids, acceptances, confidences, ratings, review_roles = batch
        loss_generation, loss_acceptance_prediction, loss_confidence_prediction, loss_rating_prediction, loss_document_type_prediction = self.shared_step(
            input_ids, document_startings, document_endings, document_levels, document_relationship_matrices,
            output_ids, acceptances, confidences, ratings, review_roles)
        lr = loss_generation.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]["lr"]
        loss = self.args.generation_loss_weight * loss_generation + self.args.acceptance_loss_weight * loss_acceptance_prediction + self.args.confidence_loss_weight * loss_confidence_prediction + self.args.rating_loss_weight * loss_rating_prediction + self.args.document_type_loss_weight * loss_document_type_prediction
        self.log("train_loss", loss)
        self.log("train_loss_generation", loss_generation)
        self.log("train_loss_acceptance", loss_acceptance_prediction)
        self.log("train_loss_rating", loss_rating_prediction)
        self.log("train_loss_confidence", loss_confidence_prediction)
        self.log("train_loss_review_role", loss_document_type_prediction)
        self.log("lr", lr)
        # print(loss.dtype)
        # loss = torch.tensor(loss, dtype=torch.float32)
        return loss

    def compute_rouge_batch(self, input_ids, document_startings, document_endings, document_levels,
                            document_relationship_matrices, gold_str, batch_idx):
        # get the input ids and attention masks together
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask = global_attention_mask.type_as(input_ids)
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1
        # if self.args.documents_concatenation == "concat_start_wdoc_global":
        #     global_attention_mask[input_ids == (
        #             self.docsep_token_id or self.threadsep_official_id or self.threadsep_public_id or self.threadsep_author_id)] = 1

        if self.args.documents_concatenation == "concat_start_wdoc_global":
            global_attention_mask[input_ids == self.docsep_token_id] = 1

        # following the api, https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/text_generation
        # in the generate() function, if the model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.
        # pdb.set_trace()
        generated_ids = self.model.generate(
            input_ids=input_ids,
            # attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            use_cache=True,
            max_length=self.args.max_length_tgt,
            min_length=self.args.min_length_tgt,
            num_beams=self.args.beam_size,
            do_sample=True,
            top_p=0.95,
            no_repeat_ngram_size=3 if self.args.apply_triblck else None,
            length_penalty=self.args.length_penalty,
            repetition_penalty=self.args.repetition_penalty,
            renormalize_logits=self.args.renormalize_logits,
            early_stopping=self.args.generation_early_stopping,
            # model specific arguments
            # encoder
            document_startings=document_startings,
            document_endings=document_endings,
            document_relationship_matrices=document_relationship_matrices,
            # decoder
            decoder_document_levels=document_levels
        )
        # pdb.set_trace()
        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )

        source_documents = self.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=False)

        if self.args.mode == "test":
            if self.args.apply_triblck:
                output_dir = os.path.join(
                    self.args.model_path,
                    "generation_%d_%s_triblck_beam=%d_%d_%d"
                    % (
                        self.args.mask_num,
                        self.args.dataset_name,
                        self.args.beam_size,
                        self.args.max_length_input,
                        self.args.max_length_tgt,
                    ),
                )
            else:
                output_dir = os.path.join(
                    self.args.model_path,
                    "generated_txt_%d_%s_beam=%d_%d_%d"
                    % (
                        self.args.mask_num,
                        self.args.dataset_name,
                        self.args.beam_size,
                        self.args.max_length_input,
                        self.args.max_length_tgt,
                    ),
                )

            if self.args.resume_ckpt is not None:  # this is for the generation results of models fine-tuned myself
                output_dir += "_myself"

            if batch_idx == 0:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                else:
                    for file in os.listdir(output_dir):
                        os.remove(os.path.join(output_dir, file))
            idx = len(os.listdir(output_dir))
        result_batch = []
        for ref, pred, source in zip(gold_str, generated_str, source_documents):
            # change <n> to \n
            pred = pred.replace("<n>", "\n")

            if self.args.mode == "test":
                result_dict = {}
                result_dict["prediction"] = pred
                result_dict["reference"] = ref
                result_dict["source_documents"] = source.replace("<pad>", "")
                with open(os.path.join(output_dir, "%d.json" % (idx)), "w") as f:
                    json.dump(result_dict, f)
                idx += 1

            s = rouge(reference=ref, candidate=pred, use_stemmer=True,
                      types=["rouge1", "rouge2", "rougeL", "rougeLsum"], split_summaries=True)
            # bertscore = self.bertscorer.compute(predictions=[pred], references=[ref], lang="en")
            result_batch.append(
                (
                    s["rouge1"]["recall"],
                    s["rouge1"]["precision"],
                    s["rouge1"]["fmeasure"],
                    s["rouge2"]["recall"],
                    s["rouge2"]["precision"],
                    s["rouge2"]["fmeasure"],
                    s["rougeL"]["recall"],
                    s["rougeL"]["precision"],
                    s["rougeL"]["fmeasure"],
                    s["rougeLsum"]["recall"],
                    s["rougeLsum"]["precision"],
                    s["rougeLsum"]["fmeasure"],
                    # bertscore["precision"][0],
                    # bertscore["recall"][0],
                    # bertscore["f1"][0]
                    0.0,
                    0.0,
                    0.0
                )
            )
        return result_batch

    def validation_step(self, batch, batch_idx):
        # pdb.set_trace()
        for p in self.model.parameters():
            p.requires_grad = False

        input_ids, document_startings, document_endings, document_levels, document_relationship_matrices, output_ids, tgt, acceptances, confidences, ratings, review_roles = batch
        loss_generation, loss_acceptance_prediction, loss_confidence_prediction, loss_rating_prediction, loss_document_type_prediction = self.shared_step(
            input_ids, document_startings, document_endings, document_levels, document_relationship_matrices,
            output_ids, acceptances, confidences, ratings, review_roles)
        loss = self.args.generation_loss_weight * loss_generation + self.args.acceptance_loss_weight * loss_acceptance_prediction + self.args.confidence_loss_weight * loss_confidence_prediction + self.args.rating_loss_weight * loss_rating_prediction + self.args.document_type_loss_weight * loss_document_type_prediction

        result_batch = self.compute_rouge_batch(input_ids, document_startings, document_endings, document_levels,
                                                document_relationship_matrices, tgt, batch_idx)
        return {"validation_loss": loss, "validation_loss_generation": loss_generation,
                "validation_loss_acceptance": loss_acceptance_prediction,
                "validation_loss_confidence": loss_confidence_prediction,
                "validation_loss_rating": loss_rating_prediction,
                "validation_loss_review_role": loss_document_type_prediction, "rouge_result": result_batch}

    def compute_rouge_all(self, outputs, output_file=None):
        rouge_result_all = [r for b in outputs for r in b["rouge_result"]]
        names = []
        for rouge in ["1", "2", "L", "Lsum"]:
            names.extend(
                [
                    "rouge-{}-r".format(rouge),
                    "rouge-{}-p".format(rouge),
                    "rouge-{}-f".format(rouge),
                ]
            )
        names.extend(["bertscore-r", "bertscore-p", "bertscore-f"])
        rouge_results = pd.DataFrame(rouge_result_all, columns=names)
        avg = [rouge_results[c].mean() for c in rouge_results.columns]
        avgf = (avg[2] + avg[5] + avg[11]) / 3
        metrics = avg

        if args.mode == "test":
            rouge_results.loc["avg_score"] = avg
            if output_file:
                csv_name = (
                        self.args.model_path
                        + output_file
                        + "-%d.csv" % (torch.distributed.get_rank() if self.use_ddp else 0)
                )
                rouge_results.to_csv(csv_name)

            print("Test Result at Step %d" % (self.global_step))
            print(
                "Rouge-1 r score: %f, Rouge-1 p score: %f, Rouge-1 f-score: %f"
                % (metrics[0], metrics[1], metrics[2])
            )
            print(
                "Rouge-2 r score: %f, Rouge-2 p score: %f, Rouge-2 f-score: %f"
                % (metrics[3], metrics[4], metrics[5])
            )
            print(
                "Rouge-L r score: %f, Rouge-L p score: %f, Rouge-L f-score: %f"
                % (metrics[6], metrics[7], metrics[8])
            )
            print(
                "Rouge-Lsum r score: %f, Rouge-Lsum p score: %f, Rouge-Lsum f-score: %f"
                % (metrics[9], metrics[10], metrics[11])
            )
        return names, metrics, avgf

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        validation_loss = torch.stack([x["validation_loss"] for x in outputs]).mean()
        self.log("validation_loss", validation_loss, sync_dist=True if self.use_ddp else False)
        validation_loss_generation = torch.stack([x["validation_loss_generation"] for x in outputs]).mean()
        self.log("validation_loss_generation", validation_loss_generation, sync_dist=True if self.use_ddp else False)
        validation_loss_acceptance = torch.stack([x["validation_loss_acceptance"] for x in outputs]).mean()
        self.log("validation_loss_acceptance", validation_loss_acceptance, sync_dist=True if self.use_ddp else False)
        validation_loss_confidence = torch.stack([x["validation_loss_confidence"] for x in outputs]).mean()
        self.log("validation_loss_confidence", validation_loss_confidence, sync_dist=True if self.use_ddp else False)
        validation_loss_rating = torch.stack([x["validation_loss_rating"] for x in outputs]).mean()
        self.log("validation_loss_rating", validation_loss_rating, sync_dist=True if self.use_ddp else False)
        validation_loss_review_role = torch.stack([x["validation_loss_review_role"] for x in outputs]).mean()
        self.log("validation_loss_review_role", validation_loss_review_role, sync_dist=True if self.use_ddp else False)

        names, metrics, avgf = self.compute_rouge_all(outputs, output_file="val")
        self.log("val_rouge1_fmeasure", metrics[2], sync_dist=True if self.use_ddp else False)
        self.log("val_rouge2_fmeasure", metrics[5], sync_dist=True if self.use_ddp else False)
        self.log("val_rougeL_fmeasure", metrics[8], sync_dist=True if self.use_ddp else False)
        self.log("val_rougeLsum_fmeasure", metrics[11], sync_dist=True if self.use_ddp else False)
        self.log("val_bertscore_fmeasure", metrics[14], sync_dist=True if self.use_ddp else False)
        metrics = [validation_loss, validation_loss_generation, validation_loss_acceptance] + metrics
        names = ["validation_loss", "validation_loss_generation", "validation_loss_acceptance"] + names
        logs = dict(zip(*[names, metrics]))
        self.log("avgf", avgf, sync_dist=True if self.use_ddp else False)
        return {
            "avg_val_loss": validation_loss,
            "avgf": avgf,
            "log": logs,
            "progress_bar": logs,
        }

    def test_step(self, batch, batch_idx):
        # pdb.set_trace()
        for p in self.model.parameters():
            p.requires_grad = False
        input_ids, document_startings, document_endings, document_levels, document_relationship_matrices, output_ids, tgt, acceptances, confidences, ratings, review_roles = batch
        loss_generation, loss_acceptance_prediction, loss_confidence_prediction, loss_rating_prediction, loss_document_type_prediction = self.shared_step(
            input_ids, document_startings, document_endings, document_levels, document_relationship_matrices,
            output_ids, acceptances, confidences, ratings, review_roles)
        loss = self.args.generation_loss_weight * loss_generation + self.args.acceptance_loss_weight * loss_acceptance_prediction + self.args.confidence_loss_weight * loss_confidence_prediction + self.args.rating_loss_weight * loss_rating_prediction + self.args.document_type_loss_weight * loss_document_type_prediction
        result_batch = self.compute_rouge_batch(input_ids, document_startings, document_endings, document_levels,
                                                document_relationship_matrices, tgt, batch_idx)
        return {"test_loss": loss, "test_loss_generation": loss_generation,
                "test_loss_acceptance": loss_acceptance_prediction, "test_loss_confidence": loss_confidence_prediction,
                "test_loss_rating": loss_rating_prediction, "test_loss_review_role": loss_document_type_prediction,
                "rouge_result": result_batch}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("test_loss", test_loss, sync_dist=True if self.use_ddp else False)
        test_loss_generation = torch.stack([x["test_loss_generation"] for x in outputs]).mean()
        self.log("test_loss_generation", test_loss_generation, sync_dist=True if self.use_ddp else False)
        test_loss_acceptance = torch.stack([x["test_loss_acceptance"] for x in outputs]).mean()
        self.log("test_loss_acceptance", test_loss_acceptance, sync_dist=True if self.use_ddp else False)
        test_loss_confidence = torch.stack([x["test_loss_confidence"] for x in outputs]).mean()
        self.log("test_loss_confidence", test_loss_confidence, sync_dist=True if self.use_ddp else False)
        test_loss_rating = torch.stack([x["test_loss_rating"] for x in outputs]).mean()
        self.log("test_loss_rating", test_loss_rating, sync_dist=True if self.use_ddp else False)
        test_loss_review_role = torch.stack([x["test_loss_review_role"] for x in outputs]).mean()
        self.log("test_loss_review_role", test_loss_review_role, sync_dist=True if self.use_ddp else False)

        output_file = "test_%s_%d_%d_beam=%d_lenPen=%.2f" % (
            self.args.dataset_name,
            self.args.max_length_input,
            self.args.max_length_tgt,
            self.args.beam_size,
            self.args.length_penalty,
        )
        output_file = (
            output_file
            + "_fewshot_%d_%d" % (self.args.num_train_data, self.args.rand_seed)
            if self.args.fewshot
            else output_file
        )
        names, metrics, avgf = self.compute_rouge_all(outputs, output_file=output_file)
        self.log("test_rouge1_fmeasure", metrics[2])
        self.log("test_rouge2_fmeasure", metrics[5])
        self.log("test_rougeL_fmeasure", metrics[8])
        self.log("test_rougeLsum_fmeasure", metrics[11])
        self.log("test_bertscore_fmeasure", metrics[14])
        metrics = [test_loss, test_loss_generation, test_loss_acceptance, avgf] + metrics
        names = ["test_loss", "test_loss_generation", "test_loss_acceptance", "avgf"] + names
        logs = dict(zip(*[names, metrics]))
        # self.log_dict(logs)
        return {"avg_test_loss": test_loss, "avgf": avgf, "log": logs, "progress_bar": logs}


def train(args):
    # initialize checkpoint
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(args.model_path, "checkpoints/")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename="{step}-{validation_loss:.2f}-{avgf:.4f}",
        save_top_k=args.save_top_k,
        monitor="avgf",
        mode="max",
        save_on_train_epoch_end=False
    )

    # early_stopping = EarlyStopping(monitor='avgf', patience=3, mode='max')
    early_stopping = EarlyStopping(monitor='validation_loss', patience=args.early_stopping_patience, mode='min')

    if args.resume_ckpt is not None:
        model = MTSummarizer.load_from_checkpoint(args.ckpt_path + args.resume_ckpt, args=args)
    else:
        model = MTSummarizer(args)

    # initialize logger
    project_name = args.model_path.split("/")[3]
    logger = WandbLogger(project=project_name)
    logger.log_hyperparams(args)

    # initialize trainer
    trainer = pl.Trainer(
        devices=args.devices,
        accelerator=args.accelerator,
        auto_select_gpus=True,
        strategy=args.speed_strategy if args.speed_strategy.startswith("ddp") else None,
        track_grad_norm=-1,
        max_steps=args.total_steps * args.accum_batch,
        replace_sampler_ddp=False,
        accumulate_grad_batches=args.accum_batch,
        val_check_interval=args.val_check_interval * args.accum_batch,
        # check_val_every_n_epoch=1 if args.num_train_data > 100 else 5,
        logger=logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, early_stopping],
        enable_checkpointing=True,
        enable_progress_bar=True,
        precision=32
    )

    # load dataset
    train_dataloader = get_dataloader_summ(args, model.tokenizer, 'train', args.num_workers, True)
    valid_dataloader = get_dataloader_summ(args, model.tokenizer, 'validation', args.num_workers, False)

    # pdb.set_trace()
    trainer.fit(model, train_dataloader, valid_dataloader)

    if args.test_imediate:
        args.resume_ckpt = checkpoint_callback.best_model_path.split("/")[-1]
        print("The best checkpoint", args.resume_ckpt)
        if args.test_batch_size != -1:
            args.batch_size = args.test_batch_size
        args.mode = "test"
        test(args)


def test(args):
    # initialize checkpoint
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(args.model_path, "checkpoints/")
    # initialize trainer
    trainer = pl.Trainer(
        devices=1,
        auto_select_gpus=True,
        accelerator=args.accelerator,
        track_grad_norm=-1,
        max_steps=args.total_steps * args.accum_batch,
        replace_sampler_ddp=False,
        log_every_n_steps=5,
        enable_checkpointing=True,
        enable_progress_bar=True,
        precision=32
    )

    if args.resume_ckpt is not None:
        model = MTSummarizer.load_from_checkpoint(args.ckpt_path + args.resume_ckpt, args=args)
    else:
        model = MTSummarizer(args)

    # load dataset
    test_dataloader = get_dataloader_summ(args, model.tokenizer, 'test', args.num_workers, False)

    # test
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--devices", default=1, type=int, help="The number of devices (cpus or gpus) to use")
    parser.add_argument("--accelerator", default="gpu", type=str, choices=["gpu", "cpu"])
    parser.add_argument("--speed_strategy", default="no_ddp", type=str,
                        help="Accelerator strategy, e.g., ddp, no_ddp, ddp_find_unused_parameters_false")
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--pretrained_model", type=str, default=None, help="The name of the pretrained model")
    parser.add_argument("--model_path", type=str, default=None, help="The path for output, checkpoints and results")
    parser.add_argument("--documents_concatenation", type=str, default="concat_start_wdoc_global",
                        help="The method to concatenate different documents")
    parser.add_argument("--debug_mode", action="store_true", help="Set true if to debug")
    parser.add_argument("--ckpt_path", type=str, default=None, help="dir to save checkpoints")
    parser.add_argument("--save_top_k", default=3, type=int)
    parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from", default=None)
    parser.add_argument("--data_path", type=str, default="../../datasets/")
    parser.add_argument("--dataset_name", type=str, default="multinews",
                        choices=["peersum_all"])
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for dataloader")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_length_input", default=4096, type=int)
    parser.add_argument("--max_length_tgt", default=1024, type=int)
    parser.add_argument("--min_length_tgt", default=0, type=int)
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--rand_seed", type=int, default=42,
                        help="Seed for random sampling, useful for few shot learning")
    parser.add_argument("--debug", action="store_true", help="debug")

    # For training
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Use limited batches in training")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Use limited batches in validation")
    parser.add_argument("--lr", type=float, default=3e-5, help="Maximum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--accum_data_per_step", type=int, default=16, help="Number of data per step")
    parser.add_argument("--total_steps", type=int, default=500000, help="Number of steps to train")
    parser.add_argument("--num_train_data", type=int, default=-1,
                        help="Number of training data, -1 for full dataset and any positive number indicates how many data to use")
    parser.add_argument("--num_val_data", type=int, default=-1, help="The number of testing data")
    parser.add_argument("--fix_lr", action="store_true", help="use fix learning rate")
    parser.add_argument("--test_imediate", action="store_true", help="test on the best checkpoint")
    parser.add_argument("--fewshot", action="store_true", help="whether this is a run for few shot learning")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="training early stopping")
    parser.add_argument("--val_check_interval", type=int, default=10)
    # balancing weights of losses
    parser.add_argument("--generation_loss_weight", type=int, default=1)
    parser.add_argument("--acceptance_loss_weight", type=int, default=1)
    parser.add_argument("--confidence_loss_weight", type=int, default=1)
    parser.add_argument("--rating_loss_weight", type=int, default=1)
    parser.add_argument("--document_type_loss_weight", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adafactor", choices=["adafactor", "adamw"], help="optimizer")
    parser.add_argument("--scheduler_type", type=str, default="linear_schedule_with_warmup",
                        choices=["linear_schedule_with_warmup", "cosine_schedule_with_warmup",
                                 "constant_schedule_with_warmup"], help="learning rate scheduler")

    # For testing
    parser.add_argument("--limit_test_batches", type=float, default=1.0,
                        help="Number of batches to test in the test mode")
    parser.add_argument("--beam_size", type=int, default=1, help="size of beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="length penalty of generated text")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="repetition penalty of generated text")
    parser.add_argument("--renormalize_logits", action="store_true", help="renormalize logits")
    parser.add_argument("--mask_num", type=int, default=0, help="Number of masks in the input of summarization data")
    parser.add_argument("--test_batch_size", type=int, default=-1,
                        help="The batch size for test, used in few shot evaluation.")
    parser.add_argument("--apply_triblck", action="store_true",
                        help="Whether apply trigram block in the evaluation phase")
    parser.add_argument("--generation_early_stopping", action="store_true", help="early stopping for generation")
    parser.add_argument("--num_test_data", type=int, default=-1, help="The number of testing data")

    args = parser.parse_args()

    args.accum_batch = args.accum_data_per_step // args.batch_size
    print(args)

    if not os.path.exists(args.model_path):  # this is used to save the checkpoints, logs, and results
        os.makedirs(args.model_path)
    with open(os.path.join(args.model_path, "args_%s.json" % args.mode), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if args.debug:
        seed_everything(42, workers=True)

    if args.mode == "train":
        train(args)
    if args.mode == "test":
        test(args)
