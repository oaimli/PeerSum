# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from tag_model.modeling import TagConfig
from data_process.datasets import SenSequence, DocSequence, QuerySequence, QueryTagSequence, \
    DocTagSequence
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceScoreTag, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tag_model.tag_tokenization import TagTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from tag_model.tagging import get_tags, SRLPredictor
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
#csv.field_size_limit(sys.maxsize)
import sklearn.metrics as mtc
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, token_tag_sequence_a, token_tag_sequence_b, len_seq_a, len_seq_b, input_tag_ids, input_tag_verbs, input_tag_len, orig_to_token_split_idx, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.token_tag_sequence_a = token_tag_sequence_a
        self.token_tag_sequence_b = token_tag_sequence_b
        self.len_seq_a = len_seq_a
        self.len_seq_b = len_seq_b
        self.input_tag_ids = input_tag_ids
        self.input_tag_verbs = input_tag_verbs
        self.input_tag_len = input_tag_len
        self.orig_to_token_split_idx = orig_to_token_split_idx
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class STSProcessor(DataProcessor):
  """Processor for the WNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv_tag")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv_tag")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv_tag")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
          if i == 0:
            continue
          guid = "%s-%s" % (set_type, i)
          if set_type == "test":
            text_a = line[-2]
            text_b = line[-1]
            label = 0
          else:
            text_a = line[-3]
            text_b = line[-2]
            label = float(line[-1])
          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

tag_vocab = []
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, srl_predictor):
    """Loads a data file into a list of `InputBatch`s."""

    max_aspect = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = []
        tokens_b = []
        tok_to_orig_index_a = []  # subword_token_index -> org_word_index
        tag_sequence = get_tags(srl_predictor, example.text_a, tag_vocab)
        token_tag_sequence_a = QueryTagSequence(tag_sequence[0], tag_sequence[1])
        tokens_a_org = tag_sequence[0]
        if len(tag_sequence[1])> max_aspect:
            max_aspect = len(tag_sequence[1])
        tok_to_orig_index_a.append(0)  # [CLS]
        for (i, token) in enumerate(tokens_a_org):
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index_a.append(i + 1)
                tokens_a.append(sub_token)
        tok_to_orig_index_b = []  # subword_token_index -> org_word_index
        token_tag_sequence_b = None
        if example.text_b:
            tag_sequence = get_tags(srl_predictor, example.text_b, tag_vocab)
            token_tag_sequence_b = QueryTagSequence(tag_sequence[0], tag_sequence[1])
            tokens_b_org = tag_sequence[0]
            if len(tag_sequence[1]) > max_aspect:
                max_aspect = len(tag_sequence[1])
            for (i, token) in enumerate(tokens_b_org):
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index_b.append(i)
                    tokens_b.append(sub_token)
            #print(len(tokens_a+tokens_b), len(tokens_a),len(tokens_b))
            if len(tokens_a+tokens_b) > max_seq_length-3:
                print("too long!!!!",len(tokens_a+tokens_b), len(tokens_a),len(tokens_b))
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, tok_to_orig_index_a, tok_to_orig_index_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                print("too long!!!!", len(tokens_a))
                tokens_a = tokens_a[:(max_seq_length - 2)]
                tok_to_orig_index_a=tok_to_orig_index_a[:max_seq_length - 1] #already has the index for [CLS]
        tok_to_orig_index_a.append(tok_to_orig_index_a[-1] + 1)  # [SEP]
        over_tok_to_orig_index = tok_to_orig_index_a
        if  example.text_b:
            tok_to_orig_index_b.append(tok_to_orig_index_b[-1] + 1)  # [SEP]
            offset = tok_to_orig_index_a[-1]
            for org_ix in tok_to_orig_index_b:
                over_tok_to_orig_index.append(offset + org_ix + 1)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        len_seq_a = tok_to_orig_index_a[len(tokens)-1] + 1
        len_seq_b = None
        if  example.text_b:
            tokens += tokens_b + ["[SEP]"]
            len_seq_b = tok_to_orig_index_b[len(tokens_b)] + 1  #+1 SEP -1 for index
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        pre_ix = -1
        start_split_ix = -1
        over_token_to_orig_map_org = []
        for value in over_tok_to_orig_index:
            over_token_to_orig_map_org.append(value)
        orig_to_token_split_idx = []
        for token_ix, org_ix in enumerate(over_token_to_orig_map_org):
            if org_ix != pre_ix:
                pre_ix = org_ix
                end_split_ix = token_ix - 1
                if start_split_ix != -1:
                    orig_to_token_split_idx.append((start_split_ix, end_split_ix))
                start_split_ix = token_ix
        if start_split_ix != -1:
            orig_to_token_split_idx.append((start_split_ix, token_ix))
        while len(orig_to_token_split_idx) < max_seq_length:
            orig_to_token_split_idx.append((-1,-1))
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        score = example.label
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("score: %s (id = %d)" % (example.label, score))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              token_tag_sequence_a = token_tag_sequence_a,
                              token_tag_sequence_b = token_tag_sequence_b,
                              len_seq_a = len_seq_a,
                              len_seq_b = len_seq_b,
                              input_tag_ids = None,
                              input_tag_verbs = None,
                              input_tag_len = None,
                              orig_to_token_split_idx=orig_to_token_split_idx,
                              label_id=score))
    return features


def transform_tag_features(features, tag_tokenizer, max_seq_length):
    #now convert the tags into ids
    #print("vocab_size: ",len(tag_vocab))
    max_num_aspect = 3
    new_features = []
    for example in features:
        token_tag_sequence_a = example.token_tag_sequence_a
        len_seq_a = example.len_seq_a
        token_tag_sequence_a.aspect_padding(max_num_aspect)
        tag_ids_list_a = token_tag_sequence_a.convert_to_ids(tag_tokenizer)
        input_tag_ids = []
        if example.token_tag_sequence_b != None:
            token_tag_sequence_b = example.token_tag_sequence_b
            token_tag_sequence_b.aspect_padding(max_num_aspect)
            tag_ids_list_b = token_tag_sequence_b.convert_to_ids(tag_tokenizer)
            len_seq_b = example.len_seq_b
            input_que_tag_ids = []
            for idx, query_tag_ids in enumerate(tag_ids_list_a):
                query_tag_ids = [1] + query_tag_ids[:len_seq_a - 2] + [2] #CLS and SEP
                input_que_tag_ids.append(query_tag_ids)
                # construct input doc tag ids with same length as input ids
            for idx, doc_tag_ids in enumerate(tag_ids_list_b):
                tmp_input_tag_ids = input_que_tag_ids[idx]
                doc_input_tag_ids = doc_tag_ids[:len_seq_b - 1] + [2] #SEP
                input_tag_id = tmp_input_tag_ids + doc_input_tag_ids
                while len(input_tag_id) < max_seq_length:
                    input_tag_id.append(0)
                assert len(input_tag_id) == len(example.input_ids)
                input_tag_ids.append(input_tag_id)
        else:
            for idx, query_tag_ids in enumerate(tag_ids_list_a):
                query_tag_ids = [1] + query_tag_ids[:len_seq_a - 2] + [2] #CLS and SEP
                input_tag_id = query_tag_ids
                while len(input_tag_id) < max_seq_length:
                    input_tag_id.append(0)
                assert len(input_tag_id) == len(example.input_ids)
                input_tag_ids.append(input_tag_id)
                # construct input doc tag ids with same length as input ids
        example.input_tag_ids = input_tag_ids
        new_features.append(example)
    return new_features

def _truncate_seq_pair(tokens_a, tokens_b, tok_to_orig_index_a, tok_to_orig_index_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            tok_to_orig_index_a.pop()
        else:
            tokens_b.pop()
            tok_to_orig_index_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def Fscore(out, labels):
    outputs = np.argmax(out, axis=1)
    TP = np.sum(np.multiply(labels, outputs))
    FP = np.sum(np.logical_and(np.equal(labels, 0), np.equal(outputs, 1)))
    FN = np.sum(np.logical_and(np.equal(labels, 1), np.equal(outputs, 0)))
    TN = np.sum(np.logical_and(np.equal(labels, 0), np.equal(outputs, 0)))
    return TP, FP, FN, TN

def mcc(out, labels):
    return mtc.matthews_corrcoef(out, labels)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="glue_data/STS-B",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default="sts",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="sts",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--tagger_path", default=None, type=str,
                        help="tagger_path for predictions if needing real-time tagging. Default: None, by loading pre-tagged data")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "sts": STSProcessor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    #num_labels = num_labels_task[task_name]
    label_list = None
    # num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    if args.tagger_path != None:
        srl_predictor = SRLPredictor(args.tagger_path)
    else:
        srl_predictor = None
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    train_features = None
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer,srl_predictor=srl_predictor )
        #TagTokenizer.make_tag_vocab("tag_vocab", tag_vocab)
    tag_tokenizer = TagTokenizer("tag_vocab")
    vocab_size = len(tag_tokenizer.ids_to_tags)
    print("tokenizer vocab size: ", str(vocab_size))
    tag_config = TagConfig(tag_vocab_size=vocab_size,
                           hidden_size=10,
                           layer_num=1,
                           output_dim=10,
                           dropout_prob=0.1,
                           num_aspect=3)
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceScoreTag.from_pretrained(args.bert_model,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),tag_config=tag_config)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    best_epoch = 0
    best_result = 0.0
    if args.do_train:
        train_features = transform_tag_features(train_features, tag_tokenizer, args.max_seq_length)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
        all_start_end_idx = torch.tensor([f.orig_to_token_split_idx for f in train_features], dtype=torch.long)
        all_input_tag_ids = torch.tensor([f.input_tag_ids for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_end_idx, all_input_tag_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        eval_dataloader = None
        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            eval_examples = processor.get_dev_examples(args.data_dir)
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer,srl_predictor=srl_predictor )
            eval_features = transform_tag_features(eval_features, tag_tokenizer, args.max_seq_length)
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
            all_start_end_idx = torch.tensor([f.orig_to_token_split_idx for f in eval_features], dtype=torch.long)
            all_input_tag_ids = torch.tensor([f.input_tag_ids for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_end_idx, all_input_tag_ids, all_label_ids)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, start_end_idx, input_tag_ids,  label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, str(epoch)+"_pytorch_model.bin")
            if args.do_train:
                torch.save(model_to_save.state_dict(), output_model_file)
            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                model_state_dict = torch.load(output_model_file)
                predict_model = BertForSequenceScoreTag.from_pretrained(args.bert_model, state_dict=model_state_dict,tag_config=tag_config)
                predict_model.to(device)
                predict_model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                total_pred, total_labels = [],[]
                total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
                for input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    start_end_idx = start_end_idx.to(device)
                    input_tag_ids = input_tag_ids.to(device)
                    with torch.no_grad():
                        tmp_eval_loss = predict_model(input_ids, segment_ids, input_mask, start_end_idx, input_tag_ids, label_ids)
                        logits = predict_model(input_ids, segment_ids, input_mask, start_end_idx,  input_tag_ids, None)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    eval_loss += tmp_eval_loss.mean().item()
                    total_pred.extend(logits.squeeze().tolist())
                    total_labels.extend(label_ids.squeeze().tolist())
                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1
                del predict_model
                eval_loss = eval_loss / nb_eval_steps
                spear = spearmanr(total_pred, total_labels)
                pear = pearsonr(total_pred, total_labels)
                mse = mean_squared_error(total_pred, total_labels)
                loss = tr_loss / nb_tr_steps if args.do_train else None
                result = {'eval_loss': eval_loss,
                          'global_step': global_step,
                          'spearson':spear,
                          'pearson':pear,
                          'mse':mse,
                          'loss': loss}

                if pear[0] > best_result:
                    best_epoch = epoch
                    best_result = pear[0]

                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "a") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("Epoch: %s,  %s = %s", str(epoch), key, str(result[key]))
                        writer.write("Epoch: %s, %s = %s\n" % (str(epoch), key, str(result[key])))
            logger.info("best epoch: %s, result:  %s", str(best_epoch), str(best_result))

    if args.do_predict:
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer,srl_predictor=srl_predictor )
        eval_features = transform_tag_features(eval_features, tag_tokenizer, args.max_seq_length)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_start_end_idx = torch.tensor([f.orig_to_token_split_idx for f in eval_features], dtype=torch.long)
        all_input_tag_ids = torch.tensor([f.input_tag_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_end_idx,
                                  all_input_tag_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        output_model_file = os.path.join(args.output_dir, str(best_epoch)+ "_pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        predict_model = BertForSequenceScoreTag.from_pretrained(args.bert_model, state_dict=model_state_dict,tag_config=tag_config)
        predict_model.to(device)
        predict_model.eval()
        predictions = []
        for input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids in tqdm(
                eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            start_end_idx = start_end_idx.to(device)
            input_tag_ids = input_tag_ids.to(device)
            with torch.no_grad():
                logits = predict_model(input_ids, segment_ids, input_mask, start_end_idx, input_tag_ids, None)
            logits = logits.detach().cpu().numpy()
            for (i, prediction) in enumerate(logits):
                predict_label = prediction.squeeze()
                predictions.append(predict_label)

        output_test_file = os.path.join(args.output_dir, "pred_results.txt")
        index = 0
        with open(output_test_file, "a") as writer:
            writer.write("index" + "\t" + "prediction" + "\n")
            for pred in predictions:
                writer.write(str(index) + "\t" + str(pred) + "\n")
                index += 1

if __name__ == "__main__":
    main()
