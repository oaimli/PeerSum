import jsonlines
from torch.utils.data import DataLoader, Dataset
import torch
import random
from nltk.tokenize import sent_tokenize
from datasets import load_dataset, concatenate_datasets


def acceptance_categorical(acceptance):
    if "eject" in acceptance:
        return 0
    else:
        return 1


def document_type_to_categorical(source_document, paper_id):
    if source_document["writer"] == "paper_abstract":
        # paper abstract
        return 6
    else:
        writer = source_document["writer"]
        reply_to = source_document["reply_to"]
        if writer == "official_reviewer":
            if reply_to == paper_id:
                # official review
                return 0
            else:
                # official response
                return 1
        if writer == "public":
            if reply_to == paper_id:
                # public comment
                return 2
            else:
                # public response
                return 3
        if writer == "author":
            if reply_to == paper_id:
                # author general response
                return 4
            else:
                # author response
                return 5



class SummarizationDataset(Dataset):
    def __init__(
        self,
        dataset,
        dataset_name,
        documents_concatenation,
        tokenizer,
        max_input_len,
        max_output_len,
        mask_num=5,
        dataset_type="train",
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.documents_concatenation = documents_concatenation
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        # self.threadsep_official_token = "<thread-official-sep>"
        # self.threadsep_public_token = "<thread-public-sep>"
        # self.threadsep_author_token = "<thread-author-sep>"
        # self.threadsep_official_id = self.tokenizer.convert_tokens_to_ids(self.threadsep_official_token)
        # self.threadsep_public_id = self.tokenizer.convert_tokens_to_ids(self.threadsep_public_token)
        # self.threadsep_author_id = self.tokenizer.convert_tokens_to_ids(self.threadsep_author_token)
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        # single doc setting
        reviews = entry["reviews"]
        tgt = entry["meta_review"]
        acceptance = acceptance_categorical(entry["paper_acceptance"])
        paper_abstract = entry["paper_abstract"]

        # the count of all source documents, including the paper abstract
        source_documents_count = len(reviews) + 1
        avg_max_length = int(self.max_input_len / source_documents_count)

        # review: document_id, writer, comment, rating, confidence, reply_to
        source_documents = []
        for i, review in enumerate(reviews):
            source_document = {}
            source_document["writer"] = review["writer"]
            comment = review["comment"]
            comment = comment.replace("\n", " ")
            comment = " ".join(comment.split())
            source_document["input_ids"] = self.tokenizer.encode(
                comment,
                truncation=True,
                max_length=avg_max_length,
            )[1:-1]
            source_document["rating"] = review["rating"]
            source_document["confidence"] = review["confidence"]
            source_document["document_id"] = review["review_id"]
            source_document["reply_to"] = review["reply_to"]
            source_documents.append(source_document)


        paper_abstract_dict = {}
        paper_abstract_dict["writer"] = "paper_abstract"
        paper_abstract_ids = self.tokenizer.encode(
            paper_abstract,
            truncation=True,
            max_length=avg_max_length,
        )[1:-1]

        paper_abstract_dict["input_ids"] = paper_abstract_ids
        paper_abstract_dict["rating"] = -1
        paper_abstract_dict["confidence"] = -1
        paper_abstract_dict["reply_to"] = entry["paper_id"]
        source_documents.append(paper_abstract_dict)
        random.shuffle(source_documents)

        input_ids = []
        ratings = []
        confidences = []
        document_types = []
        for source_document in source_documents:
            # doc-sep
            input_ids.append(self.docsep_token_id)
            document_types.append(document_type_to_categorical(source_document, entry["paper_id"]))
            ratings.append(source_document["rating"])
            confidences.append(source_document["confidence"])

            # review or comment content
            source_document_input_ids = source_document["input_ids"]
            input_ids.extend(source_document_input_ids)
            ratings.extend([-1] * len(source_document_input_ids))
            confidences.extend([-1] * len(source_document_input_ids))
            document_types.extend([-1] * len(source_document_input_ids))

        input_ids = (
                [self.tokenizer.bos_token_id]
                + input_ids
                + [self.tokenizer.eos_token_id]
        )
        ratings = ([-1] + ratings + [-1])
        confidences = ([-1] + confidences + [-1])
        document_types = ([-1] + document_types + [-1])
        # print("####", len(input_ids), self.max_input_len, avg_max_length)


        output_ids = self.tokenizer.encode(
            tgt, truncation=True, max_length=self.max_output_len
        )

        if self.dataset_type == "train":
            return torch.tensor(input_ids), torch.tensor(output_ids), acceptance, torch.tensor(confidences), torch.tensor(ratings), torch.tensor(document_types)
        else:
            return torch.tensor(input_ids), torch.tensor(output_ids), acceptance, torch.tensor(confidences), torch.tensor(ratings), torch.tensor(document_types), tgt


def collate_fn(batch):
    # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level member variables
    if batch[0][0][-1].item() == 2:
        pad_token_id = (
            1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        )
    elif batch[0][0][-1].item() == 1:
        pad_token_id = (
            0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        )
    else:
        assert False
    train = True
    if len(batch[0]) == 7:
        train = False
        tgt = [item[6] for item in batch]
        batch = [item[:6] for item in batch]
    input_ids, output_ids, acceptances, confidences, ratings, reviewer_roles = list(zip(*batch))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    confidences = torch.nn.utils.rnn.pad_sequence(
        confidences, batch_first=True, padding_value=pad_token_id
    )
    ratings = torch.nn.utils.rnn.pad_sequence(
        ratings, batch_first=True, padding_value=pad_token_id
    )
    reviewer_roles = torch.nn.utils.rnn.pad_sequence(
        reviewer_roles, batch_first=True, padding_value=pad_token_id
    )
    output_ids = torch.nn.utils.rnn.pad_sequence(
        output_ids, batch_first=True, padding_value=pad_token_id
    )
    # print("*******", acceptances, type(input_ids))
    if train:
        return input_ids, output_ids, torch.tensor(acceptances), confidences.float(), ratings.float(), reviewer_roles
    else:
        return input_ids, output_ids, tgt, torch.tensor(acceptances), confidences.float(), ratings.float(), reviewer_roles


def get_dataloader_summ(args, tokenizer, split_name, num_workers, is_shuffle):
    dataset = []
    if split_name == "train":
        with jsonlines.open(args.data_path + '%s.json' % args.dataset_name) as reader:
            for line in reader:
                if line["label"] == "train":
                    dataset.append(line)
        if args.num_train_data > 0:
            dataset = random.sample(dataset, k=args.num_train_data)
        print("training data", len(dataset))
    if split_name == "test":
        with jsonlines.open(args.data_path + '%s.json' % args.dataset_name) as reader:
            for line in reader:
                if line["label"] == "test":
                    dataset.append(line)
        if len(dataset) > args.num_test_data > 0:
            random.seed(42)
            dataset = random.sample(dataset, k=args.num_test_data)
        print("test data", len(dataset))
    if split_name == "validation":
        with jsonlines.open(args.data_path + '%s.json' % args.dataset_name) as reader:
            for line in reader:
                if line["label"] == "val":
                    dataset.append(line)
        if len(dataset) > args.num_val_data > 0:
            dataset = random.sample(dataset, k=args.num_val_data)
        print("validation data", len(dataset))

    # dataset_all = load_dataset('json', data_files=args.data_path + '%s.json' % args.dataset_name, split='all')
    # print("%s all"%args.dataset_name, len(dataset_all), args.data_path)
    #
    # random.seed(args.rand_seed)# This is to control random selection of training and testing samples
    # dataset = []
    # if split_name == "train":
    #     dataset = dataset_all.filter(lambda s: s['label'] == 'train' and s['paper_score']!=-1)
    #     print("dataset train all", len(dataset))
    #     if args.num_train_data != -1 and 0 < args.num_train_data < len(list(dataset)):
    #         dataset = dataset.select(random.choices(range(len(dataset)), k=args.num_train_data))
    #         print("dataset train selected", len(dataset))
    # if split_name == "validation":
    #     dataset = dataset_all.filter(lambda s: s['label'] == ('val' or "validation") and s['paper_score']!=-1)
    #     if len(dataset)> args.num_val_data > 0:
    #         dataset = dataset.select(random.choices(range(len(dataset)), k=args.num_val_data))
    #     print("dataset validation", len(dataset))
    # if split_name == "test":
    #     dataset = dataset_all.filter(lambda s: s['label'] == 'test' and s['paper_score']!=-1)
    #     if len(dataset)> args.num_test_data > 0:
    #         dataset = dataset.select(random.choices(range(len(dataset)), k=args.num_test_data))
    #     print("dataset test selected", len(dataset))

    # random.seed(args.rand_seed)
    # if split_name == "train":
    #     peersum_train = load_dataset("oaimli/PeerSum", split="train")
    #     print("dataset train all", len(peersum_train))
    #     if args.num_train_data != -1 and 0 < args.num_train_data < len(peersum_train):
    #         dataset = peersum_train.select(random.choices(range(len(peersum_train)), k=args.num_train_data))
    #         print("dataset train selected", len(dataset))
    # if split_name == "validation":
    #     peersum_validation = load_dataset("oaimli/PeerSum", split="validation")
    #     if len(peersum_validation) > args.num_val_data > 0:
    #         dataset = peersum_validation.select(random.choices(range(len(peersum_validation)), k=args.num_val_data))
    #     print("dataset validation", len(dataset))
    # if split_name == "test":
    #     peersum_test = load_dataset("oaimli/PeerSum", split="test")
    #     if len(peersum_test) > args.num_test_data > 0:
    #         dataset = peersum_test.select(
    #             random.choices(range(len(peersum_test)), k=args.num_test_data))
    #     print("dataset validation", len(dataset))

    summarization_dataset = SummarizationDataset(
        dataset=dataset,
        dataset_name=args.dataset_name,
        documents_concatenation=args.documents_concatenation,
        tokenizer=tokenizer,
        max_input_len=args.max_length_input,
        max_output_len=args.max_length_tgt,
        mask_num=args.mask_num,
        dataset_type=split_name,
    )

    return DataLoader(
        summarization_dataset,
        batch_size=args.batch_size,
        shuffle=is_shuffle,
        num_workers=num_workers,
        # sampler=sampler,
        collate_fn=collate_fn,
    )