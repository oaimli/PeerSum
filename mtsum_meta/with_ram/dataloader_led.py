import numpy as np
import jsonlines
from torch.utils.data import DataLoader, Dataset
import torch
import random
import pdb
from nltk.tokenize import sent_tokenize
from datasets import load_dataset, concatenate_datasets


def acceptance_categorical(acceptance):
    if "eject" in acceptance:
        return 0
    else:
        return 1


def document_type_to_categorical(source_document, paper_id):
    '''
    There are seven types of source documents in all
    '''
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


class DocumentsTree():
    '''
    each source document has the following attributes:
    - writer
    - input_ids
    - rating
    - confidence
    - document_id
    - reply_to
    '''

    def __init__(self, source_documents, paper_id):
        self.documents = source_documents
        self.documents_dict = {}
        for source_document in source_documents:
            source_document_id = source_document["document_id"]
            children_ids = []
            for tmp in source_documents:
                if tmp["reply_to"] == source_document_id:
                    children_ids.append(tmp["document_id"])
            source_document["children_ids"] = children_ids
            self.documents_dict[source_document_id] = source_document
        self.paper_id = paper_id

    def get_ancestors(self, document, n: int, ancestors):
        # n-step ancestors
        parent_document_id = document["reply_to"]
        if parent_document_id == self.paper_id or len(ancestors) == n:
            return ancestors
        else:
            document = self.documents_dict[parent_document_id]
            ancestors.append(document)
            return self.get_ancestors(document, n, ancestors)

    def distance(self, document_pre, document_suffix):
        distance = 1
        while document_suffix["reply_to"] != document_pre["document_id"]:
            distance += 1
            document_suffix = self.documents_dict[document_suffix["reply_to"]]
        return distance

    def get_descendants(self, document, max_levels: int):
        if max_levels == -1:
            # depth first, stack
            descendants = []
            tmp = []
            tmp.append(document)
            while len(tmp) > 0:
                document_tmp = tmp.pop()
                if document_tmp["document_id"] != document["document_id"]:
                    descendants.append(document_tmp)
                children_document_ids = document_tmp["children_ids"]
                for children_document_id in children_document_ids:
                    child_document = self.documents_dict[children_document_id]
                    tmp.append(child_document)
            return descendants
        else:
            # breadth first, queue
            descendants = []
            tmp = []
            tmp.append(document)
            while len(tmp) > 0:
                document_tmp = tmp.pop(0)
                if document_tmp["document_id"] != document["document_id"]:
                    if self.distance(document, document_tmp) > max_levels:
                        break
                    else:
                        descendants.append(document_tmp)
                children_document_ids = document_tmp["children_ids"]
                for children_document_id in children_document_ids:
                    child_document = self.documents_dict[children_document_id]
                    tmp.append(child_document)
            return descendants

    def get_siblings(self, document):
        document_id = document["document_id"]
        parent_document_id = document["reply_to"]
        if parent_document_id == self.paper_id:
            siblings = []
            for tmp_id in self.documents_dict.keys():
                tmp_document = self.documents_dict[tmp_id]
                if tmp_document["reply_to"] == self.paper_id and tmp_document["document_id"] != document_id:
                    siblings.append(tmp_document)
        else:
            all_siblings = self.documents_dict[parent_document_id]["children_ids"]
            siblings = []
            for id in all_siblings:
                if id != document_id:
                    siblings.append(self.documents_dict[id])
        return siblings

    def get_documents_within_same_thread(self):
        thread_starting_documents = []
        for document_id in self.documents_dict.keys():
            document = self.documents_dict[document_id]
            if document["reply_to"] == self.paper_id:
                thread_starting_documents.append(document)

        threads = []
        for thread_starting_document in thread_starting_documents:
            thread = [thread_starting_document]
            thread.extend(self.get_descendants(thread_starting_document, -1))
            threads.append(thread)

        return threads

    def get_document_level(self, document):
        ancestors = self.get_ancestors(document, -1, [])
        return len(ancestors) + 1

    def get_document_levels(self):
        # levels should be in the same order of source documents
        levels = []
        for document in self.documents:
            levels.append(self.get_document_level(document))
        return levels

    def get_document_relationships(self):
        '''
        there are eight relationships in total, represented with matrices:
        - Ancestor-1
        - Ancestor-n
        - Descendant-1
        - Descendant-n
        - Siblings (first-level review/comment, and also other sibling comments)
        - document self
        - within the same thread
        - all nodes

        relationships should be in the same order of source documents
        '''
        num_documents = len(self.documents)
        document_indexes = {}
        for i, document in enumerate(self.documents):
            document_indexes[document["document_id"]] = i

        document_relationship_matrices = []
        num_all_items = num_documents * num_documents

        # ancestor-1
        m = np.zeros(num_all_items).reshape((num_documents, num_documents))
        for document in self.documents:
            document_id = document["document_id"]
            ancestors_1 = self.get_ancestors(document, 1, [])
            for ancestor in ancestors_1:
                ancestor_id = ancestor["document_id"]
                # print(document_indexes[document_id])
                # print(document_indexes[ancestor_id])
                m[document_indexes[document_id]][document_indexes[ancestor_id]] = 1
        document_relationship_matrices.append(m.tolist())

        # ancestor-n
        m = np.zeros(num_all_items).reshape((num_documents, num_documents))
        for document in self.documents:
            document_id = document["document_id"]
            ancestors_n = self.get_ancestors(document, -1, [])
            for ancestor in ancestors_n:
                ancestor_id = ancestor["document_id"]
                m[document_indexes[document_id]][document_indexes[ancestor_id]] = 1
        document_relationship_matrices.append(m.tolist())

        # descendant-1
        m = np.zeros(num_all_items).reshape((num_documents, num_documents))
        for document in self.documents:
            document_id = document["document_id"]
            descendants_1 = self.get_descendants(document, 1)
            for descendant in descendants_1:
                descendant_id = descendant["document_id"]
                m[document_indexes[document_id]][document_indexes[descendant_id]] = 1
        document_relationship_matrices.append(m.tolist())

        # descendant-n
        m = np.zeros(num_all_items).reshape((num_documents, num_documents))
        for document in self.documents:
            document_id = document["document_id"]
            descendants_n = self.get_descendants(document, -1)
            for descendant in descendants_n:
                descendant_id = descendant["document_id"]
                m[document_indexes[document_id]][document_indexes[descendant_id]] = 1
        document_relationship_matrices.append(m.tolist())

        # siblings
        m = np.zeros(num_all_items).reshape((num_documents, num_documents))
        for document in self.documents:
            document_id = document["document_id"]
            siblings = self.get_siblings(document)
            for sibling in siblings:
                sibling_id = sibling["document_id"]
                m[document_indexes[document_id]][document_indexes[sibling_id]] = 1
        document_relationship_matrices.append(m.tolist())

        # document self
        m = np.zeros(num_all_items).reshape((num_documents, num_documents))
        for i in range(len(self.documents)):
            m[i][i] = 1
        document_relationship_matrices.append(m.tolist())

        # within the same thread
        m = np.zeros(num_all_items).reshape((num_documents, num_documents))
        threads = self.get_documents_within_same_thread()
        for thread in threads:
            for document_i in thread:
                document_i_id = document_i["document_id"]
                for document_j in thread:
                    document_j_id = document_j["document_id"]
                    m[document_indexes[document_i_id]][document_indexes[document_j_id]] = 1
        document_relationship_matrices.append(m.tolist())

        # # all nodes
        # m = np.ones(num_all_items).reshape((num_documents, num_documents))
        # document_relationship_matrices.append(m.tolist())

        return document_relationship_matrices


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
        self.docsep_token_id = self.tokenizer.sep_token_id
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

        # set the abstract as a source document
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
        paper_abstract_dict["document_id"] = "abstract"
        source_documents.append(paper_abstract_dict)
        random.shuffle(source_documents)

        input_ids = []
        ratings = []
        confidences = []
        document_types = []
        document_startings = []
        document_endings = []
        for source_document in source_documents:
            # doc-sep
            document_startings.append(len(input_ids) + 1)
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
            document_endings.append(len(input_ids))

        input_ids = (
                [self.tokenizer.bos_token_id]
                + input_ids
                + [self.tokenizer.eos_token_id]
        )
        ratings = ([-1] + ratings + [-1])
        confidences = ([-1] + confidences + [-1])
        document_types = ([-1] + document_types + [-1])
        # print("####", len(input_ids), self.max_input_len, avg_max_length)

        document_tree = DocumentsTree(source_documents, entry["paper_id"])
        document_levels = document_tree.get_document_levels()
        document_relationship_matrices = document_tree.get_document_relationships()

        output_ids = self.tokenizer.encode(
            tgt, truncation=True, max_length=self.max_output_len
        )

        # print(source_documents)
        # print("length of input ids", len(input_ids))
        # print("startings", document_startings)
        # print("endings", document_endings)
        # print("document levels", document_levels)
        # print("document_relationship_matrices", document_relationship_matrices)
        # print("confidences", confidences)
        # print("ratings", ratings)
        # print("document_types", document_types)

        # added, document_startings, document_endings, document_levels, document_relationship_matrices
        if self.dataset_type == "train":
            return torch.tensor(input_ids), torch.tensor(document_startings), torch.tensor(
                document_endings), torch.tensor(document_levels), torch.tensor(
                document_relationship_matrices), torch.tensor(output_ids), acceptance, torch.tensor(
                confidences), torch.tensor(ratings), torch.tensor(document_types)
        else:
            return torch.tensor(input_ids), torch.tensor(document_startings), torch.tensor(
                document_endings), torch.tensor(document_levels), torch.tensor(
                document_relationship_matrices), torch.tensor(output_ids), acceptance, torch.tensor(
                confidences), torch.tensor(ratings), torch.tensor(document_types), tgt


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
    if len(batch[0]) == 11:
        train = False
        tgt_batch = [item[10] for item in batch]
        batch = [item[:10] for item in batch]
    input_ids_batch, document_startings_batch, document_endings_batch, document_levels_batch, document_relationship_matrices_batch, output_ids_batch, acceptances_batch, confidences_batch, ratings_batch, reviewer_roles_batch = list(
        zip(*batch))
    input_ids_batch = torch.nn.utils.rnn.pad_sequence(
        input_ids_batch, batch_first=True, padding_value=pad_token_id
    )
    confidences_batch = torch.nn.utils.rnn.pad_sequence(
        confidences_batch, batch_first=True, padding_value=pad_token_id
    )
    ratings_batch = torch.nn.utils.rnn.pad_sequence(
        ratings_batch, batch_first=True, padding_value=pad_token_id
    )
    reviewer_roles_batch = torch.nn.utils.rnn.pad_sequence(
        reviewer_roles_batch, batch_first=True, padding_value=pad_token_id
    )
    output_ids_batch = torch.nn.utils.rnn.pad_sequence(
        output_ids_batch, batch_first=True, padding_value=pad_token_id
    )
    # print("*******", acceptances, type(input_ids))
    if train:
        return input_ids_batch, document_startings_batch, document_endings_batch, document_levels_batch, document_relationship_matrices_batch, output_ids_batch, torch.tensor(
            acceptances_batch), confidences_batch.float(), ratings_batch.float(), reviewer_roles_batch
    else:
        return input_ids_batch, document_startings_batch, document_endings_batch, document_levels_batch, document_relationship_matrices_batch, output_ids_batch, tgt_batch, torch.tensor(
            acceptances_batch), confidences_batch.float(), ratings_batch.float(), reviewer_roles_batch


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


if __name__ == "__main__":
    # test tree algorithms to get the relationships, no root node
    #        0
    #      / | \
    #     1  2  3
    #     |  |
    #     4  5
    #       / \
    #      6   7
    #      |  / \
    #      8 9  10
    #            |
    #           11
    documents = []
    document_1 = {"document_id": "1", "reply_to": "root"}
    documents.append(document_1)
    document_2 = {"document_id": "2", "reply_to": "root"}
    documents.append(document_2)
    document_3 = {"document_id": "3", "reply_to": "root"}
    documents.append(document_3)
    document_4 = {"document_id": "4", "reply_to": "1"}
    documents.append(document_4)
    document_5 = {"document_id": "5", "reply_to": "2"}
    documents.append(document_5)
    document_6 = {"document_id": "6", "reply_to": "5"}
    documents.append(document_6)
    document_7 = {"document_id": "7", "reply_to": "5"}
    documents.append(document_7)
    document_8 = {"document_id": "8", "reply_to": "6"}
    documents.append(document_8)
    document_9 = {"document_id": "9", "reply_to": "7"}
    documents.append(document_9)
    document_10 = {"document_id": "10", "reply_to": "7"}
    documents.append(document_10)
    document_11 = {"document_id": "11", "reply_to": "10"}
    documents.append(document_11)
    documents_tree = DocumentsTree(documents, "root")
    print("1-hop ancestors of document_7", documents_tree.get_ancestors(document_7, 1, []))
    print("all ancestors of document_7", documents_tree.get_ancestors(document_7, -1, []))
    print("2-hop descendants of document_2", documents_tree.get_descendants(document_2, 2))
    print("all descendants of document_2", documents_tree.get_descendants(document_2, -1))
    print("siblings of document_2", documents_tree.get_siblings(document_2))
    print("siblings of document_9", documents_tree.get_siblings(document_9))
    print("get documents within same thread", documents_tree.get_documents_within_same_thread())
    print("get document_6 level", documents_tree.get_document_level(document_6))
    print(documents_tree.get_document_levels())
    print(documents_tree.get_document_relationships())
