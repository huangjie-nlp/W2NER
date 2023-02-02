from torch.utils.data import Dataset
import torch
import numpy as np
import json
from utils.utils import convert_index_to_text, parser_indx_to_text
from transformers import BertTokenizer

dis2idx = np.zeros((1000), dtype=np.int)
dis2idx[1:] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class NerDataset(Dataset):
    def __init__(self, config, file, is_test=False):
        self.config = config
        self.is_test = is_test
        with open(file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        with open(self.config.schema_fn, "r", encoding="utf-8") as fs:
            self.label2id = json.load(fs)[0]
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        input_ids、mask、mask2d、grid_label、dist_input、piece2word、length、ner
        """
        sample = self.data[idx]
        # sentence = self.tokenizer.tokenize(sample['sentence'])
        sentence = sample['sentence']
        tokens = [self.tokenizer.tokenize(word) for word in sentence]
        pieces = [piece for pieces in tokens for piece in pieces]
        # length = len(tokens)
        length = len(sentence)

        # bert input_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + pieces + ['SEP'])
        bert_input_length = len(input_ids)
        input_ids = np.array(input_ids)

        # bert mask
        mask = [1] * bert_input_length
        mask = np.array(mask)

        # mask2d
        mask2d = np.ones((length, length), dtype=np.bool)

        # dis_input
        dist_input = np.zeros((length, length), dtype=np.int)
        # 对角线为0，上三角为负数，下三角为正数
        for k in range(length):
            dist_input[k, :] += k
            dist_input[:, k] -= k

        for i in range(length):
            for j in range(length):
                if dist_input[i, j] < 0:
                    dist_input[i, j] = dis2idx[-dist_input[i, j]] + 9
                else:
                    dist_input[i, j] = dis2idx[dist_input[i, j]]
        dist_input[dist_input == 0] = 19

        # piece2word
        pieces2word = np.zeros((length, bert_input_length), dtype=np.bool)

        start = 0
        for i, pieces in enumerate(tokens):
            if len(pieces) == 0:
                continue
            pieces = list(range(start, start+len(pieces)))
            pieces2word[i, pieces[0]+1: pieces[-1]+2] = 1
            start += len(pieces)

        if not self.is_test:
            # grid_label
            grid_label = np.zeros((length, length), dtype=np.int)
            for entity in sample['ner']:
                index = entity['index']
                for i in range(len(index)):
                    if i + 1 >= len(index):
                        break
                    grid_label[index[i], index[i+1]] = 1
                grid_label[index[-1], index[0]] = self.label2id[entity['type']]

            # ner = set([convert_index_to_text(entity['index'], entity['type']) for entity in sample['ner']])
            # ner = list(ner)
            ner = parser_indx_to_text(sample["ner"], sample["sentence"])
        else:
            grid_label = np.zeros((length, length), dtype=np.int)
            ner = []

        return input_ids, mask, mask2d, grid_label, dist_input, pieces2word, length, ner, bert_input_length, sentence

def collate_fn(batch):
    input_ids, mask, mask2d, grid_label, dist_input, pieces2word, length, ner, bert_input_length, sentence = zip(*batch)

    cur_batch = len(batch)
    max_length = max(length)
    max_bert_input_len = max(bert_input_length)

    length = torch.LongTensor(length)

    batch_input_ids = torch.LongTensor(cur_batch, max_bert_input_len).zero_()
    batch_mask = torch.LongTensor(cur_batch, max_bert_input_len).zero_()
    batch_mask2d = torch.BoolTensor(cur_batch, max_length, max_length).zero_()
    batch_pieces2word = torch.BoolTensor(cur_batch, max_length, max_bert_input_len).zero_()
    batch_grid_label = torch.LongTensor(cur_batch, max_length, max_length).zero_()
    batch_dist_input = torch.LongTensor(cur_batch, max_length, max_length).zero_()

    for i in range(cur_batch):
        batch_input_ids[i, :bert_input_length[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_mask[i, :bert_input_length[i]].copy_(torch.from_numpy(mask[i]))
        batch_mask2d[i, :length[i], :length[i]].copy_(torch.from_numpy(mask2d[i]))
        batch_pieces2word[i, :length[i], :bert_input_length[i]].copy_(torch.from_numpy(pieces2word[i]))
        batch_grid_label[i, :length[i], :length[i]].copy_(torch.from_numpy(grid_label[i]))
        batch_dist_input[i, :length[i], :length[i]].copy_(torch.from_numpy(dist_input[i]))

    return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_mask,
            "mask2d": batch_mask2d,
            "grid_label": batch_grid_label,
            "dist_input": batch_dist_input,
            "pieces2word": batch_pieces2word,
            "length": length,
            "sentence": sentence,
            "ner": ner
          }
