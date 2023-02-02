from models.model import W2NER
import torch
from torch.utils.data import DataLoader
from dataloader.dataloader import NerDataset, collate_fn
import json
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from utils.utils import decode, parser_indx_to_text

class Framework(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(self.config.schema_fn, "r", encoding="utf-8") as f:
            self.id2label = json.load(f)[1]

    def train(self):

        dataset = NerDataset(self.config, self.config.train_fn)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.config.batch_size, collate_fn=collate_fn)

        dev_dataset = NerDataset(self.config, self.config.dev_fn)
        dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn)

        model = W2NER(self.config).to(self.device)
        bert_params = set(model.bert.parameters())
        other_params = list(set(model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.config.bert_learning_rate,
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': self.config.learning_rate,
             'weight_decay': self.config.weight_decay}
        ]

        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        updates_total = len(dataloader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.config.warm_factor * updates_total,
                                                    num_training_steps=updates_total)

        loss_fn = torch.nn.CrossEntropyLoss()

        global_step, global_loss = 0, 0
        best_epoch = 0
        best_f1, recall, precision = 0, 0, 0

        for epoch in range(self.config.epochs):
            for data in tqdm(dataloader):
                logits = model(data)

                optimizer.zero_grad()
                grid_mask2d = data['mask2d'].clone()
                loss = loss_fn(logits[grid_mask2d].to(self.device), data['grid_label'][grid_mask2d].to(self.device))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_grad_norm)
                optimizer.step()
                scheduler.step()

                global_loss += loss.item()

                if global_step % self.config.step == 0:
                    print("epoch: {} global_step: {} global: {:5.4f}".format(epoch, global_step, global_loss))
                    global_loss = 0
                global_step += 1

            # if epoch % 5 == 0:
            r, p, f1_score, predict = self.evaluate(model, dev_dataloader)
            if f1_score > best_f1:
                json.dump(predict, open(self.config.dev_result, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
                best_f1 = f1_score
                best_epoch = epoch
                recall = r
                precision = p
                print("epcoh: {} precision: {:5.4f} recall: {:5.4f} best_f1: {:5.4f}".format(best_epoch, recall, precision, best_f1))
                print("save model......")
                torch.save(model.state_dict(), self.config.checkpoint)
        print("epcoh: {} precision: {:5.4f} recall: {:5.4f} best_f1: {:5.4f}".format(best_epoch, recall, precision,
                                                                                     best_f1))

    def evaluate(self, model, dataloader):

        model.eval()
        predict = []
        predict_num, correct_num, gold_num = 0, 0, 0

        print("eval......")
        with torch.no_grad():
            for data in tqdm(dataloader):
                logtis = model(data)
                output = logtis.cpu().argmax(dim=-1)
                sentence = data["sentence"][0]
                _, pred = decode(output, data["length"], sentence, self.id2label)
                # target = parser_indx_to_text(data['ner'], data['sentence'])
                target = data["ner"][0]
                predict_num += len(pred)
                gold_num += len(target)
                correct_num += len(set(pred) & set(target))
                lack = set(target) - set(pred)
                new = set(pred) - set(target)
                predict.append({"sentence": ''.join([str(i) for i in sentence]), 'gold': target,
                                "predict": pred, "lack": list(lack), "new": list(new)})

        recall = correct_num / (gold_num + 1e-10)
        precision = correct_num / (predict_num + 1e-10)
        f1_score = 2 * recall * precision / (recall + precision + 1e-10)
        print("predict_num: {} gold_num: {} correct_num: {}".format(predict_num, gold_num, correct_num))
        print("recall: {:5.4f} precision: {:5.4f} f1_score: {:5.4f}".format(recall, precision, f1_score))

        model.train()

        return recall, precision, f1_score, predict

    def test_all(self, file=None):
        if file != None:
            dataset = NerDataset(self.config, file, is_test=True)
            dataloader = DataLoader(dataset, shuffle=True, batch_size=1, collate_fn=collate_fn)

        else:
            dataset = NerDataset(self.config, self.config.test_fn)
            dataloader = DataLoader(dataset, shuffle=True, batch_size=1, collate_fn=collate_fn)

        model = W2NER(self.config)
        model.load_state_dict(torch.load(self.config.checkpoint, map_location=self.device))
        model.to(self.device)
        model.eval()
        recall, precision, f1_score, predict = self.evaluate(model, dataloader)

        if file != None:
            name = file.split("/")[-1]
            json.dump(predict, open("test/"+name, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        else:
            json.dump(predict, open(self.config.test_result, "w", encoding="utf-8"), indent=4, ensure_ascii=False)


