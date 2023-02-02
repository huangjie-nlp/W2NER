
class Config():
    def __init__(self):
        self.bert_path = "bert-base-chinese"
        self.schema_fn = "dataset/resume-zh/schema.json"
        self.train_fn = "dataset/resume-zh/train.json"
        self.dev_fn = "dataset/resume-zh/dev.json"
        self.test_fn = "dataset/resume-zh/test.json"
        self.label_num = 10
        self.batch_size = 12
        self.dist_emb_size = 20
        self.type_emb_size = 20
        self.lstm_hid_size = 512
        self.conv_hid_size = 96
        self.bert_hid_size = 768
        self.biaffine_size = 512
        self.ffnn_hid_size = 288
        self.dilation = [1, 2, 3]
        self.step = 500

        self.emb_dropout = 0.5
        self.conv_dropout = 0.5
        self.out_dropout = 0.33

        self.epochs = 10
        self.learning_rate = 1e-3
        self.weight_decay = 0
        self.clip_grad_norm = 5.0

        self.bert_learning_rate = 5e-6
        self.warm_factor = 0.1

        self.use_bert_last_4_layers = True

        self.checkpoint = "checkpoint/model.pt"
        self.dev_result = "dev_result/dev.json"
        self.test_result = "test_result/test.json"
