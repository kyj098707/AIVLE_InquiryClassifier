import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import transformers
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer,AutoConfig
from torchmetrics import Accuracy
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
class CustomDataset(Dataset):
    def __init__(self, dataframe,tokenizer, labels=None):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.labels = labels
        self.label2id = {'코드1':0, '코드2':1, '웹':2, '이론':3, '시스템 운영':4, '원격':5}
        self.max_len = 90

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text):
        tokens = self.tokenizer.tokenize(self.tokenizer.cls_token \
                                         + str(text) + self.tokenizer.sep_token)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids, len(ids)

    def _padding(self, ids):
        # padding with 'pad_token_id' of tokenizer
        while len(ids) < self.max_len:
            ids += [self.tokenizer.pad_token_id]

        if len(ids) > self.max_len:
            ids = ids[:self.max_len - 1] + [ids[-1]]
        return ids

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.label2id[self.labels[idx]]

        token_ids, ids_len = self._tokenize(text)
        token_ids = self._padding(token_ids)

        attention_masks = [float(id > 0) for id in token_ids]

        return token_ids, np.array(attention_masks), label


class InquiryClassifer(pl.LightningModule):
    def __init__(self,labels):
        super(InquiryClassifer, self).__init__()
        ckpt = "klue/roberta-base"
        config = AutoConfig.from_pretrained(ckpt,
                                            num_labels=6,
                                            id2label={str(i):label for i,label in enumerate(labels)},
                                            label2id={label:i for i,label in enumerate(labels)}
                                            )
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt,config=config)
        self.tokenizers = AutoTokenizer.from_pretrained(ckpt)
        self.accuracy = Accuracy
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
        dataset = pd.read_csv("./data/train.csv")

    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                labels=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.size(),dtype=torch.int).type_as(input_ids)
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels,
                            return_dict=True)
        return output

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch
        output = self(input_ids=input_ids, attention_mask=attention_mask,labels=label)
        probs = self.softmax(output.logits)

        self.log_dict({
            'train_loss' : self.criterion(probs,label),
        }, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch
        output = self(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        probs = self.softmax(output.logits)
        self.log_dict({
            'val_loss': self.criterion(probs, label),
        }, prog_bar=True, on_step=False, on_epoch=True)

    def validation_batch_end(self, batch, batch_idx):
        input_ids, attention_mask, label = batch
        output = self(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        probs = self.softmax(output.logits)
        self.log_dict({
            'train_loss': self.criterion(probs, label),
            'train_acc': self.accuracy(probs, label)
        }, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]

        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        dataframe = pd.read_csv("data/train.csv")
        self.train_set = CustomDataset(dataframe, self.tokenizers, dataframe['label'])
        train_dataloader = DataLoader(
            self.train_set, batch_size= 16, num_workers=0,
            shuffle=False, collate_fn=self._collate_fn
        )
        return train_dataloader

    def val_dataloader(self):
        dataframe = pd.read_csv("data/train.csv")
        self.val_set = CustomDataset(dataframe, self.tokenizers, dataframe['label'])
        val_dataloader = DataLoader(
            self.val_set, batch_size= 16, num_workers=0,
            shuffle=False, collate_fn=self._collate_fn
        )
        return val_dataloader

if __name__ == "__main__":
    model = InquiryClassifer(labels=['코드1','코드2','웹','이론','시스템 운영','원격'])
    model.train()
    trainer = Trainer(
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        logger=True,
        max_epochs=30,
        )
    trainer.fit(model)
