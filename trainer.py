import torch.optim as optim
import torch
import os
import pandas as pd
import logging
import time
from tqdm import tqdm
from transformers import AutoTokenizer
import statistics
import numpy as np

logger = logging.getLogger(__name__)


def all_metrics(y_true, y_pred, is_training=False):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training

    return f1.item(), precision.item(), recall.item(), tp.item(), tn.item(), fp.item(), fn.item()


class Trainer():
    def __init__(self, model, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.para_path)
        self.model = model
        self.metrics = {'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0}
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)
        # self.record_time = []

    def train(self, train_loader, dev_loader):
        for epoch in range(self.args.epoch):
            self.train_epoch(epoch, train_loader)
            self.eval_epoch(epoch, dev_loader)
            logging.info('Epoch %d finished' % epoch)

    def savemodel(self, k):
        # if not os.path.exists(os.path.join(self.args.savepath, self.args.dataset)):
        #     os.mkdir(os.path.join(self.args.savepath, self.args.dataset))
        torch.save({'state_dict': self.model.state_dict(),
                    'k': k,
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.args.savepath,
                                f'model_{k}.pth'))
        logger.info(f'save:{k}.pth')

    def train_epoch(self, epoch, train_loader):
        self.model.train()

        loss_num = 0.0
        pbar = tqdm(train_loader, total=len(train_loader))
        for i, (inputs, label) in enumerate(pbar):
            inputs = self.tokenizer(list(inputs), padding=True, return_tensors='pt', max_length=self.args.max_length,
                                    truncation=True)
            # ids = token['input_ids'].to(self.args.device)
            label = label.to(self.args.device)
            outputs = self.model(**inputs)
            loss = self.criterion(outputs, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_num += loss.item()
            pbar.set_description(f'epoch: {epoch}')
            pbar.set_postfix(index=i, loss=loss.sum().item())

        # self.savemodel(epoch)

    def eval_epoch(self, epoch, dev_loader):
        self.model.eval()


        y_preds = []
        y_trues = []
        pbar = tqdm(dev_loader, total=len(dev_loader))
        with torch.no_grad():
            for i, (inputs, label) in enumerate(pbar):
                inputs = self.tokenizer(list(inputs), padding=True, return_tensors='pt',
                                        max_length=self.args.max_length,
                                        truncation=True)
                # ids = token['input_ids'].to(self.args.device)
                # start_time = time.time()
                outputs = self.model(**inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                end_time = time.time()
                # self.record_time.append(end_time - start_time)
                y_preds.append(predicted.cpu().numpy())
                y_trues.append(label.cpu().numpy())

            # logger.info(f'running time is : {statistics.mean(self.record_time)}')

            y_preds = np.concatenate(y_preds, 0)
            y_trues = np.concatenate(y_trues, 0)

            if self.args.output_dim == 2:
                from sklearn.metrics import recall_score
                recall = recall_score(y_trues, y_preds)
                from sklearn.metrics import precision_score
                precision = precision_score(y_trues, y_preds)
                from sklearn.metrics import f1_score
                f1 = f1_score(y_trues, y_preds)
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(y_trues, y_preds)
            else:
                from sklearn.metrics import recall_score
                recall = recall_score(y_trues, y_preds, average='macro')
                from sklearn.metrics import precision_score
                precision = precision_score(y_trues, y_preds, average='macro')
                from sklearn.metrics import f1_score
                f1 = f1_score(y_trues, y_preds, average='macro')
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(y_trues, y_preds)



            result = {
                "eval_recall": float(recall),
                "eval_precision": float(precision),
                "eval_f1": float(f1),
                "eval_acc": float(acc)
            }
            logger.info(result)
            self.update_best_scores(epoch, result['eval_f1'], result['eval_precision'], result['eval_recall'],
                                    result['eval_acc'])

    def update_best_scores(self, epoch, f1, precision, recall, acc):
        if f1 > self.metrics['f1'] or precision > self.metrics['precision'] or recall > self.metrics['recall']:
            self.metrics['f1'] = f1
            self.metrics['precision'] = precision
            self.metrics['recall'] = recall
            self.metrics['acc'] = acc
            self.scores2file(epoch, f1, precision, recall, acc)

    def scores2file(self, epoch, f1, precision, recall, acc):
        save_path = self.args.savepath + f'/{self.args.model}_proidx_{self.args.project_idx}.csv'
        # add tp tn fp fn to self.matrix type dict
        _record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "acc": acc,
            "epoch": epoch,
            "args": self.args
        }
        result_df = pd.DataFrame(_record, index=[0])
        result_df.to_csv(save_path, mode='a', index=False, header=True)
