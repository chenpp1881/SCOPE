from torch.utils.data import DataLoader, Dataset
import logging
import re
import csv
import json
from sklearn.model_selection import train_test_split
import pandas as pd
import glob

logger = logging.getLogger(__name__)


def code_handl(code):
    unuse_item = ['\t', '\n', '  ', '  ']
    for item in unuse_item:
        code = code.replace(item, ' ')
    return code


class ContractDataSet(Dataset):
    def __init__(self, data, label):
        super(ContractDataSet, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], int(self.label[idx])


def load_dev_data():
    all_label = []
    all_data = []
    with open(r'./dataset/Devign/function.json', 'r+', encoding='utf-8') as f:
        datas = json.loads(f.read())

    for data in datas:
        all_data.append(code_handl(data['func']))
        all_label.append(data['target'])

    return all_data, all_label


def load_rev_data():
    all_label = []
    all_data = []
    with open(r'./dataset/Reveal/vulnerables.json', 'r', encoding='utf-8') as f:
        p_datas = json.loads(f.read())
    with open(r'./dataset/Reveal/non-vulnerables.json', 'r', encoding='utf-8') as f:
        n_datas = json.loads(f.read())

    # iter dic
    for p_data in p_datas:
        all_data.append(code_handl(p_data['code']))
        all_label.append(1)
    for n_data in n_datas:
        all_data.append(code_handl(n_data['code']))
        all_label.append(0)

    return all_data, all_label


def load_big_data():
    all_label = []
    all_data = []
    with open(r'./dataset/BigVul/function.json', 'r+', encoding='utf-8') as f:
        datas = json.loads(f.read())

    for data in datas:
        all_data.append(code_handl(data['code']))
        all_label.append(data['label'])

    return all_data, all_label


class MyDataSet(Dataset):
    def __init__(self, args, stage):
        super(MyDataSet, self).__init__()
        self.args = args
        if args.project_idx == 0:
            if stage == 'train':
                set_path = r'./dataset/Authorship/train.txt'
            if stage == 'test':
                set_path = r'./dataset/Authorship/valid.txt'
            with open(set_path, 'r', encoding='utf-8') as f:
                self.datas = f.readlines()
        if args.project_idx == 1:
            if stage == 'train':
                set_path = r'./dataset/DefectPre/train.txt'
            if stage == 'test':
                set_path = r'./dataset/DefectPre/valid.txt'
            with open(set_path, 'r', encoding='utf-8') as f:
                self.datas = f.readlines()
        if args.project_idx == 2:
            if stage == 'train':
                set_path = r'./dataset/Java250/train.txt'
            if stage == 'test':
                set_path = r'./dataset/Java250/test.txt'
            with open(set_path, 'r', encoding='utf-8') as f:
                self.datas = f.readlines()
        if args.project_idx == 3:
            if stage == 'train':
                set_path = r'./dataset/Python800/train.txt'
            if stage == 'test':
                set_path = r'./dataset/Python800/test.txt'
            with open(set_path, 'r', encoding='utf-8') as f:
                self.datas = f.readlines()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        # if self.args.project_idx == 4:
        #     example = json.loads(self.datas[idx])
        #     return example['func'].replace("\\n", "\n").replace('\"', '"'), example['target']

        code, label = self.datas[idx].split('<CODESPLIT>')
        return code, int(label)


class CIDataset(Dataset):

    def __init__(self, stage='train'):
        super(CIDataset, self).__init__()
        self.text_data = []
        self.label = []
        self.stage = stage
        # the data reading part
        if stage == 'train':
            text_data = json.load(open('./dataset/Dataset/data/train_fail.json', 'r'))
            feature_data = pd.read_csv('./dataset/Dataset/data/train_fail.csv', index_col=0)
            self.text_data.extend(text_data)
            self.label.extend(feature_data.iloc[:, -1].to_list())
            # text_data = json.load(open('./dataset/Dataset/data/train_pass.json', 'r'))
            # feature_data = pd.read_csv('./dataset/Dataset/data/train_pass.csv', index_col=0)
            # self.text_data.extend(text_data)
            # self.label.extend(feature_data.iloc[:, -1].to_list())

        elif stage == 'test':
            text_data = json.load(open('./dataset/Dataset/data/test_fail.json', 'r'))
            feature_data = pd.read_csv('./dataset/Dataset/data/test_fail.csv', index_col=0)
            self.text_data.extend(text_data)
            self.label.extend(feature_data.iloc[:, -1].to_list())
            # text_data = json.load(open('./dataset/Dataset/data/test_pass.json', 'r'))
            # feature_data = pd.read_csv('./dataset/Dataset/data/test_pass.csv', index_col=0)
            # self.text_data.extend(text_data)
            # self.label.extend(feature_data.iloc[:, -1].to_list())

        print(len(self.text_data), len(self.label))
        self.label = [1 if item == 0 else 0 for item in self.label]
        print(len(self.text_data), sum(self.label))
        # open csv file to save the label
        filename = f'./Results/{stage}_labels.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['label'])

            for i in range(len(self.label)):
                writer.writerow([int(self.label[i])])

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        return self.text_data[idx], int(self.label[idx])


def load_data(args):
    project = ['Authorship', 'DefectPrediction', 'Java250', 'Python800', 'BigVul', 'Devign', 'Reveal', 'CIbuild']
    if args.project_idx in [0, 1, 2, 3]:
        train_set = MyDataSet(args, 'train')
        test_set = MyDataSet(args, 'test')
        logger.info(f'loadding peoject:{project[args.project_idx]}!')
    elif args.project_idx == 7:
        train_dataset, test_dataset = CIDataset(stage='train'), CIDataset(stage='test')
        return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8), DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=8)
    else:
        # BigVul, Devign, Reveal
        if args.project_idx == 4:
            all_data, all_label = load_big_data()
        elif args.project_idx == 5:
            all_data, all_label = load_dev_data()
        elif args.project_idx == 6:
            all_data, all_label = load_rev_data()

        train_data, test_data, train_label, test_label = train_test_split(all_data, all_label, test_size=0.2,
                                                                          shuffle=True)
        p_num = sum(train_label)
        n_num = len(train_label) - p_num
        weight = n_num / p_num
        logging.info(f'The number of positive sample : {p_num}; The number of negtive sample : {n_num};')
        logging.info(f'The proportion of positive and negative samples is {weight}')
        if args.oversample == True:
            logger.info('Sampling strategy: oversample')
            positive_data = [x[0] for x in zip(train_data, train_label) if x[1] == 1]
            positive_data = positive_data * int(weight)
            positive_label = [1] * len(positive_data)
            train_data.extend(positive_data)
            train_label.extend(positive_label)

        train_set = ContractDataSet(train_data, train_label)
        test_set = ContractDataSet(test_data, test_label)

        logger.info(f'{project[args.project_idx]} dataset loaded, total data: {len(all_data)}')

    return DataLoader(train_set, batch_size=args.batch_size, shuffle=True), DataLoader(test_set,
                                                                                       batch_size=args.batch_size,
                                                                                       shuffle=True)


# def remove_javacomments(java_code):
#     pattern = r"(//.*?$|/\*.*?\*/|/\*(.|\n)*?\*/)"
#     code_without_comments = re.sub(pattern, "", java_code, flags=re.S | re.M)
#     return code_without_comments
#
#
# def remove_Ccomments(c_code):
#     pattern = r"/\*.*?\*/|//.*?$"
#     code_without_comments = re.sub(pattern, "", c_code, flags=re.S | re.M)
#     return code_without_comments
#
#
# def remove_pycomments(python_code):
#     pattern = r"(#.*?$|'''(.*?)'''|\"\"\"(.*?)\"\"\")"
#     code_without_comments = re.sub(pattern, lambda match: match.group(2) or match.group(3) or '', python_code,
#                                    flags=re.S | re.M)
#     return code_without_comments


if __name__ == '__main__':
    pass
