import argparse
import logging
import random
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from data import load_data
from trainer import Trainer
from model import *
from MHPrediction import SplitSelfAttentionWithHeads

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    return


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--savepath', type=str, default='./Results')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--output_dim', type=int)
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--project_idx', type=int, default=1)
    parser.add_argument('--max_length', type=int)
    parser.add_argument('--para_path', type=str)
    parser.add_argument('--oversample', action='store_true')
    parser.add_argument('--model', type=str, default='MHP',
                        choices=['GPA', 'CodeBert', 'CodeT5', 'Longformer', 'Unixcoder', 'MHP'])

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # parse agrs
    args = parse_args()
    logger.info(vars(args))
    # select device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device is %s', args.device)

    # set seed
    # set_seed(args.seed)

    # get data
    train_loader, test_loader = load_data(args)
    if args.project_idx == 0:
        args.output_dim = 66
    if args.project_idx == 1:
        args.output_dim = 4
    if args.project_idx == 2:
        args.output_dim = 250
    if args.project_idx == 3:
        args.output_dim = 800
    if args.project_idx in [4, 5, 6, 7]:
        args.output_dim = 2

    # get model
    if args.model == 'GPA':
        args.para_path = './model/CodeT5/'
        model = GPAResNet(d_model=200, ndead=5, batch_first=True, output_dim=args.output_dim)
        args.max_length = 6000
    elif args.model == 'CodeT5':
        args.para_path = './model/CodeT5/'
        model = CodeT5(args.output_dim)
        args.max_length = 1024
    elif args.model == 'CodeBert':
        args.para_path = './model/CodeBert/'
        model = CodeBert(args.output_dim)
        args.max_length = 512
    elif args.model == 'Unixcoder':
        args.para_path = './model/Unixcoder/'
        model = Unixcoder(args.output_dim)
        args.max_length = 512
    elif args.model == 'SCOPE':
        args.para_path = './model/CodeT5/'
        # model = SplitSelfAttentionWithHeads(split_n=8,num_classes=args.output_dim)
        model = SCOPE(d_model=200, ndead=20, output_dim=args.output_dim)
        args.max_length = 1024
    else:
        args.para_path = './model/Longformer/'
        model = Longformer(args.output_dim)
        args.max_length = 2048
    logger.info(f'Prediction model is {args.model}!~')
    model = torch.nn.DataParallel(model)
    model.to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'{total_params:,} total parameters.')
    trainer = Trainer(model, args)
    trainer.train(train_loader, test_loader)
