#!usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from src.train import start_training
from src.utils.coloured import Separator
from src.utterance import start_chat


def main(args):
    """
    Main function run training or chat
    :param args: argparse
    :return: None
    """
    Separator(50, 'BEGIN')
    if args.choice == "train":
        start_training(args)
    elif args.choice == "chat":
        start_chat(args)
    elif args.choice == "server":
        pass
    Separator(51, 'END')


parser_obj = argparse.ArgumentParser()
parser_obj.add_argument('--choice', type=str, default='train', choices={'train', 'chat'}, help='Train or utterance.')
parser_obj.add_argument('--tf', type=str, default='cpu', choices={'gpu', 'cpu'}, help='Train on GPU or CPU.')
parser_obj.add_argument('--emb', type=int, default='100', help='Embedding size.')
parser_obj.add_argument('--hidden', type=int, default='256', help='Hidden GRU cells.')
parser_obj.add_argument('--epoch', type=int, default='2', help='Number of epochs.')
parser_obj.add_argument('--batch', type=int, default='64', help='Batch size.')
parser_obj.add_argument('--maxlen', type=int, default='30', help='Max number of words in sentence.')
parser_obj.add_argument('--rate', type=float, default='0.001', help='=Learning rate.')
parser_obj.add_argument('--dropout', type=float, default='0.9', help='Dropout.')
parser_obj.add_argument('--decrate', type=float, default='0.5', help='Decrease learning rate with multiplayer.')
parser_obj.add_argument('--decstep', type=int, default='10', help='Decrease learning rate after each 5 epochs.')
parser_obj.add_argument('--lin', type=str, default='data/corpus/lines.txt', help='Lines path.')
parser_obj.add_argument('--con', type=str, default='data/corpus/conversations.txt', help='Conversations path.')
parser_obj.add_argument('--model', type=str, default='data/model', help='Model path.')

if __name__ == '__main__':
    main(parser_obj.parse_args())
