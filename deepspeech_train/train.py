#!usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import shutil

from src.utils.coloured import Print
from src.train import start_training
from src.manage import make_train_command


def main(args):
    """
    Main function run training and export normal
    :param args: argparse
    :return: None
    """
    shutil.rmtree('graph', ignore_errors=True)
    Print(f'b([INFO]) w(Start training.)')
    command = make_train_command(args.epochs).split()
    start_training(command)


parser_obj = argparse.ArgumentParser()
parser_obj.add_argument('--epochs', type=int, default=1, help='How many epochs.')
parser_obj.add_argument('--gpu', type=bool, default=True, help='If False train on cpu.')

if __name__ == '__main__':
    main(parser_obj.parse_args())
