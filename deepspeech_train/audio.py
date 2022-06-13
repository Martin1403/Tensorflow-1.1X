#!usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import shutil

from src.text import make_train_csv, make_test_csv


def main(args):
    """
    Prepare audio for training, script will convert and process text.
    :param args: argparse
    :return: None
    """
    shutil.rmtree('audio/', ignore_errors=True)
    os.makedirs('audio/', exist_ok=True)
    make_train_csv(args.wavs, 'audio/', args.meta, 'audio/train.csv', 'checkpoint/buffer.wav')
    make_test_csv('audio/train.csv')


parser_obj = argparse.ArgumentParser(description="Prepare audio for training")
parser_obj.add_argument('--wavs', type=str, required=True, help='Path to wavs folder.')
parser_obj.add_argument('--meta', type=str, default=True, help='Path to metadata csv.')

if __name__ == '__main__':
    main(parser_obj.parse_args())
