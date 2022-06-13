#!usr/bin/python3
# -*- coding: utf-8 -*-
import os
from src.train import start_training
from src.manage import manage_output


def main() -> None:
    manage_output()
    os.makedirs('graph', exist_ok=True)
    command = [
        '--alphabet_config_path', 'checkpoint/alphabet.txt',
        '--checkpoint_dir', 'checkpoint/',
        '--export_dir', 'graph/']
    start_training(command)


if __name__ == '__main__':
    main()
