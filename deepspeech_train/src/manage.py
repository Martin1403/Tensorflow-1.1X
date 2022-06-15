import os
import re


def manage_output():
    with open('checkpoint/best_dev_checkpoint', 'r', encoding='UTF-8') as r:
        line = [i for c, i in enumerate(r.read().splitlines()) if c == 0][0]
        res1 = re.sub('"', '', line.split(':')[-1].strip())
    with open('checkpoint/checkpoint', 'r', encoding='UTF-8') as r:
        line = [i for c, i in enumerate(r.read().splitlines()) if c == 0][0]
        res2 = re.sub('"', '', line.split(':')[-1].strip())
        with open('checkpoint/checkpoint', 'w', encoding='UTF-8') as w:
            w.writelines(f'{line}\n')
    for i in os.listdir('checkpoint/'):
        if (i.startswith(res1) or i.startswith(res2) or i in [
            'checkpoint', 'flags.txt', 'alphabet.txt', 'convert_graph', 'best_dev_checkpoint',
            'output_graph.scorer', 'LJSpeech-1.1', 'buffer.wav',
        ]):
            pass
        else:
            os.remove(f'checkpoint/{i}')


def make_train_command(epochs):
    command = f'--train_files train.csv ' \
              f'--test_files test.csv ' \
              f'--dev_files train.csv ' \
              f'--alphabet_config_path checkpoint/alphabet.txt ' \
              f'--checkpoint_dir checkpoint/ ' \
              '--learning_rate 0.0001 ' \
              '--n_hidden 2048 ' \
              '--test_batch_size 2 ' \
              f'--epochs {epochs} ' \
              '--load_cudnn False '
    return command


"""
def make_train_command(epochs):
    command = f'--train_files train.csv ' \
              f'--test_files test.csv ' \
              f'--dev_files train.csv ' \
              f'--alphabet_config_path checkpoint/alphabet.txt ' \
              f'--checkpoint_dir checkpoint/ ' \
              '--learning_rate 0.0001 ' \
              '--n_hidden 2048 ' \
              '--test_batch_size 2 ' \
              f'--epochs {epochs} ' \
              '--load_cudnn True ' \
              '--use_allow_growth True '
    return command
"""