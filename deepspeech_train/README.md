DeepSpeech train ![](images/logo.png)
================
### Conda env:
```shell
conda env create -n deepSpeech -f environment.yml && \
conda activate deepSpeech && \
conda install -c conda-forge deepspeech
```
### Unpack:
- ###### Download [link](https://drive.google.com/file/d/1StbE-S2W-TH2_Y6Zg0f6BX3C52vRaKB_/view?usp=sharing).

### Train:
```shell
python audio.py \
--wavs checkpoint/LJSpeech-1.1/wavs/ \
--meta checkpoint/LJSpeech-1.1/metadata.csv
python train.py --epochs 3
./export.sh
python inference.py 
```
### Train CPU:
###### ./src/manage.py
```python
def make_train_command(epochs):
    command = f'--train_files audio/train.csv ' \
              f'--test_files audio/test.csv ' \
              f'--dev_files audio/train.csv ' \
              f'--alphabet_config_path checkpoint/alphabet.txt ' \
              f'--checkpoint_dir checkpoint/ ' \
              '--learning_rate 0.0001 ' \
              '--n_hidden 2048 ' \
              '--test_batch_size 2 ' \
              f'--epochs {epochs} ' \
              '--load_cudnn False' <-- HERE
    return command
```
