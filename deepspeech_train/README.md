DeepSpeech train ![](images/logo.png)
================
### Conda env:
```
git clone https://github.com/Martin1403/Tensorflow-1.1X.git && \
cd Tensorflow-1.1X/deepspeech_train/ && \
conda env create -n deepSpeech -f environment.yml && \
conda activate deepSpeech && \
conda install -c conda-forge deepspeech
```
###### Download [checkpoint](https://drive.google.com/file/d/1StbE-S2W-TH2_Y6Zg0f6BX3C52vRaKB_/view?usp=sharing), unpack and place inside directory.

### Run:
```
python train.py --epochs 3
```
```
./export.sh
```
```
python inference.py 
```