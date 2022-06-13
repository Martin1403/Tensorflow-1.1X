Memory chatbot![](data/images/logo.png)
==============

### Run:
###### CPU/GPU
1. Create env:
```shell
conda create -n network python==3.6.9 tensorflow-gpu==1.15.0
conda activate network
python main.py --help
```
2. Unpack data:
```shell
tar -xvf data/corpus/conversations.tar.gz -C data/corpus/
```
3. Train:
```shell
python main.py --choice train
```
4. Test:
```shell
python main.py --choice chat
```