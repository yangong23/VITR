# VITR
This is the implementation of VITR.

## Requirements and installation
We recommend the following dependencies.
* ubuntu (>=18.04)

* Python 3.8

* [PyTorch](https://pytorch.org/) (1.7.1)

* [NumPy](https://numpy.org/)

* [Pandas](https://pandas.pydata.org/)

* [scikit-learn](https://scikit-learn.org/stable/)

* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger) 

* [pycocotools](https://github.com/cocodataset/cocoapi) 

* pip install transformers

* pip install ftfy regex tqdm

* Punkt Sentence Tokenizer:

``` python
import nltk
nltk.download()
> d punkt
``` 

## Datasets
### 1 Please download images:

RefCOCOg images (2014 MS-COCO images) are from [here](https://cocodataset.org/#download) or
```
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/zips/test2014.zip
```

### 2 Please copy the downloaded images to the folder 'datasets' as follows.
```
datasets
├──RefCOCOg
│  ├── precomp
│  │   ├── ......
│  │   ├── images
│  │   │   ├── train2014
│  │   │   ├── val2014
│  │   │   ├── test2014
```

## VITR
```
cd VITR
```
### Train

```
python train.py --data_path $DATA_PATH --dataset $DATASET_NAME --logger_name runs/$DATASET_NAME/log --model_name runs/$DATASET_NAME/model --bert_size 768 --embed_size 2048
```

### Evaluation

```
evaluation.evalrank("./runs/$DATASET_NAME/model/model_best.pth.tar", data_path="$DATA_PATH", split="test")
```
