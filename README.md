# CLIP-RR
This is the implementation of CLIP-RR.

## Requirements and installation
We recommended the following dependencies.
* ubuntu (>=18.04)

* Python 3.8

* [PyTorch](https://pytorch.org/) (1.7.1)

* [NumPy](https://numpy.org/) (>=1.12.1)

* [Pandas](https://pandas.pydata.org/) (>=1.2.3)

* [scikit-learn](https://scikit-learn.org/stable/) (>=0.24.1)

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

* Install CLIP (please use CLIP in this repository):
``` 
cd CLIP
python3 setup.py install
```

## Datasets
### 1 Please download images:

RefCOCOg images (2014 MS-COCO images) are from [here](https://cocodataset.org/#download) or
```
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
```

CLEVR v1.0 images are from [here](https://cs.stanford.edu/people/jcjohns/clevr/)

Flikr30K images are from [here](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)

### 2 Please copy the downloaded images to the following paths.

datasets
├──RefCOCOg
│   ├── precomp
│   │      ├── ......
│   │      ├── images
│   │      │    ├── train2014
│   │      │    ├── val2014

RefCOCOg: $PATH/datasets/RefCOCOg/precomp/images/train2014 and val2014

CLEVR: $PATH/datasets/CLEVR/precomp/images/train, val, and test

F30K: $PATH/datasets/RefCOCOg/precomp/images

## CLIP
```
cd CLIP
```
### Train

RN101
```
python train.py --data_path $DATA_PATH --dataset $DATA_NAME --model RN101
```
ViT-B/16 or ViT-L/14
```
python train.py --data_path $DATA_PATH --dataset $DATA_NAME --model ViT-B/16 or ViT-L/14
```

Please notice: 

$DATA_PATH is: $PATH/datasets

$DATASET_NAME is: one of RefCOCOg, CLEVR, and F30K

### Evaluate CLIP, and extract the precomp features for the use of CLIP-RR

RN101 is required, please use one of ViT-B/16 and ViT-L/14.

RN101
```
python extractFeaturesImages.py --data_path $DATA_PATH --dataset $DATASET_NAME --model RN101
```

ViT-B/16 or ViT-L/14
```
python extractFeaturesImages.py --data_path $DATA_PATH --dataset $DATASET_NAME --model ViT-B/16 or ViT-L/14
```
```
python extractFeaturesTexts.py --data_path $DATA_PATH --dataset $DATASET_NAME --model ViT-B/16 or ViT-L/14
```
Once finised, please go to the folder 'CLIP/features', and copy all files into datasets/$DATASET_NAME/precomp

## CLIP-RR
```
cd CLIP-RR
```
### Train

B16
```
python train.py --data_path $DATA_PATH --dataset $DATASET_NAME --logger_name runs/$DATASET_NAME/log --model_name runs/$DATASET_NAME/model --bert_size 512 --embed_size 1024
```
L14
```
python train.py --data_path $DATA_PATH --dataset $DATASET_NAME --logger_name runs/$DATASET_NAME/log --model_name runs/$DATASET_NAME/model --bert_size 768 --embed_size 2048
```

### Evaluation

B16 or L14
```
evaluation.evalrank("./runs/$DATASET_NAME/model/model_best.pth.tar", data_path="$DATA_PATH", split="test")
```
