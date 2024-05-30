# VITR
This is the implementation of VITR.

## Datasets
Please download the dataset from [here](https://cocodataset.org/#download).

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

## VITR
```
cd VITR
```
### Evaluation

```
evaluation.evalrank("./runs/$DATASET_NAME/model/model_best.pth.tar", data_path="$DATA_PATH", split="test")
```
### Train

```
python train.py --data_path $DATA_PATH --dataset $DATASET_NAME --logger_name runs/$DATASET_NAME/log --model_name runs/$DATASET_NAME/model
```
