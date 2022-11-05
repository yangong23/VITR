# CLIP-RR
The Implementation of CLIP-RR

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

* Punkt Sentence Tokenizer:

``` python
import nltk
nltk.download()
> d punkt
``` 

## Download data
Please download RefCOCOg images from [here](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) (Same images as MS-COCO), CLEVR images from [here](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), and Flikr30K images from [here](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).
