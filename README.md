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

* pip install ftfy regex tqdm

* Punkt Sentence Tokenizer:

``` python
import nltk
nltk.download()
> d punkt
``` 
cd CLIP
python3 setup.py install

## Datasets
Please download RefCOCOg images from [here](https://cocodataset.org/#download) (2014 MS-COCO images), CLEVR v1.0 images from [here](https://cs.stanford.edu/people/jcjohns/clevr/), and Flikr30K images from [here](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), then copy images to follows.
RefCOCOg: datasets/RefCOCOg/precomp/images/train2014 and val2014
CLEVR: datasets/CLEVR/precomp/images/train, val, and test
F30K: datasets/RefCOCOg/precomp/images





