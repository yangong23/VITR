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
We used [anaconda](https://www.anaconda.com/) to manage the dependencies, and the code was runing on a NVIDIA TITAN or RTX 3080 GPU.

## Download data
The datasets of Flickr30K and MS-COCO can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC) or follow [VSRN](https://github.com/KunpengLi1994/VSRN). GSMN needs additional files which can be found at [GSMN](https://github.com/CrossmodalGroup/GSMN). The dataset of IAPR TC-12 can be downloaded from [here](https://drive.google.com/drive/folders/1wXY4nOqopy4H_9Rb6RByjCrK81WQ5Fnl). For VSEinfty_LSEH project change the folder names of all datasets as 'precomp', for other projects change the folder names of the datasets as 'f30k_precomp', 'coco_precomp', and 'iaprtc12_precomp' respectively.
