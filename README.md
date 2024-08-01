# VITR
This is the PyTorch code for implementing Vision Transformers with Relation-Focused Learning (VITR).

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

## Datasets and Models
The datasets and models can be downloaded from [here](https://drive.google.com/drive/folders/1_noi3665dify3VFVf7GuZCk5ZKdPp0X8?usp=drive_link).

### Evaluation

```
evaluation.evalrank("./runs/$DATASET_NAME/model/model_best.pth.tar", data_path="$DATA_PATH", split="test")
```
### Train

```
python train.py --data_path $DATA_PATH --dataset $DATASET_NAME --logger_name runs/$DATASET_NAME/log --model_name runs/$DATASET_NAME/model
```
# Citation
If you find this code useful for your research, please consider citing:
``` 
@article{gong2023vitr,
  title={VITR: Augmenting Vision Transformers with Relation-Focused Learning for Cross-Modal Information Retrieval},
  author={Gong, Yan and Cosma, Georgina and Axel, Finke},
  year={2024}
}
``` 
