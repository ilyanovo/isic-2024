# isic-2024

This repo contains first place solution for **isic-2024** competition. The competition was hosted on [Kaggle](https://www.kaggle.com/c/isic-2024) and the task was to detect malignant skin lesions.

All training was done on a single NVIDIA A6000 GPU server with 24 cores and 64 GB RAM. All process was run inside docker container. (Ref image)[quay.io/jupyter/tensorflow-notebook:cuda-latest]

## Steps to reproduce

- Install dependences via `pip install -r requirements.txt`
- Pull competition and external data via data_pull.ipynb. Before that, make sure that kaggle package is instulled and configured.
- Train model on external data via runing NN-base-experiments-old-data.ipynb (also saves model predictions for competition data
- Train model on competition image data via runing NN-comp-experiments.ipynb. You need to run it twice, first time with MODEL_NAME = "EVA" and second with MODEL_NAME = "EDGENEXT"
- Run NN-competition-train.ipynb to train top level GBDT models

## Synthtic data
- Clone and configure (kohya-ss/sd-scripts)[https://github.com/kohya-ss/sd-scripts]
- To train SD model for synthtic data generation simply run sd_train.ipynb
- To generate synthtic data run synthetic-data-generation.ipynb
- NN-competition-train.ipynb contains code to train model on synthtic data (end of the notebook). But you can update it as well