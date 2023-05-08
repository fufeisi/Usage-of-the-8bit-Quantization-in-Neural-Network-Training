# 8-bit Quantization in Neural Network Training
This repository contains scripts to reproduce experiments in the project "Usage of 8-bit Quantization in Neural Network Training". The project aims to explore the benefits of quantizing activation maps in neural network training.

## Contents
The repository includes two directories:
- 'ImageNet': This directory contains scripts for quantizing activation maps of ResNet18 or ResNet50 on the ImageNet dataset.
- 'GLUE': This directory contains scripts for quantizing activation maps of the RoBerta-large model on the GLUE dataset.

## Requirements
To run the scripts, please install the packages listed in requirements.txt using the following command:
### install the packages in requirements.txt
- pip install -r requirements.txt
For the ImageNet experiments, you also need to download the dataset and set the path using the --data flag.

## Run
To run the scripts, navigate to the relevant directory and use the following commands:
- For ImageNet experiments: $ cd ImageNet and $ sh {quan18, quan50}.sh to train ResNet18 or ResNet50 with quantized activation maps.
- For GLUE experiments: $ cd GLUE and $ sh quan.sh to fine-tune the RoBerta-large model on all GLUE tasks with quantized activation maps.

## Credit
- Some code in this repository is modified from [Transformers](https://github.com/huggingface/transformers). 
