# Usage-of-the-8bit-Quantization-in-Neural-Network-Training
This repo has the script to reproduce the experiments in project 'Usage of the 8bit Quantization in Neural Network Training'.

## Contents
- 'ImageNet' directory is to quantize the activation maps for ResNet18 or ResNet50 on ImageNet dataset.
- 'GLUE' directory is to quantize the activation maps for RoBerta-large model on GLUE dataset.

## Requirements
### install the packages in requirements.txt
- pip install -r requirements.txt
### ImageNet only
- Download the ImageNet dataset and set --data to the path. 

## Run
- $cd ImageNet and $sh {quan18, quan50}.sh to train {ResNet18, ResNet50} with quantizing the activation maps.
- $cd GLUE and $sh quan.sh to fine-tune RoBerta-large model on all GLUE tasks with quantizing the activation maps.

## Credit
- Some code in this repository is modified from [Transformers](https://github.com/huggingface/transformers). 