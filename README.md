<p align="center">
    <!-- <a href="https://circleci.com/gh/huggingface/transformers"> -->
        <img alt="TensorFlow Logo" src="https://www.tensorflow.org/images/tf_logo_horizontal.png">
    </a>
    <!-- <a href="https://github.com/huggingface/transformers/blob/master/LICENSE"> -->
        <img alt="Python" src="https://img.shields.io/badge/python-3.6%7C3.7%7C3.8-blue">
    </a>
    <!-- <a href="https://github.com/huggingface/transformers/blob/master/LICENSE"> -->
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-1.15-blue">
    </a>

</p>

## Introduction

This repo contains all codes of my master thesis
[Second Order training for Natural Language Processing using Newton-CG Optimizer](https://mediatum.ub.tum.de/1633374)

- preprocessing directory contains **raw text processing**, **tokenization**, **word embedding**, etc.
- nmt directory contains the training and evaluation processes of **Neural Machine Translation (NMT)** using **Transformer architecture**.
- plot directory contains the **loss and accuracy plot** in both training and evaluation. Also, the **BLUE** score bar chart is included.
- blue directory contains the calculation of **BLUE** score.

## Abstract

This thesis presents a **second-order optimizer** called **Newton-CG** to solve a **Portuguese to English Neural Machine Translation (NMT) task**
on the most dominant NMT model, **Transformer**. We mainly focus on comparing the performance between Newton-CG and two popular first-order optimizers,
Adam and Stochastic gradient descent(SGD). In our previous research, the Newton-CG has already gained speed-up and accuracy in image classification.
Besides, Newton-CG has shown **higher accuracy** than other first-order optimizers in senti- ment analysis on the Attention model.
In this NMT task, Newton-CG with pre-training **outperforms** others in BLEU scores and **overcomes the overfitting**.

## Citation

Here is my [master thesis](https://mediatum.ub.tum.de/1633374). Feel free to take a look and cite it :smiley:

```bibtex
@mastersthesis{ ,
	type = {Masterarbeit},
	author = {Yi-Han Hsieh},
	title = {Second Order training for Natural Language Processing using Newton-CG Optimizer},
	year = {2021},
	school = {Technical University of Munich},
	month = {Oct},
}
```