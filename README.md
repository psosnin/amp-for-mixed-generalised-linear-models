# Approximate Message Passing Algorithms for Mixed Generalized Linear Models

This repository contains code implementing and testing Approximate Message Passing (AMP) algorithms for Generalized Linear Models (GLMs).

## Contributions

This project extends previous work on AMP algorithms for mixed GLMs [2] to four new generalised linear models:

1. Logistic Regression
2. Mixed Logistic Regression
3. Rectified Linear Regression
4. Mixed Rectified Linear Regression

Derivations and evaluation of these algorithms can be found in the thesis [here](Thesis.pdf).

## Overview

The code in this repository provides an implementation of the AMP algorithm for fitting GLMs. AMP is implemented for the following models:

- Linear Regression + Mixed Linear Regression
- Logistic Regression + Mixed Logistic Regression
- Rectified Linear Regression + Mixed Rectified Linear Regression

A detailed explanation of these models and the algorithms used can be found [here](Thesis.pdf).

The repository is organized as follows:

- `gamp/gamp.py`: This file contains the source code for the implementation of the AMP algorithm for non-mixed GLMs.

- `gamp/matrix_gamp.py`: This file contains the source code for the implementation of the AMP algorithm for mixed GLMs.

- `gamp/gk`: This directory contains code for the denoising functions gk used in the AMP algorithm for each of the GLMs.

- `gamp/generate_data.py`: Contains functions for generating data for each of the GLMs.

- `gamp/losses.py`: Contains loss functions and AMP state evolution loss prediction functions.

- `gamp/plotting.py`: Contains functions for plotting the results of the AMP algorithm.

- `gamp/run.py`: Contains functions to run trials of the AMP algorithm.

- `examples/`: This directory contains example scripts demonstrating how to use the AMP implementation.

## Getting Started

To get started, follow these steps:

1. Install the required dependencies. You can do this by navigating to the repository's root directory and running:

   ```
   pip install -r requirements.txt
   ```

2. Run the example scripts in the `examples/` directory to see how to use the AMP implementation for each of the GLMs.

## References and Acknowledgements

This project was undertaken as a research project for Part IIb of the Engineering Tripos, University of Cambridge. I would like to express gratitude to Professor Ramji Venkataramanan for his invaluable supervision and guidance throughout the project. 

The AMP algorithm for generalised linear models is based on [1]. The AMP algorithm for mixed generalised linear models and for mixed linear regression is based on [2].

[1] Feng, Oliver Y., Ramji Venkataramanan, Cynthia Rush, and Richard J. Samworth.
"A unifying tutorial on approximate message passing."
[Foundations and Trends in Machine Learning 15, no. 4 (2022): 335-536](https://www.nowpublishers.com/article/Details/MAL-092)

[2] Tan, N., & Venkataramanan, R. (2023).
"Mixed Regression via Approximate Message Passing."
[arXiv preprint arXiv:2304.02229](https://arxiv.org/abs/2304.02229)
