# Approximate Message Passing Algorithms for Mixed Generalized Linear Models

This repository contains code implementing and testing Approximate Message Passing (AMP) algorithms for Generalized Linear Models (GLMs).

## Overview

The code in this repository provides an implementation of the AMP algorithm for fitting GLMs. AMP is implemented for the following models:

- Linear Regression
- Mixed Linear Regression
- Logistic Regression
- Mixed Logistic Regression
- Rectified Linear Regression
- Mixed Rectified Linear Regression

The repository is organized as follows:

- `gamp/fitting`: This directory contains the source code for the implementation of the AMP algorithm for both mixed and non-mixed GLMs. Source code for other fitting algorithms such as Expectation-Maximisation and Alternating-Maximisation is also included as a comparison to AMP.

- `gamp/gk`: This directory contains code the denoising functions gk used in the AMP algorithm for each of the GLMs.

- `gamp/generate_data.py`: Contains functions for generating data for each of the GLMs.

- `gamp/loss.py`: Contains loss functions and AMP state evolution loss prediction functions.

- `gamp/plot.py`: Contains functions for plotting the results of the AMP algorithm.

- `examples/`: This directory contains example scripts demonstrating how to use the AMP implementation.

## Getting Started

To get started, follow these steps:

1. Install the required dependencies. You can do this by navigating to the repository's root directory and running:

   ```
   pip install -r requirements.txt
   ```

2. Run the example scripts in the `examples/` directory to see how to use the AMP implementation.
