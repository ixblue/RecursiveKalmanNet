# Recursive KalmanNet (RKN)

## About

We introduce **Recursive KalmanNet (RKN)**, a Kalman filter-informed recurrent neural network, designed to estimate both the state variables and the error covariance of a stochastic dynamical system from noisy measurements, **without requiring prior knowledge of the noise covariances**. This estimator preserves the structure of the Kalman filter while learning the gain and estimation error covariance through deep learning. It propagates the error covariance using the **recursive Joseph’s formula** and optimizes the **negative Gaussian log-likelihood**.

### Publications

- H. Mortada, C. Falcon, Y. Kahil, M. Clavaud, and J.-P. Michel,
  <a href="https://arxiv.org/abs/2506.11639">*Recursive KalmanNet: Deep Learning-Augmented Kalman Filtering for State Estimation with Consistent Uncertainty Quantification*</a>, EUSIPCO 2025.

- C. Falcon, H. Mortada, M. Clavaud, and J.-P. Michel,
  <a href="https://arxiv.org/abs/2507.14144">*Recursive KalmanNet : Analyse des capacités de généralisation d’un réseau de neurones récurrent guidé par un filtre de Kalman*</a>, GRETSI 2025.

## Python Version and Dependencies

This project was developed and tested using **Python 3.12.8**.  

Using the exact same version is not strictly necessary. You can check your Python version with:

> python --version

To create an environment

> python -m venv venv  
> venv\Scripts\activate     # On Mac, use: source venv/bin/activate  

To install all required dependencies, from the RecursiveKalmanNet folder:

> pip install -r requirements.txt

## Folder Architecture

- `.data/`: Stores all datasets used in the project.
- `.models/`: Contains saved trained models.
- `.results/`: Automatically saves training and validation loss curves, and may also include plots.
    - `.results/loss_curves/`: Saves loss curve values during training and validation.
    - `.results/plot_saves/`: Folder that can be specified to save generated plots.
- `Algo/`: Includes class definitions for dynamical systems, the Kalman filter, the Recursive KalmanNet, and loss functions.
- `Tools/`: Provides utility functions for data generation and plotting.

## Entry point

The `main_bimodal_noise.ipynb` notebook offers a practical illustration of how to use this codebase.  
The dynamical system is a one-dimensional constant-speed linear kinematic state-space model, with a position measurement. The measurement noise follows a gaussian bimodal distribution. The notebook covers:
- data generation according to the defined dynamical system
- demonstrates the use of the Kalman Filter
- demonstrates the instantiation, training, and usage of the Recursive Kalman Filter.

The notebook also includes several plots featured in the first paper listed above.







