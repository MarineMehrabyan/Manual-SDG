# Manual-SDG
This repository contains a Python implementation of a simple neural network trained using Stochastic Gradient Descent (SGD) from scratch. The neural network is built using NumPy and Pandas libraries, providing a foundational understanding of the underlying principles of neural networks and optimization algorithms.


### 1. Data Preprocessing:
- **Normalization**: 
  - ![equation](https://latex.codecogs.com/svg.latex?X%20%3D%20%5Cfrac%7BX%20-%20%5Ctext%7Bmean%7D%28X%29%7D%7B%5Ctext%7Bstd%7D%28X%29%7D)

### 2. Initialization:
- **Weights and Biases Initialization**: 
 - **Weights Initialization**: 
  - ![equation](https://latex.codecogs.com/svg.latex?W_%7B%5Ctext%7Binput_hidden%7D%7D) is initialized with random values.
  - ![equation](https://latex.codecogs.com/svg.latex?b_%7B%5Ctext%7Bhidden%7D%7D) is initialized as a vector of zeros.
  - ![equation](https://latex.codecogs.com/svg.latex?W_%7B%5Ctext%7Bhidden_output%7D%7D) is initialized with random values.
  - ![equation](https://latex.codecogs.com/svg.latex?b_%7B%5Ctext%7Boutput%7D%7D) is initialized as a vector of zeros.

### 3. Forward Pass:
- **Input to Hidden Layer**:
  - ![equation](https://latex.codecogs.com/svg.latex?%5Ctext%7Bhidden_output%7D%20%3D%20X%20%5Ccdot%20W_%7B%5Ctext%7Binput_hidden%7D%7D%20&plus;%20b_%7B%5Ctext%7Bhidden%7D%7D)
- **Activation Function (ReLU)**:
  - ![equation](https://latex.codecogs.com/svg.latex?%5Ctext%7Bhidden_activation%7D%20%3D%20%5Ctext%7Bmax%7D%280%2C%20%5Ctext%7Bhidden_output%7D%29)
- **Hidden to Output Layer**:
  - ![equation](https://latex.codecogs.com/svg.latex?%5Ctext%7Boutput%7D%20%3D%20%5Ctext%7Bhidden_activation%7D%20%5Ccdot%20W_%7B%5Ctext%7Bhidden_output%7D%7D%20&plus;%20b_%7B%5Ctext%7Boutput%7D%7D)

### 4. Loss Calculation:
- **Mean Squared Error (MSE)**:
  - ![equation](https://latex.codecogs.com/svg.latex?%5Ctext%7BLoss%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28%5Ctext%7Bpredictions%7D_i%20-%20%5Ctext%7Btargets%7D_i%29%5E2)

### 5. Backpropagation:
- **Output Layer Error**:
  - ![equation](https://latex.codecogs.com/svg.latex?%5Ctext%7Boutput_error%7D%20%3D%20%5Cfrac%7B2%7D%7BN%7D%20%28%5Ctext%7Bpredictions%7D%20-%20%5Ctext%7Btargets%7D%29)
- **Gradient Calculation for Hidden to Output Weights and Biases**:
  - ![equation](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20%5Ctext%7BLoss%7D%7D%7B%5Cpartial%20W_%7B%5Ctext%7Bhidden_output%7D%7D%7D%7D%20%3D%20%5Ctext%7Bhidden_activation%7D%5ET%20%5Ccdot%20%5Ctext%7Boutput_error%7D)
  - ![equation](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20%5Ctext%7BLoss%7D%7D%7B%5Cpartial%20b_%7B%5Ctext%7Boutput%7D%7D%7D%7D%20%3D%20%5Csum%20%5Ctext%7Boutput_error%7D)

- **Hidden Layer Error**:
  - ![equation](https://latex.codecogs.com/svg.latex?%5Ctext%7Bhidden_error%7D%20%3D%20%5Ctext%7Boutput_error%7D%20%5Ccdot%20W_%7B%5Ctext%7Bhidden_output%7D%7D%5ET)
- **Gradient Calculation for Input to Hidden Weights and Biases**:
  - ![equation](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20%5Ctext%7BLoss%7D%7D%7B%5Cpartial%20W_%7B%5Ctext%7Binput_hidden%7D%7D%7D%7D%20%3D%20X%5ET%20%5Ccdot%20%5Ctext%7Bhidden_error%7D)
  - ![equation](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20%5Ctext%7BLoss%7D%7D%7B%5Cpartial%20b_%7B%5Ctext%7Bhidden%7D%7D%7D%7D%20%3D%20%5Csum%20%5Ctext%7Bhidden_error%7D)

