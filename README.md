# Manual-SDG
This repository contains a Python implementation of a simple neural network trained using Stochastic Gradient Descent (SGD) from scratch. The neural network is built using NumPy and Pandas libraries, providing a foundational understanding of the underlying principles of neural networks and optimization algorithms.


Certainly! Let's break down the code into mathematical formulas and explanations:

### 1. Data Preprocessing:
- **Normalization**: 
  - X = (X - mean(X)) / std(X)

### 2. Initialization:
- **Weights and Biases Initialization**: 
  - W_{input\_hidden} is initialized with random values.
  - b_{hidden} is initialized as a vector of zeros.
  - W_{hidden\_output} is initialized with random values.
  - b_{output} is initialized as a vector of zeros.

### 3. Forward Pass:
- **Input to Hidden Layer**:
  - hidden_output = X \cdot W_{input\_hidden} + b_{hidden}
- **Activation Function (ReLU)**:
  - hidden_activation = max(0, hidden_output)
- **Hidden to Output Layer**:
  - output = hidden_activation \cdot W_{hidden\_output} + b_{output}

### 4. Loss Calculation:
- **Mean Squared Error (MSE)**:
  - Loss = (1/N) \sum_{i=1}^{N} (predictions_i - targets_i)^2

### 5. Backpropagation:
- **Output Layer Error**:
  - output_error = (2/N) (predictions - targets)
- **Gradient Calculation for Hidden to Output Weights and Biases**:
  - (∂Loss/∂W_{hidden\_output}) = hidden_activation^T \cdot output_error
  - (∂Loss/∂b_{output}) = sum(output_error)

- **Hidden Layer Error**:
  - hidden_error = output_error \cdot W_{hidden\_output}^T
- **Gradient Calculation for Input to Hidden Weights and Biases**:
  - (∂Loss/∂W_{input\_hidden}) = X^T \cdot hidden_error
  - (∂Loss/∂b_{hidden}) = sum(hidden_error)

### 6. Update Parameters:
- **Parameter Update using Gradient Descent**:
  - W_{input\_hidden} = W_{input\_hidden} - learning\_rate \times (∂Loss/∂W_{input\_hidden})
  - b_{hidden} = b_{hidden} - learning\_rate \times (∂Loss/∂b_{hidden})
  - W_{hidden\_output} = W_{hidden\_output} - learning\_rate \times (∂Loss/∂W_{hidden\_output})
  - b_{output} = b_{output} - learning\_rate \times (∂Loss/∂b_{output})

### 7. Inference:
- **Prediction**:
  - predictions = forward\_pass(X)

### 8. Evaluation:
- **Root Mean Squared Error (RMSE)**:
  - RMSE = sqrt((1/N) \sum_{i=1}^{N} (predictions_i - targets_i)^2)

This mathematical representation helps us understand the code in terms of the underlying operations and computations performed at each step of the neural network training process.
