# Manual-SDG
This repository contains a Python implementation of a simple neural network trained using Stochastic Gradient Descent (SGD) from scratch. The neural network is built using NumPy and Pandas libraries, providing a foundational understanding of the underlying principles of neural networks and optimization algorithms.


### 1. Data Preprocessing:
- **Normalization**: 
  - \( X = \frac{{X - \text{{mean}}(X)}}{{\text{{std}}(X)}} \)

### 2. Initialization:
- **Weights and Biases Initialization**: 
  - \( W_{\text{{input\_hidden}}} \) is initialized with random values.
  - \( b_{\text{{hidden}}} \) is initialized as a vector of zeros.
  - \( W_{\text{{hidden\_output}}} \) is initialized with random values.
  - \( b_{\text{{output}}} \) is initialized as a vector of zeros.

### 3. Forward Pass:
- **Input to Hidden Layer**:
  - \( \text{{hidden\_output}} = X \cdot W_{\text{{input\_hidden}}} + b_{\text{{hidden}}} \)
- **Activation Function (ReLU)**:
  - \( \text{{hidden\_activation}} = \text{{max}}(0, \text{{hidden\_output}}) \)
- **Hidden to Output Layer**:
  - \( \text{{output}} = \text{{hidden\_activation}} \cdot W_{\text{{hidden\_output}}} + b_{\text{{output}}} \)

### 4. Loss Calculation:
- **Mean Squared Error (MSE)**:
  - \( \text{{Loss}} = \frac{1}{N} \sum_{i=1}^{N} (\text{{predictions}}_i - \text{{targets}}_i)^2 \)

### 5. Backpropagation:
- **Output Layer Error**:
  - \( \text{{output\_error}} = \frac{2}{N} (\text{{predictions}} - \text{{targets}}) \)
- **Gradient Calculation for Hidden to Output Weights and Biases**:
  - \( \frac{{\partial \text{{Loss}}}}{{\partial W_{\text{{hidden\_output}}}}} = \text{{hidden\_activation}}^T \cdot \text{{output\_error}} \)
  - \( \frac{{\partial \text{{Loss}}}}{{\partial b_{\text{{output}}}}} = \sum \text{{output\_error}} \)

- **Hidden Layer Error**:
  - \( \text{{hidden\_error}} = \text{{output\_error}} \cdot W_{\text{{hidden\_output}}}^T \)
- **Gradient Calculation for Input to Hidden Weights and Biases**:
  - \( \frac{{\partial \text{{Loss}}}}{{\partial W_{\text{{input\_hidden}}}}} = X^T \cdot \text{{hidden\_error}} \)
  - \( \frac{{\partial \text{{Loss}}}}{{\partial b_{\text{{hidden}}}}} = \sum \text{{hidden\_error}} \)

### 6. Update Parameters:
- **Parameter Update using Gradient Descent**:
  - \( W_{\text{{input\_hidden}}} = W_{\text{{input\_hidden}}} - \text{{learning\_rate}} \times \frac{{\partial \text{{Loss}}}}{{\partial W_{\text{{input\_hidden}}}}} \)
  - \( b_{\text{{hidden}}} = b_{\text{{hidden}}} - \text{{learning\_rate}} \times \frac{{\partial \text{{Loss}}}}{{\partial b_{\text{{hidden}}}}} \)
  - \( W_{\text{{hidden\_output}}} = W_{\text{{hidden\_output}}} - \text{{learning\_rate}} \times \frac{{\partial \text{{Loss}}}}{{\partial W_{\text{{hidden\_output}}}}} \)
  - \( b_{\text{{output}}} = b_{\text{{output}}} - \text{{learning\_rate}} \times \frac{{\partial \text{{Loss}}}}{{\partial b_{\text{{output}}}}} \)

### 7. Inference:
- **Prediction**:
  - \( \text{{predictions}} = \text{{forward\_pass}}(X) \)

### 8. Evaluation:
- **Root Mean Squared Error (RMSE)**:
  - \( \text{{RMSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\text{{predictions}}_i - \text{{targets}}_i)^2} \)

This mathematical representation helps us understand the code in terms of the underlying operations and computations performed at each step of the neural network training process.
