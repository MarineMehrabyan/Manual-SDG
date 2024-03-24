# Manual-SDG
This repository contains a Python implementation of a simple neural network trained using Stochastic Gradient Descent (SGD) from scratch. The neural network is built using NumPy and Pandas libraries, providing a foundational understanding of the underlying principles of neural networks and optimization algorithms.


### Data Preprocessing:
The code first preprocesses the data. It reads a CSV file containing input features and target values. Then it standardizes the input features by subtracting the mean and dividing by the standard deviation. This is done to ensure that all features have a similar scale, which helps in training the neural network.

1. Subtract the mean: 
$$ X_{\text{mean}} = \frac{1}{n}\sum_{i=1}^{n} X_i $$

2. Divide by standard deviation: 
$$ X_{\text{std}} = \frac{X - X_{\text{mean}}}{\sigma} $$


### Neural Network Architecture:
The neural network architecture consists of an input layer, a hidden layer with ReLU activation, and an output layer.

1. **Forward Pass:**
   - Input to Hidden Layer: 
   $$ \text{hidden_output} = X \cdot W_{\text{input_hidden}} + b_{\text{hidden}} $$
   - Hidden Layer Activation (ReLU): 
   $$ \text{hidden_activation} = \max(0, \text{hidden_output}) $$
   - Hidden to Output Layer: 
   $$ \text{output} = \text{hidden_activation} \cdot W_{\text{hidden_output}} + b_{\text{output}} $$

2. **ReLU Function:**
   $$ \text{ReLU}(x) = \max(0, x) $$

3. **Loss Calculation:**
   The Mean Squared Error (MSE) loss function is used.
   $$ \text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{predicted}} - y_{\text{true}})^2 $$

4. **Backward Pass (Gradient Descent):**
   Gradients are computed and weights are updated to minimize the loss using gradient descent.
   - Output Layer Error: 
   $$ \text{output_error} = \frac{2}{n} \times (y_{\text{predicted}} - y_{\text{true}}) $$
   - Hidden Layer Error: 
   $$ \text{hidden_error} = \text{output_error} \cdot W_{\text{hidden_output}}^T $$
   - Update Weights: 
     $$ W_{\text{input_hidden}} -= \text{learning_rate} \times \frac{\partial \text{Loss}}{\partial W_{\text{input_hidden}}} $$
     $$ W_{\text{hidden\_output}} -= \text{learning_rate} \times \frac{\partial \text{Loss}}{\partial W_{\text{hidden_output}}} $$
   - Update Biases:
     $$ b_{\text{hidden}} -= \text{learning_rate} \times \frac{\partial \text{Loss}}{\partial b_{\text{hidden}}} $$
     $$ b_{\text{output}} -= \text{learning_rate} \times \frac{\partial \text{Loss}}{\partial b_{\text{output}}} $$

### Inference:
Finally, the trained model is used for inference, and Root Mean Squared Error (RMSE) is calculated to evaluate the model's performance.
$$ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{\text{predicted}} - y_{\text{true}})^2} $$

This explanation provides a mathematical understanding of the code's operations in terms of data preprocessing, neural network architecture, training (including forward and backward passes), and evaluation.
