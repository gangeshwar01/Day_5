# Neural Network for Regression from Scratch with NumPy

This project demonstrates how to build, train, and evaluate a fully connected neural network for a regression task using only the NumPy library. The implementation covers the core concepts of forward propagation, backpropagation, and optimization from the ground up.

## 🧠 Architecture

The neural network architecture was chosen to be powerful enough to capture the non-linear relationship in the data (a cubic function) but simple enough to train efficiently.

-   **Input Layer**: Takes a single feature `x`.
-   **Hidden Layer**: A `Dense` layer with **64 neurons**. This number was chosen as a balance between model capacity and computational cost. Too few neurons might underfit, while too many could overfit or slow down training.
-   **Activation Function**: A **ReLU (Rectified Linear Unit)** activation function follows the hidden layer.
    -   *Formula*: $f(x) = \max(0, x)$
    -   *Reasoning*: ReLU is computationally efficient and helps mitigate the vanishing gradient problem, making it a standard choice for hidden layers.
-   **Output Layer**: A `Dense` layer with a **single neuron**.
    -   *Activation*: No activation function is used (i.e., a linear activation). This is standard for regression tasks, as the output needs to be an unbounded continuous value.

The entire model can be summarized as: `Input(1) -> Dense(64) -> ReLU -> Dense(1)`.

## 📈 Convergence Analysis

The model is trained using **Stochastic Gradient Descent (SGD)** with mini-batches.

-   **Loss Function**: **Mean Squared Error (MSE)** was used, which is standard for regression. It penalizes larger errors more heavily.
    -   *Formula*: $L = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{pred}, i} - y_{\text{true}, i})^2$
-   **Optimizer**: A custom SGD optimizer updates the model's weights and biases. The learning rate was set to `0.001`, a small value to ensure stable convergence without overshooting the minimum. The mini-batch size was `32`, a common choice that offers a good trade-off between gradient accuracy and computational speed.

### Results

1.  **Loss Curve**: The training loss plot shows a steep initial decline, followed by a gradual and steady decrease. This indicates that the model is learning effectively and converging towards a minimum. The absence of large oscillations suggests that the learning rate is appropriate.

2.  **Prediction vs. Ground Truth**: The final plot shows the model's predictions overlaid on the original noisy data. The predicted curve successfully captures the underlying cubic trend of the data, demonstrating that the network has learned the non-linear relationship between the input `X` and output `y`. It effectively acts as a "smoother," ignoring the random noise, which is the desired outcome for a regression model.