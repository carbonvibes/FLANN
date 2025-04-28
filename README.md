# Convex Combination of FLANNs for System Identification

This implementation demonstrates an adaptive system identification approach using a convex combination of two Functional Link Artificial Neural Networks (FLANNs) with different step sizes.

![FLANN Convex Combination Results](/flann.png)

## Group Members

- Arjun Sekar (22110034)
- Naga Sheshu (22110178)

## Youtube Link
[Youtube Playlist](https://youtube.com/playlist?list=PLMHHV60z4KgEpT_k9kgo4eLVq6iRVf8WL&si=6VK-1Unl51huvg4f)

## Overview

System identification aims to build mathematical models of dynamic systems based on observed input-output data. In this implementation, we use a convex combination of two FLANNs with different learning rates to achieve better performance than either network alone. The system adaptively adjusts the combination parameter to optimize performance across different learning phases.

## Implementation Details

### FLANN (Functional Link Artificial Neural Network)

A FLANN is a single-layer neural network that uses functional expansion of the input pattern to increase the dimensionality of the input, making the problem linearly separable in the expanded space. 

In our implementation:

1. **Input Expansion**: Each input is expanded using trigonometric functions (sin and cos) of various frequencies:
   ```python
   def expand_input(self, x: np.ndarray) -> np.ndarray:
       expanded = []
       # For each input dimension
       for i in range(self.input_size):
           # Add the original feature
           expanded.append(x[i])
           # Add sin and cos terms for each degree
           for j in range(1, self.expansion_degree + 1):
               expanded.append(np.sin(j * np.pi * x[i]))  # sin terms
               expanded.append(np.cos(j * np.pi * x[i]))  # cos terms
       return np.array(expanded)
   ```

2. **Linear Combiner**: The expanded inputs are linearly combined using weights.
3. **Learning Algorithm**: The weights are updated using the LMS (Least Mean Square) algorithm:
   ```python
   weight_update = self.learning_rate * error * x_expanded
   self.weights += weight_update
   ```

### Convex Combination of FLANNs

The convex combination approach combines the outputs of two FLANNs with a parameter 'a':

```
y_combined = a * y1 + (1-a) * y2
```

Where:
- y1 and y2 are the outputs of the two FLANNs
- 'a' is the combination parameter (between 0 and 1)
- y_combined is the final output

The combination parameter 'a' is dynamically adjusted during training to minimize the overall error:

```python
# Update combination parameter using gradient descent
e1 = y - y1
e2 = y - y2
grad_a = e1 - e2  # Gradient of error with respect to a

# Update auxiliary parameter
a_param_update = self.lambda_ * error_combined * grad_a
self.a_param = self.a_param + a_param_update

# Apply sigmoid to keep a between 0 and 1
self.a = 1 / (1 + np.exp(-self.a_param))
```

This allows the system to automatically leverage the strengths of each FLANN.

## Key Features

1. **Two FLANNs with Different Step Sizes**: 
   - **FLANN 1**: Smaller learning rate (0.01) - stable but slower convergence
   - **FLANN 2**: Larger learning rate (0.1) - faster convergence but potentially less stable

2. **Adaptive Combination Parameter**:
   - Updated using gradient descent
   - Sigmoidal mapping ensures 'a' stays between 0 and 1
   - Adaptive learning rate (lambda) for the combination parameter

3. **Performance Tracking**:
   - Convergence characteristics for both individual FLANNs and combined system
   - Visualization of how the combination parameter varies during training
   - Tracking of which network performs better on each sample

## Results and Analysis

Our implementation demonstrates how the convex combination of two FLANNs can achieve better performance than either network alone. Below are sample results from our testing:

```
Network Configuration:
  Input size: 1
  Expansion degree: 5
  Expanded feature size: 11
  FLANN1 learning rate: 0.01
  FLANN2 learning rate: 0.1
  Combination parameter a (initial): 0.5
  Combination parameter learning rate (lambda): 0.05
```

After 5 epochs of training, the system achieves the following performance on the test set:

```
Test set mean squared errors:
  FLANN1: 0.01003090
  FLANN2: 0.01345518
  Combined: 0.01011033
```

The final combination parameter 'a' converges to approximately 0.82, showing the system's preference for FLANN1 while still utilizing some information from FLANN2.

### Combination Rate and Adaptation Rate

Two key parameters that vary during training provide insight into the system's adaptive behavior:

1. **Combination Rate (a)**:
   - Determines how the outputs of the two FLANNs are weighted
   - Ranges from 0 (rely on FLANN2) to 1 (rely on FLANN1)
   - In our experiments, 'a' evolved from 0.5 to ~0.82, indicating a learned preference for the more stable FLANN1

2. **Adaptation Rate (lambda)**:
   - Controls how quickly the combination parameter adapts
   - Higher values allow faster adaptation when performance differences are clear
   - Lower values provide stability once a good combination is found
   - Our adaptive implementation adjusts lambda based on the relative performance of the networks

### Epoch Summary

The system shows progressive improvement over training epochs:

```
Epoch 5 Summary:
  Average squared errors:
    FLANN1: 0.01064720
    FLANN2: 0.01368214
    Combined: 0.01069938
  Combination parameter a: 0.8215
  Weight norms: FLANN1=0.6434, FLANN2=0.7272
```

## Nonlinear System Model

The implementation identifies a nonlinear system represented by:

```
y = 0.6*sin(π*x) + 0.3*cos(2π*x) + 0.1*sin(3π*x) + noise
```

Sample data points from this system:
```
x = -0.2509, y = -0.4795, true y (without noise) = -0.4973, noise = 0.0178
x = 0.9014, y = 0.3737, true y (without noise) = 0.5072, noise = -0.1335
x = 0.4640, y = 0.2475, true y (without noise) = 0.2095, noise = 0.0380
```

## Usage

To run this implementation:

1. Ensure you have the required dependencies:
   ```
   numpy
   matplotlib
   ```

2. Run the script:
   ```
   python flann_convex_combination.py
   ```

3. Observe the generated plots showing:
   - Convergence characteristics of both individual FLANNs and the combined system
   - Variation of the combination parameter 'a' over training iterations
   - Adaptation rate (lambda) variation

The results are saved to 'flann_convex_combination_results.png'.

## References

1. Pao, Y. H. (1989). Adaptive pattern recognition and neural networks.
2. Vural, R. A., Özen, T., & Çelebi, A. (2013). A new approach for functional link neural network with trigonometric and polynomial terms. Neural Computing and Applications, 22(3-4), 721-729.
3. Arenas-Garcia, J., Figueiras-Vidal, A. R., & Sayed, A. H. (2006). Mean-square performance of a convex combination of two adaptive filters. IEEE transactions on signal processing, 54(3), 1078-1090.
