# ZNN
Zig Neural Network. Rolling my own implementation of a neural network in Zig. Goal is to solve MNIST or some similar problem efficiently with it. Shamelessely steals concepts from the [ZML](https://github.com/zml/zml) library.

> **README Out of date!**

**!This is a work in progress and for my learning only!**

## Structure:
- main.zig: Entry point of the program.
- network.zig: Contains the neural network struct and functions.
- inference.zig: Contains the forward pass inference functions
- training.zig: Contains training functions incl. backpropagation to update the model based on inputs
- serialize.zig: Used to persist the model following training and read it for use.

### Network:
- Contains network structure
- ActivationFn enum: Contains the activation functions.
    - Sigmoid, ReLu, TanH, Softmax
- Layer struct: Structure containing the W&B & activation function.
- Network struct: Contains the layers and the forward function.

### Inference:
- Contains inference context, lightweight to use for predictions (only forward pass)

### Training:
- Contains training context, used to update the model with backpropagation

### Serialize:
- Contains JSON logic, to persist the model as required.