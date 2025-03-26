# ZNN - A Neural Network Implementation in Zig

A hand-rolled neural network implementation written in Zig that solves for the MNIST handwritten digit recognition task. 

I'd not written any Zig code before this, so it was a fun project to get to grips with the language.

## Project Status

✅ **COMPLETED** - This repo is in cryogenic storage 🧊. Signs of life may be rare.

## Overview

ZNN was created as a learning project to understand both neural networks and the Zig programming language. The implementation includes:

- Feed-forward neural networks with customizable architectures
- Backpropagation for training
- Multiple activation functions (Sigmoid, ReLU, Tanh)
- MNIST data loading and preprocessing
- Training and evaluation pipelines

## Project Structure

- `main.zig`: Entry point of the program, handles MNIST data loading, training, and evaluation
- `src/network.zig`: Core neural network implementation with layers, activation functions, and both forward and backward passes
- `src/data.zig`: Data loading utilities for MNIST dataset, including preprocessing functions
- `src/training.zig`: Training loop implementation for the neural network
- `src/helpers.zig`: Utility functions like progress bar visualization

## Usage Example

```zig
// Load and preprocess MNIST data
const data = try data_loader.MNISTData.loadMNIST(
    train_images_path,
    train_labels_path,
    test_images_path,
    test_labels_path,
    allocator,
);

// Create normalized inputs and one-hot encoded targets
var inputs = try allocator.alloc([]f32, data.train_data.images.len);
var targets = try allocator.alloc([]f32, data.train_data.labels.len);
for (data.train_data.images, data.train_data.labels, 0..) |image, label, i| {
    inputs[i] = try data_loader.imageToInput(image, allocator);
    targets[i] = try data_loader.labelToTarget(label, allocator);
}

// Initialize a neural network
const layer_sizes = [_]usize{ 784, 128, 64, 10 };
var network = try Network.init(allocator, &layer_sizes, .Sigmoid);
defer network.deinit();

// Train the network
try training.trainNetwork(
    &network,
    inputs,
    targets,
    3,      // epochs
    0.5,    // learning rate
    0.1,    // error threshold
    1,      // min epochs
    true,   // verbose output
);
```

## Lessons Learned

- ReLU activation can suffer from the "dying ReLU" problem (especially when in such a rudimentary architecture). Sigmoid or Leaky ReLU are recommended alternatives for small networks (this one stumped me for a while - thought it was a memory issue)
- Weight initialization is critical for proper convergence
- I've learnt to appreciate the python ecosystem even more than I already did...
- Zig needs to sort out its documentation situation.
- AI assistants are great until the going gets tough. Relatedly I totally understand the AI safety folk now. LLMs are really good at gaslighting you and it's kind of terrifying.

## Performance
Not worth talking too much about, but it generalises to the test set completely adequately, with a 0.04 RMSE after 1 epoch.

Training completes in a reasonable time (98s per epoch on my machine).

## Building and Running

1. Ensure you have Zig installed
2. Download the MNIST dataset files to a `data/` directory:
   - `train-images.idx3-ubyte`
   - `train-labels.idx1-ubyte`
   - `t10k-images.idx3-ubyte`
   - `t10k-labels.idx1-ubyte`
3. Run with:
   ```
   zig run main.zig
   ```

## License

This project is open source, feel free to use and learn from it.
