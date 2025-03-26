//! This module contains the structures for networks and layers
const std = @import("std");

/// Enum containing activation function types
pub const ActivationFn = enum {
    ReLu,
    Sigmoid,
    Tanh,
    None,

    /// Logic for activation functions
    pub fn apply(self: ActivationFn, x: f32) f32 {
        return switch (self) {
            .ReLu => if (x > 0) x else 0,
            .Sigmoid => 1.0 / (1.0 + std.math.exp(-x)),
            .Tanh => std.math.tanh(x),
            .None => x,
        };
    }

    /// Derivative of activation function w.r.t its output value
    /// (i.e. derivative(activation(x)) w.r.t. "activation(x)", not w.r.t. x)
    pub fn derivative(self: ActivationFn, x: f32) f32 {
        return switch (self) {
            .ReLu => if (x > 0) 1.0 else 0.0,
            .Sigmoid => x * (1.0 - x),
            .Tanh => 1.0 - (x * x),
            .None => 1.0,
        };
    }
};

// /// Modify in-place the y array with a linear y = Ax + b
// pub fn linearForwardOld(y: []f32, weights: []const []const f32, x: []const f32, b: []const f32) void {
//     for (y, 0..) |*y_i, i| {
//         y_i.* = b[i];
//         for (weights, 0..) |row, j| {
//             y_i.* += row[i] * x[j];
//         }
//     }
// }

/// Modify in-place the y array with a linear y = Ax + b
pub fn linearForward(y: []f32, weights: []const []const f32, x: []const f32, b: []const f32) void {
    for (0..b.len) |i| {
        y[i] = b[i];
        for (0..x.len) |j| {
            y[i] += weights[j][i] * x[j];
        }
    }
}

/// Data structure for layer representation
pub const Layer = struct {
    weights: [][]f32, // each weights[i] contains array of weights connecting to all downstream nodes
    biases: []f32, // dimension of biases is same as num of neurons in layer
    activation: ActivationFn,
};

/// Data structure for gradients, used for batched gradient descent
pub const GradientAccumulator = struct {
    layers: []Layer,
    arena: std.heap.ArenaAllocator,

    pub fn init(
        network: Network,
        backing_allocator: std.mem.Allocator,
    ) !GradientAccumulator {
        var arena = std.heap.ArenaAllocator.init(backing_allocator);
        errdefer arena.deinit();
        const mem_alloc = arena.allocator();

        // Over first dimension
        const grad_layers = try mem_alloc.alloc(Layer, network.layers.len - 1); // Allocate empty layers

        for (grad_layers, 0..) |*layer, i| {
            const grad_weights = try mem_alloc.alloc([]f32, network.layers[i].weights.len);

            for (grad_weights) |*row| {
                row.* = try mem_alloc.alloc(f32, network.layers[i + 1].weights.len);
                @memset(row.*, 0);
            }

            const grad_biases = try mem_alloc.alloc(f32, network.layers[i + 1].biases.len); // Allocate and initialises biases for the layer
            @memset(grad_biases, 0);

            layer.* = Layer{ .weights = grad_weights, .biases = grad_biases, .activation = undefined };
        }

        return GradientAccumulator{
            .layers = grad_layers,
            .arena = arena,
        };
    }

    pub fn deinit(self: *GradientAccumulator) void {
        self.arena.deinit();
    }
};

/// Network Struct
pub const Network = struct {
    layers: []Layer,
    arena: std.heap.ArenaAllocator,

    /// Initialise the network
    /// Caller owns the memory?
    pub fn init(backing_allocator: std.mem.Allocator, layer_sizes: []const usize, activation_fn: ActivationFn) !Network {
        if (layer_sizes.len < 2) return error.InvalidArgument;

        var arena = std.heap.ArenaAllocator.init(backing_allocator); // Create an arena and deinit if there's an error in this function to prevent leaks
        errdefer arena.deinit();
        const mem_alloc = arena.allocator(); // Create the allocator explicitly as a constant

        // Create our random number generator
        var seed: u64 = undefined;
        std.crypto.random.bytes(std.mem.asBytes(&seed));
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();

        const layers = try mem_alloc.alloc(Layer, layer_sizes.len - 1); // Allocate the empty layers for the layers struct

        for (layers, 0..) |*layer, i| {
            const weights = try mem_alloc.alloc([]f32, layer_sizes[i]); // For each layer - initialise the memory of a weight array

            for (weights) |*row| {
                row.* = try mem_alloc.alloc(f32, layer_sizes[i + 1]); // For each row , create another pointer array connecting it to each neuron in the next layer

                // He/Kaiming initialization: scale = sqrt(2 / fan_in)
                const scale = std.math.sqrt(2.0 / @as(f32, @floatFromInt(layer_sizes[i])));

                for (row.*) |*weight| {
                    weight.* = rand.floatNorm(f32) * scale; // For each individual weight in the row, initialise it randomly and resolve the pointer
                }
            }

            const biases = try mem_alloc.alloc(f32, layer_sizes[i + 1]); // Allocate and initialises biases for the layer
            for (biases) |*bias| {
                bias.* = 0.0; // Initialise biases to 0
            }

            layer.* = Layer{ .weights = weights, .biases = biases, .activation = activation_fn }; // Create the layer struct
        }

        // Create the network struct
        return Network{
            .layers = layers,
            .arena = arena,
        };
    }

    /// Deinitialise the network
    pub fn deinit(self: *Network) void {
        self.arena.deinit();
    }

    /// Compute maximum layer size for allocation to a temporary buffer.
    fn maxLayerSize(self: *Network) usize {
        var max_size: usize = 0;
        for (self.layers) |layer| {
            if (layer.biases.len > max_size) {
                max_size = layer.biases.len;
            }
        }

        return max_size;
    }

    /// Perform a forward pass using the network. Caller owns memory for output.
    pub fn forward(
        self: *Network,
        input: []const f32,
        output_allocator: std.mem.Allocator,
    ) ![]f32 {
        // // Verify input size matches first layer
        // if (self.layers.len == 0 or input.len != self.layers[0].weights.len) {
        //     return error.InvalidInput;
        // }

        var temp_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer temp_arena.deinit();
        const temp_allocator = temp_arena.allocator();

        // Buffer for temp allocations
        var buffer = try temp_allocator.alloc(f32, self.maxLayerSize());
        _ = &buffer; // Acknowledge potential mutation through pointers

        // Store for current dat
        var current = try temp_allocator.dupe(f32, input);

        for (self.layers) |layer| {
            var next = try temp_allocator.alloc(f32, layer.biases.len);
            _ = &next; // Acknowledge potential mutation through pointers

            // Forward pass for each layer from scratch to store intermediate state
            // linearForwardOld(next, layer.weights, current, layer.biases);
            linearForward(next, layer.weights, current, layer.biases);

            // Calculate the activation func for each value before we copy to the next layer
            for (next) |*val| {
                val.* = layer.activation.apply(val.*);
            }

            // linearForward(buffer, layer.weights, current, layer.biases);

            // for (0..layer.biases.len) |i| {
            //     buffer[i] = layer.activation.apply(buffer[i]);
            // }
            // Replace our current with our next
            current = next;
            // current = buffer[0..layer.biases.len];
        }

        // Once we reach the final layer, return the output
        const output = try output_allocator.dupe(f32, current);
        return output;
    }

    /// Mutate the network in-place following a backward pass.
    /// Implements a MeanSquaredError loss function
    /// Returns the error for the pass
    pub fn backward(
        self: *Network,
        input: []const f32,
        target: []const f32,
        learning_rate: f32,
    ) !f32 {
        // Validate input
        if (self.layers.len == 0 or input.len != self.layers[0].weights.len) return error.InvalidInput;
        if (target.len != self.layers[self.layers.len - 1].biases.len) return error.InvalidInput;

        var temp_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer temp_arena.deinit();
        const temp_allocator = temp_arena.allocator();

        // Create an array list for our forward passes
        var activations = std.ArrayList([]f32).init(temp_allocator);
        defer activations.deinit();

        // Create a buffer for forward pass
        var buffer = try temp_allocator.alloc(f32, self.maxLayerSize());
        _ = &buffer; // Acknowledge potential mutation through pointers

        // Store for current size
        var current = try temp_allocator.dupe(f32, input);

        var pass_err: f32 = 0.0;

        // Perform a forward pass, storing our activations this time.
        try activations.append(current); // Start with the input as the first activation (for our first layer)
        for (self.layers) |layer| {
            var next = try temp_allocator.alloc(f32, layer.biases.len);
            _ = &next; // Acknowledge potential mutation through pointers

            // Forward pass for each layer from scratch to store intermediate state
            // linearForwardOld(next, layer.weights, current, layer.biases);
            linearForward(next, layer.weights, current, layer.biases);

            // Calculate the activation func for each value before we copy to the next layer
            for (next) |*val| {
                val.* = layer.activation.apply(val.*);
            }

            // // Forward pass for each layer from scratch to store intermediate state
            // linearForward(buffer, layer.weights, current, layer.biases);

            // for (0..layer.biases.len) |i| {
            //     buffer[i] = layer.activation.apply(buffer[i]);
            // }

            try activations.append(next);
            current = next;

            // // Store our activation for the rest of the backward pass
            // try activations.append(buffer);
            // current = buffer[0..layer.biases.len]; // Replace current with next and move to the next layer
        }

        // ================= Do backward pass
        // (1) Compute error using Mean Squared Error (derives to predictions[i] - target[i])
        const predicted = activations.items[self.layers.len]; // Get the predictions from the final layer
        var current_delta = try temp_allocator.alloc(f32, predicted.len); // Current for each output neuron

        // For each output neuron
        for (predicted, 0..) |pred, i| {
            // Component 1: partial derivative of error wrt output. Calculate dError_total / dOutput_i
            const err = pred - target[i]; // This is the same for output and hidden neurons
            pass_err += err * err;
            // Component 2: partial derivative of output wrt sum. Calculate dOutput_i / dSum_i
            current_delta[i] = err * self.layers[self.layers.len - 1].activation.derivative(pred); // This is the same for output and hidden neurons
        }

        // (2) Propagate backwards
        // Idiomatic reverse indexing through array is odd in zig.
        var layer_idx: usize = self.layers.len - 1;
        while (true) : (layer_idx -= 1) {
            const layer = self.layers[layer_idx];
            const activation_in = activations.items[layer_idx]; // Input to this layer

            // Component 3: Partial derivative of sum wrt weight. Calculate dSum_i / dWeight_i
            // Allocate next delta - the gradient wrt the *previous* layer's output.
            var next_delta = try temp_allocator.alloc(f32, activation_in.len);
            // Initialise to 0
            for (next_delta) |*nd| {
                nd.* = 0;
            }

            // Update each weight and bias, accumulating the delta to calculate the next layer
            for (0..layer.biases.len) |n| {
                // 1. Update bias: dBias = current_delta[n]
                layer.biases[n] -= learning_rate * current_delta[n];

                // 2. Update each weight & accumulate gradient
                for (0..activation_in.len) |m| {
                    const input_val = activation_in[m];
                    const grad_w = input_val * current_delta[n];

                    // Accumulate delta for previous layer. Calculate next_delta[m] += (delta for this neuron) * (current weight)
                    next_delta[m] += current_delta[n] * layer.weights[m][n];

                    // Update the weight
                    layer.weights[m][n] -= learning_rate * grad_w;
                }
            }

            // If we're not at the first layer - need the previous layers' contributions. Since we're working backwards...
            if (layer_idx > 0) {
                for (0..activation_in.len) |m| {
                    next_delta[m] = next_delta[m] * self.layers[layer_idx - 1].activation.derivative(activation_in[m]);
                }
            }

            // Prep for next iteration
            current_delta = next_delta;

            // Stop if we just updated the first layer
            if (layer_idx == 0) break;
        }

        pass_err = std.math.sqrt(pass_err / @as(f32, @floatFromInt(predicted.len)));

        return pass_err;
    }
};

test "Network constructor - first" {
    const layer_sizes = [_]usize{ 3, 4, 2 };
    var network = try Network.init(std.testing.allocator, &layer_sizes, ActivationFn.ReLu);
    defer network.deinit();

    // Check the number of layers
    try std.testing.expectEqual(network.layers.len, layer_sizes.len - 1);

    // Check first layer dimensions
    try std.testing.expectEqual(3, network.layers[0].weights.len);
    try std.testing.expectEqual(4, network.layers[0].weights[0].len);
    try std.testing.expectEqual(4, network.layers[0].biases.len);

    // Check output layer dimensions
    try std.testing.expectEqual(4, network.layers[1].weights.len);
    try std.testing.expectEqual(2, network.layers[1].weights[0].len);
    try std.testing.expectEqual(2, network.layers[1].biases.len);

    // Print some layer weights & biases
    // std.debug.print("Layer 0 Node 0 Weight 0: {d} \n", .{network.layers[0].weights[0][0]});
    // std.debug.print("Layer 1 Node 1 Weight 1: {d} \n", .{network.layers[1].weights[1][1]});
}

test "Network constructor - deep network" {
    const layer_sizes = [_]usize{ 2, 4, 4, 3, 1 };
    var network = try Network.init(std.testing.allocator, &layer_sizes, ActivationFn.ReLu);
    defer network.deinit();

    try std.testing.expectEqual(4, network.layers.len);
    try std.testing.expectEqual(ActivationFn.ReLu, network.layers[0].activation);
}

test "Network constructor - minimal network" {
    const layer_sizes = [_]usize{ 1, 1 };
    var network = try Network.init(std.testing.allocator, &layer_sizes, ActivationFn.ReLu);
    defer network.deinit();

    try std.testing.expectEqual(1, network.layers.len);
    try std.testing.expectEqual(network.layers[0].weights.len, 1);
    try std.testing.expectEqual(network.layers[0].weights[0].len, 1);
    try std.testing.expectEqual(network.layers[0].biases.len, 1);
}

test "VecOps - linearForward basic operation" {
    // Test a 2->3 network layer
    var y = [_]f32{ 0.0, 0.0, 0.0 }; // 3 outputs
    const x = [_]f32{ 1.0, 2.0 }; // 2 inputs
    const b = [_]f32{ 0.1, 0.2, 0.3 }; // 3 biases

    const weights = [_][]const f32{
        &[_]f32{ 1.0, 0.5, 0.0 }, // weights from input 0 to all outputs -> first neuron
        &[_]f32{ 2.0, 0.5, 1.0 }, // weights from input 1 to all outputs -> second neuron
    };

    linearForward(&y, &weights, &x, &b);

    try std.testing.expectApproxEqAbs(5.1, y[0], 0.0001);
    try std.testing.expectApproxEqAbs(1.7, y[1], 0.0001);
    try std.testing.expectApproxEqAbs(2.3, y[2], 0.0001);
}

test "VecOps - linearForward on a 2-2 network" {
    var y = [_]f32{ 0.0, 0.0 };
    const x = [_]f32{ 0.1, 0.5 };
    const b = [_]f32{ 0.25, 0.25 };

    const weights = [_][]const f32{
        &[_]f32{ 0.1, 0.2 }, // weights from input 0 to all outputs -> weights of the first neuron
        &[_]f32{ 0.3, 0.4 }, // weights from input 1 to all outputs -> weights of the second neuron
    };

    linearForward(&y, &weights, &x, &b);

    try std.testing.expectApproxEqAbs(0.41, y[0], 1e-3);
    try std.testing.expectApproxEqAbs(0.47, y[1], 1e-3);
}

test "VecOps - linearForward super simple test" {
    const w = [_][]const f32{
        &[_]f32{10.0}, // input #0
        &[_]f32{100.0}, // input #1
    };
    const x = [_]f32{ 2.0, 3.0 };
    const b = [_]f32{1.0};
    var y = [_]f32{0.0};

    linearForward(&y, &w, &x, &b);

    try std.testing.expectApproxEqAbs(y[0], 321, 0);
}

test "Network - single layer forward pass" {
    // Create a simple 2->1 network
    var network = try Network.init(std.testing.allocator, &[_]usize{ 2, 1 }, ActivationFn.ReLu);
    defer network.deinit();

    // Configure the weights and biases for a simple sum operation
    network.layers[0].weights[0][0] = 1.0;
    network.layers[0].weights[1][0] = 1.0;
    network.layers[0].biases[0] = 0.0;
    network.layers[0].activation = .ReLu;

    // Input: [1.0, 2.0] should give us 3.0 (1.0 + 2.0 + 0.0 bias)
    const input = [_]f32{ 1.0, 2.0 };
    const output = try network.forward(&input, std.testing.allocator);
    defer std.testing.allocator.free(output);

    try std.testing.expectEqual(@as(usize, 1), output.len);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), output[0], 1e-4);
}

test "Forward and Backprop on 2-2-1 Network" {
    std.debug.print("Forwarding and backpropping on a 2-2-1 network...\n", .{});

    const layer_sizes = [_]usize{ 2, 2, 1 };
    var network = try Network.init(std.testing.allocator, &layer_sizes, .ReLu);
    defer network.deinit();

    // Overwrite our Ws&Bs. Layer->neuron->weight for *each* output neuron
    network.layers[0].weights[0][0] = 0.11;
    // network.layers[0].weights[0][1] = 0.12;
    // network.layers[0].weights[1][0] = 0.21;
    network.layers[0].weights[0][1] = 0.12;
    network.layers[0].weights[1][0] = 0.21;
    network.layers[0].weights[1][1] = 0.08;

    network.layers[1].weights[0][0] = 0.14;
    network.layers[1].weights[1][0] = 0.15;

    network.layers[0].biases[0] = 0;
    network.layers[0].biases[1] = 0;
    network.layers[1].biases[0] = 0;

    const input_data = [_]f32{ 2, 3 };
    const output_before = try network.forward(&input_data, std.testing.allocator);
    defer std.testing.allocator.free(output_before);
    std.debug.print("   Output before single backprop: {d:.6}\n", .{output_before});

    // Test whether the forward pass has the right results
    try std.testing.expectApproxEqAbs(0.191, output_before[0], 1e-3);

    // -------------- Test backpropagation
    const target_data = [_]f32{1};
    const current_err = try network.backward(&input_data, &target_data, 0.05);

    std.debug.print("   Current error is: {d:.6}\n", .{current_err});

    try std.testing.expectApproxEqAbs(0.12, network.layers[0].weights[0][0], 1e-2);
    try std.testing.expectApproxEqAbs(0.13, network.layers[0].weights[0][1], 1e-2);
    try std.testing.expectApproxEqAbs(0.23, network.layers[0].weights[1][0], 1e-2);
    try std.testing.expectApproxEqAbs(0.10, network.layers[0].weights[1][1], 1e-2);
    try std.testing.expectApproxEqAbs(0.17, network.layers[1].weights[0][0], 1e-2);
    try std.testing.expectApproxEqAbs(0.17, network.layers[1].weights[1][0], 1e-2);

    // Reset our biases as the example I used for these numbers didn't have those updateable
    network.layers[0].biases[0] = 0;
    network.layers[0].biases[1] = 0;
    network.layers[1].biases[0] = 0;

    const output_after = try network.forward(&input_data, std.testing.allocator);
    defer std.testing.allocator.free(output_after);
    std.debug.print("   Output after single backprop: {d:.6}\n", .{output_after});

    // Test whether the updated outputs are right
    try std.testing.expectApproxEqAbs(0.26, output_after[0], 0.01);

    // ------------------------------------------------------------

    // Run another round of backprop to see if the network weights continue updating correctly
    // const current_err_2 = try network.backward(&input_data, &target_data, 0.05);
    // defer std.testing.allocator.free(current_err_2);
    // std.debug.print("   Error after second backprop: {d:.6}\n", .{current_err_2});

    // const output_after_2 = try network.forward(&input_data, std.testing.allocator);
    // defer std.testing.allocator.free(output_after_2);
    // std.debug.print("   Output after second backprop: {d:.6}\n", .{output_after_2});
}

test "Network - Forward on a [2,2,2] network - v1" {
    // 1) Construct a small [2, 2, 2] network
    const layer_sizes = [_]usize{ 2, 2, 2 };
    var network = try Network.init(std.testing.allocator, &layer_sizes, ActivationFn.Sigmoid);
    defer network.deinit();

    // For reproducibility, override the randomly initialized weights/biases with known constants:
    // -- First layer (index 0)
    network.layers[0].weights[0][0] = 0.15;
    network.layers[0].weights[0][1] = 0.30;
    network.layers[0].weights[1][0] = 0.20;
    network.layers[0].weights[1][1] = 0.25;

    network.layers[0].biases[0] = 0.35;
    network.layers[0].biases[1] = 0.35;

    // -- Second layer (index 1)
    network.layers[1].weights[0][0] = 0.40;
    network.layers[1].weights[0][1] = 0.55;
    network.layers[1].weights[1][0] = 0.45;
    network.layers[1].weights[1][1] = 0.50;

    network.layers[1].biases[0] = 0.60;
    network.layers[1].biases[1] = 0.60;

    // 2) Do a forward pass with a known input
    const input_data = [_]f32{ 0.05, 0.10 };
    const output_before = try network.forward(&input_data, std.testing.allocator);
    defer std.testing.allocator.free(output_before);

    // 2a) Test that the output is correct
    try std.testing.expectApproxEqAbs(0.75136507, output_before[0], 1e-3);
    try std.testing.expectApproxEqAbs(0.77292847, output_before[1], 1e-4);

    // // 3) Run a backward pass with a known target
    // const target_data = [_]f32{ 0.01, 0.99 };
    // try network.backward(&input_data, &target_data, 0.5);

    // try std.testing.expectApproxEqAbs(0.149780716, network.layers[0].weights[0][0], 1e-4);
    // try std.testing.expectApproxEqAbs(0.19956143, network.layers[0].weights[0][1], 1e-4);
    // try std.testing.expectApproxEqAbs(0.24975114, network.layers[0].weights[1][0], 1e-4);
    // try std.testing.expectApproxEqAbs(0.29950299, network.layers[0].weights[1][1], 1e-4);
    // try std.testing.expectApproxEqAbs(0.35891648, network.layers[1].weights[0][0], 1e-4);
    // try std.testing.expectApproxEqAbs(0.408666186, network.layers[1].weights[0][1], 1e-4);
    // try std.testing.expectApproxEqAbs(0.51130270, network.layers[1].weights[1][0], 1e-4);
    // try std.testing.expectApproxEqAbs(0.561370121, network.layers[1].weights[1][1], 1e-4);
}

test "Forward on a [2, 2, 2] network - v2" {
    const input = [_]f32{ 0.1, 0.5 };
    // const target = [_]f32{ 0.05, 0.95 };
    const layer_sizes = [_]usize{ 2, 2, 2 };

    var network = try Network.init(std.testing.allocator, &layer_sizes, .Sigmoid);
    defer network.deinit();

    // Fix the W&Bs
    network.layers[0].weights[0][0] = 0.10;
    network.layers[0].weights[0][1] = 0.20;
    network.layers[0].weights[1][0] = 0.30;
    network.layers[0].weights[1][1] = 0.40;

    network.layers[1].weights[0][0] = 0.50;
    network.layers[1].weights[0][1] = 0.70; // ! Flipping these two makes the test pass?
    network.layers[1].weights[1][0] = 0.60; // ! We seem to be using the wrong weight in the 2nd layer...
    network.layers[1].weights[1][1] = 0.80;

    network.layers[0].biases[0] = 0.25;
    network.layers[0].biases[1] = 0.25;
    network.layers[1].biases[0] = 0.35;
    network.layers[1].biases[1] = 0.35;

    // Run a forward pass
    const output_before = try network.forward(&input, std.testing.allocator);
    defer std.testing.allocator.free(output_before);

    try std.testing.expectApproxEqAbs(0.73492, output_before[0], 1e-4);
    try std.testing.expectApproxEqAbs(0.77955, output_before[1], 1e-4);
}

test "Initialise GradientAccumulator" {
    // var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    // defer arena.deinit();
    // const mem_alloc = arena.allocator();

    // Create a network
    const layer_sizes = [_]usize{ 2, 2, 2 };
    var network = try Network.init(std.testing.allocator, &layer_sizes, .Sigmoid);
    defer network.deinit();

    // Create our accumulator
    var grad_accumulator = try GradientAccumulator.init(network, std.testing.allocator);
    defer grad_accumulator.deinit();

    // Check the structures align
    try std.testing.expectEqual(grad_accumulator.layers[0].weights[0][0], @as(f32, @floatFromInt(0)));
    try std.testing.expectEqual(grad_accumulator.layers[0].biases[0], @as(f32, @floatFromInt(0)));
}
