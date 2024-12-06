//! This module contains the structures for networks and layers
const std = @import("std");
// const stdout = std.io.getStdOut().writer();

/// Enum containing activation function types
pub const ActivationFn = enum {
    ReLu,
    Sigmoid,
    Tanh,

    /// Logic for activation functions
    pub fn apply(self: ActivationFn, x: f32) f32 {
        return switch (self) {
            .ReLu => if (x > 0) x else 0,
            .Sigmoid => 1.0 / (1.0 + std.math.exp(-x)),
            .Tanh => std.math.tanh(x),
        };
    }
};

/// Layer struct using vectors
pub const Layer = struct {
    weights: [][]f32, // each weights[i] contains array of weights connecting to all downstream nodes
    biases: []f32, // dimension of biases is same as num of neurons in layer
    activation: ActivationFn,
};

/// Network Struct
pub const Network = struct {
    layers: []Layer,
    arena: std.heap.ArenaAllocator,

    /// Initialise the network
    /// Caller owns the memory?
    pub fn init(backing_allocator: std.mem.Allocator, layer_sizes: []const usize) !Network {
        var arena = std.heap.ArenaAllocator.init(backing_allocator); // Create an arena and deinit if there's an error in this function to prevent leaks
        errdefer arena.deinit();
        const mem_alloc = arena.allocator(); // Create the allocator explicitly as a constant

        // Create our random number generator
        var seed: u64 = undefined;
        std.crypto.random.bytes(std.mem.asBytes(&seed));
        var prng = std.rand.DefaultPrng.init(seed);
        const rand = prng.random();

        const layers = try mem_alloc.alloc(Layer, layer_sizes.len - 1); // Allocate the empty memory for the layers struct

        for (layers, 0..) |*layer, i| {
            const weights = try mem_alloc.alloc([]f32, layer_sizes[i]); // For each layer - initialise the memory of a weight array

            for (weights) |*row| {
                row.* = try mem_alloc.alloc(f32, layer_sizes[i + 1]); // For each row , create another pointer array connecting it to each neuron in the next layer

                for (row.*) |*weight| {
                    weight.* = rand.floatNorm(f32); // For each individual weight in the row, initialise it randomly and resolve the pointer
                }
            }

            const biases = try mem_alloc.alloc(f32, layer_sizes[i + 1]); // Allocate and initialises biases for the layer
            for (biases) |*bias| {
                bias.* = rand.floatNorm(f32);
            }

            layer.* = Layer{ .weights = weights, .biases = biases, .activation = ActivationFn.ReLu }; // Create the layer struct
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
};

test "Network constructor - first" {
    const layer_sizes = [_]usize{ 3, 4, 2 };
    var network = try Network.init(std.testing.allocator, &layer_sizes);
    defer network.deinit();

    // Check the number of layers
    try std.testing.expectEqual(network.layers.len, layer_sizes.len - 1);

    // Check first layer dimensions
    try std.testing.expectEqual(network.layers[0].weights.len, 3);
    try std.testing.expectEqual(network.layers[0].weights[0].len, 4);
    try std.testing.expectEqual(network.layers[0].biases.len, 4);

    // Check output layer dimensions
    try std.testing.expectEqual(network.layers[1].weights.len, 4);
    try std.testing.expectEqual(network.layers[1].weights[0].len, 2);
    try std.testing.expectEqual(network.layers[1].biases.len, 2);
}

test "Network constructor - deep network" {
    const layer_sizes = [_]usize{ 2, 4, 4, 3, 1 };
    var network = try Network.init(std.testing.allocator, &layer_sizes);
    defer network.deinit();

    try std.testing.expectEqual(4, network.layers.len);
    try std.testing.expectEqual(ActivationFn.ReLu, network.layers[0].activation);
}

test "Network constructor - minimal network" {
    const layer_sizes = [_]usize{ 1, 1 };
    var network = try Network.init(std.testing.allocator, &layer_sizes);
    defer network.deinit();

    try std.testing.expectEqual(1, network.layers.len);
    try std.testing.expectEqual(network.layers[0].weights.len, 1);
    try std.testing.expectEqual(network.layers[0].weights[0].len, 1);
    try std.testing.expectEqual(network.layers[0].biases.len, 1);
}
