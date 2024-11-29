//! This module contains the structures for networks and layers
const std = @import("std");

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
    weights: [][]@Vector(usize, f32), // each weights[i] contains vector of weights connecting to all downstream nodes
    biases: []@Vector(usize, f32), // dimension of biases is same as num of neurons in layer
    activation: ActivationFn,
};

/// Network Struct
pub const Network = struct {
    layers: []Layer,
    arena: std.heap.ArenaAllocator,

    /// Initialise the network
    /// Caller owns the defered memory - take care with pointers :)
    pub fn init(backing_allocator: std.mem.Allocator, layer_sizes: []const usize) !Network {
        var arena = std.heap.ArenaAllocator.init(backing_allocator); // Create an arena and deinit if there's an error in this function to prevent leaks
        errdefer arena.deinit();
        const mem_alloc = arena.allocator; // Create the allocator explicitly as a constant

        var layers = try mem_alloc.alloc(Layer, layer_sizes.len - 1); // Allocate the empty memory for the layers struct

        for (layers, 0..) |*layer, i| {
            const weights = try mem_alloc.alloc(@Vector(layer_sizes[i], f32)); // For each layer - initialise the memory of a weight vector

            for (weights) |*row| {
                row.* = try mem_alloc.alloc(@Vector(layer_sizes[i + 1], f32)); // For each row , create another vector connecting it to the next layer

                for (row.*) |*weight| {
                    weight.* = std.Random.floatNorm(); // For each individual weight in the row, initialise it randomly
                }
            }

            var biases = try mem_alloc.alloc(f32, layer_sizes[i + 1]); // Allocate and initialises biases for the layer
            for (biases) |*bias| {
                bias.* = std.Random.floatNorm();
            }

            layer.* = Layer{ .weights = weights, .biases = biases, .activation = ActivationFn.ReLu }; // Create the layer struct
        }

        // Create the network struct
        return Network{
            .layers = layers,
            .arena = arena,
        };
    }
};
