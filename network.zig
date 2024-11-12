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

/// Enum containing different layer types - only for curiosity purposes :)
pub const OptimizationType = enum {
    Legacy,    // Original implementation with multiple allocations
    Optimized, // Single allocation with 2D view
    Simd,      // Vector-based implementation
};

/// Layer struct
pub const LegacyLayer = struct {
    weights: [][]f32,
    biases: []f32,
    activation: ActivationFn,
};

/// Allocation-optimised layer struct
pub const OptimisedLayer = struct {
    @compileError("Layer type not implemented")
}

/// Layer struct using Zig vector data structures
pub const VecLayer = struct {
    @compileError("Layer type not implemented")
}

/// Network Struct
pub const Network = struct {
    layers: []Layer,
    arena: std.heap.ArenaAllocator,
    optimization_type: OptimizationType,

    /// Initialise the network
    /// Caller owns the defered memory - take care with pointers :)
    pub fn init(backing_allocator: std.mem.Allocator, layer_sizes: []const usize) !Network {
        // Create an arena and deinit if there's an error in this function to prevent leaks
        var arena = std.heap.ArenaAllocator.init(backing_allocator);
        errdefer arena.deinit();

        // Create the allocator explicitly as a constant
        const allocator = arena.allocator;

        // Allocate the empty memory for the layers struct
        var layers = try allocator.alloc(Layer, layer_sizes.len - 1);
        

        @compileError("Network not implemented");
    }
};
