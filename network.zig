//! Network.zig. Contains Network and layer definitions.
const std = @import("std");

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

pub const Layer = struct {
    weights: [][]f32,
    biases: []f32,
    activation: ActivationFn,
};

pub const Network = struct {
    layers: []Layer,
    arena: std.heap.ArenaAllocator,

    /// Initialise the network
    /// Caller owns the defered memory - take care with pointers :)
    pub fn _init(backing_allocator: std.mem.Allocator, layer_sizes: []const usize) !Network {
        var arena = std.heap.ArenaAllocator.init(backing_allocator);
        errdefer arena.deinit();

        const allocator = arena.allocator;
        var layers = try allocator.alloc(Layer, layer_sizes.len - 1);

        @compileError("Network not implemented");
    }
};
