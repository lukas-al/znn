//! Inference context.
//! Simple forward passess used for prediction when W&Bs are fixed.
const std = @import("std");
const Network = @import("network.zig").Network;
const VecOps = @import("vectors.zig").VecOps;

/// Inference context - forward pass and computational context
/// Caller owns memory
pub const InferenceContext = struct {
    arena: std.heap.ArenaAllocator,

    /// Initialise memory
    pub fn init(backing_allocator: std.mem.Allocator) InferenceContext {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
        };
    }

    /// Deinitialise
    pub fn deinit(self: *InferenceContext) void {
        self.arena.deinit();
    }

    /// Reset the memory & context
    pub fn reset(self: *InferenceContext) void {
        _ = self.arena.reset(.free_all);
    }

    /// Simple forward pass implementation for the network. Returns an f32 array of the same dim as the network final layer.
    pub fn forward(
        self: *InferenceContext,
        network: *Network,
        input: []const f32,
        output_alloc: std.mem.Allocator,
    ) ![]f32 {
        // Validate our inputs
        if (network.layers.len == 0) return error.InvalidArgument;
        const first_layer_inputs = network.layers[0].weights.len;
        if (input.len != first_layer_inputs) return error.InvalidArgument;

        defer self.reset();
        const allocator = self.arena.allocator();
        var current = try allocator.dupe(f32, input);

        for (network.layers) |layer| {
            // Allocate temp memory of the right size for our next layer
            var next = try allocator.alloc(f32, layer.biases.len);
            // @constCast(&next); // Acknowledge potential mutation through pointers
            _ = &next;

            // Transform the next layer in place
            VecOps.linearForward(next, layer.weights, current, layer.biases);

            // Calculate the activation func for each value before we copy to the next layer
            for (next) |*val| {
                val.* = layer.activation.apply(val.*);
            }

            // Replace our current with our next
            current = next;
        }

        // Return our output to the peristent output allocator
        const output = try output_alloc.dupe(f32, current);
        return output;
    }
};

test "InferenceContext - basic initialization and cleanup" {
    var ctx = InferenceContext.init(std.testing.allocator);
    defer ctx.deinit();
}

test "InferenceContext - single layer forward pass" {
    // Initialize context
    var ctx = InferenceContext.init(std.testing.allocator);
    defer ctx.deinit();

    // Create a simple 2->1 network
    var network = try Network.init(std.testing.allocator, &[_]usize{ 2, 1 });
    defer network.deinit();

    // Configure the weights and biases for a simple sum operation
    network.layers[0].weights[0][0] = 1.0;
    network.layers[0].weights[1][0] = 1.0;
    network.layers[0].biases[0] = 0.0;
    network.layers[0].activation = .ReLu;

    // Input: [1.0, 2.0] should give us 3.0 (1.0 + 2.0 + 0.0 bias)
    const input = [_]f32{ 1.0, 2.0 };
    const output = try ctx.forward(&network, &input, std.testing.allocator);
    defer std.testing.allocator.free(output);

    try std.testing.expectEqual(@as(usize, 1), output.len);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), output[0], 0.0001);
}

test "InferenceContext - test different activation functions" {
    var ctx = InferenceContext.init(std.testing.allocator);
    defer ctx.deinit();

    // Create a simple 2->1 network
    var network = try Network.init(std.testing.allocator, &[_]usize{ 2, 1 });
    defer network.deinit();

    // Set up weights and biases for predictable results
    network.layers[0].weights[0][0] = 0.5;
    network.layers[0].weights[1][0] = -0.5;
    network.layers[0].biases[0] = 0.0;

    const input = [_]f32{ 1.0, 1.0 };

    // Test ReLU
    network.layers[0].activation = .ReLu;
    const output_relu = try ctx.forward(&network, &input, std.testing.allocator);
    defer std.testing.allocator.free(output_relu);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output_relu[0], 0.0001);

    // Test Sigmoid
    network.layers[0].activation = .Sigmoid;
    const output_sigmoid = try ctx.forward(&network, &input, std.testing.allocator);
    defer std.testing.allocator.free(output_sigmoid);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output_sigmoid[0], 0.0001);

    // Test Tanh
    network.layers[0].activation = .Tanh;
    const output_tanh = try ctx.forward(&network, &input, std.testing.allocator);
    defer std.testing.allocator.free(output_tanh);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output_tanh[0], 0.0001);
}
