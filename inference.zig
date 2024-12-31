//! Inference context.
//! Simple forward passess used for prediction when W&Bs are fixed.
const std = @import("std");
const Network = @import("network.zig").Network;

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
        self.arena.reset();
    }

    /// Mutate the network state to be after a single pass
    pub fn forward(self: *InferenceContext, network: *Network) void {
        // input = ActivationFn(Layer Input)
        // layer output = input * weights of layer + bias
        // Need dot product of matrices, need input to be stored somewhere, need network state to be stored somewhere

        // Reset our temp memory once complete
        self.reset();
    }
};

test "Inference context" {
    var test_network = try Network.init(std.testing.allocator);
    defer test_network.deinit();

    var ctx = try InferenceContext.init(std.testing.allocator);
    defer ctx.deinit();

    ctx.forward(test_network);
}
