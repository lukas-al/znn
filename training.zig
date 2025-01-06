//! Training context.
//! Used to update the model based on an error term, and then persist

const std = @import("std");

pub const TrainingContext = struct {
    arena: std.heap.ArenaAllocator,
    // Store layer activations and pre-activation values for backprop
    layer_activations: [][]f32,
    layer_z_values: [][]f32,
    // Store gradients
    weight_gradients: [][]f32,
    bias_gradients: []f32,

    /// Initialise with backing allocator. Caller owns memory.
    pub fn init(backing_allocator: std.mem.Allocator) TrainingContext {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .layer_activations = undefined,
            .layer_z_values = undefined,
            .weight_gradients = undefined,
            .bias_gradients = undefined,
        };
    }

    /// Deinitialise
    pub fn deinit(self: *TrainingContext) void {
        self.arena.deinit();
    }

    /// Reset the memory & context
    pub fn reset(self: *TrainingContext) void {
        _ = self.arena.reset(.free_all);
    }
};
