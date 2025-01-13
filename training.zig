//! Training the network

const std = @import("std");
const Network = @import("network.zig").Network;
const ProgressBar = @import("helpers.zig").ProgressBar;

/// Train the network on a dataset. Caller owns memory.
pub fn trainNetwork(
    network: *Network,
    inputs: []const []const f32, // N inputs of dimension X
    targets: []const []const f32, // N targets of dimension Y
    max_epochs: usize,
    learning_rate: f32,
    error_threshold: f32,
    min_epochs: usize,
    verbose: bool,
) !void {
    // Validate some inputs
    if (inputs.len == 0 or inputs.len != targets.len) return error.InvalidInput;

    // Create a progress bar
    var progress = if (verbose) ProgressBar.init(max_epochs, 30) else undefined;

    // Create a temp buffer for our results
    var temp_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer temp_arena.deinit();
    const temp_allocator = temp_arena.allocator();

    var epoch: usize = 0;
    var current_err: f32 = 0;

    // Iterate for our epochs
    while (epoch < max_epochs and current_err > error_threshold or epoch < min_epochs) : (epoch += 1) {

        // 1) Iterate over our training and test set
        for (inputs, targets) |input, target| {
            // 1) Forward pass
            const output = try network.forward(input, temp_allocator);
            defer temp_allocator.free(output);

            // 2) Calculate error for the sample (MSE)
            var sample_err: f32 = 0;
            for (output, target) |out, targ| {
                const diff = out - targ;
                sample_err += diff * diff;
            }
            sample_err /= @as(f32, @floatFromInt(output.len));
            current_err += sample_err;

            // 3) Update our network state
            try network.backward(input, target, learning_rate);
        }

        // 2) Calculate our average error over the dataset
        current_err /= @as(f32, @floatFromInt(inputs.len));

        // 3) Report
        if (verbose) {
            const status = try std.fmt.allocPrint(temp_allocator, "Error: {d:.6} ", .{current_err});
            try progress.update(epoch, status);
        }
    }
    // Report
    if (verbose) {
        std.debug.print("Training complete after {} epochs with final error: {} \n", .{ epoch, current_err });
    }
}

test "Integration Test: Does it solve a real problem?" {
    const layer_sizes = [_]usize{ 2, 4, 6, 1 };
    var network = try Network.init(std.testing.allocator, &layer_sizes, .Sigmoid);
    defer network.deinit();

    // XOR training data
    const inputs = [_][]const f32{
        &[_]f32{ 0, 0 },
        &[_]f32{ 0, 1 },
        &[_]f32{ 1, 0 },
        &[_]f32{ 1, 1 },
        &[_]f32{ 0, 0 },
        &[_]f32{ 0, 1 },
        &[_]f32{ 1, 0 },
        &[_]f32{ 1, 1 },
    };
    const targets = [_][]const f32{
        &[_]f32{0},
        &[_]f32{1},
        &[_]f32{1},
        &[_]f32{0},
        &[_]f32{0},
        &[_]f32{1},
        &[_]f32{1},
        &[_]f32{0},
    };

    try trainNetwork(&network, &inputs, &targets, 1000, 0.5, 0.01, 15, true);

    // Verify the network learned XOR
    for (inputs, targets) |input, target| {
        const output = try network.forward(input, std.testing.allocator);
        defer std.testing.allocator.free(output);

        try std.testing.expectApproxEqAbs(target[0], output[0], 0.2);
    }
}
