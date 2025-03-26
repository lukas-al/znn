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

    // Create a temp buffer for our results
    var temp_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer temp_arena.deinit();
    const temp_allocator = temp_arena.allocator();

    var epoch: usize = 0;
    var epoch_err: f32 = 0.0;
    var training_err: f32 = 9999.0;
    var timer = try std.time.Timer.start();

    // Iterate for our epochs
    std.debug.print("Iterating to max epoch {} ...\n", .{max_epochs});
    while (epoch < max_epochs and epoch_err > error_threshold or epoch < min_epochs) : (epoch += 1) {
        // Reset our epoch error
        epoch_err = 0.0;

        // Create a progress bar for the dataset that we're going over
        var progress = if (verbose) ProgressBar.init(inputs.len, 40) else undefined;

        // 1) Iterate over our training and test set -> in totality for each epoch
        for (inputs, targets, 0..) |input, target, i| {
            // 1) Update our network state
            training_err = try network.backward(input, target, learning_rate);

            epoch_err += training_err;

            // 2) Report
            if (verbose) {
                const status = try std.fmt.allocPrint(temp_allocator, "|| Epoch no. {} || Error: {d:.4} ", .{ epoch, training_err });
                try progress.update(i, status);
            }

            // 3) DEBUG: LETS SEE IF THE WEIGHTS ARE UPDATING CORRECTLY
            // Print a specific weight to track changes during training
            // std.debug.print("Layer 1 Weight[0][0]: {d:.8}, Layer 2 Weight[0][1]: {d:.8}\n", .{ network.layers[1].weights[0][0], network.layers[2].weights[0][1] });
            // std.debug.print("Error: {d:.6}, Learning rate: {d:.6}\n", .{ training_err, learning_rate });
        }

        // Get the average training set error for this epoch
        epoch_err /= @as(f32, @floatFromInt(inputs.len));

        // Write a new line for the next epoch
        std.debug.print("\n", .{});
        // 3) Return the time it took for the epoch, and project out
        if (verbose) {
            // Calculate and log epoch duration
            const epoch_time_ns = timer.lap();
            const epoch_time_ms = epoch_time_ns / 1_000_000;
            std.debug.print("\nEpoch {d} completed in {d} ms ({d:.2} s) || Average epoch error: {d:.6}\n", .{ epoch + 1, epoch_time_ms, @as(f32, @floatFromInt(epoch_time_ms)) / 1000.0, epoch_err });
        }
    }
    // Report
    if (verbose) {
        std.debug.print("Training complete after {} epochs with average epoch error: {d:.6} \n", .{ epoch, epoch_err / @as(f32, @floatFromInt(epoch)) });
    }
}

// /// Same as train network, but passes evaluates the error stop condition based on batches
// pub fn trainNetworkBatched(
//     network: *Network,
//     inputs: []const []const f32, // N inputs of dimension X
//     targets: []const []const f32, // N targets of dimension Y
//     batch_size: usize,
//     max_epochs: usize,
//     learning_rate: f32,
//     error_threshold: f32,
//     min_epochs: usize,
//     verbose: bool,
// ) !void {
//     // Validate some inputs
//     if (inputs.len == 0 or inputs.len != targets.len) return error.InvalidInput;

//     // Create a temp buffer for our results
//     var temp_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer temp_arena.deinit();
//     const temp_allocator = temp_arena.allocator();

//     var epoch: usize = 0;
//     var current_err: f32 = 0;

//     var timer = try std.time.Timer.start();

//     // Iterate for our epochs
//     std.debug.print("Iterating to max epoch {} ...\n", .{max_epochs});
//     while (epoch < max_epochs and current_err > error_threshold or epoch < min_epochs) : (epoch += 1) {

//         // Create a progress bar for the dataset that we're going over
//         var progress = if (verbose) ProgressBar.init(inputs.len, 40) else undefined;

//         // 1) Iterate over our training and test set -> in totality for each epoch
//         for (inputs, targets, 0..) |input, target, i| {
//             // 1) Forward pass
//             const output = try network.forward(input, temp_allocator);
//             defer temp_allocator.free(output);

//             // 1a) print our output
//             std.debug.print("Output: {any} \n", .{output});
//             // std.debug.print("Input: {any} \n", .{input});

//             // 2) Calculate error for the sample (MSE)
//             var sample_err: f32 = 0;
//             for (output, target) |out, targ| {
//                 const diff = out - targ;
//                 sample_err += diff * diff;
//             }
//             sample_err /= @as(f32, @floatFromInt(output.len));
//             current_err = sample_err;

//             // 3) Update our network state
//             try network.backward(input, target, learning_rate);

//             // 3) Report
//             if (verbose) {
//                 const status = try std.fmt.allocPrint(temp_allocator, "|| Epoch no. {} || Error: {d:.6} ", .{ epoch, sample_err });
//                 try progress.update(i, status);
//             }
//         }

//         // 2) Calculate our average error over the dataset
//         current_err /= @as(f32, @floatFromInt(inputs.len));

//         // 3) Return the time it took for the epoch, and project out
//         const epoch_time = timer.lap();
//         // std.debug.print("Current epoch took {}s", .{(epoch_time / @as(u64, @intCast(std.time.ns_per_s)))});
//         std.debug.print("   \n Current epoch took {}ns \n", .{epoch_time / @as(u64, @intCast(std.time.ns_per_s))});
//     }
//     // Report
//     if (verbose) {
//         std.debug.print("Training complete after {} epochs with final error: {d:.6} \n", .{ epoch, current_err });
//     }
// }

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
        &[_]f32{ 0, 0 },
        &[_]f32{ 0, 1 },
        &[_]f32{ 1, 0 },
        &[_]f32{ 1, 1 },
        &[_]f32{ 0, 0 },
        &[_]f32{ 0, 1 },
        &[_]f32{ 1, 0 },
        &[_]f32{ 1, 1 },
        &[_]f32{ 0, 0 },
        &[_]f32{ 0, 1 },
        &[_]f32{ 1, 0 },
        &[_]f32{ 1, 1 },
        &[_]f32{ 0, 0 },
        &[_]f32{ 0, 1 },
        &[_]f32{ 1, 0 },
        &[_]f32{ 1, 1 },
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
        &[_]f32{0},
        &[_]f32{1},
        &[_]f32{1},
        &[_]f32{0},
        &[_]f32{0},
        &[_]f32{1},
        &[_]f32{1},
        &[_]f32{0},
        &[_]f32{0},
        &[_]f32{1},
        &[_]f32{1},
        &[_]f32{0},
        &[_]f32{0},
        &[_]f32{1},
        &[_]f32{1},
        &[_]f32{0},
        &[_]f32{0},
        &[_]f32{1},
        &[_]f32{1},
        &[_]f32{0},
        &[_]f32{0},
        &[_]f32{1},
        &[_]f32{1},
        &[_]f32{0},
    };

    try trainNetwork(&network, &inputs, &targets, 1000, 0.5, 0.0001, 15, true);

    // Verify the network learned XOR
    for (inputs, targets) |input, target| {
        const output = try network.forward(input, std.testing.allocator);
        defer std.testing.allocator.free(output);

        for (output, target) |output_val, target_val| {
            try std.testing.expectApproxEqAbs(target_val, output_val, 0.1);
        }
        // try std.testing.expectApproxEqAbs(target[0], output[0], 0.5);
    }
}
