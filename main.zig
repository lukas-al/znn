//! Main entry point for module
const std = @import("std");
const data_loader = @import("src/data.zig");
const Network = @import("src/network.zig").Network;
const ProgressBar = @import("src/helpers.zig").ProgressBar;
const training = @import("src/training.zig");

/// Convert MNIST image to input array for neural network
fn imageToInput(image: data_loader.Image, allocator: std.mem.Allocator) ![]f32 {
    var input = try allocator.alloc(f32, image.width * image.height);
    for (image.pixels, 0..) |pixel, i| {
        // Normalize pixel values to [0,1] range
        input[i] = @as(f32, @floatFromInt(pixel)) / 255.0;
    }
    return input;
}

/// Convert MNIST label to target array for neural network
fn labelToTarget(label: u8, allocator: std.mem.Allocator) ![]f32 {
    // One-hot encode the label -> fill array with 0 and then set the right label to 1
    var target = try allocator.alloc(f32, 10);
    @memset(target, 0);
    target[label] = 1;
    return target;
}

pub fn main() !void {

    // Load the data
    const train_images_path = "data/train-images.idx3-ubyte";
    const train_labels_path = "data/train-labels.idx1-ubyte";
    const test_images_path = "data/t10k-images.idx3-ubyte";
    const test_labels_path = "data/t10k-labels.idx1-ubyte";

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Load the data
    std.debug.print("Loading training data... \n", .{});
    const data = try data_loader.MNISTData.loadMNIST(
        train_images_path,
        train_labels_path,
        test_images_path,
        test_labels_path,
        allocator,
    );

    // Transform our input data
    std.debug.print("Preparing input data \n", .{});
    var inputs = try allocator.alloc([]f32, data.train_data.images.len);
    var targets = try allocator.alloc([]f32, data.train_data.labels.len);
    var test_inputs = try allocator.alloc([]f32, data.train_data.images.len);
    var test_targets = try allocator.alloc([]f32, data.train_data.labels.len);

    var train_progress = ProgressBar.init(data.train_data.images.len, 40);
    for (data.train_data.images, data.train_data.labels, 0..) |image, label, i| {
        try train_progress.update(i, "Converting training data...");
        inputs[i] = try imageToInput(image, allocator);
        targets[i] = try labelToTarget(label, allocator);
    }

    var test_progress = ProgressBar.init(data.test_data.images.len, 40);
    for (data.test_data.images, data.test_data.labels, 0..) |test_image, test_label, i| {
        try test_progress.update(i, "Converting testing data...");
        test_inputs[i] = try imageToInput(test_image, allocator);
        test_targets[i] = try labelToTarget(test_label, allocator);
    }

    // Instantiate our network
    std.debug.print("\n Instantiating our network... \n", .{});
    const layer_sizes = [_]usize{ 784, 128, 64, 10 };
    var network = try Network.init(allocator, &layer_sizes, .ReLu);
    defer network.deinit();

    // Train our network
    std.debug.print("Training... \n", .{});
    try training.trainNetwork(
        &network,
        inputs,
        targets,
        2,
        0.5,
        0.1,
        1,
        true,
    );

    // Evaluate on our test set. Calculate RMSE
    std.debug.print("\nEvaluating on test set...\n", .{});
    var test_err: f32 = 0.0;
    var sample_err: f32 = 0.0;
    for (test_inputs, test_targets, 0..) |image, label, i| {
        sample_err = 0;
        const output = try network.forward(image, allocator);
        for (output, label) |out_i, targ_i| {
            const diff = out_i - targ_i; // Calculate err
            sample_err += diff * diff; // Square it
        }
        sample_err /= @as(f32, @floatFromInt(output.len)); // Mean
        test_err += std.math.sqrt(sample_err); // Root

        const status = try std.fmt.allocPrint(allocator, "|| Test evaluation error: {}", .{sample_err});
        try test_progress.update(i, status);
    }

    test_err = test_err / @as(f32, @floatFromInt(data.test_data.images.len));
    std.debug.print("Overall error on test is: {d:.6}", .{test_err});
}
