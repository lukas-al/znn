//! Data loader for MNIST
const std = @import("std");

pub const MNISTData = struct {
    training_images: [][]f32,
    training_labels: []f32,
    test_images: [][]f32,
    test_labels: []f32,

    pub fn deinit(self: *MNISTData) void {
        // Free the training images arrays
        for (self.training_images) |image| {
            self.allocator.free(image);
        }
        self.allocator.free(self.training_images);
        self.allocator.free(self.training_labels);

        // Free the test images arrays
        for (self.test_images) |image| {
            self.allocator.free(image);
        }
        self.allocator.free(self.test_images);
        self.allocator.free(self.test_labels);
    }

    pub fn load(train_images_path: []u8, train_labels_path: []u8, test_images_path: []u8, test_labels_path: []u8, allocator: std.mem.Allocator) !MNISTData {
        const train_images_file = try std.fs.cwd().openFile(train_images_path);
        defer train_images_file.close();
        var train_images_reader = train_images_file.reader();

        const train_labels_file = try std.fs.cwd().openFile(train_labels_path);
        defer train_labels_file.close();
        var train_labels_reader = train_labels_file.reader();

        const test_images_file = try std.fs.cwd().openFile(test_images_path);
        defer test_images_file.close();
        var test_images_reader = test_images_file.reader();

        const test_labels_file = try std.fs.cwd().openFile(test_labels_path);
        defer test_labels_file.close();
        var test_labels_reader = test_labels_file.reader();

        // Now to decode the dataset

    }
};
