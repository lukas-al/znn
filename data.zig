//! Data loader for MNIST
const std = @import("std");

/// Decode the IDX types stored as a hex value
pub fn TypeFromHex(hex: u8) ?type {
    return switch (hex) {
        0x08 => u8,
        0x09 => i8,
        0x0B => i16,
        0x0C => i32,
        0x0D => f32,
        0x0E => f64,
        else => null,
    };
}

/// Image struct
pub const Image = struct {
    width: u32 = 28,
    height: u32 = 28,
    pixels: []u8,
};

/// Create a single dataset struct
pub const Dataset = struct {
    images: []Image,
    labels: []u8,
};

/// MNIST Loaders and such
pub const MNISTData = struct {
    // !IDX FORMAT INFO: https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
    /// Struct for MetadataReturn types.
    const MetadataReturn = union(enum) { image_file: struct {
        type_info: u8,
        num_images: u32,
        width: u32,
        height: u32,
    }, label_file: struct {
        type_info: u8,
        num_labels: u32,
    } };

    /// Some custom error types
    const MNISTError = error{
        InvalidDataType,
        InvalidDimensions,
        InvalidFileType,
        IncompleteRead,
    };

    /// Read the header of our file
    fn readMetadata(reader: anytype) !MetadataReturn {
        const magic = try reader.readInt(u32, std.builtin.Endian.big); // Read magic number which contains type info and dimensionality
        const magic_bytes = @as([4]u8, @bitCast(magic)); // Convert to array of bytes for easy access

        const data_type = magic_bytes[1];
        const num_dims = magic_bytes[0];

        // // Validate data type
        // if (TypeFromHex(data_type) == null) {
        //     return MNISTError.InvalidDataType;
        // }

        switch (num_dims) {
            // label
            1 => {
                const num_labels = try reader.readInt(u32, std.builtin.Endian.big);
                return MetadataReturn{ .label_file = .{ .type_info = data_type, .num_labels = num_labels } };
            },
            3 => {
                const num_images = try reader.readInt(u32, std.builtin.Endian.big);
                const height = try reader.readInt(u32, std.builtin.Endian.big);
                const width = try reader.readInt(u32, std.builtin.Endian.big);
                return MetadataReturn{ .image_file = .{
                    .type_info = data_type,
                    .num_images = num_images,
                    .height = @intCast(height),
                    .width = @intCast(width),
                } };
            },
            else => {
                std.debug.print("Magic metadata is {any} \n", .{magic_bytes});
                std.debug.print("Number of dimensions is {} \n", .{num_dims});
                return MNISTError.InvalidDimensions;
            },
        }
    }

    /// Load image data from IDX file
    pub fn loadImages(filepath: []const u8, allocator: std.mem.Allocator) ![]Image {
        const file = try std.fs.cwd().openFile(filepath, .{});
        defer file.close();
        var file_reader = file.reader();

        const metadata = try readMetadata(&file_reader);
        switch (metadata) {
            .image_file => |info| {
                const num_images = info.num_images;
                const height = info.height;
                const width = info.width;

                var images = try allocator.alloc(Image, num_images);
                errdefer allocator.free(images);

                for (0..num_images) |i| {
                    var pixels = try allocator.alloc(u8, width * height);
                    errdefer allocator.free(pixels);
                    _ = &pixels;

                    const num_bytes_read = try file_reader.readAll(pixels);
                    if (num_bytes_read == 0) break; // readAll returns 0 on the end of stream
                    if (num_bytes_read != width * height) return MNISTError.IncompleteRead;

                    images[i] = Image{
                        .width = width,
                        .height = height,
                        .pixels = pixels,
                    };
                }

                return images;
            },
            .label_file => return MNISTError.InvalidFileType,
        }
    }

    /// Load label data from IDX file
    pub fn loadLabels(filepath: []const u8, allocator: std.mem.Allocator) ![]u8 {
        const file = try std.fs.cwd().openFile(filepath, .{});
        defer file.close();
        var file_reader = file.reader();

        const metadata = try readMetadata(&file_reader);
        switch (metadata) {
            .label_file => |info| {
                const num_labels = info.num_labels;
                var labels = try allocator.alloc(u8, num_labels);
                _ = &labels;
                errdefer allocator.free(labels);

                const num_bytes_read = try file_reader.readAll(labels);
                if (num_bytes_read != num_labels) {
                    allocator.free(labels);
                    return MNISTError.IncompleteRead;
                }

                return labels;
            },
            .image_file => return MNISTError.InvalidFileType,
        }
    }

    const TrainTest = struct {
        train_data: Dataset = undefined,
        test_data: Dataset = undefined,
    };

    /// Load both training and test datasets
    pub fn loadMNIST(
        train_images_path: []const u8,
        train_labels_path: []const u8,
        test_images_path: []const u8,
        test_labels_path: []const u8,
        allocator: std.mem.Allocator,
    ) !TrainTest {
        // Load training data
        const train_dataset = Dataset{
            .images = try loadImages(train_images_path, allocator),
            .labels = try loadLabels(train_labels_path, allocator),
        };

        // Load test data
        const test_dataset = Dataset{
            .images = try loadImages(test_images_path, allocator),
            .labels = try loadLabels(test_labels_path, allocator),
        };

        return .{ .train_data = train_dataset, .test_data = test_dataset };
    }
};

// Tests
test "Load and validate MNIST data" {
    // Setup paths to your MNIST data files
    const train_images_path = "data/train-images.idx3-ubyte";
    const train_labels_path = "data/train-labels.idx1-ubyte";
    const test_images_path = "data/t10k-images.idx3-ubyte";
    const test_labels_path = "data/t10k-labels.idx1-ubyte";

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const mem_alloc = arena.allocator();

    // Load the data
    const data = try MNISTData.loadMNIST(
        train_images_path,
        train_labels_path,
        test_images_path,
        test_labels_path,
        mem_alloc,
    );
    // defer {
    //     data.train_data.deinit(alloc);
    //     data.test_data.deinit(alloc);
    // }

    // Verify training data
    try validateDataset("Training", &data.train_data, 60000);

    // Verify test data
    try validateDataset("Test", &data.test_data, 10000);

    // Print some example digits
    printExampleDigits(&data.train_data);
}

/// Helper function to validate dataset's structure and contents
fn validateDataset(name: []const u8, dataset: *const Dataset, expected_size: usize) !void {
    std.debug.print("\nValidating {s} Dataset:\n", .{name});

    // Check dataset size
    try std.testing.expect(dataset.images.len == expected_size);
    try std.testing.expect(dataset.labels.len == expected_size);
    std.debug.print("✓ Dataset size matches expected ({d} samples)\n", .{expected_size});

    // Validate image dimensions
    for (dataset.images, 0..) |image, i| {
        try std.testing.expect(image.width == 28);
        try std.testing.expect(image.height == 28);
        try std.testing.expect(image.pixels.len == 28 * 28);

        // Validate pixel values are in range 0-255
        var pixel_sum: u32 = 0;
        for (image.pixels) |pixel| {
            try std.testing.expect(pixel <= 255);
            pixel_sum += pixel;
        }

        // Validate we don't have a bunch of 0s
        try std.testing.expect(pixel_sum / image.pixels.len >= 1);

        // Validate label is 0-9
        try std.testing.expect(dataset.labels[i] <= 9);
    }
    std.debug.print("✓ All images are 28x28 with valid pixel values\n", .{});
    std.debug.print("✓ All labels are valid digits (0-9)\n", .{});

    // Basic statistical checks
    var label_counts = [_]usize{0} ** 10;
    var total_pixel_sum: usize = 0;
    var non_zero_pixels: usize = 0;

    for (dataset.images, dataset.labels) |image, label| {
        label_counts[label] += 1;
        for (image.pixels) |pixel| {
            total_pixel_sum += pixel;
            if (pixel > 0) non_zero_pixels += 1;
        }
    }

    // Verify we have samples of all digits
    for (label_counts, 0..) |count, digit| {
        try std.testing.expect(count > 0);
        std.debug.print("  Digit {d}: {d} samples\n", .{ digit, count });
    }

    const avg_pixel_value = @as(f32, @floatFromInt(total_pixel_sum)) /
        @as(f32, @floatFromInt(expected_size * 28 * 28));
    const pixel_density = @as(f32, @floatFromInt(non_zero_pixels)) /
        @as(f32, @floatFromInt(expected_size * 28 * 28)) * 100;

    std.debug.print("✓ Average pixel value: {d:.2}\n", .{avg_pixel_value});
    std.debug.print("✓ Pixel density: {d:.1}%\n", .{pixel_density});
}

/// Helper function to print example digits as ASCII art
fn printExampleDigits(dataset: *const Dataset) void {
    std.debug.print("\nExample Digits:\n", .{});
    const examples_to_print = 5;

    for (0..examples_to_print) |i| {
        const image = dataset.images[i];
        const label = dataset.labels[i];

        std.debug.print("\nDigit {d}:\n", .{label});
        for (0..28) |y| {
            for (0..28) |x| {
                const pixel = image.pixels[y * 28 + x];
                // Convert pixel value to ASCII character based on intensity
                const char: u8 = if (pixel < 50) ' ' else if (pixel < 100) '.' else if (pixel < 150) '+' else if (pixel < 200) '*' else '#';
                std.debug.print("{c}", .{char});
            }
            std.debug.print("\n", .{});
        }
    }
}
