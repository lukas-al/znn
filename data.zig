//! Data loader for MNIST
const std = @import("std");

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

pub const MNISTData = struct {
    training_images: [][]f32,
    training_labels: []f32,
    test_images: [][]f32,
    test_labels: []f32,

    // !IDX FORMAT INFO: https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
    // pub fn deinit(self: *MNISTData) void {
    //     // Free the training images arrays
    //     for (self.training_images) |image| {
    //         self.allocator.free(image);
    //     }
    //     self.allocator.free(self.training_images);
    //     self.allocator.free(self.training_labels);

    //     // Free the test images arrays
    //     for (self.test_images) |image| {
    //         self.allocator.free(image);
    //     }
    //     self.allocator.free(self.test_images);
    //     self.allocator.free(self.test_labels);
    // }

    // /// Extract out same logic
    // fn readHeader(reader: anytype) !struct { u32, u32 } {
    //     // Read magic number and number of items (big endian)
    //     const magic = try reader.readIntBig(u32);
    //     const num_items = try reader.readIntBig(u32);
    //     return .{ magic, num_items };
    // }

    // /// Extract out logic
    // fn readImageDimensions(reader: anytype) !struct { u32, u32 } {
    //     // Read number of rows and columns (big endian)
    //     const num_rows = try reader.readIntBig(u32);
    //     const num_cols = try reader.readIntBig(u32);
    //     return .{ num_rows, num_cols };
    // }

    /// Extract out logic
    fn readMetadata(reader: anytype) !struct { u32, u32, u32, u32 } {
        // Read out all metadata from the file at once
        const magic = try reader.readIntBig(u32);
        const num_items = try reader.readIntBig(u32);
        const num_rows = try reader.readIntBig(u32);
        const num_cols = try reader.readIntBig(u32);

        return .{ magic, num_items, num_rows, num_cols };
    }

    /// Logic to decode each individual image and shape it into the NN input data
    pub fn decodeImageBinary(filepath: []u8, allocator: std.mem.Allocator) [][]f32 {
        const file = try std.fs.cwd().openFile(filepath);
        defer file.close();
        var file_reader = file.reader();

        const file_metadata = try readMetadata(file_reader);

        // Check that the metadata is correct
        // First two bytes are set to 0.
        if (file_metadata[0] != 0) return error.InvalidInput;
        if (file_metadata[1] != 0) return error.InvalidInput;

        // Third byte defines data type.
        const data_type = TypeFromHex(file_metadata[2]);

        // Fourth defines number of dimensions of the stored arrays
        const array_dim = file_metadata[3];
        if (array_dim != )

        // Assign a memory space array of the expected file length

    }

    /// Logic to decode
    pub fn decodeLabelBinary(filepath: []u8) []f32 {
        //
    }

    // pub fn load(train_images_path: []u8, train_labels_path: []u8, test_images_path: []u8, test_labels_path: []u8, allocator: std.mem.Allocator) !MNISTData {
    //     const train_images_file = try std.fs.cwd().openFile(train_images_path);
    //     defer train_images_file.close();
    //     var train_images_reader = train_images_file.reader();

    //     const train_labels_file = try std.fs.cwd().openFile(train_labels_path);
    //     defer train_labels_file.close();
    //     var train_labels_reader = train_labels_file.reader();

    //     const test_images_file = try std.fs.cwd().openFile(test_images_path);
    //     defer test_images_file.close();
    //     var test_images_reader = test_images_file.reader();

    //     const test_labels_file = try std.fs.cwd().openFile(test_labels_path);
    //     defer test_labels_file.close();
    //     var test_labels_reader = test_labels_file.reader();

    //     // Now to decode the datasets from their binary format. We don't really care about most of the information in the headers.
    //     // Read headers
    //     const train_images_metadata = readMetadata(train_images_reader);
    //     const train_labels_metadata = readMetadata(train_labels_reader);
    //     const test_images_metadata = readMetadata(test_images_reader);
    //     const test_labels_metadata = readMetadata(test_labels_reader);
    // }
};
