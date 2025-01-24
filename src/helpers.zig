const std = @import("std");

/// A terminal progress bar that can show progress and a status message
pub const ProgressBar = struct {
    total: usize,
    current: usize,
    width: usize,
    writer: std.fs.File.Writer,

    const Self = @This();

    pub fn init(total: usize, width: usize) Self {
        return .{
            .total = total,
            .current = 0,
            .width = width,
            .writer = std.io.getStdOut().writer(),
        };
    }

    /// Update progress and display the bar with a status message
    pub fn update(self: *Self, current: usize, status: []const u8) !void {
        self.current = current;

        // Move to start of line
        try self.writer.writeAll("\r");

        const progress = @as(f32, @floatFromInt(self.current)) / @as(f32, @floatFromInt(self.total));
        const filled_width = @as(usize, @intFromFloat(progress * @as(f32, @floatFromInt(self.width))));

        // Print progress percentage and opening bracket
        try self.writer.print("{d:>3.0}% [", .{progress * 100});

        // Print the filled portion
        var i: usize = 0;
        while (i < filled_width) : (i += 1) {
            try self.writer.writeAll("=");
        }

        // Print the unfilled portion
        while (i < self.width) : (i += 1) {
            try self.writer.writeAll(" ");
        }

        // Print closing bracket and status (but no newline)
        try self.writer.print("] {s}", .{status});
    }
};

test "progress bar basic functionality" {
    var progress = ProgressBar.init(100, 20);

    var i: usize = 0;
    while (i <= 100) : (i += 20) {
        const status = std.fmt.allocPrint(std.testing.allocator, "Processing... error={d:.4}", .{1.0 / @as(f32, @floatFromInt(i + 1))}) catch unreachable;
        defer std.testing.allocator.free(status);

        try progress.update(i, status);

        // Add a 500ms delay
        std.time.sleep(500 * std.time.ns_per_ms);
    }
}
