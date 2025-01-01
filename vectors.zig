//! Vector calculations
const std = @import("std");
const Network = @import("network").Network;

pub const VecOps = struct {
    // y = Ax + b (simple matrix multiplication with a bias)
    // Modifies the memory of y in-place
    pub fn linearForward(y: []f32, weights: []const []const f32, x: []const f32, b: []const f32) void {
        // For each output
        for (y, 0..) |*y_i, i| {
            y_i.* = b[i]; // Initialise with the bias
            // std.debug.print("Output {}: Starting with bias {d}\n", .{ i, b[i] });

            // For each input
            for (weights, 0..) |w_row, j| {
                y_i.* += w_row[i] * x[j];
                // std.debug.print("  Input {}: weight={d}, x={d}, contribution={d}, running_sum={d}\n", .{ j, w_row[i], x[j], w_row[i] * x[j], y_i.* });
            }
        }
    }
};

test "VecOps - linearForward basic operation" {
    // Test a 2->3 network layer
    var y = [_]f32{ 0.0, 0.0, 0.0 }; // 3 outputs
    const x = [_]f32{ 1.0, 2.0 }; // 2 inputs
    const b = [_]f32{ 0.1, 0.2, 0.3 }; // 3 biases

    // weights[i][j] where:
    // i is the input index (2 inputs)
    // j is the output index (3 outputs)
    const weights = [_][]const f32{
        &[_]f32{ 1.0, 0.5, 0.0 }, // weights from input 0 to all outputs
        &[_]f32{ 2.0, 0.5, 1.0 }, // weights from input 1 to all outputs
    };

    VecOps.linearForward(&y, &weights, &x, &b);

    // y[0] = 1.0*x[0] + 2.0*x[1] + b[0] = 1.0*1.0 + 2.0*2.0 + 0.1 = 1.0 + 4.0 + 0.1 = 5.1
    // y[1] = 0.5*x[0] + 0.5*x[1] + b[1] = 0.5*1.0 + 0.5*2.0 + 0.2 = 0.5 + 1.0 + 0.2 = 1.7
    // y[2] = 0.0*x[0] + 1.0*x[1] + b[2] = 0.0*1.0 + 1.0*2.0 + 0.3 = 0.0 + 2.0 + 0.3 = 2.3
    try std.testing.expectApproxEqAbs(@as(f32, 5.1), y[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.7), y[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.3), y[2], 0.0001);
}
