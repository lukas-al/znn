//! Vector calculations
const std = @import("std");
const Network = @import("network").Network;

pub fn linearForward(y: []f32, weights: []const []const f32, x: []const f32, b: []const f32) void {
    for (y, 0..) |*y_i, i| {
        y_i.* = b[i];
        for (weights, 0..) |row, j| {
            y_i.* += row[i] * x[j];
        }
    }
}

test "VecOps - linearForward basic operation" {
    // Test a 2->3 network layer
    var y = [_]f32{ 0.0, 0.0, 0.0 }; // 3 outputs
    const x = [_]f32{ 1.0, 2.0 }; // 2 inputs
    const b = [_]f32{ 0.1, 0.2, 0.3 }; // 3 biases

    const weights = [_][]const f32{
        &[_]f32{ 1.0, 0.5, 0.0 }, // weights from input 0 to all outputs -> first neuron
        &[_]f32{ 2.0, 0.5, 1.0 }, // weights from input 1 to all outputs -> second neuron
    };

    linearForward(&y, &weights, &x, &b);

    try std.testing.expectApproxEqAbs(5.1, y[0], 0.0001);
    try std.testing.expectApproxEqAbs(1.7, y[1], 0.0001);
    try std.testing.expectApproxEqAbs(2.3, y[2], 0.0001);
}

test "VecOps - linearForward on a 2-2 network" {
    var y = [_]f32{ 0.0, 0.0 };
    const x = [_]f32{ 0.1, 0.5 };
    const b = [_]f32{ 0.25, 0.25 };

    const weights = [_][]const f32{
        &[_]f32{ 0.1, 0.2 }, // weights from input 0 to all outputs -> weights of the first neuron
        &[_]f32{ 0.3, 0.4 }, // weights from input 1 to all outputs -> weights of the second neuron
    };

    linearForward(&y, &weights, &x, &b);

    try std.testing.expectApproxEqAbs(0.41, y[0], 1e-3);
    try std.testing.expectApproxEqAbs(0.47, y[1], 1e-3);
}
