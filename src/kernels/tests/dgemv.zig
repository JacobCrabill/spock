//! Tests for the dense double-precision matrix-vector kernel.

const std = @import("std");
const spock = @import("../../spock.zig");
const harness = @import("harness.zig");

const dgemv = @import("../blas/dgemv.zig");
const dgemv_spv = @embedFile("spock/dgemv.spv");

test "dgemv" {
    const debug_print: bool = false;

    // Deliberately non-square, so a row/column mix-up cannot pass by accident.
    const M: u32 = 8;
    const N: u32 = 4;

    // A_ij = j + i*N, x_j = j, so y_i = sum_j (j + i*N) * j = 14 + 24*i
    const expected_ab = [8]f64{ 14.0, 38.0, 62.0, 86.0, 110.0, 134.0, 158.0, 182.0 };
    // Re-running with alpha=2, beta=3 over that result: 2*(14+24i) + 3*(14+24i)
    const expected_scaled = [8]f64{ 70.0, 190.0, 310.0, 430.0, 550.0, 670.0, 790.0, 910.0 };

    const ctx = try harness.context();

    var Amat = try spock.Buffer(f64).create(ctx, M * N);
    defer Amat.deinit();
    var xvec = try spock.Buffer(f64).create(ctx, N);
    defer xvec.deinit();
    var yvec = try spock.Buffer(f64).create(ctx, M);
    defer yvec.deinit();

    for (Amat.slice(), 0..) |*v, i| v.* = @floatFromInt(i);
    for (xvec.slice(), 0..) |*v, i| v.* = @floatFromInt(i);
    yvec.fill(0);

    if (debug_print) harness.printMatrix("A", Amat.slice(), M, N);

    const groups = harness.groupsFor(M, dgemv.WgSize.x);
    const buffers = [_]spock.vk.Buffer{ Amat.raw(), xvec.raw(), yvec.raw() };

    const host = try std.testing.allocator.alloc(f64, M);
    defer std.testing.allocator.free(host);

    // alpha = 1, beta = 0 — the plain product, with `y` never read back.
    const pc = dgemv.PushConstants{ .M = M, .N = N, .alpha = 1.0, .beta = 0.0 };
    try harness.dispatch(dgemv_spv, "dgemv", &buffers, &pc, groups);

    yvec.copyToHost(host);
    if (debug_print) harness.printMatrix("y", host, 1, M);
    try harness.expectApproxEqualSlices(f64, expected_ab[0..], host, harness.tolFor(f64));

    // Re-dispatch over the result with alpha != 1 and beta != 0, so the
    // accumulate path is covered too.
    const pc_scaled = dgemv.PushConstants{ .M = M, .N = N, .alpha = 2.0, .beta = 3.0 };
    try harness.dispatch(dgemv_spv, "dgemv", &buffers, &pc_scaled, groups);

    yvec.copyToHost(host);
    try harness.expectApproxEqualSlices(f64, expected_scaled[0..], host, harness.tolFor(f64));
}
