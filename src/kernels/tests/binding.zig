//! Tests for binding part of a buffer rather than all of it.
//!
//! The case this exists for: one array holding several blocks -- the per-stage
//! residuals of a Runge-Kutta step, say -- with each dispatch aimed at its own
//! block and the others left untouched.

const std = @import("std");
const spock = @import("../../spock.zig");
const harness = @import("harness.zig");

const dgemm = @import("../blas/dgemm.zig");
const dgemm_spv = @embedFile("spock/dgemm.spv");

test "a binding can point at one block of a buffer" {
    const ctx = try harness.context();

    const M: u32 = 4;
    const N: u32 = 4;
    const K: u32 = 2;

    // A storage-buffer offset has to be a multiple of the device's alignment, so
    // pad each block out to one rather than assuming M*N*8 already is.
    const align_to: usize = @intCast(ctx.props.limits.min_storage_buffer_offset_alignment);
    const per_align = @max(1, align_to / @sizeOf(f64));
    const block = std.mem.alignForward(usize, M * N, per_align);

    var a = try spock.Buffer(f64).create(ctx, M * K);
    defer a.deinit();
    var b = try spock.Buffer(f64).create(ctx, K * N);
    defer b.deinit();
    var c = try spock.Buffer(f64).create(ctx, 2 * block);
    defer c.deinit();

    for (a.slice(), 0..) |*v, i| v.* = @floatFromInt(i);
    for (b.slice(), 0..) |*v, i| v.* = @floatFromInt(i);
    c.fill(-1.0);

    // Aim the product at the *second* block
    const buffers = [_]spock.Kernel.Binding{
        .whole(a.raw()),
        .whole(b.raw()),
        .{ .buffer = c.raw(), .offset = block * @sizeOf(f64), .size = M * N * @sizeOf(f64) },
    };
    const pc = dgemm.PushConstants{ .M = M, .N = N, .K = K, .alpha = 1.0, .beta = 0.0 };
    try harness.dispatch(dgemm_spv, "dgemm", &buffers, &pc, harness.groupsFor(M * N, dgemm.WgSize.x));

    // The first block is exactly as it was
    for (c.slice()[0..block]) |v| try std.testing.expectEqual(@as(f64, -1.0), v);

    // ...and the second holds A*B
    const got = c.slice()[block..][0 .. M * N];
    for (0..M) |i| {
        for (0..N) |j| {
            var want: f64 = 0.0;
            for (0..K) |k| want += a.slice()[i * K + k] * b.slice()[k * N + j];
            try std.testing.expectApproxEqAbs(want, got[i * N + j], 1e-12);
        }
    }
}

test "a misaligned binding offset is rejected" {
    const ctx = try harness.context();

    const align_to = ctx.props.limits.min_storage_buffer_offset_alignment;
    if (align_to <= 1) return error.SkipZigTest; // nothing to misalign

    var buf = try spock.Buffer(f64).create(ctx, 64);
    defer buf.deinit();

    var kernel = try spock.Kernel.create(ctx, .{
        .spirv = dgemm_spv,
        .entry = "dgemm",
        .buffers = 3,
        .push_constant_size = @sizeOf(dgemm.PushConstants),
    });
    defer kernel.deinit();

    // Vulkan would reject this too, but only through a validation message the
    // caller may not have switched on -- so it fails here, plainly.
    const buffers = [_]spock.Kernel.Binding{
        .whole(buf.raw()),
        .whole(buf.raw()),
        .{ .buffer = buf.raw(), .offset = align_to + 1 },
    };
    const pc = dgemm.PushConstants{ .M = 1, .N = 1, .K = 1, .alpha = 1.0, .beta = 0.0 };
    try std.testing.expectError(error.BadDispatchArgs, kernel.dispatch(.{
        .buffers = &buffers,
        .push_constant = std.mem.asBytes(&pc),
        .groups = .{ 1, 1, 1 },
    }));
}
