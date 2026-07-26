//! Tests for single-precision vector addition.

const std = @import("std");
const spock = @import("../../spock.zig");
const harness = @import("harness.zig");

const addvecf32 = @import("../basic/addvecf32.zig");
const addvecf32_spv = @embedFile("spock/addvecf32.spv");

test "addvecf32" {
    const N: u32 = 256;

    var expected_z = std.mem.zeroes([N]f32);

    const ctx = try harness.context();

    var xvec = try spock.Buffer(f32).create(ctx, N);
    defer xvec.deinit();
    var yvec = try spock.Buffer(f32).create(ctx, N);
    defer yvec.deinit();
    var zvec = try spock.Buffer(f32).create(ctx, N);
    defer zvec.deinit();

    for (xvec.slice(), 0..) |*v, i| v.* = @floatFromInt(i);
    for (yvec.slice(), 0..) |*v, i| v.* = @floatFromInt(2 * i + 1);
    zvec.fill(0);
    for (expected_z[0..], 0..) |*v, i| v.* = @floatFromInt(3 * i + 1);

    const groups = harness.groupsFor(N, addvecf32.WgSize.x);
    const buffers = [_]spock.vk.Buffer{ xvec.raw(), yvec.raw(), zvec.raw() };

    const host = try std.testing.allocator.alloc(f32, N);
    defer std.testing.allocator.free(host);

    const pc = addvecf32.PushConstants{ .N = N };
    try harness.dispatch(addvecf32_spv, "addvecf32", &buffers, &pc, groups);

    zvec.copyToHost(host);
    try harness.expectApproxEqualSlices(f32, expected_z[0..], host, harness.tolFor(f32));
}
