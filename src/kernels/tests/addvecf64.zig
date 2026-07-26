//! Tests for single-precision vector addition.

const std = @import("std");
const spock = @import("../../spock.zig");
const harness = @import("harness.zig");

const addvecf64 = @import("../basic/addvecf64.zig");
const addvecf64_spv = @embedFile("spock/addvecf64.spv");

test "addvecf64" {
    const N: u32 = 256;

    var expected_z = std.mem.zeroes([N]f64);

    const ctx = try harness.context();

    var xvec = try spock.Buffer(f64).create(ctx, N);
    defer xvec.deinit();
    var yvec = try spock.Buffer(f64).create(ctx, N);
    defer yvec.deinit();
    var zvec = try spock.Buffer(f64).create(ctx, N);
    defer zvec.deinit();

    for (xvec.slice(), 0..) |*v, i| v.* = @floatFromInt(i);
    for (yvec.slice(), 0..) |*v, i| v.* = @floatFromInt(2 * i + 1);
    zvec.fill(0);
    for (expected_z[0..], 0..) |*v, i| v.* = @floatFromInt((3 * i) + 1);

    const groups = harness.groupsFor(N, addvecf64.WgSize.x);
    const buffers = [_]spock.vk.Buffer{ xvec.raw(), yvec.raw(), zvec.raw() };

    const host = try std.testing.allocator.alloc(f64, N);
    defer std.testing.allocator.free(host);

    const pc = addvecf64.PushConstants{ .N = N };
    try harness.dispatch(addvecf64_spv, "addvecf64", &buffers, &pc, groups);

    zvec.copyToHost(host);
    try harness.expectApproxEqualSlices(f64, expected_z[0..], host, harness.tolFor(f64));
}
