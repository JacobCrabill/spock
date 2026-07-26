//! Tests for single-precision vector addition.

const std = @import("std");
const spock = @import("../../spock.zig");
const harness = @import("harness.zig");

const addveci32 = @import("../basic/addveci32.zig");
const addveci32_spv = @embedFile("spock/addveci32.spv");

test "addveci32" {
    const N: u32 = 256;

    var expected_z = std.mem.zeroes([N]i32);

    const ctx = try harness.context();

    var xvec = try spock.Buffer(i32).create(ctx, N);
    defer xvec.deinit();
    var yvec = try spock.Buffer(i32).create(ctx, N);
    defer yvec.deinit();
    var zvec = try spock.Buffer(i32).create(ctx, N);
    defer zvec.deinit();

    for (xvec.slice(), 0..) |*v, i| v.* = @intCast(i);
    for (yvec.slice(), 0..) |*v, i| v.* = @intCast(2 * i + 1);
    zvec.fill(0);
    for (expected_z[0..], 0..) |*v, i| v.* = @intCast(3 * i + 1);

    const groups = harness.groupsFor(N, addveci32.WgSize.x);
    const buffers = [_]spock.vk.Buffer{ xvec.raw(), yvec.raw(), zvec.raw() };

    const host = try std.testing.allocator.alloc(i32, N);
    defer std.testing.allocator.free(host);

    const pc = addveci32.PushConstants{ .N = N };
    try harness.dispatch(addveci32_spv, "addveci32", &buffers, &pc, groups);

    zvec.copyToHost(host);
    try std.testing.expectEqualSlices(i32, expected_z[0..], host);
}
