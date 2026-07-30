//! Tests for single-precision vector addition.

const std = @import("std");
const spock = @import("../../spock.zig");
const harness = @import("harness.zig");

const addveci64 = @import("../basic/addveci64.zig");
const addveci64_spv = @embedFile("spock/addveci64.spv");

test "addveci64" {
    const N: u32 = 256;

    var expected_z = std.mem.zeroes([N]i64);

    const ctx = try harness.context();

    var xvec = try spock.Buffer(i64).create(ctx, N);
    defer xvec.deinit();
    var yvec = try spock.Buffer(i64).create(ctx, N);
    defer yvec.deinit();
    var zvec = try spock.Buffer(i64).create(ctx, N);
    defer zvec.deinit();

    for (xvec.slice(), 0..) |*v, i| v.* = @intCast(i);
    for (yvec.slice(), 0..) |*v, i| v.* = @intCast(2 * i + 1);
    zvec.fill(0);
    for (expected_z[0..], 0..) |*v, i| v.* = @intCast((3 * i) + 1);

    const groups = harness.groupsFor(N, addveci64.WgSize.x);
    const buffers = [_]spock.Kernel.Binding{ .whole(xvec.raw()), .whole(yvec.raw()), .whole(zvec.raw()) };

    const host = try std.testing.allocator.alloc(i64, N);
    defer std.testing.allocator.free(host);

    const pc = addveci64.PushConstants{ .N = N };
    try harness.dispatch(addveci64_spv, "addveci64", &buffers, &pc, .{ groups, 1, 1 });

    zvec.copyToHost(host);
    try std.testing.expectEqualSlices(i64, expected_z[0..], host);
}
