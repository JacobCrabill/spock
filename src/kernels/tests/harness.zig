//! Shared scaffolding for the compute-kernel tests.
//!
//! Every test needs the same three things: a device, a way to run one dispatch and
//! read the result back, and a float comparison that tolerates reassociation. Doing
//! them here keeps each `tests/<kernel>.zig` down to setup, dispatch, expectation.

const std = @import("std");
const spock = @import("../../spock.zig");

/// Deliberately *not* `std.testing.allocator`: the context outlives each individual
/// test, and the testing allocator reports still-live allocations as leaks when a
/// test ends. Nothing here is freed before process exit by design.
const ctx_gpa = std.heap.smp_allocator;

var ctx_state: ?spock.Context = null;

/// The process-wide compute context, created on first use.
///
/// Building a Vulkan instance + device costs far more than any single kernel
/// dispatch, so all tests share one. The test runner executes tests sequentially in
/// a single process, which is what makes this safe.
pub fn context() !*spock.Context {
    if (ctx_state == null) {
        ctx_state = try spock.Context.init(ctx_gpa, .{ .pick = .largest, .validate = true });
    }
    return &ctx_state.?;
}

/// One kernel, one dispatch, results left in the bound buffers.
///
/// `push_constant` is taken as a pointer to any extern struct; its size doubles as
/// the pipeline's push-constant range, so host and device cannot disagree.
pub fn dispatch(
    spirv: []const u8,
    entry: [*:0]const u8,
    buffers: []const spock.Kernel.Binding,
    push_constant: anytype,
    groups: u32,
) !void {
    const ctx = try context();
    const Pc = @typeInfo(@TypeOf(push_constant)).pointer.child;

    var kernel = try spock.Kernel.create(ctx, .{
        .spirv = spirv,
        .buffers = @intCast(buffers.len),
        .push_constant_size = @sizeOf(Pc),
        .entry = entry,
    });
    defer kernel.deinit();

    try kernel.dispatch(.{
        .buffers = buffers,
        .push_constant = std.mem.asBytes(push_constant),
        .groups = .{ groups, 1, 1 },
    });
    try ctx.wait();
}

/// Number of workgroups needed to cover `n` threads at `wg_size` threads per group.
pub fn groupsFor(n: u32, wg_size: u32) u32 {
    return @divCeil(n, wg_size);
}

/// Relative-error comparison for floating-point calculations.
///
/// Comparison is always done in f64 so a single `rel_tol` means the same thing for
/// both precisions; pick it with `tolFor(T)`.
pub fn expectApproxEqualSlices(comptime T: type, expected: []const T, actual: []const T, rel_tol: f64) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |ev, av, i| {
        const e: f64 = ev;
        const a: f64 = av;
        const scale = @max(@abs(e), @abs(a));
        const err = @abs(e - a);
        if (err > rel_tol * @max(scale, 1.0)) {
            std.debug.print(
                "index {d}: expected {d}, found {d} (abs err {d}, rel tol {d})\n",
                .{ i, e, a, err, rel_tol },
            );
            return error.TestExpectedApproxEqual;
        }
    }
}

/// Default tolerance for element type `T`: loose enough to survive a reordered
/// summation, tight enough that a genuinely wrong result cannot slip through.
pub fn tolFor(comptime T: type) f64 {
    return switch (T) {
        f64 => 1e-12,
        f32 => 1e-5,
        else => @compileError("no default tolerance for " ++ @typeName(T)),
    };
}

/// Dump a row-major matrix; handy when a test fails.
pub fn printMatrix(name: []const u8, m: anytype, rows: u32, cols: u32) void {
    std.debug.print("{s}:\n", .{name});
    for (0..rows) |i| {
        for (0..cols) |j| std.debug.print("{d:>8} ", .{m[j + i * cols]});
        std.debug.print("\n", .{});
    }
    std.debug.print("\n", .{});
}
