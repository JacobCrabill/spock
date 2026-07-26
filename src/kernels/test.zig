//! Compute kernel unit tests
const std = @import("std");
const spock = @import("../spock.zig");

const PushConstants = @import("blas/dgemm.zig").PushConstants;
const WgSize = @import("blas/dgemm.zig").WgSize;
const dgemm_spv = @embedFile("dgemm.spv");

const dgemv = @import("blas/dgemv.zig");
const dgemv_spv = @embedFile("dgemv.spv");

test "dgemm" {
    const gpa = std.testing.allocator;
    const validate: bool = true;
    const debug_print: bool = false;

    const M: u32 = 8;
    const N: u32 = 8;
    const K: u32 = 2;
    const expected = [64]f64{
        8.0,   9.0,   10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
        24.0,  29.0,  34.0,  39.0,  44.0,  49.0,  54.0,  59.0,
        40.0,  49.0,  58.0,  67.0,  76.0,  85.0,  94.0,  103.0,
        56.0,  69.0,  82.0,  95.0,  108.0, 121.0, 134.0, 147.0,
        72.0,  89.0,  106.0, 123.0, 140.0, 157.0, 174.0, 191.0,
        88.0,  109.0, 130.0, 151.0, 172.0, 193.0, 214.0, 235.0,
        104.0, 129.0, 154.0, 179.0, 204.0, 229.0, 254.0, 279.0,
        120.0, 149.0, 178.0, 207.0, 236.0, 265.0, 294.0, 323.0,
    };

    // 1. Device setup — pick the largest GPU on the system.
    var ctx = try spock.Context.init(gpa, .{ .pick = .largest, .validate = validate });
    defer ctx.deinit();

    if (debug_print) {
        std.debug.print("[spock] device: {s}\n", .{ctx.deviceName()});
    }

    // 2. Array creation — allocate, then fill the mapped host slice directly.
    var Amat = try spock.Buffer(f64).create(&ctx, M * K);
    defer Amat.deinit();
    var Bmat = try spock.Buffer(f64).create(&ctx, K * N);
    defer Bmat.deinit();
    var Cmat = try spock.Buffer(f64).create(&ctx, M * N);
    defer Cmat.deinit();

    for (Amat.slice(), 0..) |*v, i| v.* = @floatFromInt(i);
    for (Bmat.slice(), 0..) |*v, i| v.* = @floatFromInt(i);
    Cmat.fill(0);

    if (debug_print) {
        std.debug.print("A:\n", .{});
        for (0..M) |i| {
            for (0..K) |j| {
                std.debug.print("{d:>4} ", .{Amat.slice()[j + i * K]});
            }
            std.debug.print("\n", .{});
        }
        std.debug.print("\n", .{});

        std.debug.print("B:\n", .{});
        for (0..K) |i| {
            for (0..N) |j| {
                std.debug.print("{d:>4} ", .{Bmat.slice()[j + i * N]});
            }
            std.debug.print("\n", .{});
        }
        std.debug.print("\n", .{});
    }

    const wg_size = WgSize.x; // Number of threads per block
    const groups = @divCeil(M * N, wg_size);

    // 3a. Single dispatch (async), then wait.
    var kernel = try spock.Kernel.create(&ctx, .{
        .spirv = dgemm_spv,
        .buffers = 3,
        .push_constant_size = @sizeOf(PushConstants),
        .entry = "dgemm",
    });
    defer kernel.deinit();

    const pc = PushConstants{
        .M = M,
        .N = N,
        .K = K,
        .alpha = 1.0,
        .beta = 0.0,
    };

    try kernel.dispatch(.{
        .buffers = &.{ Amat.raw(), Bmat.raw(), Cmat.raw() },
        .push_constant = std.mem.asBytes(&pc),
        .groups = .{ groups, 1, 1 },
    });
    try ctx.wait();

    // 5. Copy results back to a plain host array.
    const host = try gpa.alloc(f64, M * N);
    defer gpa.free(host);
    Cmat.copyToHost(host);

    // Report.
    if (debug_print) {
        std.debug.print("C:\n", .{});
        for (0..M) |i| {
            for (0..N) |j| {
                std.debug.print("{d:>4} ", .{host[j + i * N]});
            }
            std.debug.print("\n", .{});
        }
        std.debug.print("\n", .{});
    }

    try std.testing.expectEqualSlices(f64, expected[0..], host);
}

test "dgemv" {
    const gpa = std.testing.allocator;
    const validate: bool = true;
    const debug_print: bool = false;

    // Deliberately non-square, so a row/column mix-up cannot pass by accident.
    const M: u32 = 8;
    const N: u32 = 4;

    // A_ij = j + i*N, x_j = j, so y_i = sum_j (j + i*N) * j = 14 + 24*i
    const expected_ab = [8]f64{ 14.0, 38.0, 62.0, 86.0, 110.0, 134.0, 158.0, 182.0 };
    // Re-running with alpha=2, beta=3 over that result: 2*(14+24i) + 3*(14+24i)
    const expected_scaled = [8]f64{ 70.0, 190.0, 310.0, 430.0, 550.0, 670.0, 790.0, 910.0 };

    // 1. Device setup — pick the largest GPU on the system.
    var ctx = try spock.Context.init(gpa, .{ .pick = .largest, .validate = validate });
    defer ctx.deinit();

    if (debug_print) {
        std.debug.print("[spock] device: {s}\n", .{ctx.deviceName()});
    }

    // 2. Array creation — allocate, then fill the mapped host slice directly.
    var Amat = try spock.Buffer(f64).create(&ctx, M * N);
    defer Amat.deinit();
    var xvec = try spock.Buffer(f64).create(&ctx, N);
    defer xvec.deinit();
    var yvec = try spock.Buffer(f64).create(&ctx, M);
    defer yvec.deinit();

    for (Amat.slice(), 0..) |*v, i| v.* = @floatFromInt(i);
    for (xvec.slice(), 0..) |*v, i| v.* = @floatFromInt(i);
    yvec.fill(0);

    const wg_size = dgemv.WgSize.x; // Number of threads per block
    const groups = @divCeil(M, wg_size);

    var kernel = try spock.Kernel.create(&ctx, .{
        .spirv = dgemv_spv,
        .buffers = 3,
        .push_constant_size = @sizeOf(dgemv.PushConstants),
        .entry = "dgemv",
    });
    defer kernel.deinit();

    const host = try gpa.alloc(f64, M);
    defer gpa.free(host);

    // 3a. alpha = 1, beta = 0 — the plain product, with `y` never read back.
    const pc = dgemv.PushConstants{ .M = M, .N = N, .alpha = 1.0, .beta = 0.0 };
    try kernel.dispatch(.{
        .buffers = &.{ Amat.raw(), xvec.raw(), yvec.raw() },
        .push_constant = std.mem.asBytes(&pc),
        .groups = .{ groups, 1, 1 },
    });
    try ctx.wait();

    yvec.copyToHost(host);

    if (debug_print) {
        std.debug.print("y:\n", .{});
        for (host) |v| std.debug.print("{d:>6} ", .{v});
        std.debug.print("\n", .{});
    }

    try std.testing.expectEqualSlices(f64, expected_ab[0..], host);

    // 3b. Re-dispatch over the result with alpha != 1 and beta != 0, so the
    //     accumulate path is covered too.
    const pc_scaled = dgemv.PushConstants{ .M = M, .N = N, .alpha = 2.0, .beta = 3.0 };
    try kernel.dispatch(.{
        .buffers = &.{ Amat.raw(), xvec.raw(), yvec.raw() },
        .push_constant = std.mem.asBytes(&pc_scaled),
        .groups = .{ groups, 1, 1 },
    });
    try ctx.wait();

    yvec.copyToHost(host);

    try std.testing.expectEqualSlices(f64, expected_scaled[0..], host);
}
