//! Tests for the dense double-precision matrix-matrix kernel.

const std = @import("std");
const spock = @import("../../spock.zig");
const harness = @import("harness.zig");

const dgemm = @import("../blas/dgemm.zig");
const dgemm_spv = @embedFile("spock/dgemm.spv");

test "dgemm" {
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

    const ctx = try harness.context();

    // Array creation — allocate, then fill the mapped host slice directly.
    var Amat = try spock.Buffer(f64).create(ctx, M * K);
    defer Amat.deinit();
    var Bmat = try spock.Buffer(f64).create(ctx, K * N);
    defer Bmat.deinit();
    var Cmat = try spock.Buffer(f64).create(ctx, M * N);
    defer Cmat.deinit();

    for (Amat.slice(), 0..) |*v, i| v.* = @floatFromInt(i);
    for (Bmat.slice(), 0..) |*v, i| v.* = @floatFromInt(i);
    Cmat.fill(0);

    if (debug_print) {
        harness.printMatrix("A", Amat.slice(), M, K);
        harness.printMatrix("B", Bmat.slice(), K, N);
    }

    const buffers = [_]spock.Kernel.Binding{ .whole(Amat.raw()), .whole(Bmat.raw()), .whole(Cmat.raw()) };

    const host = try std.testing.allocator.alloc(f64, M * N);
    defer std.testing.allocator.free(host);

    // alpha = 1, beta = 0 — the plain product, with `C` never read back.
    const pc = dgemm.PushConstants{ .M = M, .N = N, .K = K, .alpha = 1.0, .beta = 0.0 };
    try harness.dispatch(dgemm_spv, "dgemm", &buffers, &pc, dgemm.groups(M, N));

    Cmat.copyToHost(host);
    if (debug_print) harness.printMatrix("C", host, M, N);
    try harness.expectApproxEqualSlices(f64, expected[0..], host, harness.tolFor(f64));

    // Re-dispatch over the result with alpha != 1 and beta != 0. A*B is unchanged,
    // so C := 2*(A*B) + 3*C collapses to 5*C — which no kernel that drops either
    // scalar can produce (ignoring both gives 1x, alpha only 2x, beta only 3x).
    var scaled: [64]f64 = undefined;
    for (expected, &scaled) |e, *s| s.* = 5.0 * e;

    const pc_scaled = dgemm.PushConstants{ .M = M, .N = N, .K = K, .alpha = 2.0, .beta = 3.0 };
    try harness.dispatch(dgemm_spv, "dgemm", &buffers, &pc_scaled, dgemm.groups(M, N));

    Cmat.copyToHost(host);
    try harness.expectApproxEqualSlices(f64, scaled[0..], host, harness.tolFor(f64));

    // alpha = 0 skips the product entirely and degenerates to C := beta * C.
    // A fractional beta keeps every expectation exactly representable in f64.
    var halved: [64]f64 = undefined;
    for (scaled, &halved) |s, *h| h.* = 0.5 * s;

    const pc_scale_only = dgemm.PushConstants{ .M = M, .N = N, .K = K, .alpha = 0.0, .beta = 0.5 };
    try harness.dispatch(dgemm_spv, "dgemm", &buffers, &pc_scale_only, dgemm.groups(M, N));

    Cmat.copyToHost(host);
    try harness.expectApproxEqualSlices(f64, halved[0..], host, harness.tolFor(f64));
}

test "dgemm - assorted shapes against a host reference" {
    // The hand-written case above is 8x8, which is exactly one strip and so
    // never exercises the row-tiling. These do: row counts either side of
    // `tile_m`, and one that is not a multiple of it, checked against the
    // definition rather than against tabulated numbers.
    const ctx = try harness.context();
    const gpa = std.testing.allocator;

    var prng: std.Random.DefaultPrng = .init(0x5EED_5EED);
    const rand = prng.random();

    const shapes = [_][3]u32{
        .{ 1, 1, 1 },
        .{ 3, 5, 2 },
        .{ 8, 64, 8 }, // exactly one strip
        .{ 16, 129, 16 }, // two strips, ragged column count
        .{ 13, 37, 11 }, // a strip that runs off the end
        .{ 33, 17, 4 },
    };

    for (shapes) |shape| {
        const M, const N, const K = shape;

        var a = try spock.Buffer(f64).create(ctx, M * K);
        defer a.deinit();
        var b = try spock.Buffer(f64).create(ctx, K * N);
        defer b.deinit();
        var c = try spock.Buffer(f64).create(ctx, M * N);
        defer c.deinit();

        for (a.slice()) |*v| v.* = rand.floatNorm(f64);
        for (b.slice()) |*v| v.* = rand.floatNorm(f64);
        for (c.slice()) |*v| v.* = rand.floatNorm(f64);

        const want = try gpa.alloc(f64, M * N);
        defer gpa.free(want);

        const alpha = 2.0;
        const beta = -0.5;
        for (0..M) |i| {
            for (0..N) |j| {
                var sum: f64 = 0.0;
                for (0..K) |k| sum += a.slice()[i * K + k] * b.slice()[k * N + j];
                want[i * N + j] = alpha * sum + beta * c.slice()[i * N + j];
            }
        }

        const buffers = [_]spock.Kernel.Binding{
            .whole(a.raw()),
            .whole(b.raw()),
            .whole(c.raw()),
        };
        const pc = dgemm.PushConstants{ .M = M, .N = N, .K = K, .alpha = alpha, .beta = beta };
        try harness.dispatch(dgemm_spv, "dgemm", &buffers, &pc, dgemm.groups(M, N));

        harness.expectApproxEqualSlices(f64, want, c.slice(), harness.tolFor(f64)) catch |err| {
            std.debug.print("shape M={d} N={d} K={d}\n", .{ M, N, K });
            return err;
        };
    }
}
