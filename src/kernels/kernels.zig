/// Built-in linear-algebra kernels.
///
/// Each namespace exposes the `PushConstants` and `WgSize` that the matching
/// pre-compiled module was built with, so a consumer never has to restate them:
///
///     // build.zig
///     const spv = spock_dep.namedLazyPath("spock/dgemm.spv");
///     exe.root_module.addAnonymousImport("spock/dgemm.spv", .{ .root_source_file = spv });
///
///     // host code
///     const dgemm = spock.kernels.blas.dgemm;
///     const pc = dgemm.PushConstants{ .M = m, .N = n, .K = k, .alpha = 1, .beta = 0 };
///
/// Only the host-side declarations are reachable from here; the SPIR-V globals in
/// these files are never analysed when compiling for the host.
pub const blas = struct {
    pub const dgemm = @import("blas/dgemm.zig");
    pub const dgemv = @import("blas/dgemv.zig");
    pub const sgemm = @import("blas/sgemm.zig");
    pub const sgemv = @import("blas/sgemv.zig");
};

pub const basic = struct {
    pub const addvecf32 = @import("basic/addvecf32.zig");
    pub const addvecf64 = @import("basic/addvecf64.zig");
    pub const addveci32 = @import("basic/addveci32.zig");
    pub const addveci64 = @import("basic/addveci64.zig");
};

test {
    _ = @import("tests/dgemm.zig");
    _ = @import("tests/dgemv.zig");
    _ = @import("tests/sgemm.zig");
    _ = @import("tests/sgemv.zig");
    _ = @import("tests/addvecf32.zig");
    _ = @import("tests/addvecf64.zig");
    _ = @import("tests/addveci32.zig");
    _ = @import("tests/addveci64.zig");
}
