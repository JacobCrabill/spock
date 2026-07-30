//! Performs a dense double-precision matrix-matrix multiplication.
//!
//!     C := alpha * A * B + beta * C
//!
//! The dimension of each matrix is:
//!   C: M * N
//!   A: M * K
//!   B: K * N
//! Such that: C_ij = A_ik * B_kj
//!
//! All matrices are assumed to be row-major with strides equal to the number of
//! columns.
//!
//! Each thread accumulates `tile_m` rows of one column of `C`, rather than a
//! single element. A thread-per-element kernel reads `K` values of `B` for every
//! element of `C`, so `B` is read `M` times over -- `M * N * K` loads against
//! the `K * N` values it holds. Owning a strip of rows means each `B` value is
//! loaded once and used `tile_m` times, cutting that to `ceil(M / tile_m)`.
//!
//! It matters most for the skinny shapes a physics solver dispatches, where a
//! small operator is applied to a large batch: at M = 16, K = 16, N = 65536 on a
//! Quadro T1000 this is 1.7x the thread-per-element version, and lands at ~64%
//! of the card's double-precision peak, where it is compute-bound rather than
//! bandwidth-bound. Strip sizes from 4 to 16 measure within a few percent of
//! each other for that reason.
//!
//! Dispatched over two dimensions -- x over columns of `C`, y over row strips --
//! so a workgroup shares one strip. Use `groups()` rather than working the
//! mapping out at the call site.
//!
//! TODO: Allow stride != columns for cache alignment
//! TODO: Implement transposed variants?
//! TODO: Stage A in shared memory

// Export the entrypoint - _but only when compiling for spir-v_
// This allows the host-side code to @import("kernels/dgemm.zig")
comptime {
    if (@import("builtin").target.cpu.arch.isSpirV()) {
        @export(&dgemm, .{ .name = "dgemm" });
    }
}

/// Rows of `C` one thread accumulates.
///
/// The accumulators are registers, so this trades register pressure against `B`
/// traffic: 8 doubles per thread costs 16 registers and cuts the reads eightfold
/// for an operator with 8 or more rows.
pub const tile_m = 8;

/// Push constants: Up to 128 bytes of raw data sent to the GPU as part of the dispatch.
///
/// The extern struct is required to ensure consistent memory layout (field ordering and padding)
/// between host and device.
pub const PushConstants = extern struct {
    M: u32,
    K: u32,
    N: u32,
    alpha: f64,
    beta: f64,
};

/// The number of threads per workgroup
pub const WgSize = extern struct {
    pub const x: u32 = 128;
    pub const y: u32 = 1;
    pub const z: u32 = 1;
};

/// Workgroups needed for an `M` by `N` result: x covers the columns, y the row
/// strips. Pass straight to `Kernel.DispatchArgs.groups`.
pub fn groups(M: u32, N: u32) [3]u32 {
    return .{
        (N + WgSize.x - 1) / WgSize.x,
        (M + tile_m - 1) / tile_m,
        1,
    };
}

const VkArray = @SpirvType(.{ .runtime_array = f64 });
const VkBuf = extern struct { data: VkArray };

const Amat = @extern(*addrspace(.storage_buffer) const VkBuf, .{
    .name = "A",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } },
});
const Bmat = @extern(*addrspace(.storage_buffer) const VkBuf, .{
    .name = "B",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } },
});
const Cmat = @extern(*addrspace(.storage_buffer) VkBuf, .{
    .name = "C",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 2 } },
});

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

fn dgemm() callconv(.{ .spirv_kernel = .{ .x = WgSize.x, .y = WgSize.y, .z = WgSize.z } }) void {
    const j = std.spirv.global_invocation_id[0];
    const strip = std.spirv.global_invocation_id[1];
    if (j >= pc.N) return;

    const m0 = strip * tile_m;
    if (m0 >= pc.M) return;

    var acc: [tile_m]f64 = @splat(0.0);

    if (pc.alpha != 0.0) {
        if (m0 + tile_m <= pc.M) {
            // The whole strip is in range, so the inner loop needs no bounds
            // check -- the case every dispatch here takes.
            for (0..pc.K) |k| {
                const b = Bmat.data[j + k * pc.N];
                inline for (0..tile_m) |t| {
                    acc[t] += Amat.data[k + (m0 + t) * pc.K] * b;
                }
            }
        } else {
            for (0..pc.K) |k| {
                const b = Bmat.data[j + k * pc.N];
                inline for (0..tile_m) |t| {
                    if (m0 + t < pc.M) acc[t] += Amat.data[k + (m0 + t) * pc.K] * b;
                }
            }
        }
    }

    // Written as an explicit load, scale and store: the compound form is
    // silently dropped by the SPIR-V backend here -- see the note in `dgemv.zig`.
    // Do not "simplify" it back.
    inline for (0..tile_m) |t| {
        const m = m0 + t;
        if (m < pc.M) {
            const idx = j + m * pc.N;
            const prev: f64 = if (pc.beta == 0.0) 0.0 else pc.beta * Cmat.data[idx];
            Cmat.data[idx] = pc.alpha * acc[t] + prev;
        }
    }
}

const std = @import("std");
