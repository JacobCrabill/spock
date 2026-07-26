//! Performs a dense double-precision matrix-vector multiplication.
//!
//!     y := alpha * A * x + beta * y
//!
//! The dimension of each operand is:
//!   A: M * N
//!   x: N
//!   y: M
//! Such that: y_i = alpha * A_ij * x_j + beta * y_i
//!
//! The matrix is assumed to be row-major with stride equal to the number of columns.
//!
//! TODO: Allow stride != columns for cache alignment
//! TODO: Implement the transposed variant (y := alpha * A^T * x + beta * y)
//! TODO: Reduce across a workgroup instead of one thread per row, which serialises
//!       the whole dot product onto a single lane for wide matrices

// Export the entrypoint - _but only when compiling for spir-v_
// This allows the host-side code to @import("kernels/dgemv.zig")
comptime {
    if (@import("builtin").target.cpu.arch.isSpirV()) {
        @export(&dgemv, .{ .name = "dgemv" });
    }
}

/// Push constants: Up to 128 bytes of raw data sent to the GPU as part of the dispatch.
///
/// The extern struct is required to ensure consistent memory layout (field ordering and padding)
/// between host and device.
pub const PushConstants = extern struct {
    M: u32,
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

// Types used by input parameters (descriptors).
//
// The pattern is to define the SPIR-V type, then define an extern struct containing
// all SPIR-V types (Objects in the 'storage_buffer' address space must be structs,
// and 'runtime_array' objects must be the last member of a struct, as the length is
// not known at compile time).
const VkArray = @SpirvType(.{ .runtime_array = f64 });
const VkBuf = extern struct { data: VkArray };

const Amat = @extern(*addrspace(.storage_buffer) const VkBuf, .{
    .name = "A",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } },
});
const xvec = @extern(*addrspace(.storage_buffer) const VkBuf, .{
    .name = "x",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } },
});
// Read-write: the `beta * y` term reads the previous contents back.
const yvec = @extern(*addrspace(.storage_buffer) VkBuf, .{
    .name = "y",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 2 } },
});

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

/// Naive DGEMV
///
/// Use one thread per output value.
/// Each thread calculates y_i = alpha * A_ij * x_j + beta * y_i for its row i
fn dgemv() callconv(.{ .spirv_kernel = .{ .x = WgSize.x, .y = WgSize.y, .z = WgSize.z } }) void {
    const i = std.spirv.global_invocation_id[0];

    if (i >= pc.M) return;

    var sum: f64 = 0.0;
    for (0..pc.N) |j| {
        sum += Amat.data[j + i * pc.N] * xvec.data[j];
    }

    // BLAS semantics: when beta is zero `y` is *not* read, so an uninitialised or
    // NaN-filled output buffer is still valid input.
    const prev: f64 = if (pc.beta == 0.0) 0.0 else pc.beta * yvec.data[i];
    yvec.data[i] = pc.alpha * sum + prev;
}

const std = @import("std");
