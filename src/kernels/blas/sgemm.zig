//! Performs a dense single-precision matrix-matrix multiplication.
//!
//!     C := alpha * A * B + beta * C
//!
//! The dimension of each matrix is:
//!   C: M * N
//!   A: M * K
//!   B: K * N
//! Such that: C_ij = A_ik * B_kj
//!
//! All matrices are assumed to be row-major with strides equal to the number of columns.
//!
//! The single-precision twin of `dgemm.zig`; keep the two in step.
//!
//! TODO: Allow stride != columns for cache alignment
//! TODO: Implement transposed variants?
//! TODO: Implement blocked version with shared memory

// Export the entrypoint - _but only when compiling for spir-v_
// This allows the host-side code to @import("sgemm.zig")
comptime {
    if (@import("builtin").target.cpu.arch.isSpirV()) {
        @export(&sgemm, .{ .name = "sgemm" });
    }
}

/// Push constants: Up to 128 bytes of raw data sent to the GPU as part of the dispatch.
///
/// The extern struct is required to ensure consistent memory layout (field ordering and padding)
/// between host and device.
pub const PushConstants = extern struct {
    M: u32,
    K: u32,
    N: u32,
    alpha: f32,
    beta: f32,
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
const VkArray = @SpirvType(.{ .runtime_array = f32 });
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

/// Naive SGEMM
///
/// Use one thread per output value.
/// Each thread calculates C_ij = alpha * A_ik * B_kj + beta * C_ij for its (i, j)
///
/// NOTE: the final store is written as an explicit load/scale/store rather than
/// `C *= beta; C += sum;`. The compound form is silently dropped by the SPIR-V
/// backend here — see the note in `dgemv.zig`. Do not "simplify" it back.
fn sgemm() callconv(.{ .spirv_kernel = .{ .x = WgSize.x, .y = WgSize.y, .z = WgSize.z } }) void {
    const tid = std.spirv.global_invocation_id[0];

    if (tid >= pc.M * pc.N) return;

    const i: u32 = @divTrunc(tid, pc.N);
    const j: u32 = @mod(tid, pc.N);

    var sum: f32 = 0.0;
    if (pc.alpha != 0.0) {
        for (0..pc.K) |k| {
            sum += Amat.data[k + i * pc.K] * Bmat.data[j + k * pc.N];
        }
        sum *= pc.alpha;
    }
    const idx = j + i * pc.N;
    const prev: f32 = if (pc.beta == 0.0) 0.0 else pc.beta * Cmat.data[idx];
    Cmat.data[idx] = sum + prev;
}

const std = @import("std");
