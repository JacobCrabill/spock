//! Adds 2 double-precision vectors.
//!
//!     z := x + y
//!
//! The length of each operand is 'N'.

// Export the entrypoint - _but only when compiling for spir-v_
// This allows the host-side code to @import("sgemv.zig")
comptime {
    if (@import("builtin").target.cpu.arch.isSpirV()) {
        @export(&addvecf64, .{ .name = "addvecf64" });
    }
}

/// Push constants: Up to 128 bytes of raw data sent to the GPU as part of the dispatch.
///
/// The extern struct is required to ensure consistent memory layout (field ordering and padding)
/// between host and device.
pub const PushConstants = extern struct {
    N: u32,
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

const xvec = @extern(*addrspace(.storage_buffer) const VkBuf, .{
    .name = "x",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } },
});
const yvec = @extern(*addrspace(.storage_buffer) const VkBuf, .{
    .name = "y",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } },
});
// Read-write: the `beta * y` term reads the previous contents back.
const zvec = @extern(*addrspace(.storage_buffer) VkBuf, .{
    .name = "z",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 2 } },
});

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

/// Basic vector addition.
/// Use one thread per output value.
fn addvecf64() callconv(.{ .spirv_kernel = .{ .x = WgSize.x, .y = WgSize.y, .z = WgSize.z } }) void {
    const i = std.spirv.global_invocation_id[0];
    if (i >= pc.N) return;
    zvec.data[i] = xvec.data[i] + yvec.data[i];
}

const std = @import("std");
