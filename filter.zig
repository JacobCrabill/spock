const std = @import("std");

// OpenCL-style compute kernel. Pointer parameters live in the .global address space.
// Parallel filter: out[i] = in[i] if in[i] > threshold else 0.
export fn filter(
    input: [*]addrspace(.global) const f32,
    output: [*]addrspace(.global) f32,
    threshold: f32,
    n: u32,
) callconv(.kernel) void {
    const i = std.spirv.global_invocation_id[0];
    if (i >= n) return;
    const v = input[i];
    output[i] = if (v > threshold) v else 0.0;
}
