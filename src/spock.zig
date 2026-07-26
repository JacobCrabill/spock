//! A small, physics-friendly compute layer over Vulkan (vulkan-zig bindings).
//!
//! Deliberately minimal and opinionated:
//!   * One compute device, one compute queue, one command pool, one fence.
//!   * Buffers are host-visible + host-coherent and persistently mapped, so you
//!     read/write them like normal Zig slices — no staging, no manual flushes.
//!     Great for iterative solvers on integrated / resizable-BAR GPUs; for a
//!     huge discrete-GPU working set you'd switch to device-local + staging.
//!
//! Typical use:
//!   var ctx = try gpu.Context.init(gpa, .{ .pick = .largest });
//!   defer ctx.deinit();
//!   var x = try ctx.alloc(f32, n); defer x.deinit();
//!   for (x.slice(), 0..) |*v, i| v.* = ...;              // fill on host
//!   var k = try ctx.kernel(.{ .spirv = @embedFile("k.spv"), .buffers = 2 });
//!   defer k.deinit();
//!   try k.dispatch(.{ .buffers = &.{ x.raw(), y.raw() }, .groups = .{ g, 1, 1 } });
//!   try ctx.wait();
//!   y.copyToHost(host_slice);                             // read results back

const std = @import("std");

/// Re-export the Vulkan bindings from vulkan-zig
pub const vk = @import("vulkan");

pub const Context = @import("core/context.zig").Context;
pub const Buffer = @import("core/buffer.zig").Buffer;
pub const Kernel = @import("core/kernel.zig").Kernel;
pub const Pipeline = @import("core/pipeline.zig").Pipeline;

test {
    _ = @import("kernels/tests/dgemm.zig");
    _ = @import("kernels/tests/dgemv.zig");
    _ = @import("kernels/tests/sgemm.zig");
    _ = @import("kernels/tests/sgemv.zig");
}
