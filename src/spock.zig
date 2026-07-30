//! A small, physics-friendly compute layer over Vulkan (vulkan-zig bindings).
//!
//! Deliberately minimal and opinionated:
//!   * One compute device, one compute queue, one command pool, one fence.
//!   * Buffers are host-visible + host-coherent and persistently mapped, so you
//!     read/write them like normal Zig slices — no staging, no manual flushes.
//!     Great for iterative solvers on integrated / resizable-BAR GPUs; for a
//!     huge discrete-GPU working set you'd switch to device-local + staging.
//!
//! See tests or the examples dir for representative usage.

/// Re-export the Vulkan bindings from vulkan-zig.
/// This points to the zig build cache if the bindings were regenerated from
/// the user's host `vk.xml` file, otherwise `bindings/vk.zig` is used.
pub const vk = @import("vulkan");

pub const Context = @import("core/context.zig").Context;
pub const Buffer = @import("core/buffer.zig").Buffer;
pub const Kernel = @import("core/kernel.zig").Kernel;
pub const Pipeline = @import("core/pipeline.zig").Pipeline;

pub const kernels = @import("kernels/kernels.zig");

test "spock" {
    @import("std").testing.refAllDecls(@This());

    _ = @import("core/buffer.zig");
}
