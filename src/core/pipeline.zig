//! Pipelines of compute kernels
const std = @import("std");
pub const vk = @import("vulkan");

const context = @import("context.zig");
const Kernel = @import("kernel.zig").Kernel;
const Context = context.Context;
const Error = context.Error;

/// A sequential batch of compute kernels
///
/// A chain of dispatches recorded into a single command buffer and submitted
/// once. A storage-buffer memory barrier is inserted before every dispatch after
/// the first, so kernel N+1 sees kernel N's writes without a host round-trip —
/// the usual pattern for a multi-stage physics step (advect → project → update).
///
/// Reusable: for a time loop, call `reset()` at the top of each step, record the
/// dispatches, `submit()`, then `ctx.wait()`. Free once with `deinit()`.
pub const Pipeline = struct {
    ctx: *Context,
    cmd: vk.CommandBuffer,
    count: u32,

    /// Begin a multi-dispatch sequence: several kernels recorded into one command
    /// buffer, with automatic storage-buffer barriers between them, submitted once.
    /// The returned `Pipeline` is reusable across steps via `reset()`; free it with
    /// `deinit()`.
    pub fn create(ctx: *Context) !Pipeline {
        var cmd: vk.CommandBuffer = undefined;
        try ctx.device.allocateCommandBuffers(&.{
            .command_pool = ctx.cmd_pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, @ptrCast(&cmd));
        try ctx.device.beginCommandBuffer(cmd, &.{ .flags = .{ .one_time_submit = true } });
        return .{ .ctx = ctx, .cmd = cmd, .count = 0 };
    }

    /// Record a dispatch. Inserts a compute→compute storage barrier first if this
    /// is not the first dispatch in the sequence.
    /// If the dispatches are independent, use `dispatchNoBarrier` instead..
    pub fn addKernel(self: *Pipeline, kernel: *Kernel, args: Kernel.DispatchArgs) !void {
        if (self.count > 0) self.barrier();
        try kernel.record(self.cmd, args);
        self.count += 1;
    }

    /// Record a dispatch without an automatic preceding barrier (for independent
    /// kernels that can overlap).
    pub fn dispatchNoBarrier(self: *Pipeline, kernel: *Kernel, args: Kernel.DispatchArgs) !void {
        try kernel.record(self.cmd, args);
        self.count += 1;
    }

    /// Insert an explicit storage-buffer barrier: makes prior compute shader writes
    /// visible to subsequent compute shader reads/writes.
    pub fn barrier(self: *Pipeline) void {
        self.ctx.device.cmdPipelineBarrier(
            self.cmd,
            .{ .compute_shader = true },
            .{ .compute_shader = true },
            .{},
            &[_]vk.MemoryBarrier{.{
                .src_access_mask = .{ .shader_write = true },
                .dst_access_mask = .{ .shader_read = true, .shader_write = true },
            }},
            null,
            null,
        );
    }

    /// End recording and submit the whole sequence as one batch, signalling the
    /// context fence so `ctx.wait()` can block on it.
    pub fn submit(self: *Pipeline) !void {
        const dev = self.ctx.device;
        try dev.endCommandBuffer(self.cmd);
        try dev.resetFences(&[_]vk.Fence{self.ctx.fence});
        try dev.queueSubmit(self.ctx.queue, &[_]vk.SubmitInfo{.{
            .command_buffer_count = 1,
            .p_command_buffers = &[_]vk.CommandBuffer{self.cmd},
        }}, self.ctx.fence);
    }

    /// Re-begin recording to reuse this batch for another step. Call after the
    /// previous submission has completed (`ctx.wait()`).
    pub fn reset(self: *Pipeline) !void {
        try self.ctx.device.resetCommandBuffer(self.cmd, .{});
        try self.ctx.device.beginCommandBuffer(self.cmd, &.{ .flags = .{ .one_time_submit = true } });
        self.count = 0;
    }

    pub fn deinit(self: *Pipeline) void {
        self.ctx.device.freeCommandBuffers(self.ctx.cmd_pool, &[_]vk.CommandBuffer{self.cmd});
    }
};
