//! Compute kernel definition and dispatch
const std = @import("std");
pub const vk = @import("vulkan");

const context = @import("context.zig");

const Context = context.Context;
const Error = context.Error;

/// A single compute kernel instance.
///
/// In Vulkan terms, this is a compiled compute pipeline plus its descriptor/command state.
pub const Kernel = struct {
    ctx: *Context,
    dsl: vk.DescriptorSetLayout,
    layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,
    pool: vk.DescriptorPool,
    dset: vk.DescriptorSet,
    cmd: vk.CommandBuffer,
    n_buffers: u32,
    pc_size: u32,

    pub const Config = struct {
        /// SPIR-V module bytes (e.g. `@embedFile("kernel.spv")`). Any alignment is fine.
        spirv: []const u8,
        /// The entrypoint of the kernel (typically just "main").
        entry: [*:0]const u8 = "main",
        /// Number of storage-buffer arguments (descriptor bindings 0..buffers-1).
        buffers: u32,
        /// Size of the push-constant block, in bytes (0 = none).
        push_constant_size: u32 = 0,
    };

    pub const DispatchArgs = struct {
        /// Storage-buffer arguments, in binding order. Length must equal the
        /// kernel's `buffers` count.
        buffers: []const vk.Buffer,
        /// Push-constant bytes (e.g. `std.mem.asBytes(&pc)`). Length must equal
        /// the kernel's `push_constant_size`.
        push_constant: []const u8 = &.{},
        /// Number of workgroups to launch in x, y, z.
        groups: [3]u32 = .{ 1, 1, 1 },
    };

    /// Build a compute pipeline from a SPIR-V module.
    pub fn create(ctx: *Context, desc: Config) !Kernel {
        const dev = ctx.device;

        // Descriptor set layout: N storage buffers on the compute stage.
        const bindings = try ctx.gpa.alloc(vk.DescriptorSetLayoutBinding, desc.buffers);
        defer ctx.gpa.free(bindings);
        for (bindings, 0..) |*b, i| b.* = .{
            .binding = @intCast(i),
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{ .compute = true },
        };
        const dsl = try dev.createDescriptorSetLayout(&.{
            .binding_count = desc.buffers,
            .p_bindings = bindings.ptr,
        }, null);
        errdefer dev.destroyDescriptorSetLayout(dsl, null);

        const pc_ranges = [_]vk.PushConstantRange{.{
            .stage_flags = .{ .compute = true },
            .offset = 0,
            .size = desc.push_constant_size,
        }};
        const layout = try dev.createPipelineLayout(&.{
            .set_layout_count = 1,
            .p_set_layouts = &[_]vk.DescriptorSetLayout{dsl},
            .push_constant_range_count = if (desc.push_constant_size > 0) 1 else 0,
            .p_push_constant_ranges = &pc_ranges,
        }, null);
        errdefer dev.destroyPipelineLayout(layout, null);

        // Shader module. Copy to a u32 buffer (naturally 4-aligned) so any input
        // alignment works for SPIR-V's p_code requirement.
        const code = try ctx.gpa.alloc(u32, (desc.spirv.len + 3) / 4);
        defer ctx.gpa.free(code);
        @memcpy(std.mem.sliceAsBytes(code)[0..desc.spirv.len], desc.spirv);
        const module = try dev.createShaderModule(&.{
            .code_size = desc.spirv.len,
            .p_code = code.ptr,
        }, null);
        defer dev.destroyShaderModule(module, null); // baked into the pipeline; not needed after

        var pipelines: [1]vk.Pipeline = undefined;
        _ = try dev.createComputePipelines(.null_handle, &[_]vk.ComputePipelineCreateInfo{.{
            .stage = .{ .stage = .{ .compute = true }, .module = module, .p_name = desc.entry },
            .layout = layout,
            .base_pipeline_index = -1,
        }}, null, &pipelines);
        errdefer dev.destroyPipeline(pipelines[0], null);

        // Descriptor pool + set (reused; rebound each dispatch).
        const pool = try dev.createDescriptorPool(&.{
            .max_sets = 1,
            .pool_size_count = 1,
            .p_pool_sizes = &[_]vk.DescriptorPoolSize{.{ .type = .storage_buffer, .descriptor_count = desc.buffers }},
        }, null);
        errdefer dev.destroyDescriptorPool(pool, null);

        var dset: vk.DescriptorSet = undefined;
        try dev.allocateDescriptorSets(&.{
            .descriptor_pool = pool,
            .descriptor_set_count = 1,
            .p_set_layouts = &[_]vk.DescriptorSetLayout{dsl},
        }, @ptrCast(&dset));

        var cmd: vk.CommandBuffer = undefined;
        try dev.allocateCommandBuffers(&.{
            .command_pool = ctx.cmd_pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, @ptrCast(&cmd));

        return .{
            .ctx = ctx,
            .dsl = dsl,
            .layout = layout,
            .pipeline = pipelines[0],
            .pool = pool,
            .dset = dset,
            .cmd = cmd,
            .n_buffers = desc.buffers,
            .pc_size = desc.push_constant_size,
        };
    }

    /// Bind arguments and dispatch as a standalone, one-shot submission. Returns
    /// immediately (async); call `ctx.wait()` to block for completion. For chaining
    /// several kernels in one submission, use `Batch.create()`.
    pub fn dispatch(self: *Kernel, args: DispatchArgs) !void {
        const dev = self.ctx.device;
        try dev.resetCommandBuffer(self.cmd, .{});
        try dev.beginCommandBuffer(self.cmd, &.{ .flags = .{ .one_time_submit = true } });
        try self.record(self.cmd, args);
        try dev.endCommandBuffer(self.cmd);

        try dev.resetFences(&[_]vk.Fence{self.ctx.fence});
        try dev.queueSubmit(self.ctx.queue, &[_]vk.SubmitInfo{.{
            .command_buffer_count = 1,
            .p_command_buffers = &[_]vk.CommandBuffer{self.cmd},
        }}, self.ctx.fence);
    }

    /// Bind arguments and record a dispatch into an already-begun command buffer.
    /// Updates this kernel's (single) descriptor set, so a given kernel may be
    /// recorded at most once per in-flight command buffer; for ping-pong within a
    /// batch, use two kernel instances.
    pub fn record(self: *Kernel, cmd: vk.CommandBuffer, args: DispatchArgs) !void {
        if (args.buffers.len != self.n_buffers or args.push_constant.len != self.pc_size)
            return Error.BadDispatchArgs;

        var arena = std.heap.ArenaAllocator.init(self.ctx.gpa);
        defer arena.deinit();
        const alloc = arena.allocator();

        const dev = self.ctx.device;

        // Point the descriptor set at the caller's buffers.
        const infos = try alloc.alloc(vk.DescriptorBufferInfo, self.n_buffers);
        const writes = try alloc.alloc(vk.WriteDescriptorSet, self.n_buffers);
        for (args.buffers, infos, writes, 0..) |b, *info, *w, i| {
            info.* = .{ .buffer = b, .offset = 0, .range = vk.WHOLE_SIZE };
            w.* = .{
                .dst_set = self.dset,
                .dst_binding = @intCast(i),
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_buffer_info = info[0..1].ptr,
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            };
        }
        dev.updateDescriptorSets(writes, null);

        dev.cmdBindPipeline(cmd, .compute, self.pipeline);
        dev.cmdBindDescriptorSets(cmd, .compute, self.layout, 0, &[_]vk.DescriptorSet{self.dset}, null);
        if (self.pc_size > 0)
            dev.cmdPushConstants(cmd, self.layout, .{ .compute = true }, 0, self.pc_size, args.push_constant.ptr);
        dev.cmdDispatch(cmd, args.groups[0], args.groups[1], args.groups[2]);
    }

    pub fn deinit(self: *Kernel) void {
        const dev = self.ctx.device;
        dev.freeCommandBuffers(self.ctx.cmd_pool, &[_]vk.CommandBuffer{self.cmd});
        dev.destroyDescriptorPool(self.pool, null);
        dev.destroyPipeline(self.pipeline, null);
        dev.destroyPipelineLayout(self.layout, null);
        dev.destroyDescriptorSetLayout(self.dsl, null);
    }
};
