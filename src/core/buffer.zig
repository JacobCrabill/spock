//! Host/Device buffer management
const std = @import("std");
pub const vk = @import("vulkan");

const context = @import("context.zig");

const Context = context.Context;
const Error = context.Error;

/// A typed, host-visible, persistently mapped GPU array.
pub fn Buffer(comptime T: type) type {
    return struct {
        const Self = @This();

        ctx: *Context,
        buf: vk.Buffer,
        mem: vk.DeviceMemory,
        len: usize,
        ptr: [*]T,

        /// Allocate a host-visible, persistently device-mapped array of `n` elements of `T`.
        pub fn create(ctx: *Context, n: usize) !Buffer(T) {
            const bytes: vk.DeviceSize = @sizeOf(T) * n;
            const buf = try ctx.device.createBuffer(&.{
                .size = bytes,
                .usage = .{ .storage_buffer = true, .transfer_src = true, .transfer_dst = true },
                .sharing_mode = .exclusive,
            }, null);
            errdefer ctx.device.destroyBuffer(buf, null);

            const req = ctx.device.getBufferMemoryRequirements(buf);
            const want: vk.MemoryPropertyFlags = .{ .host_visible = true, .host_coherent = true };
            const idx = findMemoryType(ctx, req.memory_type_bits, want) orelse return Error.NoHostVisibleMemory;

            const mem = try ctx.device.allocateMemory(&.{
                .allocation_size = req.size,
                .memory_type_index = idx,
            }, null);
            errdefer ctx.device.freeMemory(mem, null);
            try ctx.device.bindBufferMemory(buf, mem, 0);

            const mapped = try ctx.device.mapMemory(mem, 0, bytes, .{});
            return .{
                .ctx = ctx,
                .buf = buf,
                .mem = mem,
                .len = n,
                .ptr = @ptrCast(@alignCast(mapped.?)),
            };
        }

        /// Direct host view of the array (read + write; coherent with the device).
        pub fn slice(self: Self) []T {
            return self.ptr[0..self.len];
        }

        /// The underlying Vulkan handle, for passing to `Kernel.dispatch`.
        pub fn raw(self: Self) vk.Buffer {
            return self.buf;
        }

        pub fn fill(self: Self, value: T) void {
            @memset(self.slice(), value);
        }

        pub fn copyFromHost(self: Self, src: []const T) void {
            @memcpy(self.slice()[0..src.len], src);
        }

        /// Copy the array's contents into a host buffer. (Coherent memory, so this
        /// is a plain memcpy once the producing dispatch has been waited on.)
        pub fn copyToHost(self: Self, dst: []T) void {
            const n = @min(dst.len, self.len);
            @memcpy(dst[0..n], self.slice()[0..n]);
        }

        pub fn deinit(self: Self) void {
            self.ctx.device.unmapMemory(self.mem);
            self.ctx.device.destroyBuffer(self.buf, null);
            self.ctx.device.freeMemory(self.mem, null);
        }
    };
}

fn findMemoryType(ctx: *const Context, type_bits: u32, want: vk.MemoryPropertyFlags) ?u32 {
    var i: u32 = 0;
    while (i < ctx.mem_props.memory_type_count) : (i += 1) {
        const usable = type_bits & (@as(u32, 1) << @intCast(i)) != 0;
        if (usable and ctx.mem_props.memory_types[i].property_flags.contains(want)) return i;
    }
    return null;
}
