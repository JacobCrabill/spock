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
        /// Index into `ctx.mem_props.memory_types` of the heap this came from.
        /// Worth having: whether it is cached decides how expensive `slice()` is
        /// to read, and that is otherwise invisible.
        mem_type: u32,

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

            // These buffers are persistently mapped and read back through
            // `slice()`, so the host reads them as ordinary memory -- and an
            // uncached (write-combined) heap makes that punishingly slow: reads
            // measured ~25x slower than from a normal allocation on an NVIDIA
            // discrete GPU, where the first host-visible type is uncached and a
            // cached one sits immediately after it. Ask for cached first and
            // fall back only if the device has nothing better.
            const want: vk.MemoryPropertyFlags = .{ .host_visible = true, .host_coherent = true };
            const prefer: vk.MemoryPropertyFlags = .{ .host_visible = true, .host_coherent = true, .host_cached = true };
            const idx = findMemoryType(ctx, req.memory_type_bits, prefer) orelse
                findMemoryType(ctx, req.memory_type_bits, want) orelse
                return Error.NoHostVisibleMemory;

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
                .mem_type = idx,
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

test "buffers prefer cached host-visible memory" {
    // An uncached (write-combined) heap is cheap to write and punishing to read,
    // and `slice()` exists precisely to invite the host to read. Where the device
    // offers a cached host-visible type, that is the one to take.
    var ctx = Context.init(std.testing.allocator, .{}) catch |err| switch (err) {
        error.NoComputeDevice => return error.SkipZigTest,
        else => return err,
    };
    defer ctx.deinit();

    var buf = try Buffer(f64).create(&ctx, 1024);
    defer buf.deinit();

    const chosen = ctx.mem_props.memory_types[buf.mem_type].property_flags;
    try std.testing.expect(chosen.host_visible and chosen.host_coherent);

    // Only demand it where the device has one this buffer could have used.
    const req = ctx.device.getBufferMemoryRequirements(buf.raw());
    var cached_available = false;
    for (0..ctx.mem_props.memory_type_count) |i| {
        if (req.memory_type_bits & (@as(u32, 1) << @intCast(i)) == 0) continue;
        const f = ctx.mem_props.memory_types[i].property_flags;
        if (f.host_visible and f.host_coherent and f.host_cached) cached_available = true;
    }
    if (cached_available) try std.testing.expect(chosen.host_cached);
}
