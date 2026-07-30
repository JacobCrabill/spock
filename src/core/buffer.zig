//! Host/Device buffer management
const std = @import("std");
pub const vk = @import("vulkan");

const context = @import("context.zig");

const Context = context.Context;
const Error = context.Error;

/// Where a buffer's memory lives.
pub const Location = enum {
    /// Host-visible and persistently mapped, so `slice()` reads and writes it
    /// like any other memory. The device reaches it over the bus, which for a
    /// discrete GPU is an order of magnitude slower than its own memory --
    /// convenient, and the wrong place for anything a kernel touches often.
    host,
    /// The device's own memory. Not addressable from the host at all: move data
    /// with `copyFrom`, which goes through a host-visible staging buffer.
    device,
};

/// A typed GPU array, host-visible and persistently mapped by default.
pub fn Buffer(comptime T: type) type {
    return struct {
        const Self = @This();

        ctx: *Context,
        buf: vk.Buffer,
        mem: vk.DeviceMemory,
        len: usize,
        location: Location,
        /// Null for a device-local buffer, which the host cannot address
        ptr: ?[*]T,
        /// Index into `ctx.mem_props.memory_types` of the heap this came from.
        /// Worth having: whether it is cached decides how expensive `slice()` is
        /// to read, and that is otherwise invisible.
        mem_type: u32,

        /// Allocate a host-visible, persistently device-mapped array of `n` elements of `T`.
        pub fn create(ctx: *Context, n: usize) !Buffer(T) {
            return createIn(ctx, n, .host);
        }

        /// Allocate `n` elements of `T` in the given memory.
        pub fn createIn(ctx: *Context, n: usize, location: Location) !Buffer(T) {
            const bytes: vk.DeviceSize = @sizeOf(T) * n;
            const buf = try ctx.device.createBuffer(&.{
                .size = bytes,
                .usage = .{ .storage_buffer = true, .transfer_src = true, .transfer_dst = true },
                .sharing_mode = .exclusive,
            }, null);
            errdefer ctx.device.destroyBuffer(buf, null);

            const req = ctx.device.getBufferMemoryRequirements(buf);

            // A host buffer is persistently mapped and read back through
            // `slice()`, so the host reads it as ordinary memory -- and an
            // uncached (write-combined) heap makes that punishingly slow: reads
            // measured ~25x slower than from a normal allocation on an NVIDIA
            // discrete GPU, where the first host-visible type is uncached and a
            // cached one sits immediately after it. Ask for cached first and
            // fall back only if the device has nothing better.
            const want: vk.MemoryPropertyFlags = switch (location) {
                .host => .{ .host_visible = true, .host_coherent = true },
                .device => .{ .device_local = true },
            };
            const idx = switch (location) {
                .host => findMemoryType(ctx, req.memory_type_bits, .{
                    .host_visible = true,
                    .host_coherent = true,
                    .host_cached = true,
                }) orelse findMemoryType(ctx, req.memory_type_bits, want),
                .device => findMemoryType(ctx, req.memory_type_bits, want),
            } orelse return Error.NoHostVisibleMemory;

            const mem = try ctx.device.allocateMemory(&.{
                .allocation_size = req.size,
                .memory_type_index = idx,
            }, null);
            errdefer ctx.device.freeMemory(mem, null);
            try ctx.device.bindBufferMemory(buf, mem, 0);

            const mapped: ?[*]T = switch (location) {
                .host => @ptrCast(@alignCast((try ctx.device.mapMemory(mem, 0, bytes, .{})).?)),
                .device => null,
            };

            return .{
                .ctx = ctx,
                .buf = buf,
                .mem = mem,
                .len = n,
                .location = location,
                .ptr = mapped,
                .mem_type = idx,
            };
        }

        /// Direct host view of the array (read + write; coherent with the device).
        ///
        /// Host buffers only -- a device-local one has no host address, and its
        /// contents move through `copyFrom`.
        pub fn slice(self: Self) []T {
            std.debug.assert(self.location == .host);
            return self.ptr.?[0..self.len];
        }

        /// The underlying Vulkan handle, for passing to `Kernel.dispatch`.
        pub fn raw(self: Self) vk.Buffer {
            return self.buf;
        }

        /// Whether the host can address this buffer's contents directly.
        pub fn isMapped(self: Self) bool {
            return self.ptr != null;
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

        /// Copy `src` into this buffer on the device, and wait for it.
        ///
        /// The two may be in different memory: this is how data reaches a
        /// device-local buffer, with a host one as the staging area. Copies the
        /// smaller of the two lengths.
        pub fn copyFrom(self: Self, src: Self) !void {
            const n = @min(self.len, src.len);
            if (n == 0) return;

            const dev = self.ctx.device;
            var cmd: vk.CommandBuffer = undefined;
            try dev.allocateCommandBuffers(&.{
                .command_pool = self.ctx.cmd_pool,
                .level = .primary,
                .command_buffer_count = 1,
            }, @ptrCast(&cmd));
            defer dev.freeCommandBuffers(self.ctx.cmd_pool, &[_]vk.CommandBuffer{cmd});

            try dev.beginCommandBuffer(cmd, &.{ .flags = .{ .one_time_submit = true } });
            dev.cmdCopyBuffer(cmd, src.buf, self.buf, &[_]vk.BufferCopy{.{
                .src_offset = 0,
                .dst_offset = 0,
                .size = @sizeOf(T) * n,
            }});
            try dev.endCommandBuffer(cmd);

            try dev.resetFences(&[_]vk.Fence{self.ctx.fence});
            try dev.queueSubmit(self.ctx.queue, &[_]vk.SubmitInfo{.{
                .command_buffer_count = 1,
                .p_command_buffers = &[_]vk.CommandBuffer{cmd},
            }}, self.ctx.fence);
            try self.ctx.wait();
        }

        pub fn deinit(self: Self) void {
            if (self.location == .host) self.ctx.device.unmapMemory(self.mem);
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

test "a device-local buffer round-trips through a host one" {
    // The point of device-local memory is that the host cannot reach it, so the
    // only way data gets in or out is a copy -- and the only way to know the
    // copy works is to send something through and read it back.
    var ctx = Context.init(std.testing.allocator, .{}) catch |err| switch (err) {
        error.NoComputeDevice => return error.SkipZigTest,
        else => return err,
    };
    defer ctx.deinit();

    const n = 4096;
    var host = try Buffer(f64).create(&ctx, n);
    defer host.deinit();
    var dev = try Buffer(f64).createIn(&ctx, n, .device);
    defer dev.deinit();

    try std.testing.expect(host.isMapped());
    try std.testing.expect(!dev.isMapped());

    // Device memory really is device memory, where the device has any
    const flags = ctx.mem_props.memory_types[dev.mem_type].property_flags;
    try std.testing.expect(flags.device_local);

    for (host.slice(), 0..) |*v, i| v.* = @floatFromInt(i * 3);
    try dev.copyFrom(host);

    host.fill(-1.0);
    try host.copyFrom(dev);

    for (host.slice(), 0..) |v, i| {
        try std.testing.expectEqual(@as(f64, @floatFromInt(i * 3)), v);
    }
}

test "a kernel can be dispatched over device-local memory" {
    const spock = @import("../spock.zig");
    const dgemm = @import("../kernels/blas/dgemm.zig");

    var ctx = Context.init(std.testing.allocator, .{}) catch |err| switch (err) {
        error.NoComputeDevice => return error.SkipZigTest,
        else => return err,
    };
    defer ctx.deinit();

    const M: u32 = 4;
    const N: u32 = 8;
    const K: u32 = 3;

    var stage = try Buffer(f64).create(&ctx, M * N);
    defer stage.deinit();
    var a = try Buffer(f64).createIn(&ctx, M * K, .device);
    defer a.deinit();
    var b = try Buffer(f64).createIn(&ctx, K * N, .device);
    defer b.deinit();
    var c = try Buffer(f64).createIn(&ctx, M * N, .device);
    defer c.deinit();

    for (stage.slice()[0 .. M * K], 0..) |*v, i| v.* = @floatFromInt(i);
    try a.copyFrom(stage);
    for (stage.slice()[0 .. K * N], 0..) |*v, i| v.* = @floatFromInt(i);
    try b.copyFrom(stage);

    var kernel = try spock.Kernel.create(&ctx, .{
        .spirv = @embedFile("spock/dgemm.spv"),
        .entry = "dgemm",
        .buffers = 3,
        .push_constant_size = @sizeOf(dgemm.PushConstants),
    });
    defer kernel.deinit();

    const pc = dgemm.PushConstants{ .M = M, .N = N, .K = K, .alpha = 1.0, .beta = 0.0 };
    try kernel.dispatch(.{
        .buffers = &.{ .whole(a.raw()), .whole(b.raw()), .whole(c.raw()) },
        .push_constant = std.mem.asBytes(&pc),
        .groups = .{ (M * N + 127) / 128, 1, 1 },
    });
    try ctx.wait();

    try stage.copyFrom(c);
    for (0..M) |i| {
        for (0..N) |j| {
            var want: f64 = 0.0;
            for (0..K) |k| {
                want += @as(f64, @floatFromInt(i * K + k)) * @as(f64, @floatFromInt(k * N + j));
            }
            try std.testing.expectApproxEqAbs(want, stage.slice()[i * N + j], 1e-12);
        }
    }
}
