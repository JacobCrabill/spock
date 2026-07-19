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
pub const vk = @import("vulkan");

const Instance = vk.InstanceProxy;
const Device = vk.DeviceProxy;

extern fn vkGetInstanceProcAddr(instance: vk.Instance, p_name: [*:0]const u8) vk.PfnVoidFunction;

const validation_layer: [*:0]const u8 = "VK_LAYER_KHRONOS_validation";

/// How to choose among multiple GPUs.
pub const Pick = enum {
    /// First device that supports compute.
    first,
    /// Device with the most device-local memory (usually the beefiest discrete GPU).
    largest,
};

pub const Options = struct {
    pick: Pick = .largest,
    validate: bool = false,
    app_name: [*:0]const u8 = "zig-gpu",
};

pub const Error = error{
    NoComputeDevice,
    NoHostVisibleMemory,
    BadDispatchArgs,
} || std.mem.Allocator.Error;

pub const Context = struct {
    gpa: std.mem.Allocator,

    vkb: vk.BaseWrapper,
    vki: *vk.InstanceWrapper,
    vkd: *vk.DeviceWrapper,
    instance: Instance,
    device: Device,

    phys: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    qfi: u32,
    queue: vk.Queue,
    cmd_pool: vk.CommandPool,
    fence: vk.Fence,

    pub fn init(gpa: std.mem.Allocator, opts: Options) !Context {
        const vkb = vk.BaseWrapper.load(vkGetInstanceProcAddr);

        const app: vk.ApplicationInfo = .{
            .p_application_name = opts.app_name,
            .application_version = 0,
            .engine_version = 0,
            .api_version = vk.API_VERSION_1_2.toU32(),
        };
        var ici: vk.InstanceCreateInfo = .{ .p_application_info = &app };
        const layers = [_][*:0]const u8{validation_layer};
        if (opts.validate and validationAvailable(vkb, gpa)) {
            ici.enabled_layer_count = 1;
            ici.pp_enabled_layer_names = &layers;
        }
        const instance_handle = try vkb.createInstance(&ici, null);

        const vki = try gpa.create(vk.InstanceWrapper);
        errdefer gpa.destroy(vki);
        vki.* = vk.InstanceWrapper.load(instance_handle, vkb.dispatch.vkGetInstanceProcAddr.?);
        const instance = Instance.init(instance_handle, vki);
        errdefer instance.destroyInstance(null);

        // --- Pick a physical device + compute queue family ---------------------
        const devices = try instance.enumeratePhysicalDevicesAlloc(gpa);
        defer gpa.free(devices);

        var chosen: ?vk.PhysicalDevice = null;
        var chosen_qfi: u32 = 0;
        var best_score: u64 = 0;
        for (devices) |pd| {
            const qfi = findComputeQueue(instance, pd, gpa) orelse continue;
            // Score = device-type class in the high bits (so a real GPU always
            // beats a CPU software rasterizer like llvmpipe), device-local memory
            // in the low bits. `.first` keeps the earliest by not overwriting ties.
            const score = deviceScore(instance, pd);
            const better = switch (opts.pick) {
                .largest => score > best_score,
                .first => (score >> 56) > (best_score >> 56), // only jump to a better class
            };
            if (chosen == null or better) {
                chosen = pd;
                chosen_qfi = qfi;
                best_score = score;
            }
        }
        const phys = chosen orelse return Error.NoComputeDevice;

        // --- Logical device + queue -------------------------------------------
        const prio = [_]f32{1.0};
        const qci = [_]vk.DeviceQueueCreateInfo{.{
            .queue_family_index = chosen_qfi,
            .queue_count = 1,
            .p_queue_priorities = &prio,
        }};
        const features: vk.PhysicalDeviceFeatures = .{ .shader_float_64 = .true };
        const dev_handle = try instance.createDevice(phys, &.{
            .queue_create_info_count = 1,
            .p_queue_create_infos = &qci,
            .p_enabled_features = &features,
        }, null);

        const vkd = try gpa.create(vk.DeviceWrapper);
        errdefer gpa.destroy(vkd);
        vkd.* = vk.DeviceWrapper.load(dev_handle, instance.wrapper.dispatch.vkGetDeviceProcAddr.?);
        const device = Device.init(dev_handle, vkd);
        errdefer device.destroyDevice(null);

        const cmd_pool = try device.createCommandPool(&.{
            .queue_family_index = chosen_qfi,
            .flags = .{ .reset_command_buffer = true },
        }, null);
        errdefer device.destroyCommandPool(cmd_pool, null);

        const fence = try device.createFence(&.{}, null);

        return .{
            .gpa = gpa,
            .vkb = vkb,
            .vki = vki,
            .vkd = vkd,
            .instance = instance,
            .device = device,
            .phys = phys,
            .props = instance.getPhysicalDeviceProperties(phys),
            .mem_props = instance.getPhysicalDeviceMemoryProperties(phys),
            .qfi = chosen_qfi,
            .queue = device.getDeviceQueue(chosen_qfi, 0),
            .cmd_pool = cmd_pool,
            .fence = fence,
        };
    }

    pub fn deinit(self: *Context) void {
        self.device.destroyFence(self.fence, null);
        self.device.destroyCommandPool(self.cmd_pool, null);
        self.device.destroyDevice(null);
        self.gpa.destroy(self.vkd);
        self.instance.destroyInstance(null);
        self.gpa.destroy(self.vki);
        self.* = undefined;
    }

    /// Name of the selected GPU (for logging).
    pub fn deviceName(self: *const Context) []const u8 {
        return std.mem.sliceTo(&self.props.device_name, 0);
    }

    /// Block until all submitted work has finished.
    pub fn wait(self: *Context) !void {
        _ = try self.device.waitForFences(&[_]vk.Fence{self.fence}, .true, std.math.maxInt(u64));
    }
};

fn validationAvailable(vkb: vk.BaseWrapper, gpa: std.mem.Allocator) bool {
    const props = vkb.enumerateInstanceLayerPropertiesAlloc(gpa) catch return false;
    defer gpa.free(props);
    for (props) |p| {
        if (std.mem.eql(u8, std.mem.sliceTo(&p.layer_name, 0), std.mem.span(validation_layer))) return true;
    }
    return false;
}

fn findComputeQueue(instance: Instance, pd: vk.PhysicalDevice, gpa: std.mem.Allocator) ?u32 {
    const qfams = instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(pd, gpa) catch return null;
    defer gpa.free(qfams);
    for (qfams, 0..) |qf, i| {
        if (qf.queue_flags.contains(.{ .compute = true })) return @intCast(i);
    }
    return null;
}

fn deviceLocalMemory(instance: Instance, pd: vk.PhysicalDevice) u64 {
    const mp = instance.getPhysicalDeviceMemoryProperties(pd);
    var total: u64 = 0;
    var i: u32 = 0;
    while (i < mp.memory_heap_count) : (i += 1) {
        if (mp.memory_heaps[i].flags.device_local) total += mp.memory_heaps[i].size;
    }
    return total;
}

/// Rank device classes so real GPUs outrank software/CPU implementations.
fn typeRank(t: vk.PhysicalDeviceType) u8 {
    return switch (t) {
        .discrete_gpu => 4,
        .integrated_gpu => 3,
        .virtual_gpu => 2,
        .other => 1,
        .cpu => 0,
        _ => 0,
    };
}

/// device-type class in the top byte, device-local memory (clamped) below it.
fn deviceScore(instance: Instance, pd: vk.PhysicalDevice) u64 {
    const props = instance.getPhysicalDeviceProperties(pd);
    const mem = @min(deviceLocalMemory(instance, pd), (@as(u64, 1) << 56) - 1);
    return (@as(u64, typeRank(props.device_type)) << 56) | mem;
}
