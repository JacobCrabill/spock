const std = @import("std");

// Minimal OpenCL bindings (no headers needed — just the symbols we call).
const cl_int = i32;
const cl_uint = u32;
const cl_ulong = u64;
const CL_SUCCESS = 0;
const CL_DEVICE_TYPE_GPU: cl_ulong = 1 << 2;
const CL_MEM_READ_ONLY: cl_ulong = 1 << 2;
const CL_MEM_WRITE_ONLY: cl_ulong = 1 << 1;
const CL_TRUE: cl_uint = 1;

extern fn clGetPlatformIDs(cl_uint, ?[*]?*anyopaque, ?*cl_uint) cl_int;
extern fn clGetDeviceIDs(?*anyopaque, cl_ulong, cl_uint, ?[*]?*anyopaque, ?*cl_uint) cl_int;
extern fn clCreateContext(?*const anyopaque, cl_uint, [*]const ?*anyopaque, ?*anyopaque, ?*anyopaque, *cl_int) ?*anyopaque;
extern fn clCreateCommandQueue(?*anyopaque, ?*anyopaque, cl_ulong, *cl_int) ?*anyopaque;
extern fn clCreateProgramWithIL(?*anyopaque, *const anyopaque, usize, *cl_int) ?*anyopaque;
extern fn clBuildProgram(?*anyopaque, cl_uint, ?[*]const ?*anyopaque, ?[*:0]const u8, ?*anyopaque, ?*anyopaque) cl_int;
extern fn clGetProgramBuildInfo(?*anyopaque, ?*anyopaque, cl_uint, usize, ?*anyopaque, ?*usize) cl_int;
extern fn clCreateKernel(?*anyopaque, [*:0]const u8, *cl_int) ?*anyopaque;
extern fn clCreateBuffer(?*anyopaque, cl_ulong, usize, ?*anyopaque, *cl_int) ?*anyopaque;
extern fn clSetKernelArg(?*anyopaque, cl_uint, usize, ?*const anyopaque) cl_int;
extern fn clEnqueueWriteBuffer(?*anyopaque, ?*anyopaque, cl_uint, usize, usize, *const anyopaque, cl_uint, ?*anyopaque, ?*anyopaque) cl_int;
extern fn clEnqueueReadBuffer(?*anyopaque, ?*anyopaque, cl_uint, usize, usize, *anyopaque, cl_uint, ?*anyopaque, ?*anyopaque) cl_int;
extern fn clEnqueueNDRangeKernel(?*anyopaque, ?*anyopaque, cl_uint, ?[*]const usize, [*]const usize, ?[*]const usize, cl_uint, ?*anyopaque, ?*anyopaque) cl_int;
extern fn clFinish(?*anyopaque) cl_int;

fn check(err: cl_int, what: []const u8) void {
    if (err != CL_SUCCESS) std.debug.panic("{s} failed: {d}\n", .{ what, err });
}

pub fn main() !void {
    var err: cl_int = 0;

    var platform: ?*anyopaque = null;
    check(clGetPlatformIDs(1, @ptrCast(&platform), null), "clGetPlatformIDs");
    var device: ?*anyopaque = null;
    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, @ptrCast(&device), null), "clGetDeviceIDs");

    const ctx = clCreateContext(null, 1, @ptrCast(&device), null, null, &err);
    check(err, "clCreateContext");
    const queue = clCreateCommandQueue(ctx, device, 0, &err);
    check(err, "clCreateCommandQueue");

    // Load the SPIR-V produced by the Zig compiler.
    const spv = @embedFile("filter.spv");
    const program = clCreateProgramWithIL(ctx, spv.ptr, spv.len, &err);
    check(err, "clCreateProgramWithIL");
    if (clBuildProgram(program, 1, @ptrCast(&device), null, null, null) != CL_SUCCESS) {
        var log: [4096]u8 = undefined;
        var len: usize = 0;
        _ = clGetProgramBuildInfo(program, device, 0x1183, log.len, &log, &len); // CL_PROGRAM_BUILD_LOG
        std.debug.panic("build log:\n{s}\n", .{log[0..len]});
    }
    const kernel = clCreateKernel(program, "filter", &err);
    check(err, "clCreateKernel");

    const n: u32 = 16;
    var in: [n]f32 = undefined;
    for (&in, 0..) |*x, i| x.* = @floatFromInt(i); // 0,1,2,...,15
    const threshold: f32 = 8.0;

    const bytes = n * @sizeOf(f32);
    const d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, null, &err);
    check(err, "clCreateBuffer in");
    const d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, null, &err);
    check(err, "clCreateBuffer out");
    check(clEnqueueWriteBuffer(queue, d_in, CL_TRUE, 0, bytes, &in, 0, null, null), "write");

    check(clSetKernelArg(kernel, 0, @sizeOf(?*anyopaque), @ptrCast(&d_in)), "arg0");
    check(clSetKernelArg(kernel, 1, @sizeOf(?*anyopaque), @ptrCast(&d_out)), "arg1");
    check(clSetKernelArg(kernel, 2, @sizeOf(f32), &threshold), "arg2");
    check(clSetKernelArg(kernel, 3, @sizeOf(u32), &n), "arg3");

    const global = [_]usize{n};
    check(clEnqueueNDRangeKernel(queue, kernel, 1, null, &global, null, 0, null, null), "ndrange");
    check(clFinish(queue), "finish");

    var out: [n]f32 = undefined;
    check(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, bytes, &out, 0, null, null), "read");

    std.debug.print("threshold = {d}\n in : ", .{threshold});
    for (in) |x| std.debug.print("{d:>4} ", .{x});
    std.debug.print("\nout : ", .{});
    for (out) |x| std.debug.print("{d:>4} ", .{x});
    std.debug.print("\n", .{});
}
