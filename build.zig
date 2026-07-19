const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- Configurable knobs -------------------------------------------------
    // Compile-time: the kernel's local workgroup size is baked into the SPIR-V,
    // so it must be known when the kernel is compiled. Shared with the host so
    // both agree on how many groups to dispatch.
    const wgsize = b.option(u32, "wgsize", "Kernel workgroup size (X); recompiles the kernel") orelse 64;
    // Runtime: forwarded to the program as CLI flags; no rebuild needed.
    const threshold = b.option([]const u8, "threshold", "Filter cutoff; keep values > threshold") orelse "128";
    const n = b.option([]const u8, "n", "Number of elements to process") orelse "256";
    const validate = b.option(bool, "validate", "Enable Vulkan validation layer if installed") orelse false;

    // A generated `config` module carries the compile-time workgroup size to
    // both the kernel and the host, keeping them in sync from one -Dwgsize.
    const cfg = b.addOptions();
    cfg.addOption(u32, "wg_size", wgsize);
    const cfg_mod = cfg.createModule();

    // 1. Compile the GPU kernel to a SPIR-V module with the self-hosted backend.
    //    `spirv32-vulkan` + `vulkan_v1_2` matches what Vulkan drivers ingest.
    const spirv_target = b.resolveTargetQuery(std.Target.Query.parse(.{
        .arch_os_abi = "spirv32-vulkan",
        .cpu_features = "vulkan_v1_2",
    }) catch @panic("bad spirv target"));

    const kernel = b.addObject(.{
        .name = "vk_filter",
        .root_module = b.createModule(.{
            .root_source_file = b.path("vk_filter.zig"),
            .target = spirv_target,
            .optimize = optimize,
        }),
    });
    kernel.root_module.addImport("config", cfg_mod);
    const kernel_spv = kernel.getEmittedBin(); // LazyPath to vk_filter.spv

    // 2. Translate <vulkan/vulkan.h> into a Zig module (@cImport is gone in 0.17-dev).
    const vk = b.addTranslateC(.{
        .root_source_file = b.path("vkimport.c"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // 3. The host executable: links Vulkan, imports the bindings, and embeds the .spv.
    const exe = b.addExecutable(.{
        .name = "vk_host",
        .root_module = b.createModule(.{
            .root_source_file = b.path("vk_host.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    exe.root_module.addImport("vk", vk.createModule());
    exe.root_module.addImport("config", cfg_mod);
    // Makes `@embedFile("vk_filter.spv")` resolve to the compiled kernel.
    exe.root_module.addAnonymousImport("vk_filter.spv", .{ .root_source_file = kernel_spv });
    exe.root_module.linkSystemLibrary("vulkan", .{});

    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.addArgs(&.{
        b.fmt("--threshold={s}", .{threshold}),
        b.fmt("--n={s}", .{n}),
    });
    if (validate) run.addArg("--validate");
    run.step.dependOn(b.getInstallStep());
    b.step("run", "Build the kernel + host and run the compute demo").dependOn(&run.step);
}
