const std = @import("std");

const spock = @import("spock");

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
    const chain = b.option(bool, "chain", "run: chain two filter passes in one submission (barrier demo)") orelse false;

    // --- Build Config -------------------------------------------------------
    // Regenerate the vulkan-zig bindings from the Vulkan registry via the
    // vulkan-zig generator (a lazy dependency, only fetched when this is set).
    // Off by default: the vkzig/gpu hosts use the vendored src/bingings/vk.zig.
    const gen_vk = b.option(bool, "gen-vk", "Regenerate vulkan-zig bindings from the registry instead of using the vendored src/vulkan/vk.zig") orelse false;
    const vk_registry = b.option([]const u8, "vk-registry", "Path to the Vulkan registry (vk.xml) used by -Dgen-vk") orelse "/usr/share/vulkan/registry/vk.xml";

    // Import the spock module
    const spock_mod = b.dependency("spock", .{
        .@"gen-vk" = gen_vk,
        .@"vk-registry" = vk_registry,
        .target = target,
        .optimize = optimize,
    }).module("spock");

    // A generated `config` module carries the compile-time workgroup size to
    // both the kernel and the host, keeping them in sync from one -Dwgsize.
    const cfg = b.addOptions();
    cfg.addOption(u32, "wg_size", wgsize);
    const cfg_mod = cfg.createModule();

    // 1. Compile the GPU kernel to a SPIR-V module with the self-hosted backend.
    //    `spirv32-vulkan` + `vulkan_v1_2` matches what Vulkan drivers ingest.
    const kernel_spv = spock.addSpirvKernel(b, .{
        .name = "filter",
        .root_source_file = b.path("src/kernels/filter.zig"),
        .imports = &.{.{ .name = "config", .mod = cfg_mod }},
        .optimize = optimize,
    });

    const dgemm_spv = spock.addSpirvKernel(b, .{
        .name = "dgemm",
        .root_source_file = b.path("src/kernels/dgemm.zig"),
        .optimize = optimize,
    });

    // Demo Executable
    const exe = b.addExecutable(.{
        .name = "spock_demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .strip = optimize != .debug,
        }),
    });
    exe.root_module.addImport("config", cfg_mod);
    exe.root_module.addImport("spock", spock_mod);
    exe.root_module.addAnonymousImport("filter.spv", .{ .root_source_file = kernel_spv });
    exe.root_module.addAnonymousImport("dgemm.spv", .{ .root_source_file = dgemm_spv });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.addArgs(&.{
        b.fmt("--threshold={s}", .{threshold}),
        b.fmt("--n={s}", .{n}),
    });
    if (validate) run.addArg("--validate");
    if (chain) run.addArg("--chain");
    run.step.dependOn(b.getInstallStep());
    b.step("run", "Run the Spock demo").dependOn(&run.step);
}
