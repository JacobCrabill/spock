const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- Module Config -------------------------------------------------------

    // Regenerate the vulkan-zig bindings from the Vulkan registry via the
    // vulkan-zig generator (a lazy dependency, only fetched when this is set).
    // Off by default: the vkzig/gpu hosts use the vendored src/bingings/vk.zig.
    const gen_vk = b.option(bool, "gen-vk", "Regenerate vulkan-zig bindings from the registry instead of using the vendored src/vulkan/vk.zig") orelse false;
    const vk_registry = b.option([]const u8, "vk-registry", "Path to the Vulkan registry (vk.xml) used by -Dgen-vk") orelse "/usr/share/vulkan/registry/vk.xml";

    // --- vulkan-zig bindings source -------------------------------------------
    // Default: the vkzig/gpu hosts import the vendored static file by *relative
    // path* (`@import("vk.zig")`, resolved next to each source file), which keeps
    // language-server goto-definition working. With -Dgen-vk we instead run the
    // vulkan-zig generator over the registry and override the `vk.zig` import on
    // those modules to point at the freshly generated file.
    const vk_zig_generated: ?std.Build.LazyPath = if (gen_vk) blk: {
        // Lazy: the dependency is only fetched when -Dgen-vk is actually passed.
        const vulkan_zig = b.lazyDependency("vulkan_zig", .{
            .target = b.graph.host,
            .optimize = .ReleaseSafe,
        }) orelse break :blk null;
        const gen = b.addRunArtifact(vulkan_zig.artifact("vulkan-zig-generator"));
        gen.addFileArg(.{ .cwd_relative = vk_registry }); // registry lives outside the repo
        break :blk gen.addOutputFileArg("vk.zig");
    } else null;

    // ------ The Spock Module -------
    const spock = b.addModule("spock", .{
        .root_source_file = b.path("src/spock.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true, // the Vulkan loader (libvulkan.so) needs libc
    });
    if (vk_zig_generated) |vkzig| {
        spock.addImport("vulkan", b.createModule(.{ .root_source_file = vkzig }));
    } else {
        spock.addImport("vulkan", b.createModule(.{ .root_source_file = b.path("src/bindings/vk.zig") }));
    }
    spock.linkSystemLibrary("vulkan", .{});
}

/// Zig build module definition for `addImport()`
pub const ModImport = struct {
    name: []const u8,
    mod: *std.Build.Module,
};

/// Options used to build a Vulkan SPIR-V kernel
pub const KernelOpts = struct {
    name: []const u8,
    root_source_file: std.Build.LazyPath,
    optimize: std.builtin.OptimizeMode,
    imports: ?[]const ModImport = null,
};

/// Add a compiled SPIR-V kernel
///
/// Compiles the 'main()' entrypoint to a Vulkan compute kernel
///
/// Returns a LazyPath to the generated .spv file
///
/// After adding 'spock' as a dependency to your `build.zig.zon`,
/// you can use this in your `build.zig` like:
///
///     const spv_file: std.Build.LazyPath = @import("spock").addSpirvKernel(
///         .name = "filter",
///         .root_source_file = b.path("src/kernels/filter.zig"),
///         .imports = &.{.{ .name = "config", .mod = cfg_mod }},
///         .optimize = optimize,
///     );
///     my_exe.root_module.addAnonymousImport("filter.spv", .{ .root_source_file = kernel_spv });
///
/// Then in your host-side .zig file:
///
///     const filter_spv = @embedFile("filter.spv");
pub fn addSpirvKernel(b: *std.Build, opts: KernelOpts) std.Build.LazyPath {
    const spirv_target = b.resolveTargetQuery(std.Target.Query.parse(.{
        .arch_os_abi = "spirv32-vulkan",
        .cpu_features = "vulkan_v1_2+float64",
    }) catch @panic("bad spirv target"));

    const kernel = b.addObject(.{
        .name = opts.name,
        .root_module = b.createModule(.{
            .root_source_file = opts.root_source_file,
            .target = spirv_target,
            .optimize = opts.optimize,
        }),
        .use_llvm = false, // SPIR-V is only supported by Zig's self-hosted backend.
    });

    if (opts.imports) |imports| {
        for (imports) |mod| {
            kernel.root_module.addImport(mod.name, mod.mod);
        }
    }

    return kernel.getEmittedBin();
}
