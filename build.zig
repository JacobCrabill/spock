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

    // ------ Built-In Kernels for Linear Algebra ------

    // ------ Unit Tests ------

    const test_exe = b.addTest(.{ .name = "test", .root_module = spock });

    // Each kernel is compiled to SPIR-V, exposed to downstream builds as a named
    // LazyPath, and embedded into the test exe as `<name>.spv`. The imports land on
    // the *root* module, so any file in the spock module can `@embedFile` them.
    inline for (blas_kernels) |name| {
        const spv = addSpirvKernel(b, .{
            .name = name,
            .root_source_file = b.path(b.fmt("src/kernels/blas/{s}.zig", .{name})),
            .optimize = optimize,
        });
        const spv_name = "spock/" ++ name ++ ".spv";
        b.addNamedLazyPath(spv_name, spv);
        test_exe.root_module.addAnonymousImport(spv_name, .{ .root_source_file = spv });
    }
    inline for (basic_kernels) |name| {
        const spv = addSpirvKernel(b, .{
            .name = name,
            .root_source_file = b.path(b.fmt("src/kernels/basic/{s}.zig", .{name})),
            .optimize = optimize,
        });
        const spv_name = "spock/" ++ name ++ ".spv";
        b.addNamedLazyPath(spv_name, spv);
        test_exe.root_module.addAnonymousImport(spv_name, .{ .root_source_file = spv });
    }

    const run_step = b.addRunArtifact(test_exe);
    run_step.has_side_effects = true; // Force the test to always be run on command
    const step = b.step("test", "Run all compute-kernel unit tests");
    step.dependOn(&run_step.step);
}

/// Built-in linear-algebra kernels. Each entry `x` expects `src/kernels/blas/x.zig`
/// with an entrypoint exported as `x`, and a test in `src/kernels/tests/x.zig`.
const blas_kernels = [_][]const u8{
    "dgemm",
    "dgemv",
    "sgemm",
    "sgemv",
};

const basic_kernels = [_][]const u8{
    "addvecf32",
    "addvecf64",
    "addveci32",
    "addveci64",
};

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
    /// Spock's own dependency handle, i.e. the result of `b.dependency("spock", ...)`.
    ///
    /// Required when calling from a *consumer's* build.zig: `addSpirvKernel` receives
    /// the caller's `b`, so it cannot otherwise locate `tools/spv_patch.zig` inside
    /// spock's source tree. Leave null only when called from spock's own build.zig.
    spock_dep: ?*std.Build.Dependency = null,
    /// Run the emitted module through `tools/spv_patch.zig`. See that file for why
    /// this is needed; turn it off once Zig emits a Vulkan-clean module.
    patch: bool = true,
};

/// Build the host-side SPIR-V patch tool (see `tools/spv_patch.zig`).
fn addSpvPatchTool(b: *std.Build, opts: KernelOpts) *std.Build.Step.Compile {
    const src = if (opts.spock_dep) |dep|
        dep.path("tools/spv_patch.zig")
    else
        b.path("tools/spv_patch.zig");

    return b.addExecutable(.{
        .name = "spv-patch",
        .root_module = b.createModule(.{
            .root_source_file = src,
            .target = b.graph.host,
            .optimize = .ReleaseSafe,
        }),
    });
}

/// Add a compiled SPIR-V kernel.
///
/// Compiles the 'main()' entrypoint to a Vulkan compute kernel.
/// Returns a LazyPath to the generated .spv file.
///
/// After adding 'spock' as a dependency to your `build.zig.zon`,
/// you can use this in your `build.zig` like:
///
///     const spock_dep = b.dependency("spock", .{ .target = target, .optimize = optimize });
///     const spv_file: std.Build.LazyPath = @import("spock").addSpirvKernel(b, .{
///         .name = "filter",
///         .root_source_file = b.path("src/kernels/filter.zig"),
///         .imports = &.{.{ .name = "config", .mod = cfg_mod }},
///         .optimize = optimize,
///         .spock_dep = spock_dep,
///     });
///     my_exe.root_module.addAnonymousImport("filter.spv", .{ .root_source_file = kernel_spv });
///
/// Then in your host-side .zig file:
///
///     const filter_spv = @embedFile("filter.spv");
///
/// For a complete usage example, see the `example/` directory.
pub fn addSpirvKernel(b: *std.Build, opts: KernelOpts) std.Build.LazyPath {
    const spirv_target = b.resolveTargetQuery(std.Target.Query.parse(.{
        .arch_os_abi = "spirv32-vulkan",
        .cpu_features = "vulkan_v1_2+float64+int64",
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

    if (!opts.patch) return kernel.getEmittedBin();

    // Zig emits a SPIR-V *object*, which declares things Vulkan rejects. Rewrite it
    // into a conformant module before handing it to any host code.
    const run = b.addRunArtifact(addSpvPatchTool(b, opts));
    run.addFileArg(kernel.getEmittedBin());
    return run.addOutputFileArg(b.fmt("{s}.spv", .{opts.name}));
}
