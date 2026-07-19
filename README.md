# Spock: The Human-Friendly Vulkan for Science

Spock is a highly opinionated set of Vulkan wrappers intended for computational physics.

The goal is to minimize the amount of boilerplate required to implement basic compute-kernel
pipelines for computational physics applications, abstracting the process of creating and managing
device data arrays, binding kernel descriptors (function arguments), and invoking pipelines of
compute kernels.

> [!NOTE]
> This is very much a WIP!

## Use

In your `build.zig.zon` file:

```zig
    .dependencies = .{
        .spock = .{
            .url = "...",
            .hash = "...",
        },
    },
```

Then in your `build.zig` file(s):

```zig
const spock = @import("spock");
const spv_file: std.Build.LazyPath = @import("spock").addSpirvKernel(
    .name = "filter",
    .root_source_file = b.path("src/kernels/filter.zig"),
    .imports = &.{.{ .name = "config", .mod = cfg_mod }},
    .optimize = optimize,
);
my_exe.root_module.addAnonymousImport("my-filter.spv", .{ .root_source_file = kernel_spv });
```

Then in your host-side .zig file:

```zig
const filter_spv = @embedFile("my-filter.spv");
```

See `example/` for a complete example project using Spock with a simple dummy kernel.
