# zig-gpu-demo

A minimal, working example of authoring a **GPU compute kernel in Zig**, compiling it to **SPIR-V**
with Zig's self-hosted backend, and running it via **Vulkan compute** — no LLVM, no external
shader compiler (`glslc`/ `clang`), no GPU vendor SDK.

The demo is a **parallel filter**: `out[i] = in[i] if in[i] > threshold else 0`, over an array
`0, 1, 2, …, n-1`.

> Built and verified with **Zig `0.17.0-dev.1422+e863bf3be`**. The SPIR-V backend is experimental
> and the `std`/build APIs move fast — see [API drift](#api-drift-in-this-dev-build).

## Requirements

- A Zig `0.17-dev` build with the SPIR-V backend (this is in mainline Zig; no special flags).
- A Vulkan loader + a driver with compute (`libvulkan.so`, plus any ICD — a real GPU, or the
  `lavapipe` software device works too).
- Vulkan headers (`/usr/include/vulkan/vulkan.h`) for the `translate-c` step.
- Optional: `vulkan-validationlayers` for `-Dvalidate=true`.

## Quick start

```sh
zig build run                                   # defaults: n=256, wgsize=64, threshold=128
zig build run -Dthreshold=200                   # runtime cutoff (push constant, no recompile)
zig build run -Dn=1000 -Dthreshold=990          # any n; kernel bounds-checks the tail
zig build run -Dwgsize=32                        # recompiles the kernel with local size 32
zig build run -Dvalidate=true                    # enable the Vulkan validation layer

./zig-out/bin/vk_host --threshold=200 --n=1000 --validate   # run the binary directly

zig build run-vkzig                              # same demo, host uses Snektron/vulkan-zig bindings
zig build run-vkzig -Dn=1000 -Dvalidate=true    # (accepts the same -D options)
```

Expected output (defaults):

```
config: n=256 wg_size=64 groups=4 threshold=128 validate=false
device: Intel(R) UHD Graphics (CML GT2)
out[240..256] (input is 0..255):
   240  241  242  243  244  245  246  247  248  249  250  251  252  253  254  255
nonzero elements: 127 (expected 127)
```

## Files

| File                             | Role                                                                                                                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build.zig`                      | Builds two hosts: the translate-c one (`run`) and the vulkan-zig one (`run-vkzig`); compiles the kernel → SPIR-V; defines all `-D` options.                                   |
| `build.zig.zon`                  | Package manifest; declares the `vulkan` (vulkan-zig) dependency.                                                                                                                |
| `vk_filter.zig`                  | **The compute kernel.** Vulkan/shader model: storage buffers + push constants. This is what runs on the GPU.                                                                    |
| `vk_host.zig`                    | Host program using **raw translate-c bindings**: sets up Vulkan, uploads data, dispatches, reads back.                                                                          |
| `vk_host_vkzig.zig`              | Same host, ported to **Snektron/vulkan-zig** typed bindings (wrappers/proxies, typed enums/flags, errors). See [below](#variant-vulkan-zig-bindings).                           |
| `vk.zig`                         | **Vendored** vulkan-zig bindings (generated from `vk.xml`); regenerate with `tools/gen-vk.sh`.                                                                                  |
| `gpu.zig` / `example_filter.zig` | High-level compute API and a demo using it (`zig build run-gpu`).                                                                                                               |
| `vkimport.c`                     | One line (`#include <vulkan/vulkan.h>`) fed to `zig translate-c` to generate the bindings for `vk_host.zig`.                                                                    |
| `filter.zig`                     | **Reference only.** OpenCL-flavored kernel (pointer params, `.global` addrspace). Compiles to `spirv64-opencl`.                                                                 |
| `host.zig`                       | **Reference only.** OpenCL host harness (`clCreateProgramWithIL`). Works on runtimes that ingest SPIR-V IL — _not_ NVIDIA's OpenCL (see [below](#why-vulkan-and-not-opencl)). |

## Configuration knobs

Two timing domains — this is the important mental model:

| Knob             | Build flag        | Runtime flag   | Domain           | How it's plumbed                                                      |
| ---------------- | ----------------- | -------------- | ---------------- | --------------------------------------------------------------------- |
| Filter cutoff    | `-Dthreshold=`    | `--threshold=` | **runtime**      | push constant                                                         |
| Element count    | `-Dn=`            | `--n=`         | **runtime**      | push constant + buffer sizing + kernel bounds check                   |
| Workgroup size   | `-Dwgsize=`       | —            | **compile-time** | `b.addOptions` → `config` module imported by _both_ kernel and host |
| Validation layer | `-Dvalidate=true` | `--validate`   | **runtime**      | layer enumerated & enabled at instance creation                       |

`threshold` and `n` are just data, so they travel to the GPU as **push constants** and need no
rebuild. The **workgroup size is baked into the SPIR-V** (`OpExecutionMode LocalSize`), so it _must_
be known at kernel-compile time. One `-Dwgsize` feeds a generated `config` module that both the
kernel and host import, so they can never disagree on the group count:

```zig
// build.zig
const cfg = b.addOptions();
cfg.addOption(u32, "wg_size", wgsize);
const cfg_mod = cfg.createModule();
kernel.root_module.addImport("config", cfg_mod);  // kernel: callconv .x = cfg.wg_size
exe.root_module.addImport("config", cfg_mod);     // host: dispatch ceil(n / wg_size) groups
```

Because `n` needn't be a multiple of the workgroup size, the host dispatches `ceil(n / wg_size)`
groups and the kernel guards the overhang:

```zig
if (i >= pc.n) return;   // last group may run past the buffer
```

## How it works (build pipeline)

```
vk_filter.zig ──(zig build-obj -target spirv32-vulkan)──▶ vk_filter.spv
                                                              │
vulkan.h ──(zig translate-c)──▶ vk (Zig module)              │ @embedFile via addAnonymousImport
                                     │                        ▼
                                     └──────────────▶ vk_host.zig ──▶ zig-out/bin/vk_host
```

1. **Kernel → SPIR-V.** `b.addObject` with target `spirv32-vulkan` + `vulkan_v1_2` runs the
   self-hosted SPIR-V backend. `kernel.getEmittedBin()` yields a `LazyPath` to the `.spv`.
2. **Bindings.** `b.addTranslateC` turns `vulkan.h` into a Zig module (the replacement for the
   now-removed `@cImport`).
3. **Host.** The `.spv` is embedded into the executable via
   `addAnonymousImport("vk_filter.spv", …)` so `@embedFile("vk_filter.spv")` resolves to the
   compiled kernel — nothing loose to ship. Then `vkCreateShaderModule` hands those bytes straight
   to the driver.

## Kernel authoring reference (`std.spirv` + builtins)

What the Vulkan-flavored kernel uses (all in mainline `lib/std/spirv.zig` and the language):

- **Buffers** — `@SpirvType(.{ .runtime_array = T })` is an unsized SSBO array (
  `OpTypeRuntimeArray`), wrapped in an `extern struct`, bound via:
  ```zig
  const input = @extern(*addrspace(.storage_buffer) const Buf, .{
      .name = "input",
      .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } },
  });
  ```
- **Push constants** — small per-dispatch data, no descriptor/buffer:
  ```zig
  const PushConstants = extern struct { threshold: f32, n: u32 };
  const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });
  ```
- **Entry point** — `callconv` marks it and sets the local size:
  ```zig
  export fn main() callconv(.{ .spirv_kernel = .{ .x = 64, .y = 1, .z = 1 } }) void { … }
  // callconv(.kernel) is a shortcut defaulting to 1×1×1.
  ```
- **Builtins** — `std.spirv.global_invocation_id`, `local_invocation_id`, `workgroup_id`,
  `num_workgroups`, `workgroup_size` (all `@Vector(3, u32)`).
- **Sync within a workgroup** — `std.spirv.workgroupBarrier()`, or the lower-level
  `controlBarrier` / `memoryBarrier`.
- **Address spaces** — `.storage_buffer`, `.push_constant`, `.uniform`, `.workgroup` (shared),
  `.global`/ `.constant` (OpenCL model), `.input`/ `.output` (graphics stages).

### Calling functions from kernels

- **Plain functions are just normal Zig** — call them directly; they lower to `OpFunctionCall`.
  Helpers can take parameters, including `addrspace`-qualified pointers, and be shared by multiple
  kernels.
- **You cannot call one kernel from another.** A `callconv(.spirv_kernel)` function is an _entry
  point_; SPIR-V forbids calling it. The compiler rejects it:
  `error: unable to call function with calling convention 'spirv_kernel'`.
- **Composing kernels** (output of A feeds B) is done on the **host**: dispatch A, insert a
  `vkCmdPipelineBarrier` (`SHADER_WRITE → SHADER_READ`), dispatch B — not by calling A from B.

## Variant: vulkan-zig bindings

`vk_host_vkzig.zig` is the same demo with the host ported from raw translate-c'd headers to
[Snektron/vulkan-zig](https://github.com/Snektron/vulkan-zig), a generator that emits a typed Zig
binding from the Vulkan XML registry. Run it with `zig build run-vkzig`.

The bindings are **vendored** as a static file at `src/vulkan/vk.zig` and imported by **relative
path** — no build-time dependency, no named module:

```zig
const vk = @import("vk.zig"); // in src/vulkan/*.zig, next to vk.zig
```

Two deliberate choices here, both for language-server goto-definition / type-info:

- **Vendored, not generated at build time** — LSPs resolve into a static source file, but not
  reliably into a module whose root is produced by a build step in the cache.
- **Relative import, not a named build module** — zigscient resolves relative-path imports and
  named modules that map to _generated_ files (e.g. `@import("config")`), but _not_ a named module
  that maps to a _source-path_ file. A relative `@import("vk.zig")` sidesteps that.

### Regenerating the bindings

The bindings track the **Vulkan XML registry (`vk.xml`)**, which ships with the system Vulkan
headers/SDK — so yes, they're tied to your installed Vulkan version (header `280` /
`/usr/share/vulkan/registry/vk.xml` here). They also depend on the pinned **vulkan-zig** generator
commit. Regenerate after bumping either:

```sh
tools/gen-vk.sh                       # uses /usr/share/vulkan/registry/vk.xml
tools/gen-vk.sh path/to/other/vk.xml  # or point at a specific registry
```

The script (git-clones the pinned vulkan-zig commit and runs its generator) writes
`src/vulkan/vk.zig`; commit the result. To move to a newer registry/generator, edit
`VULKAN_ZIG_COMMIT` in `tools/gen-vk.sh` and rerun.

### What the port looks like vs. translate-c

| Concern           | translate-c (`vk_host.zig`)                   | vulkan-zig (`vk_host_vkzig.zig`)            |
| ----------------- | --------------------------------------------- | ------------------------------------------- |
| Constants         | `c.VK_SHADER_STAGE_COMPUTE_BIT` (int)         | typed flags: `.{ .compute = true }`         |
| Enums             | `c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`         | `.storage_buffer`                           |
| `sType`           | set every struct by hand                      | defaulted by the struct                     |
| Errors            | check `VkResult` ints manually                | `try device.createBuffer(...)` (Zig errors) |
| Calls             | free functions: `vkCreateBuffer(device, …)` | proxy methods: `device.createBuffer(…)`   |
| count+ptr pairs   | `count`, `pData` fields                       | Zig slices (`&writes`, `&cpci`)             |
| Function pointers | resolved by the linker                        | loaded into dispatch tables at runtime      |

The runtime model is dispatch tables + proxies:

```zig
const vkb = vk.BaseWrapper.load(vkGetInstanceProcAddr);          // global fns
const instance_handle = try vkb.createInstance(&ici, null);
var vki = vk.InstanceWrapper.load(instance_handle, vkb.dispatch.vkGetInstanceProcAddr.?);
const instance = vk.InstanceProxy.init(instance_handle, &vki);   // carries the handle
// ...createDevice, then DeviceWrapper.load + DeviceProxy.init the same way...
device.createBuffer(&.{ .size = bytes, .usage = .{ .storage_buffer = true }, .sharing_mode = .exclusive }, null);
```

Only one raw symbol is still needed to bootstrap — the loader:
`extern fn vkGetInstanceProcAddr(instance: vk.Instance, p_name: [*:0]const u8) vk.PfnVoidFunction;`

### Gotchas hit during the port

- **The Vulkan loader needs libc.** The module must set `.link_libc = true`. Without it the program
  links `libvulkan` but the loader segfaults on the _first_ call (even a raw
  `vkEnumerateInstanceVersion`) because libc isn't initialized. This bites the vulkan-zig host
  specifically because, unlike the translate-c host, it has no other reason to pull in libc.
- **Generated field names have no `_bit` suffix.** The registry-driven generator emits `.compute`,
  `.storage_buffer`, `.host_visible` — not `.compute_bit` etc. (Some READMEs show the `_bit` form;
  the generated code is the source of truth — generate `vk.zig` and grep it.)
- **Explicit cleanup or validation complains.** This version adds `defer device.destroyX(...)` for
  every object (buffers, memory, layouts, pipeline, pools). Because it also destroys the
  device/instance, skipping child cleanup triggers `VUID-vkDestroyDevice-device-05137`. The
  translate-c host sidesteps this only by never destroying anything and relying on process exit.
- **SPIR-V needs u32 alignment.** `const spv align(@alignOf(u32)) = @embedFile("vk_filter.spv").*;`
  forces 4-byte alignment for `p_code`.

Both hosts produce identical results and surface the same benign SPIR-V source-language validation
note (see [below](#validation-findings)).

## Why Vulkan and not OpenCL?

Both flavors compile to valid SPIR-V. But **NVIDIA's OpenCL does not ingest SPIR-V** (
`CL_DEVICE_IL_VERSION` is empty, no `cl_khr_il_program`), so `clCreateProgramWithIL` fails with
`CL_INVALID_OPERATION` (-59) on NVIDIA hardware. **Vulkan drivers consume SPIR-V natively**, so the
Vulkan path is the portable way to actually execute here. The OpenCL files (`filter.zig`, `host.zig`
) are kept for reference and will run on a SPIR-V-capable OpenCL runtime (PoCL, Intel CPU runtime,
etc.).

## Validation findings

With `-Dvalidate=true`, the Khronos layer flags two things in the Zig-emitted SPIR-V (the kernel
still runs and results stay correct on Mesa/Intel drivers):

- `VUID-…-08737` _"Invalid source language operand: 12"_ — Zig tags the module with SPIR-V
  source-language id `12` (= Zig); the bundled `spirv-val` is too old to recognize it. Metadata
  only.
- `VUID-…-08739` _"Unhandled OpCapability"_ — the module declares a capability the validator
  doesn't handle for the Vulkan environment.

Both are consistent with the backend being experimental — cosmetic here, but exactly what you'd
want the layer catching on a real project.

## API drift in this dev build

Things that differ from older tutorials/docs (all encountered building this):

- `@cImport`** is removed.** Use `zig translate-c` (CLI) or `b.addTranslateC` (build) instead.
- **Build API is module-first.** `addExecutable`/ `addObject` take `root_module: *Module` built with
  `b.createModule`; `linkSystemLibrary`, `addImport`, `addAnonymousImport` live on `root_module`,
  not the `Compile` step.
- `b.args`** (run-arg forwarding after `--`) is gone.** This demo forwards options as explicit run
  args via `run.addArgs(...)`.
- `std.process.argsAlloc`** is gone.** argv now arrives through `std.process.Init`: `main` takes it
  as its first parameter and you iterate `init.minimal.args.iterate()`. `Init` also carries `gpa`,
  `arena`, `io`, `environ_map` — the blessed way to get allocators/IO into `main`.
- `std.io.getStdOut()`** reorganized.** This demo uses `std.debug.print`.

## CUDA → Vulkan / Zig-SPIR-V cheat-sheet

### Mental model

CUDA gives you a single-source, tightly integrated runtime. Vulkan compute is explicit and verbose:
you build the pipeline, descriptors, and command buffers by hand. The _compute_ concepts map almost
1:1; the _host_ side is where the ceremony lives.

### Kernel / device code

| CUDA                                       | Zig + SPIR-V (Vulkan)                                                            | Notes                                                           |
| ------------------------------------------ | -------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `__global__ void k(...)`                   | `export fn k() callconv(.{ .spirv_kernel = .{ .x=…, .y=…, .z=… } }) void`  | The launch bounds (block dim) are baked in here, not at launch. |
| `__device__` helper                        | plain `fn` (default callconv)                                                    | Just normal Zig; lowers to `OpFunctionCall`.                    |
| calling a `__global__` from another kernel | **not allowed**                                                                  | Entry points can't be called; factor shared code into a `fn`.   |
| `blockIdx`                                 | `std.spirv.workgroup_id`                                                         | `@Vector(3, u32)`                                               |
| `blockDim`                                 | `std.spirv.workgroup_size` (or the compile-time `wg_size`)                       | fixed at compile time in SPIR-V                                 |
| `threadIdx`                                | `std.spirv.local_invocation_id`                                                  |                                                                 |
| `blockIdx*blockDim+threadIdx`              | `std.spirv.global_invocation_id`                                                 | the global lane id — the usual index                          |
| `gridDim`                                  | `std.spirv.num_workgroups`                                                       |                                                                 |
| `__shared__ T buf[N]`                      | `var buf: [N]T addrspace(.workgroup) = …`                                      | workgroup-shared memory                                         |
| `__syncthreads()`                          | `std.spirv.workgroupBarrier()`                                                   | also `controlBarrier`/ `memoryBarrier`                          |
| `__constant__`                             | `addrspace(.uniform)` / `.push_constant`                                         | small read-only data                                            |
| kernel pointer params `float* a`           | `@extern(*addrspace(.storage_buffer) Buf, .{…descriptor…})`                  | Vulkan: bound via descriptor sets, not passed as args           |
| launch config `<<<grid, block>>>`          | `block` → kernel `callconv` local size; `grid` → host `vkCmdDispatch(x,y,z)` | grid = number of workgroups                                     |
| `atomicAdd`, etc.                          | SPIR-V atomics via inline asm / builtins                                         | see `lib/std/spirv.zig`                                         |
| `printf` in kernel                         | —                                                                              | no device printf; debug via readback or validation layers       |

> **OpenCL model instead of Vulkan?** If you target `spirv64-opencl`, kernels take pointer params
> directly (`[*]addrspace(.global) f32`) — much closer to CUDA's `float*` signature — and there
> are no descriptor sets. See `filter.zig`.

### Host / runtime code

| CUDA                                  | Vulkan (this demo)                                                                  | Notes                                   |
| ------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------- |
| (implicit context)                    | `vkCreateInstance` → pick `VkPhysicalDevice` → `vkCreateDevice` + compute queue | fully manual                            |
| `cudaMalloc`                          | `vkCreateBuffer` + `vkAllocateMemory` + `vkBindBufferMemory`                        | pick a memory type yourself             |
| `cudaMemcpy` H→D / D→H            | `vkMapMemory` + write/read (host-visible+coherent memory)                           | or staging buffers for device-local     |
| pass args to `<<<>>>`                 | descriptor set (buffers) + `vkCmdPushConstants` (scalars)                           | two different mechanisms                |
| `k<<<grid, block>>>(...)`             | bind pipeline + descriptors + push constants, then `vkCmdDispatch(grid…)`         | recorded into a command buffer          |
| kernel launch (async)                 | `vkQueueSubmit`                                                                     |                                         |
| `cudaDeviceSynchronize`               | `vkQueueWaitIdle` (or fences)                                                       |                                         |
| CUDA stream ordering                  | command-buffer order + `vkCmdPipelineBarrier`                                       | barrier between dependent dispatches    |
| `nvcc` compiles kernel                | `zig build-obj -target spirv32-vulkan`                                              | emits `.spv`; embedded via `@embedFile` |
| `cuModuleLoad` (PTX/cubin)            | `vkCreateShaderModule(spv)` + `vkCreateComputePipelines`                            |                                         |
| `cuda-memcheck` / `compute-sanitizer` | Vulkan validation layers (`-Dvalidate=true`)                                        |                                         |
| `-arch=sm_XX`                         | `-mcpu vulkan_v1_2` (target CPU features)                                           | selects the SPIR-V/Vulkan feature level |

### Concept glossary

| CUDA term            | Vulkan/SPIR-V term                  |
| -------------------- | ----------------------------------- |
| thread               | invocation                          |
| warp                 | subgroup                            |
| block / thread block | workgroup / local workgroup         |
| grid                 | dispatch (set of workgroups)        |
| shared memory        | workgroup storage class             |
| global memory        | storage buffer (SSBO)               |
| constant memory      | uniform buffer / push constants     |
| PTX / cubin          | SPIR-V                              |
| streams              | queues + command buffers + barriers |
