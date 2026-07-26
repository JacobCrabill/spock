//! Patch a Zig-emitted SPIR-V object into a module the Vulkan environment accepts.
//!
//! Zig's SPIR-V backend emits an *object* module, which trips two Vulkan rules:
//!
//!   1. `OpCapability Linkage` — an OpenCL/library-linking capability Vulkan
//!      forbids outright (VUID-VkShaderModuleCreateInfo-pCode-08739). Zig declares
//!      it unconditionally but emits no matching `LinkageAttributes` decoration,
//!      so it is vestigial and safe to drop.
//!   2. `OpSource` with source language 12 (Zig) — not a value in the SPIR-V
//!      headers spirv-val was built against, so validation rejects the module
//!      (VUID-VkShaderModuleCreateInfo-pCode-08737). Rewritten to 0 (Unknown),
//!      which is what the validator explicitly asks for.
//!
//! Both are upstream Zig issues; delete this step once the backend emits a
//! Vulkan-clean module.
//!
//!     spv_patch <in.spv> <out.spv>

const std = @import("std");
const Io = std.Io;
const fatal = std.process.fatal;

const magic = 0x07230203;
const header_words = 5;

const op_source = 3;
const op_capability = 17;
const op_decorate = 71;

const cap_linkage = 5;
const dec_linkage_attributes = 41;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 3) fatal("usage: spv_patch <in.spv> <out.spv>", .{});
    const in_path = args[1];
    const out_path = args[2];

    const in_bytes = try Io.Dir.cwd().readFileAllocOptions(
        io,
        in_path,
        arena,
        .limited(1 << 30),
        .of(u32),
        null,
    );
    if (in_bytes.len % 4 != 0) fatal("{s}: not a SPIR-V module (size not a multiple of 4)", .{in_path});

    const words = std.mem.bytesAsSlice(u32, in_bytes);
    if (words.len < header_words or words[0] != magic)
        fatal("{s}: not a little-endian SPIR-V module", .{in_path});

    var out: std.ArrayList(u32) = try .initCapacity(arena, words.len);
    try out.appendSlice(arena, words[0..header_words]);

    var i: usize = header_words;
    while (i < words.len) {
        const len = words[i] >> 16;
        const op = words[i] & 0xffff;
        if (len == 0 or i + len > words.len) fatal("{s}: malformed instruction at word {d}", .{ in_path, i });
        const inst = words[i .. i + len];
        i += len;

        // A real linkage decoration means the capability is load-bearing and this
        // genuinely is an unlinked object. Refuse rather than silently corrupt it.
        if (op == op_decorate and len >= 3 and inst[2] == dec_linkage_attributes)
            fatal("{s}: module has LinkageAttributes decorations; cannot strip Linkage", .{in_path});

        if (op == op_capability and len == 2 and inst[1] == cap_linkage)
            continue; // drop the instruction entirely

        const start = out.items.len;
        try out.appendSlice(arena, inst);

        if (op == op_source and len >= 2)
            out.items[start + 1] = 0; // SourceLanguage -> Unknown
    }

    try Io.Dir.cwd().writeFile(io, .{
        .sub_path = out_path,
        .data = std.mem.sliceAsBytes(out.items),
    });
}
