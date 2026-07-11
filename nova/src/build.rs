// Phase 6.1: `nova build` — standalone executables.
//
// The built binary is a copy of the running `nova` executable with the program
// source appended, followed by an 8-byte little-endian length and a magic tag:
//
//     [nova runtime][source bytes][len: u64 LE][b"NOVA_EMBED_v1"]
//
// On startup `main()` calls `embedded_source()`, which reads only the trailer
// (a seek, not a full read); when present, the binary runs the embedded program
// directly on the tiered VM/JIT (interpreter fallback for non-VM `main`s).
// Because the runtime IS the shipped binary, output is byte-identical to
// `nova run` by construction — for every Nova program, no exceptions.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::Command;

// AOT build: emit C or LLVM IR, compile with the system toolchain, then verify
// the binary against `nova run` byte-for-byte. Returns Ok(true) if the AOT
// binary shipped, Ok(false) if the program isn't AOT-able or diverged (caller
// falls back to the embed build — never wrong output, only a different tier).
// the AOT runtime, embedded so built binaries need no Nova installation
const NOVA_RT: &str = include_str!("../runtime/nova_rt.c");
// runtime primitives for the native object backend (arena + fmod/fpow + float
// printer), linked alongside the Cranelift-emitted object — not program logic.
#[cfg(feature = "jit")]
const NOVA_NATIVE_RT: &str = include_str!("../runtime/nova_native_rt.c");

pub fn build_aot(entry: &str, out: &Path, backend: &crate::aot::Backend, extra_flags: &[String])
    -> Result<Option<crate::aot::Tier>, String>
{
    let source = std::fs::read_to_string(entry)
        .map_err(|e| format!("cannot read {}: {}", entry, e))?;
    let mut program = crate::parser::parse_program(&source)?;
    crate::interp::fold_program(&mut program);
    // Native object backend: Cranelift IR -> relocatable .o -> link. Separate
    // path (no C/LLVM text). Host, or a cross target (aarch64/riscv64).
    let native_target = match backend {
        crate::aot::Backend::Native => Some(NativeArch::Host),
        crate::aot::Backend::NativeArm64 => Some(NativeArch::Aarch64),
        crate::aot::Backend::NativeRiscv64 => Some(NativeArch::Riscv64),
        _ => None,
    };
    if let Some(arch) = native_target {
        return build_native(entry, out, &program, arch);
    }
    let (code, tier) = match crate::aot::emit(&program, backend) {
        Some(ct) => ct,
        None => return Ok(None),
    };
    if let Some(dir) = out.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
    }
    // WASM takes a separate compile+verify path (clang freestanding wasm32, run
    // and byte-diff via node). Only the typed tier reaches here for wasm.
    if matches!(backend, crate::aot::Backend::Wasm) {
        return build_wasm(entry, out, &code, tier);
    }
    let arm = matches!(backend, crate::aot::Backend::Arm);
    let arm32 = matches!(backend, crate::aot::Backend::Arm32);
    let (ext, cc) = match backend {
        crate::aot::Backend::C => ("c", "cc"),
        crate::aot::Backend::Llvm => ("ll", "clang"),
        // ARM64: cross-compile the portable C with the aarch64 gcc, run under qemu.
        crate::aot::Backend::Arm => ("c", "aarch64-linux-gnu-gcc"),
        // ARMv7 (32-bit hard-float): older / weaker phones.
        crate::aot::Backend::Arm32 => ("c", "arm-linux-gnueabihf-gcc"),
        crate::aot::Backend::Wasm => unreachable!("wasm handled by build_wasm above"),
        crate::aot::Backend::Native | crate::aot::Backend::NativeArm64
        | crate::aot::Backend::NativeRiscv64 => unreachable!("native handled by build_native above"),
    };
    let tmp = out.with_extension(ext);
    std::fs::write(&tmp, &code).map_err(|e| e.to_string())?;
    // the boxed tier compiles against runtime/nova_rt.c. The C backend #includes
    // it as one translation unit (so -O3 -flto inlines refcount/value ops). The
    // LLVM backend can't #include C into textual IR, so the runtime is compiled
    // as a second translation unit; `-Dstatic=` gives its otherwise file-local
    // functions external linkage so the IR's `declare @nv_*` calls resolve.
    let rt = out.parent().unwrap_or(Path::new(".")).join("nova_rt.c");
    let boxed = matches!(tier, crate::aot::Tier::Boxed);
    let llvm_boxed = boxed && matches!(backend, crate::aot::Backend::Llvm);
    if boxed {
        std::fs::write(&rt, NOVA_RT).map_err(|e| e.to_string())?;
    }
    let mut cmd = Command::new(cc);
    // ARM cross gcc: build a static binary (so qemu needs no sysroot libs) and
    // skip -flto (the cross toolchain's LTO plugin isn't always present). ARMv7
    // uses -marm (A32) for stable output.
    if arm { cmd.arg("-O2").arg("-static"); }
    else if arm32 { cmd.arg("-O2").arg("-static").arg("-marm"); }
    // Native host build: tune for this machine. `-march=native` lets the C
    // compiler use the host's full instruction set (wider vectors, etc.) and
    // `-fno-math-errno` frees float ops from errno side effects. Portable
    // targets (arm/arm32/wasm) keep generic codegen. Kept off the oracle path's
    // correctness — the AOT build is still byte-verified against `nova run`.
    else { cmd.arg("-O3").arg("-flto").arg("-march=native").arg("-fno-math-errno"); }
    if matches!(backend, crate::aot::Backend::Llvm) { cmd.arg("-Wno-override-module"); }
    if llvm_boxed { cmd.arg("-Dstatic="); }
    for f in extra_flags { cmd.arg(f); }
    cmd.arg("-o").arg(out).arg(&tmp);
    if llvm_boxed { cmd.arg(&rt); } // compile the runtime as a second TU
    let status = cmd.status()
        .map_err(|e| format!("cannot run {}: {}", cc, e))?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Ok(None);
    }
    // the oracle gate: ship only if byte-identical to `nova run`. ARM binaries
    // run under qemu-aarch64 (the emulator present in this environment).
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let expect = Command::new(&exe).arg("run").arg(entry).output()
        .map_err(|e| e.to_string())?;
    let got = if arm {
        Command::new("qemu-aarch64").arg(out).output()
    } else if arm32 {
        Command::new("qemu-arm").arg(out).output()
    } else {
        Command::new(out).output()
    }.map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&rt);
    if expect.stdout == got.stdout && expect.status.code() == got.status.code() {
        Ok(Some(tier))
    } else {
        let _ = std::fs::remove_file(out);
        Ok(None)
    }
}

// Which architecture the native object is built for, and the toolchain to link
// and run it. Host links with `cc` and runs the binary directly; cross targets
// link a static binary with a cross-gcc and run it under qemu (the same qemu the
// C-backend ARM path already uses).
#[cfg(feature = "jit")]
#[derive(Clone, Copy, PartialEq)]
enum NativeArch { Host, Aarch64, Riscv64 }

#[cfg(feature = "jit")]
impl NativeArch {
    fn target(self) -> crate::jit::NativeTarget {
        match self {
            NativeArch::Host => crate::jit::NativeTarget::Host,
            NativeArch::Aarch64 => crate::jit::NativeTarget::Aarch64,
            NativeArch::Riscv64 => crate::jit::NativeTarget::Riscv64,
        }
    }
    // (C compiler / linker, qemu runner or None for host). Cross targets link
    // static so qemu needs no target sysroot at run time.
    fn tools(self) -> (&'static str, Option<&'static str>) {
        match self {
            NativeArch::Host => ("cc", None),
            NativeArch::Aarch64 => ("aarch64-linux-gnu-gcc", Some("qemu-aarch64")),
            NativeArch::Riscv64 => ("riscv64-linux-gnu-gcc", Some("qemu-riscv64")),
        }
    }
}

// The native object-code AOT path: lower the program's numeric core to a real
// relocatable object with Cranelift (`jit::compile_object` — the SAME codegen the
// JIT uses, no C for program logic), write `build/<name>.o`, then link it with
// the target's toolchain (used only as the libc linker driver). Ship the binary
// only if its output is byte-identical to `nova run` (the oracle gate, cross
// binaries run under qemu). Returns Ok(None) when the program isn't native-
// eligible, the cross toolchain is absent, or output diverges — the caller falls
// back to the C/embed AOT, so output is never wrong.
#[cfg(feature = "jit")]
fn build_native(entry: &str, out: &Path, program: &crate::ast::Program, arch: NativeArch)
    -> Result<Option<crate::aot::Tier>, String>
{
    let (obj, needs_runtime) = match crate::jit::compile_object(program, arch.target()) {
        Some(v) => v,
        None => return Ok(None), // not native-able -> fall back
    };
    let (cc, runner) = arch.tools();
    // a cross target we have no linker for -> fall back (host C/embed AOT).
    if arch != NativeArch::Host && which(cc).is_none() { return Ok(None); }
    if let Some(dir) = out.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
    }
    let objp = out.with_extension("o");
    std::fs::write(&objp, &obj).map_err(|e| e.to_string())?;
    // For float/array/numeric programs, compile the runtime primitives object
    // (arena arrays + fmod/fpow + the fmt_f64 float printer) with the target's cc
    // and link it in. This is runtime, NOT program logic — exactly like the JIT
    // linking Rust helpers.
    let rtp = out.parent().unwrap_or(Path::new(".")).join("nova_native_rt.o");
    let rtc = out.parent().unwrap_or(Path::new(".")).join("nova_native_rt.c");
    let cleanup_rt = || {
        let _ = std::fs::remove_file(&rtp);
        let _ = std::fs::remove_file(&rtc);
    };
    if needs_runtime {
        std::fs::write(&rtc, NOVA_NATIVE_RT).map_err(|e| e.to_string())?;
        let st = Command::new(cc).arg("-O2").arg("-c").arg(&rtc).arg("-o").arg(&rtp)
            .status().map_err(|e| format!("cannot run {}: {}", cc, e))?;
        if !st.success() {
            let _ = std::fs::remove_file(&objp);
            cleanup_rt();
            return Ok(None);
        }
    }
    // link (libc + crt; cc compiles no program logic here). Cross targets link
    // static + non-PIE so qemu needs no sysroot; `-lm` for the runtime's fmod/pow.
    let mut cmd = Command::new(cc);
    cmd.arg("-O2");
    if arch != NativeArch::Host { cmd.arg("-static").arg("-no-pie"); }
    cmd.arg("-o").arg(out).arg(&objp);
    if needs_runtime { cmd.arg(&rtp).arg("-lm"); }
    let status = cmd.status().map_err(|e| format!("cannot run {}: {}", cc, e))?;
    if !status.success() {
        let _ = std::fs::remove_file(&objp);
        cleanup_rt();
        return Ok(None);
    }
    // oracle gate: keep the native binary only on a byte-identical match. Cross
    // binaries run under qemu (present in this environment).
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let expect = Command::new(&exe).arg("run").arg(entry).output()
        .map_err(|e| e.to_string())?;
    let got = match runner {
        Some(q) => Command::new(q).arg(out).output(),
        None => Command::new(out).output(),
    }.map_err(|e| e.to_string())?;
    cleanup_rt();
    if expect.stdout == got.stdout && expect.status.code() == got.status.code() {
        // keep the .o alongside the binary as proof of the object path
        Ok(Some(crate::aot::Tier::Typed))
    } else {
        let _ = std::fs::remove_file(out);
        let _ = std::fs::remove_file(&objp);
        Ok(None)
    }
}

// Without the jit feature there is no Cranelift, so the native object backend is
// unavailable — fall back to the embed build honestly.
#[cfg(not(feature = "jit"))]
#[derive(Clone, Copy)]
enum NativeArch { Host, Aarch64, Riscv64 }

#[cfg(not(feature = "jit"))]
fn build_native(_entry: &str, _out: &Path, _program: &crate::ast::Program, _arch: NativeArch)
    -> Result<Option<crate::aot::Tier>, String>
{
    Ok(None)
}

// Node harness: instantiate the wasm32-wasi module, run it under node's WASI
// (stdout captured), so the byte-diff gate can compare against `nova run`.
const WASM_HARNESS: &str = r#"import { readFileSync } from 'node:fs';
import { WASI } from 'node:wasi';
const wasi = new WASI({ version: 'preview1', args: [], env: {} });
const wasm = await WebAssembly.compile(readFileSync(process.argv[2]));
const inst = await WebAssembly.instantiate(wasm, wasi.getImportObject());
wasi.start(inst);
"#;

// a wasi sysroot: a dir whose lib/wasm32-wasi has libc.a (apt's wasi-libc puts
// it at /usr/lib/wasm32-wasi, i.e. sysroot = /usr)
fn wasi_sysroot() -> Option<&'static str> {
    for root in ["/usr", "/opt/wasi-sysroot", "/usr/share/wasi-sysroot"] {
        if Path::new(root).join("lib/wasm32-wasi/libc.a").is_file() { return Some(root); }
    }
    None
}

// Compile the portable AOT C (typed OR boxed, incl. nova_rt.c) to wasm32-wasi
// with clang + a wasi-libc sysroot, run it under node's WASI, and ship the
// `.wasm` only if its output is byte-identical to `nova run`. Requires `clang`
// (wasm32 target), a wasi sysroot, and `node`; absent any, returns Ok(None) (no
// wasm, honest fallback — never wrong output).
fn build_wasm(entry: &str, out: &Path, code: &str, tier: crate::aot::Tier)
    -> Result<Option<crate::aot::Tier>, String>
{
    let node = which("node");
    let clang = which("clang");
    let sysroot = wasi_sysroot();
    let (Some(node), Some(clang), Some(sysroot)) = (node, clang, sysroot) else { return Ok(None) };
    let cdir = out.parent().unwrap_or(Path::new("."));
    let tmp_c = out.with_extension("wasm.c");
    let wasm = out.with_extension("wasm");
    let harness = cdir.join(".nova_wasm_harness.mjs");
    let rt = cdir.join("nova_rt.c");
    let boxed = matches!(tier, crate::aot::Tier::Boxed);
    std::fs::write(&tmp_c, code).map_err(|e| e.to_string())?;
    std::fs::write(&harness, WASM_HARNESS).map_err(|e| e.to_string())?;
    if boxed { std::fs::write(&rt, NOVA_RT).map_err(|e| e.to_string())?; }
    let cleanup = |extra: bool| {
        let _ = std::fs::remove_file(&tmp_c);
        let _ = std::fs::remove_file(&harness);
        let _ = std::fs::remove_file(&rt);
        if extra { let _ = std::fs::remove_file(&wasm); }
    };
    let status = Command::new(&clang)
        .arg("--target=wasm32-wasi").arg(format!("--sysroot={}", sysroot))
        .arg("-O2").arg("-o").arg(&wasm).arg(&tmp_c)
        .status().map_err(|e| format!("cannot run clang: {}", e))?;
    if !status.success() { cleanup(true); return Ok(None); }
    // oracle gate: node-run wasm output must equal `nova run` byte-for-byte
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let expect = Command::new(&exe).arg("run").arg(entry).output().map_err(|e| e.to_string())?;
    let got = Command::new(&node).arg("--no-warnings").arg(&harness).arg(&wasm).output()
        .map_err(|e| format!("cannot run node: {}", e))?;
    cleanup(false);
    if expect.stdout == got.stdout && got.status.success() {
        Ok(Some(tier))
    } else {
        let _ = std::fs::remove_file(&wasm);
        Ok(None)
    }
}

// first matching executable on PATH, if any
fn which(bin: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let p = dir.join(bin);
        if p.is_file() { return Some(p); }
    }
    None
}

const MAGIC: &[u8; 13] = b"NOVA_EMBED_v1";
const TRAILER: u64 = 13 + 8;

pub fn embedded_source() -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    let mut f = std::fs::File::open(exe).ok()?;
    let size = f.seek(SeekFrom::End(0)).ok()?;
    if size < TRAILER { return None; }
    f.seek(SeekFrom::End(-(TRAILER as i64))).ok()?;
    let mut tail = [0u8; TRAILER as usize];
    f.read_exact(&mut tail).ok()?;
    if &tail[8..] != MAGIC { return None; }
    let len = u64::from_le_bytes(tail[..8].try_into().unwrap());
    if len > size - TRAILER { return None; }
    f.seek(SeekFrom::End(-((TRAILER + len) as i64))).ok()?;
    let mut src = vec![0u8; len as usize];
    f.read_exact(&mut src).ok()?;
    String::from_utf8(src).ok()
}

// Inline `use "file.nova";` imports textually (recursive, deduplicated) so a
// built binary is self-contained regardless of where it runs from.
fn flatten_imports(path: &Path, visited: &mut std::collections::HashSet<std::path::PathBuf>)
    -> Result<String, String>
{
    let canon = std::fs::canonicalize(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    if !visited.insert(canon.clone()) { return Ok(String::new()); }
    let source = std::fs::read_to_string(&canon)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    let base = canon.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let mut out = String::new();
    for line in source.lines() {
        let t = line.trim();
        let imported = t.strip_prefix("use \"").and_then(|r| {
            r.split('"').next().filter(|p| p.ends_with(".nova")).map(|p| p.to_string())
        });
        match imported {
            Some(rel) => {
                let full = crate::resolve_import(&base, &rel);
                out.push_str(&flatten_imports(&full, visited)?);
                out.push('\n');
            }
            None => { out.push_str(line); out.push('\n'); }
        }
    }
    Ok(out)
}

pub fn build(entry: &str, out: &Path) -> Result<(), String> {
    let mut visited = std::collections::HashSet::new();
    let source = flatten_imports(Path::new(entry), &mut visited)?;
    // must still parse as a self-contained program
    let _program = crate::parser::parse_program(&source)
        .map_err(|e| format!("in {} (after inlining imports):\n{}", entry, e))?;
    let exe = std::env::current_exe()
        .map_err(|e| format!("cannot locate the nova runtime: {}", e))?;
    let runtime = std::fs::read(&exe)
        .map_err(|e| format!("cannot read the nova runtime: {}", e))?;

    if let Some(dir) = out.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("cannot create {}: {}", dir.display(), e))?;
        }
    }
    let mut f = std::fs::File::create(out)
        .map_err(|e| format!("cannot write {}: {}", out.display(), e))?;
    f.write_all(&runtime).map_err(|e| e.to_string())?;
    f.write_all(source.as_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(source.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(MAGIC).map_err(|e| e.to_string())?;
    drop(f);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(out, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("cannot chmod {}: {}", out.display(), e))?;
    }
    Ok(())
}
