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

pub fn build_aot(entry: &str, out: &Path, backend: &crate::aot::Backend, extra_flags: &[String])
    -> Result<Option<crate::aot::Tier>, String>
{
    let source = std::fs::read_to_string(entry)
        .map_err(|e| format!("cannot read {}: {}", entry, e))?;
    let mut program = crate::parser::parse_program(&source)?;
    crate::interp::fold_program(&mut program);
    let (code, tier) = match crate::aot::emit(&program, backend) {
        Some(ct) => ct,
        None => return Ok(None),
    };
    if let Some(dir) = out.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
    }
    let (ext, cc) = match backend {
        crate::aot::Backend::C => ("c", "cc"),
        crate::aot::Backend::Llvm => ("ll", "clang"),
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
    cmd.arg("-O3").arg("-flto");
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
    // the oracle gate: ship only if byte-identical to `nova run`
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let expect = Command::new(&exe).arg("run").arg(entry).output()
        .map_err(|e| e.to_string())?;
    let got = Command::new(out).output().map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&rt);
    if expect.stdout == got.stdout && expect.status.code() == got.status.code() {
        Ok(Some(tier))
    } else {
        let _ = std::fs::remove_file(out);
        Ok(None)
    }
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
