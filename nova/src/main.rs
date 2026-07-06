mod ast;
mod parser;
mod interp;
mod types;
mod fmt;
mod bytecode;
mod jit;
mod config;
mod build;
mod aot;
mod diag;
mod obfuscate;
mod lsp;
mod registry;
mod tokens;
mod astdump;

use std::io::{self, Write, BufRead};
use std::process::exit;

const VERSION: &str = "3.28.0";

fn main() {
    // a binary produced by `nova build` carries its program in a trailer:
    // run it directly (this executable IS the runtime)
    if let Some(src) = build::embedded_source() {
        run_embedded(&src);
        return;
    }

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        repl();
        return;
    }

    match args[1].as_str() {
        "help" | "--help" | "-h" => { print!("{}", usage()); }
        "build" => {
            let (entry, hgx) = resolve_input(&args);
            let name = hgx.as_ref().map(|c| c.name.clone()).unwrap_or_else(|| {
                std::path::Path::new(&entry).file_stem()
                    .and_then(|s| s.to_str()).unwrap_or("app").to_string()
            });
            let out = std::path::PathBuf::from("build").join(&name);
            // --aot / --aot=c / --aot=llvm: pure native binary via cc/clang,
            // verified byte-identical vs `nova run`; falls back to embed
            let aot_backend = args.iter().skip(2).find_map(|a| match a.as_str() {
                "--aot" | "--aot=c" => Some(aot::Backend::C),
                "--aot=llvm" => Some(aot::Backend::Llvm),
                "--aot=wasm" => Some(aot::Backend::Wasm),
                "--aot=arm" | "--aot=arm64" | "--aot=aarch64" => Some(aot::Backend::Arm),
                "--aot=arm32" | "--aot=armv7" | "--aot=armhf" => Some(aot::Backend::Arm32),
                _ => None,
            });
            if let Some(bk) = aot_backend {
                let extra: Vec<String> = args.iter().skip(2)
                    .find_map(|a| a.strip_prefix("--aot-flags="))
                    .map(|s| s.split_whitespace().map(|w| w.to_string()).collect())
                    .unwrap_or_default();
                match build::build_aot(&entry, &out, &bk, &extra) {
                    Ok(Some(tier)) => {
                        let (which, kind) = match bk {
                            aot::Backend::C => ("c", "native"),
                            aot::Backend::Llvm => ("llvm", "native"),
                            aot::Backend::Wasm => ("wasm", "wasm32"),
                            aot::Backend::Arm => ("arm", "aarch64"),
                            aot::Backend::Arm32 => ("arm32", "armv7"),
                        };
                        let art = if matches!(bk, aot::Backend::Wasm) {
                            out.with_extension("wasm").display().to_string()
                        } else { out.display().to_string() };
                        println!("built {} (aot-{}, {} tier, {})", art, which, tier.name(), kind);
                        return;
                    }
                    Ok(None) if matches!(bk, aot::Backend::Wasm) => {
                        eprintln!("note: program not WASM-able (typed/boxed tier only, not embed; needs clang wasm32 + a wasi-sysroot + node, verified vs `nova run`); no .wasm emitted");
                        exit(1);
                    }
                    Ok(None) if matches!(bk, aot::Backend::Arm) => {
                        eprintln!("note: program not ARM-AOT-able or diverged in verify (needs aarch64-linux-gnu-gcc + qemu-aarch64); no arm binary emitted");
                        exit(1);
                    }
                    Ok(None) if matches!(bk, aot::Backend::Arm32) => {
                        eprintln!("note: program not ARMv7-AOT-able or diverged in verify (needs arm-linux-gnueabihf-gcc + qemu-arm); no arm32 binary emitted");
                        exit(1);
                    }
                    Ok(None) => eprintln!(
                        "note: program not fully AOT-able (or diverged in verify); using the embedded runtime build"),
                    Err(e) => { eprintln!("error: {}", e); exit(1); }
                }
            }
            match build::build(&entry, &out) {
                Ok(()) => println!("built {}", out.display()),
                Err(e) => { eprintln!("error: {}", e); exit(1); }
            }
        }
        "run" => {
            let path = require_path(&args);
            let src = std::fs::read_to_string(&path).unwrap_or_default();
            let mut program = match load_program(&path) {
                Ok(p) => p,
                Err(e) => { eprintln!("{}", e); exit(1); }
            };
            interp::fold_program(&mut program); // constant-fold before execution
            let interp = match interp::Interp::new(&program) {
                Ok(i) => i,
                Err(e) => { eprintln!("{}", diag::render("error", &path, &src, &e)); exit(1); }
            };
            interp.set_args(program_args(&args, &path));
            match interp.run() {
                Ok(_) => {}
                Err(e) => { eprintln!("{}", diag::render("runtime error", &path, &src, &e)); exit(1); }
            }
        }
        "check" => {
            let path = require_path(&args);
            let src = std::fs::read_to_string(&path).unwrap_or_default();
            match load_program(&path) {
                Ok(p) => {
                    let (errors, warnings) = types::Checker::new(&p).check(&p);
                    for w in &warnings {
                        eprintln!("{}", diag::render("warning", &path, &src, w));
                    }
                    if errors.is_empty() {
                        println!("OK: parsed {} item(s), no type errors", p.items.len());
                    } else {
                        eprintln!("found {} type error(s):\n", errors.len());
                        for e in &errors {
                            eprintln!("{}\n", diag::render("error", &path, &src, e));
                        }
                        exit(1);
                    }
                }
                Err(e) => { eprintln!("{}", e); exit(1); }
            }
        }
        "vm" => {
            let (path, hgx) = resolve_input(&args);
            let src = std::fs::read_to_string(&path).unwrap_or_default();
            let optimize = !flag_present(&args, "--no-opt");
            let dump = flag_present(&args, "--dump");
            let mut program = match load_program(&path) {
                Ok(p) => p,
                Err(e) => { eprintln!("{}", e); exit(1); }
            };
            interp::fold_program(&mut program);
            let compiled = match bytecode::compile_program_opt(&program, optimize) {
                Ok(c) => c,
                Err(e) => { eprintln!("{} (run it with `nova run` instead)", e); exit(1); }
            };
            if dump { print!("{}", bytecode::disassemble(&compiled)); return; }
            let interp = match interp::Interp::new(&program) {
                Ok(i) => i,
                Err(e) => { eprintln!("error: {}", e); exit(1); }
            };
            interp.set_args(program_args(&args, &path));
            // JIT modes: default = tiered (compile a function only after it has
            // been called `--jit-threshold=N` times); `--jit` = eager
            // compile-everything; `--no-jit` = pure VM. Default threshold 100,
            // picked by benchmark: at 1000 a medium-hot function (1.5k calls)
            // ran 5x slower than at 100, while one-shot programs showed no
            // measurable counter overhead and per-function compile cost is ~ms.
            let threshold: u64 = args.iter().skip(2)
                .find_map(|a| a.strip_prefix("--jit-threshold="))
                .and_then(|s| s.parse().ok())
                .or(hgx.as_ref().and_then(|c| c.jit_threshold))
                .unwrap_or(100);
            let result = if flag_present(&args, "--no-jit") {
                bytecode::run(&compiled, &interp)
            } else if flag_present(&args, "--jit") {
                match jit::Jit::compile(&program) {
                    Some(j) => bytecode::run_jit(&compiled, &interp, &j),
                    None => bytecode::run(&compiled, &interp), // nothing eligible
                }
            } else {
                let t = jit::TieredJit::new(&program, threshold);
                t.warm_loops(); // compile loop kernels up-front, even if called once
                let r = bytecode::run_tiered(&compiled, &interp, &t);
                if flag_present(&args, "--jit-stats") {
                    let names = t.compiled_functions();
                    eprintln!("jit-stats: threshold={} compiled={} {:?}",
                              t.threshold, names.len(), names);
                }
                r
            };
            if let Err(e) = result {
                eprintln!("{}", diag::render("runtime error", &path, &src, &e));
                exit(1);
            }
        }
        "jit" => {
            // `nova jit --dump <file>`: show the Cranelift IR for JIT-eligible functions
            let path = path_arg(&args);
            let mut program = match load_program(&path) {
                Ok(p) => p,
                Err(e) => { eprintln!("{}", e); exit(1); }
            };
            interp::fold_program(&mut program);
            match jit::Jit::compile(&program) {
                Some(j) => print!("{}", j.dump()),
                None => eprintln!("jit: no integer-pure functions are eligible"),
            }
        }
        "disasm" => {
            let path = path_arg(&args);
            let optimize = !flag_present(&args, "--no-opt");
            let mut program = match load_program(&path) {
                Ok(p) => p,
                Err(e) => { eprintln!("{}", e); exit(1); }
            };
            interp::fold_program(&mut program);
            match bytecode::compile_program_opt(&program, optimize) {
                Ok(c) => print!("{}", bytecode::disassemble(&c)),
                Err(e) => { eprintln!("{} (run it with `nova run` instead)", e); exit(1); }
            }
        }
        "bench" => {
            let path = require_path(&args);
            let mut program = match load_program(&path) {
                Ok(p) => p,
                Err(e) => { eprintln!("{}", e); exit(1); }
            };
            interp::fold_program(&mut program);
            // tree-walking interpreter
            let t0 = std::time::Instant::now();
            match interp::Interp::new(&program).and_then(|i| i.run()) {
                Ok(_) => {}
                Err(e) => { eprintln!("interp error: {}", e); exit(1); }
            }
            let interp_t = t0.elapsed();
            // bytecode VM
            match bytecode::compile_program(&program) {
                Ok(c) => {
                    let interp = match interp::Interp::new(&program) {
                        Ok(i) => i,
                        Err(e) => { eprintln!("error: {}", e); exit(1); }
                    };
                    let t1 = std::time::Instant::now();
                    if let Err(e) = bytecode::run(&c, &interp) { eprintln!("vm error: {}", e); exit(1); }
                    let vm_t = t1.elapsed();
                    let speedup = interp_t.as_secs_f64() / vm_t.as_secs_f64().max(1e-9);
                    eprintln!("interp: {:?}   vm: {:?}   speedup: {:.2}x", interp_t, vm_t, speedup);
                    // JIT, when any function is eligible
                    if let Some(jit) = jit::Jit::compile(&program) {
                        let interp2 = match interp::Interp::new(&program) {
                            Ok(i) => i,
                            Err(e) => { eprintln!("error: {}", e); exit(1); }
                        };
                        let t2 = std::time::Instant::now();
                        if let Err(e) = bytecode::run_jit(&c, &interp2, &jit) { eprintln!("jit error: {}", e); exit(1); }
                        let jit_t = t2.elapsed();
                        let jspeed = interp_t.as_secs_f64() / jit_t.as_secs_f64().max(1e-9);
                        eprintln!("jit: {:?}   speedup vs interp: {:.2}x", jit_t, jspeed);
                    }
                }
                Err(e) => eprintln!("vm: not compilable ({})", e),
            }
        }
        "repl" => repl(),
        "test" => {
            let path = require_path(&args);
            let mut program = match load_program(&path) {
                Ok(p) => p,
                Err(e) => { eprintln!("{}", e); exit(1); }
            };
            interp::fold_program(&mut program); // constant-fold before execution
            let interp = match interp::Interp::new(&program) {
                Ok(i) => i,
                Err(e) => { eprintln!("error: {}", e); exit(1); }
            };
            let failures = interp.run_tests();
            if failures > 0 { exit(1); }
        }
        "doc" => {
            let path = require_path(&args);
            let src = read(&path);
            print!("{}", doc_extract(&path, &src));
        }
        "tokens" => {
            // the canonical token dump — the self-hosting stage-1 reference
            // (selfhost/lexer.nova must produce byte-identical output)
            let path = require_path(&args);
            let src = read(&path);
            print!("{}", tokens::dump(&src));
        }
        "ast" => {
            // the canonical AST dump — the self-hosting stage-2 reference
            // (selfhost/parser.nova must produce byte-identical output)
            let path = require_path(&args);
            let src = read(&path);
            match parser::parse_program(&src) {
                Ok(p) => print!("{}", astdump::dump(&p)),
                Err(e) => { eprintln!("{}", e); exit(1); }
            }
        }
        "fmt" => {
            // `nova fmt <file>` prints to stdout; `nova fmt -w <file>` rewrites in place
            let write = args.get(2).map(|s| s == "-w").unwrap_or(false);
            let path = if write { args.get(3) } else { args.get(2) };
            let path = match path {
                Some(p) => p.clone(),
                None => { eprintln!("error: missing file path"); exit(2); }
            };
            let src = read(&path);
            match parser::parse_program(&src) {
                Ok(p) => {
                    let out = fmt::format_program(&p);
                    if write {
                        if let Err(e) = std::fs::write(&path, &out) {
                            eprintln!("error: cannot write {}: {}", path, e);
                            exit(1);
                        }
                        println!("formatted {}", path);
                    } else {
                        print!("{}", out);
                    }
                }
                Err(e) => { eprintln!("{}", e); exit(1); }
            }
        }
        "obfuscate" => {
            // `nova obfuscate <file>` prints an obfuscated copy to stdout;
            // `-w` rewrites in place. Local identifiers (params + body bindings)
            // are alpha-renamed to opaque names; behaviour is byte-identical.
            // `#[obfuscate]` on functions selects which to transform; if no
            // function is marked, every user function is obfuscated.
            let write = args.get(2).map(|s| s == "-w").unwrap_or(false);
            let path = if write { args.get(3) } else { args.get(2) };
            let path = match path {
                Some(p) => p.clone(),
                None => { eprintln!("error: missing file path"); exit(2); }
            };
            let src = read(&path);
            match parser::parse_program(&src) {
                Ok(mut p) => {
                    let marked: std::collections::HashSet<String> = p.items.iter().filter_map(|it| match it {
                        ast::Item::Func(f) if f.attrs.iter().any(|a| a.name == "obfuscate") => Some(f.name.clone()),
                        _ => None,
                    }).collect();
                    let targets = if marked.is_empty() { None } else { Some(marked) };
                    obfuscate::obfuscate_program(&mut p, &targets);
                    let out = fmt::format_program(&p);
                    if write {
                        if let Err(e) = std::fs::write(&path, &out) {
                            eprintln!("error: cannot write {}: {}", path, e);
                            exit(1);
                        }
                        println!("obfuscated {}", path);
                    } else {
                        print!("{}", out);
                    }
                }
                Err(e) => { eprintln!("{}", e); exit(1); }
            }
        }
        "lsp" => lsp::run(),
        "add" => pkg_add(&args),
        "remove" | "rm" => pkg_remove(&args),
        "deps" => pkg_deps(),
        "install" | "fetch" => pkg_install(false),
        "update" => pkg_install(true),
        "tree" => pkg_tree(),
        "publish" => pkg_publish(&args),
        "registry" => pkg_registry(&args),
        "daemon" => run_daemon(),
        "version" | "--version" | "-v" => println!("Nova {}", VERSION),
        other => {
            eprintln!("unknown command: {}", other);
            eprintln!("usage: nova [run <file> | vm <file> | bench <file> | check <file> | test <file> | doc <file> | fmt [-w] <file> | obfuscate [-w] <file> | lsp | add <src|name@req> [name] | remove <name> | deps | install | update | tree | publish <index-dir> | registry <index-dir> [--port N] | repl | daemon | version]");
            exit(2);
        }
    }
}

// Persistent compiler service. Reads commands from stdin and keeps parsed
// programs in memory so repeated builds don't re-read cold state — real Daemon
// Mode. `reload` re-parses and reports exactly which functions changed
// (Incremental Compilation: unchanged functions are reused from cache), and a
// subsequent `run` executes the new code without restarting the process (Hot
// Reload). Commands: load/reload/run/funcs/stats/quit.
fn run_daemon() {
    use std::collections::HashMap;
    // path -> (program, function-name -> body hash)
    let mut cache: HashMap<String, (ast::Program, HashMap<String, u64>)> = HashMap::new();
    let mut loads = 0usize;
    let mut reuses = 0usize;

    let hashes = |p: &ast::Program| -> HashMap<String, u64> {
        let mut m = HashMap::new();
        for it in &p.items {
            if let ast::Item::Func(f) = it {
                let text = format!("{:?}", f);
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for b in text.bytes() { h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3); }
                m.insert(f.name.clone(), h);
            }
        }
        m
    };

    println!("Nova {} — daemon ready (commands: load/reload/run/funcs/stats/quit)", VERSION);
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        let mut it = line.trim().splitn(2, char::is_whitespace);
        let cmd = it.next().unwrap_or("");
        let arg = it.next().unwrap_or("").trim();
        match cmd {
            "" => {}
            "quit" | "exit" => { println!("bye"); break; }
            "load" | "reload" => {
                let mut prog = match load_program(arg) {
                    Ok(p) => p, Err(e) => { println!("error: {}", e); continue; }
                };
                interp::fold_program(&mut prog);
                let new = hashes(&prog);
                if cmd == "reload" {
                    if let Some((_, old)) = cache.get(arg) {
                        let mut changed: Vec<&str> = Vec::new();
                        for (n, h) in &new {
                            if old.get(n) != Some(h) { changed.push(n); }
                        }
                        let unchanged = new.len() - changed.len();
                        reuses += unchanged;
                        changed.sort();
                        println!("reloaded {}: {} changed {:?}, {} reused",
                                 arg, changed.len(), changed, unchanged);
                    } else {
                        println!("reloaded {}: {} functions (first load)", arg, new.len());
                    }
                } else {
                    println!("loaded {}: {} functions", arg, new.len());
                }
                loads += 1;
                cache.insert(arg.to_string(), (prog, new));
            }
            "run" => {
                let prog = match cache.get(arg) {
                    Some((p, _)) => p.clone(),
                    None => match load_program(arg) {
                        Ok(mut p) => { interp::fold_program(&mut p); p }
                        Err(e) => { println!("error: {}", e); continue; }
                    }
                };
                match interp::Interp::new(&prog) {
                    Ok(i) => { if let Err(e) = i.run() { println!("runtime error: {}", e); } }
                    Err(e) => println!("error: {}", e),
                }
            }
            "funcs" => {
                match cache.get(arg) {
                    Some((_, h)) => {
                        let mut names: Vec<&String> = h.keys().collect();
                        names.sort();
                        println!("{:?}", names);
                    }
                    None => println!("not loaded: {}", arg),
                }
            }
            "stats" => println!("cached={} loads={} reused_functions={}", cache.len(), loads, reuses),
            other => println!("unknown command: {}", other),
        }
    }
}

fn repl() {
    println!("Nova {} — interactive REPL", VERSION);
    println!("Type expressions or definitions. Commands: :help  :quit\n");

    // An empty program seeds an interpreter; the REPL grows its tables over time.
    let empty = ast::Program { items: Vec::new() };
    let mut interp = match interp::Interp::new(&empty) {
        Ok(i) => i,
        Err(e) => { eprintln!("init error: {}", e); exit(1); }
    };
    let mut scope = interp::Scope::new();

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut buffer = String::new(); // accumulates a multi-line input
    loop {
        print!("{}", if buffer.is_empty() { "nova> " } else { "....> " });
        io::stdout().flush().ok();
        let line = match lines.next() {
            Some(Ok(l)) => l,
            _ => { println!("\nbye!"); break; }
        };
        // commands are only recognized at the start of a fresh input
        if buffer.is_empty() {
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }
            match trimmed {
                ":quit" | ":q" | ":exit" => { println!("bye!"); break; }
                ":help" | ":h" => {
                    println!("  type any Nova expression to evaluate it");
                    println!("  definitions persist; open braces continue on the next line");
                    println!("  :quit  exit    :help  this message");
                    continue;
                }
                _ => {}
            }
        }
        if !buffer.is_empty() { buffer.push('\n'); }
        buffer.push_str(&line);
        // keep reading while delimiters are still open (a multi-line definition)
        if delim_balance(&buffer) > 0 { continue; }
        let input = std::mem::take(&mut buffer);
        match parser::parse_repl(&input) {
            Ok((prog, stmts)) => {
                if !prog.items.is_empty() {
                    if let Err(e) = interp.register_items(&prog) {
                        println!("error: {}", e);
                        continue;
                    }
                    println!("ok");
                } else {
                    match interp.eval_repl(&stmts, &mut scope) {
                        Ok(Some(v)) => {
                            if !matches!(v, interp::Value::Null) {
                                println!("{}", v);
                            }
                        }
                        Ok(None) => {}
                        Err(e) => println!("error: {}", e),
                    }
                }
            }
            Err(e) => println!("{}", e.lines().last().unwrap_or("parse error")),
        }
    }
}

fn usage() -> String {
    format!(
"nova {} — the Nova programming language

USAGE:
  nova <command> [flags] [file.nova]
  (with a nova.hgx in the current directory, the file argument is optional)

COMMANDS:
  run     <file>     run on the tree-walking interpreter (reference semantics)
  vm      <file>     run on the bytecode VM with the tiered JIT (fastest)
  build   <file>     produce a standalone executable in ./build/
  check   <file>     gradual type checking
  test    <file>     run the file's `test` blocks
  bench   <file>     time interpreter vs VM vs JIT
  fmt [-w] <file>    format (print, or rewrite with -w)
  disasm  <file>     show compiled bytecode
  jit --dump <file>  show generated Cranelift IR
  doc     <file>     extract documentation comments
  repl               interactive session (no arguments)
  version            print the version

VM FLAGS:
  --jit                eager-compile every eligible function up front
  --no-jit             disable native compilation
  --jit-threshold=N    tiering threshold (default 100 calls)
  --jit-stats          report which functions were natively compiled (stderr)
  --no-opt             skip the bytecode optimizer
  --dump               print bytecode instead of running

BUILD FLAGS:
  --aot | --aot=c      pure native binary via the C backend (cc -O2)
  --aot=llvm           pure native binary via the LLVM backend (clang -O2)
  --aot=wasm           freestanding wasm32 module, typed tier (clang, node-verified)
  --aot=arm            static aarch64 binary, typed+boxed (cross gcc, qemu-verified)
  --aot=arm32          static ARMv7 (32-bit) binary — older/weaker phones (qemu-verified)
                       (AOT output is verified byte-identical to `nova run`
                        at build time; non-AOT-able programs fall back to the
                        embedded-runtime build automatically)
", VERSION)
}

// Run a program embedded by `nova build`: tiered VM/JIT when `main` is
// VM-compilable, interpreter otherwise — same outputs as `nova run`.
fn run_embedded(src: &str) {
    let mut program = match parser::parse_program(src) {
        Ok(p) => p,
        Err(e) => { eprintln!("{}", e); exit(1); }
    };
    interp::fold_program(&mut program);
    let interp = match interp::Interp::new(&program) {
        Ok(i) => i,
        Err(e) => { eprintln!("error: {}", e); exit(1); }
    };
    interp.set_args(std::env::args().skip(1).collect());
    let result = match bytecode::compile_program(&program) {
        Ok(c) => {
            let t = jit::TieredJit::new(&program, 100);
            t.warm_loops();
            bytecode::run_tiered(&c, &interp, &t)
        }
        Err(_) => interp.run().map(|_| ()),
    };
    if let Err(e) = result {
        eprintln!("{}", diag::render("runtime error", "<program>", src, &e));
        exit(1);
    }
}

// The input file: the first non-flag CLI argument, else `entry` from a
// `nova.hgx` in the current directory (backward compatible either way).
fn resolve_input(args: &[String]) -> (String, Option<config::HgxConfig>) {
    let explicit = args.iter().skip(2).find(|a| !a.starts_with("--")).cloned();
    let cfg = match config::load_hgx(std::path::Path::new(".")) {
        Some(Ok(c)) => Some(c),
        Some(Err(e)) => { eprintln!("{}", e); exit(1); }
        None => None,
    };
    match explicit {
        Some(p) => (p, cfg),
        None => match cfg {
            Some(c) => (c.entry.clone(), Some(c)),
            None => { eprintln!("error: missing file path (and no nova.hgx found)"); exit(2); }
        },
    }
}

fn require_path(args: &[String]) -> String { resolve_input(args).0 }

// program-visible argv: everything after the input file token
fn program_args(args: &[String], path: &str) -> Vec<String> {
    match args.iter().position(|a| a == path) {
        Some(i) => args[i + 1..].to_vec(),
        None => Vec::new(),
    }
}

fn path_arg(args: &[String]) -> String { resolve_input(args).0 }

fn flag_present(args: &[String], flag: &str) -> bool {
    args.iter().skip(2).any(|a| a == flag)
}

fn read(path: &str) -> String {
    match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("error: cannot read {}: {}", path, e); exit(1); }
    }
}

// Net open-delimiter count of a source fragment, ignoring strings and line
// comments. Used by the REPL to keep reading while braces/parens are still open.
fn delim_balance(s: &str) -> i32 {
    let b = s.as_bytes();
    let (mut depth, mut i, mut in_str, mut esc) = (0i32, 0usize, false, false);
    while i < b.len() {
        let c = b[i] as char;
        if in_str {
            if esc { esc = false; }
            else if c == '\\' { esc = true; }
            else if c == '"' { in_str = false; }
            i += 1;
            continue;
        }
        match c {
            '"' => in_str = true,
            '/' if i + 1 < b.len() && b[i + 1] == b'/' => {
                while i < b.len() && b[i] != b'\n' { i += 1; }
                continue;
            }
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    depth
}

// Parse a file and recursively inline every `use "file.nova"` import. Paths are
// resolved relative to the importing file; each file is loaded once (so diamond
// imports and cycles are safe), and all items are merged into one Program.
fn load_program(path: &str) -> Result<ast::Program, String> {
    let mut visited = std::collections::HashSet::new();
    let mut program = load_module(path, &mut visited)?;
    // Two-mode abilities: merge any `[abilities]` declared in nova.hgx onto the
    // program's functions before execution/checking (see registry::apply_abilities).
    if let Some(Ok(cfg)) = config::load_hgx(std::path::Path::new(".")) {
        if !cfg.abilities.is_empty() { registry::apply_abilities(&mut program, &cfg.abilities); }
    }
    Ok(program)
}

// Import search order: relative to the importing file, then $NOVA_STD's
// parent, then next to the nova executable (so `use "std/list.nova"` finds the
// shipped standard library from any project directory).
pub(crate) fn resolve_import(base: &std::path::Path, rel: &str) -> std::path::PathBuf {
    let local = base.join(rel);
    if local.exists() { return local; }
    if let Ok(root) = std::env::var("NOVA_STD") {
        let p = std::path::Path::new(&root).parent().unwrap_or(std::path::Path::new(&root)).join(rel);
        if p.exists() { return p; }
        let p = std::path::Path::new(&root).join(rel.strip_prefix("std/").unwrap_or(rel));
        if p.exists() { return p; }
    }
    // vendored dependencies (the package manager): `nova add` copies deps into
    // ./nova_modules, so `use "dep"` / `use "dep.nova"` resolves from any project.
    for cand in [
        std::path::Path::new("nova_modules").join(rel),
        std::path::Path::new("nova_modules").join(format!("{}.nova", rel.trim_end_matches(".nova"))),
    ] {
        if cand.exists() { return cand; }
    }
    if let Ok(exe) = std::env::current_exe() {
        for up in [exe.parent(), exe.parent().and_then(|p| p.parent()),
                   exe.parent().and_then(|p| p.parent()).and_then(|p| p.parent())] {
            if let Some(dir) = up {
                let p = dir.join(rel);
                if p.exists() { return p; }
            }
        }
    }
    local
}

// The package manager: `nova add <src.nova> [name]` vendors a dependency into
// ./nova_modules/<name>.nova and records it in ./nova.deps; `nova deps` lists
// them. Vendored deps are resolved by `resolve_import` above, so a program can
// `use "name"` regardless of where the source lived. Minimal but real — no
// registry/network, just local/path deps (the honest, verifiable core).
// `nova add` has two modes:
//   * `nova add <file.nova> [name]` — vendor a local file into nova_modules and
//     record it in nova.deps (the original quick, offline path — unchanged).
//   * `nova add <name>@<req>` (or `<name> --version <req>` / `--git <url>` /
//     `--path <p>`) — add a `[dependencies]` entry to nova.hgx, then `install`.
fn pkg_add(args: &[String]) {
    let spec = match args.get(2) { Some(s) => s.clone(), None => { eprintln!("usage: nova add <src.nova> [name] | <name>@<req>"); exit(2); } };
    // local-file mode: the arg names an existing .nova file
    if std::path::Path::new(&spec).is_file() {
        let stem = std::path::Path::new(&spec).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let name = args.get(3).filter(|a| !a.starts_with("--")).cloned().unwrap_or(stem);
        let content = match std::fs::read_to_string(&spec) { Ok(c) => c, Err(e) => { eprintln!("error: cannot read {}: {}", spec, e); exit(1); } };
        if let Err(e) = std::fs::create_dir_all("nova_modules") { eprintln!("error: {}", e); exit(1); }
        let dest = format!("nova_modules/{}.nova", name);
        if let Err(e) = std::fs::write(&dest, &content) { eprintln!("error: cannot write {}: {}", dest, e); exit(1); }
        let manifest = std::fs::read_to_string("nova.deps").unwrap_or_default();
        let mut lines: Vec<String> = manifest.lines().filter(|l| !l.starts_with(&format!("{} =", name))).map(|s| s.to_string()).collect();
        lines.push(format!("{} = {}", name, spec));
        let _ = std::fs::write("nova.deps", lines.join("\n") + "\n");
        println!("added `{}` -> {} (use \"{}\")", name, dest, name);
        return;
    }
    // registry/git/path mode: mutate nova.hgx [dependencies], then install
    let (name, ver_inline) = match spec.split_once('@') { Some((n, v)) => (n.to_string(), Some(v.to_string())), None => (spec.clone(), None) };
    let flag = |k: &str| args.iter().skip(3).position(|a| a == k).and_then(|i| args.get(3 + i + 1)).cloned();
    let entry = if let Some(g) = flag("--git") {
        match flag("--rev") { Some(r) => format!("{{ git = \"{}\", rev = \"{}\" }}", g, r), None => format!("{{ git = \"{}\" }}", g) }
    } else if let Some(p) = flag("--path") {
        format!("{{ path = \"{}\" }}", p)
    } else {
        format!("\"{}\"", ver_inline.or_else(|| flag("--version")).unwrap_or_else(|| "*".into()))
    };
    if let Err(e) = hgx_upsert_dependency(&name, &entry) { eprintln!("error: {}", e); exit(1); }
    println!("added `{}` to nova.hgx [dependencies]", name);
    pkg_install(false);
}

// Insert/replace a `name = <entry>` line under `[dependencies]` in nova.hgx,
// creating the file / section if needed. Preserves all other content.
fn hgx_upsert_dependency(name: &str, entry: &str) -> Result<(), String> {
    let path = std::path::Path::new("nova.hgx");
    let text = std::fs::read_to_string(path).unwrap_or_else(|_| "[package]\nname = \"app\"\nversion = \"0.1.0\"\n".to_string());
    let mut lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
    let dep_hdr = lines.iter().position(|l| l.trim() == "[dependencies]");
    let new_line = format!("{} = {}", name, entry);
    match dep_hdr {
        Some(h) => {
            // find the extent of the section, replace an existing key or append
            let end = lines[h + 1..].iter().position(|l| l.trim_start().starts_with('[')).map(|i| h + 1 + i).unwrap_or(lines.len());
            if let Some(k) = lines[h + 1..end].iter().position(|l| l.split('=').next().map(|s| s.trim() == name).unwrap_or(false)) {
                lines[h + 1 + k] = new_line;
            } else {
                lines.insert(end, new_line);
            }
        }
        None => { lines.push("[dependencies]".into()); lines.push(new_line); }
    }
    std::fs::write(path, lines.join("\n") + "\n").map_err(|e| e.to_string())
}

fn pkg_remove(args: &[String]) {
    let name = match args.get(2) { Some(s) => s.clone(), None => { eprintln!("usage: nova remove <name>"); exit(2); } };
    let mut changed = false;
    if let Ok(text) = std::fs::read_to_string("nova.hgx") {
        let kept: Vec<String> = text.lines().filter(|l| !l.split('=').next().map(|s| s.trim() == name).unwrap_or(false)).map(|s| s.to_string()).collect();
        if kept.len() != text.lines().count() { let _ = std::fs::write("nova.hgx", kept.join("\n") + "\n"); changed = true; }
    }
    let _ = std::fs::remove_file(format!("nova_modules/{}.nova", name));
    let _ = std::fs::remove_dir_all(format!("nova_modules/{}", name));
    if let Ok(text) = std::fs::read_to_string("nova.deps") {
        let kept: Vec<String> = text.lines().filter(|l| !l.starts_with(&format!("{} =", name))).map(|s| s.to_string()).collect();
        let _ = std::fs::write("nova.deps", kept.join("\n") + "\n");
    }
    println!("{} `{}`", if changed { "removed" } else { "removed (vendored files)" }, name);
}

fn pkg_deps() {
    if let Some(Ok(cfg)) = config::load_hgx(std::path::Path::new(".")) {
        if !cfg.dependencies.is_empty() {
            println!("dependencies (nova.hgx):");
            for d in &cfg.dependencies {
                let src = d.git.clone().map(|g| format!("git {}", g))
                    .or_else(|| d.path.clone().map(|p| format!("path {}", p)))
                    .unwrap_or_else(|| d.version.clone().unwrap_or_else(|| "*".into()));
                println!("  {} = {}", d.name, src);
            }
            return;
        }
    }
    match std::fs::read_to_string("nova.deps") {
        Ok(m) if !m.trim().is_empty() => { print!("{}", m); }
        _ => println!("no dependencies (use `nova add <name>@<req>` or `nova add <src.nova> [name]`)"),
    }
}

fn load_cfg_or_exit() -> config::HgxConfig {
    match config::load_hgx(std::path::Path::new(".")) {
        Some(Ok(c)) => c,
        Some(Err(e)) => { eprintln!("{}", e); exit(1); }
        None => { eprintln!("error: no nova.hgx in this directory"); exit(1); }
    }
}

fn pkg_install(update: bool) {
    let cfg = load_cfg_or_exit();
    match registry::install(&cfg, update) {
        Ok(locked) if locked.is_empty() => println!("no dependencies to install"),
        Ok(locked) => {
            println!("{} {} package(s), locked in nova.lock:", if update { "updated" } else { "installed" }, locked.len());
            print!("{}", registry::tree(&locked));
        }
        Err(e) => { eprintln!("error: {}", e); exit(1); }
    }
}

fn pkg_tree() {
    // prefer the lock (reproducible); fall back to a fresh resolve
    let locked = if let Ok(text) = std::fs::read_to_string("nova.lock") {
        registry::read_lock(&text)
    } else {
        let cfg = load_cfg_or_exit();
        match registry::install(&cfg, false) { Ok(l) => l, Err(e) => { eprintln!("error: {}", e); exit(1); } }
    };
    if locked.is_empty() { println!("no dependencies"); } else { print!("{}", registry::tree(&locked)); }
}

fn pkg_publish(args: &[String]) {
    let index_dir = match args.get(2) { Some(s) => s.clone(), None => { eprintln!("usage: nova publish <index-dir>"); exit(2); } };
    let cfg = load_cfg_or_exit();
    match registry::publish(std::path::Path::new("."), std::path::Path::new(&index_dir), &cfg) {
        Ok(sha) => println!("published {} {} to {} (sha256 {})", cfg.name, cfg.version, index_dir, &sha[..16]),
        Err(e) => { eprintln!("error: {}", e); exit(1); }
    }
}

fn pkg_registry(args: &[String]) {
    let dir = match args.iter().skip(2).find(|a| !a.starts_with("--")) { Some(s) => s.clone(), None => { eprintln!("usage: nova registry <index-dir> [--port N]"); exit(2); } };
    let port: u16 = args.iter().skip(2).find_map(|a| a.strip_prefix("--port=")).and_then(|s| s.parse().ok())
        .or_else(|| args.iter().position(|a| a == "--port").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()))
        .unwrap_or(7878);
    if let Err(e) = registry::serve(std::path::Path::new(&dir), port) { eprintln!("error: {}", e); exit(1); }
}

fn load_module(
    path: &str,
    visited: &mut std::collections::HashSet<std::path::PathBuf>,
) -> Result<ast::Program, String> {
    let canon = std::fs::canonicalize(path)
        .map_err(|e| format!("cannot import `{}`: {}", path, e))?;
    if !visited.insert(canon.clone()) {
        // already loaded on this run (or a cycle): contribute nothing more
        return Ok(ast::Program { items: Vec::new() });
    }
    let src = std::fs::read_to_string(&canon)
        .map_err(|e| format!("cannot read `{}`: {}", path, e))?;
    let prog = parser::parse_program(&src).map_err(|e| format!("in {}:\n{}", path, e))?;
    let base = canon.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let mut items = Vec::new();
    for item in prog.items {
        match item {
            ast::Item::Import { path: rel } => {
                let full = resolve_import(&base, &rel);
                let sub = load_module(&full.to_string_lossy(), visited)?;
                items.extend(sub.items);
            }
            other => items.push(other),
        }
    }
    Ok(ast::Program { items })
}

// Generate Markdown API docs from a Nova source file.
//
// Doc comments are `///` lines (and module-level `//!`). The grammar treats
// comments as whitespace, so the AST drops them — instead we scan the source
// text: contiguous `///` lines attach to the declaration on the next code line
// (fn / struct / enum / trait / impl / const / static / type / data / macro).
fn doc_extract(path: &str, src: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Documentation for `{}`\n\n", path));

    let mut module_doc: Vec<String> = Vec::new();
    let mut pending: Vec<String> = Vec::new();
    let mut entries: Vec<(String, String)> = Vec::new();

    for raw in src.lines() {
        let line = raw.trim();
        if let Some(rest) = line.strip_prefix("///") {
            pending.push(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("//!") {
            module_doc.push(rest.trim().to_string());
        } else if let Some(sig) = decl_signature(line) {
            entries.push((sig, pending.join(" ").trim().to_string()));
            pending.clear();
        } else {
            // a blank or unrelated line breaks the doc/declaration association
            pending.clear();
        }
    }

    if !module_doc.is_empty() {
        out.push_str(&module_doc.join(" "));
        out.push_str("\n\n");
    }
    if entries.is_empty() {
        out.push_str("_No documented declarations found._\n");
        return out;
    }
    for (sig, doc) in entries {
        out.push_str(&format!("### `{}`\n\n", sig));
        if doc.is_empty() {
            out.push_str("_No documentation._\n\n");
        } else {
            out.push_str(&doc);
            out.push_str("\n\n");
        }
    }
    out
}

// If `line` begins a top-level declaration, return its signature (the text up to
// the body `{`, the `;`, or an `=>`/`=` introducer), else None.
fn decl_signature(line: &str) -> Option<String> {
    let head = line.strip_prefix("pub ").unwrap_or(line).trim_start();
    const KEYWORDS: [&str; 11] = [
        "fn ", "async fn", "struct ", "enum ", "trait ", "impl ",
        "const ", "static ", "type ", "data ", "macro ",
    ];
    if !KEYWORDS.iter().any(|k| head.starts_with(k)) {
        return None;
    }
    let cut = line.find('{')
        .or_else(|| line.find("=>"))
        .or_else(|| line.find(';'))
        .unwrap_or(line.len());
    let sig = line[..cut].trim().to_string();
    if sig.is_empty() { None } else { Some(sig) }
}
