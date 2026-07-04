// Nova bytecode compiler + stack VM.
//
// Phase 1 gave a fast slot-based VM for the scalar/compute core. Phase 2 extends
// coverage to the WHOLE language: the compute core (literals, locals, arithmetic,
// control flow, function calls, loops) is compiled to native bytecode, and every
// other expression is delegated to the interpreter via an `EvalAst` op. That is
// sound because `Interp::eval` takes the scope by immutable reference — it can
// never rebind the VM's locals; heap mutations propagate through `Rc` exactly as
// in `nova run`. Results are therefore identical to the tree-walker, which the
// test suite verifies byte-for-byte on every example.

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use crate::ast::*;
use crate::interp::{
    Interp, Scope, Value, Flow, eval_binop, eval_unop, norm_big,
    build_map, build_set, build_range, index_get, field_get, safe_field_get,
    index_set, field_set, do_slice, THROW_SENTINEL, YIELD_STOP,
};

#[derive(Debug, Clone)]
enum Op {
    Const(u32),
    ConstNull,
    ConstBool(bool),
    LoadLocal(u16),
    StoreLocal(u16),
    Pop,
    Nop,                   // placeholder left by the optimizer before compaction
    Bin(BinOp),
    Un(UnOp),
    Truthy,
    // --- native heap reads / literals (Phase 3A) ---
    MakeArray(u32),        // pop n values -> Array
    MakeMap(u32),          // pop n key/value pairs (2n values) -> Map
    MakeSet(u32),          // pop n values -> Set (map elem->null)
    MakeRange(bool),       // pop hi, lo -> Array of ints (bool = inclusive)
    Index,                 // pop idx, base -> base[idx]
    Slice(bool, bool, bool), // (has_lo, has_hi, inclusive); pop hi?, lo?, base -> slice
    GetField(u32),         // name-pool index; pop base -> base.field
    SafeField(u32),        // name-pool index; pop base -> base?.field
    MakeStruct(u32, Vec<u32>), // struct-name idx + field-name idxs; pop n values -> Struct
    Fmt(u32),              // pop n string parts -> concatenated Str
    IndexSet,              // pop value, idx, base -> base[idx] = value
    SetField(u32),         // name-pool index; pop value, base -> base.field = value
    // match: test match_tests[i] against locals[scrut]; on success bind into the
    // recorded slots and fall through, else jump to the fail ip (next arm).
    MatchTest(u32, u16, usize),
    NoMatch,               // no arm matched -> non-exhaustive error
    // lazy for-each step: it_slot, idx_slot, var_slot, end_ip. Fetches the
    // idx-th item of the iterable in it_slot; if exhausted jumps to end_ip,
    // otherwise binds it to var_slot and advances idx_slot.
    IterStep(u16, u16, u16, usize),
    Jump(usize),
    JumpIfFalse(usize),
    JumpIfTrue(usize),
    Call(u32, u8),         // compiled-chunk index, argc
    CallDyn(u32, u8),      // name-pool index, argc -> interp.call_named (args evaluated natively)
    Method(u32, u8),       // method-name pool idx, argc; pop args + receiver -> call_method_vals
    MakeClosure(u32),      // lambda-table idx -> Value::Closure capturing the current frame
    CallValue(u8),         // argc; pop args + callee value -> call the closure
    EvalAst(u32),          // delegate del_exprs[i] to interp.eval
    Return,
    // --- flow / exceptions / defers (Phase 5) ---
    Pos(u32, u32),         // statement position marker (mirrors the interp's cur_pos)
    Throw,                 // pop v -> unwind with Flow::Throw(v)
    FlowBreak,             // pop v -> unwind with Flow::Break(v) out of a region
    FlowContinue,          // unwind with Flow::Continue out of a region
    Try(u32),              // try_meta idx; drives body/catch/finally sub-runs
    PushDefer(u16, u32),   // (block depth, defer_meta idx): register a pending defer
    RunDefers(u16),        // run + pop defers registered at depth >= d (LIFO)
    // --- fused superinstructions (peephole; semantics = the unfused sequence) ---
    IncLocal(u16, u32, BinOp),           // locals[s] = locals[s] <op> consts[k]
    LocalsBinJf(u16, u16, BinOp, usize), // if !(locals[a] <op> locals[b]) jump
    BinLL(u16, u16, BinOp),              // push(locals[a] <op> locals[b])
    BinLC(u16, u32, BinOp),              // push(locals[a] <op> consts[k])
}

// Everything `Op::Try` needs at runtime. The body/catch/finally code ranges live
// in the same chunk and are executed as sub-runs over the same locals frame,
// transcribing the tree-walker's TryCatch logic op-for-op. The optional stub ips
// are compile-time continuations for flows that must keep travelling after the
// try resolves (a `break` to an enclosing loop, a `return` to an enclosing
// expression block); `None` means "propagate the Flow out of this run".
struct TryMeta {
    body: (usize, usize),
    catch: Option<(usize, usize)>,
    finally: Option<(usize, usize)>,
    catch_slot: Option<u16>,
    has_catch: bool,
    body_depth: u16,
    after: usize,
    brk: Option<usize>,   // stub: value pushed, stub pops + jumps
    cont: Option<usize>,  // stub: nothing pushed
    ret: Option<usize>,   // stub: value pushed, stub keeps it
    thr: Option<usize>,   // stub: value pushed, stub pops + jumps
}

struct Chunk {
    name: String,
    n_params: usize,
    n_locals: usize,
    consts: Vec<Value>,
    code: Vec<Op>,
    // delegated expressions: (node, snapshot of visible name->slot)
    del_exprs: Vec<(Expr, Vec<(String, u16)>)>,
    // match arms: (pattern, bound name -> slot) used by `MatchTest`
    match_tests: Vec<(Pattern, Vec<(String, u16)>)>,
    // for a lambda chunk: the captured names, in order, occupying slots
    // n_params..n_params+captures.len() (filled by `CallValue` from the closure).
    captures: Vec<String>,
    // side tables for Op::Try / Op::PushDefer (code ranges into `code`);
    // chunks with entries here skip the optimizer so the ranges stay valid.
    try_meta: Vec<TryMeta>,
    defer_meta: Vec<(usize, usize)>,
}

// Everything `MakeClosure` needs to build a `Value::Closure` at runtime.
struct LambdaInfo {
    chunk_id: usize,
    params: Vec<String>,
    body: LambdaBody,
    // captured names -> the PARENT frame slot to read at closure-creation time.
    captures: Vec<(String, u16)>,
}

pub struct Compiled {
    chunks: Vec<Chunk>,
    names: Vec<String>, // CallDyn / field / method name pool
    lambdas: Vec<LambdaInfo>,
    main: usize,
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

pub fn compile_program(prog: &Program) -> Result<Compiled, String> {
    compile_program_opt(prog, true)
}

pub fn compile_program_opt(prog: &Program, optimize_flag: bool) -> Result<Compiled, String> {
    let mut funcs: HashMap<String, &Func> = HashMap::new();
    for item in &prog.items {
        if let Item::Func(f) = item { funcs.insert(f.name.clone(), f); }
    }
    let main = funcs.get("main").ok_or("vm: no `main` function")?;
    if !main.params.is_empty() {
        return Err("vm: `main` must take no arguments".into());
    }
    // `let x: T = v` where T is a refinement type runs the predicate in the
    // interpreter; the VM doesn't compile those checks, so such functions stay
    // interp-only (rare, and the fallback message documents it)
    let refined: std::collections::HashSet<&str> = prog.items.iter()
        .filter_map(|i| match i {
            Item::TypeAlias { name, refinement: Some(_), .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    // functions whose statements are all VM-native; the rest run on the interpreter.
    // Functions carrying BEHAVIOURAL attributes (`#[self_healing]`, `#[memo]`,
    // contracts, `#[trace]`, …) are kept interp-only so their semantics — applied
    // at the interpreter's `call` boundary — take effect on every tier (the VM
    // reaches them through `CallDyn` -> `call_named` -> `call`). Pure optimisation
    // hints / metadata (`#[hot]`, `#[cold]`, `#[version]`, …) don't change
    // behaviour, so such functions may still be compiled.
    let mut compiled_names: Vec<String> = funcs.values()
        .filter(|f| func_compilable(f) && !uses_refined_let(&f.body, &refined)
            && !f.attrs.iter().any(|a| is_behavioural_attr(&a.name)))
        .map(|f| f.name.clone())
        .collect();
    compiled_names.sort();
    if !compiled_names.iter().any(|n| n == "main") {
        return Err("vm: `main` uses features not supported by the VM yet — use `nova run`".into());
    }
    let index: HashMap<String, usize> =
        compiled_names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();

    // Chunk ids 0..F are the top-level functions; lambda chunks are appended.
    let mut ctx = Ctx {
        index: &index,
        names: Vec::new(),
        next_chunk_id: compiled_names.len(),
        jobs: std::collections::VecDeque::new(),
        lambdas: Vec::new(),
    };
    // slots keyed by chunk id; functions fill 0..F, lambdas append beyond
    let mut slots: Vec<Option<Chunk>> = (0..compiled_names.len()).map(|_| None).collect();

    for name in &compiled_names {
        let f = funcs[name];
        let mut fc = FnCompiler::new(&mut ctx);
        for p in &f.params { fc.define(p); }
        fc.compile_body(&f.body)?;
        let id = index[name];
        slots[id] = Some(fc.finish(name.clone(), f.params.len(), optimize_flag, Vec::new()));
    }

    // Compile every queued lambda body to its own chunk (may queue more).
    while let Some(job) = ctx.jobs.pop_front() {
        let mut fc = FnCompiler::new(&mut ctx);
        for p in &job.params { fc.define(p); }
        for c in &job.capture_names { fc.define(c); }
        fc.compile_lambda_body(&job.body)?;
        let chunk = fc.finish(
            format!("<lambda#{}>", job.chunk_id),
            job.params.len(), optimize_flag, job.capture_names);
        if job.chunk_id >= slots.len() { slots.resize_with(job.chunk_id + 1, || None); }
        slots[job.chunk_id] = Some(chunk);
    }

    let chunks: Vec<Chunk> = slots.into_iter().map(|c| c.expect("chunk slot filled")).collect();
    Ok(Compiled { chunks, names: ctx.names, lambdas: ctx.lambdas, main: index["main"] })
}

// A queued lambda body awaiting compilation into its own chunk.
struct LambdaJob {
    chunk_id: usize,
    params: Vec<String>,
    body: LambdaBody,
    // enclosing local names, in order, captured into child slots n_params.. .
    capture_names: Vec<String>,
}

// Shared compilation state: the function index, the name pool, and the lambda
// worklist + table, all grown as functions and lambdas are compiled.
struct Ctx<'i> {
    index: &'i HashMap<String, usize>,
    names: Vec<String>,
    next_chunk_id: usize,
    jobs: std::collections::VecDeque<LambdaJob>,
    lambdas: Vec<LambdaInfo>,
}

// A function is VM-compilable iff every statement is in the native set (its
// expressions can always be delegated). Only `yield` at statement level makes a
// function interp-only now: generators replay their body through the tree-walker.
fn func_compilable(f: &Func) -> bool {
    f.body.iter().all(stmt_compilable)
}

// Attributes that change a function's runtime behaviour (so the function must run
// through the interpreter on every tier). Pure hints/metadata are not listed and
// don't block VM/JIT compilation.
pub(crate) fn is_behavioural_attr(name: &str) -> bool {
    matches!(name,
        "self_healing" | "retry" | "hot_swap" | "memo" | "memoize"
        | "requires" | "ensures" | "assumes" | "trace" | "log" | "audit"
        | "profile" | "deprecate" | "deprecated" | "time_travel"
        | "anti_debug" | "anti_tamper" | "integrity")
}

// Does any (nested) `let` in this body carry a refinement-type annotation?
fn uses_refined_let(body: &[Stmt], refined: &std::collections::HashSet<&str>) -> bool {
    body.iter().any(|s| match s {
        Stmt::Let { ty: Some(t), .. } if refined.contains(t.as_str()) => true,
        Stmt::If { then, els, .. } =>
            uses_refined_let(then, refined)
            || els.as_ref().map_or(false, |e| uses_refined_let(e, refined)),
        Stmt::While { body, .. } | Stmt::ForRange { body, .. }
        | Stmt::ForEach { body, .. } | Stmt::Defer(body) => uses_refined_let(body, refined),
        Stmt::TryCatch { body, catch_body, finally_body, .. } =>
            uses_refined_let(body, refined)
            || catch_body.as_ref().map_or(false, |b| uses_refined_let(b, refined))
            || finally_body.as_ref().map_or(false, |b| uses_refined_let(b, refined)),
        _ => false,
    })
}

fn stmt_compilable(s: &Stmt) -> bool {
    match s {
        Stmt::Let { .. } | Stmt::Assign { .. } | Stmt::Expr(_) | Stmt::Return(_)
        | Stmt::IndexAssign { .. } | Stmt::FieldAssign { .. } | Stmt::Break(_)
        | Stmt::Continue | Stmt::Throw(_) => true,
        Stmt::If { then, els, .. } =>
            then.iter().all(stmt_compilable)
            && els.as_ref().map_or(true, |e| e.iter().all(stmt_compilable)),
        Stmt::While { body, .. } | Stmt::ForRange { body, .. } | Stmt::ForEach { body, .. }
        | Stmt::Defer(body) =>
            body.iter().all(stmt_compilable),
        Stmt::TryCatch { body, catch_body, finally_body, .. } =>
            body.iter().all(stmt_compilable)
            && catch_body.as_ref().map_or(true, |b| b.iter().all(stmt_compilable))
            && finally_body.as_ref().map_or(true, |b| b.iter().all(stmt_compilable)),
        _ => false, // Yield
    }
}

// A lambda body can be compiled to a chunk iff (for a block) all its statements
// are native; an expression body is always fine. Otherwise the lambda is
// delegated to the interpreter.
fn lambda_compilable(body: &LambdaBody) -> bool {
    match body {
        LambdaBody::Expr(_) => true,
        LambdaBody::Block(stmts) => stmts.iter().all(stmt_compilable),
    }
}

struct FnCompiler<'a, 'i> {
    code: Vec<Op>,
    consts: Vec<Value>,
    // The interpreter has ONE flat mutable scope per function call: statement
    // blocks (if/while/for bodies) share it, so bindings made inside them persist
    // after the block. Only match arms and expression blocks get their own layer
    // (the tree-walker clones the scope there). We mirror that exactly: extra
    // scope layers are pushed only for those, and `barriers` marks expression-
    // block entries where WRITES must shadow instead of mutating outer slots
    // (`{ x = 2 }` must not leak — the interp writes into a clone).
    scopes: Vec<HashMap<String, u16>>,
    barriers: Vec<usize>,
    n_locals: u16,
    ctx: &'a mut Ctx<'i>,
    flows: Vec<FlowCtx>,
    cur_depth: u16,
    any_defers: bool,
    del_exprs: Vec<(Expr, Vec<(String, u16)>)>,
    match_tests: Vec<(Pattern, Vec<(String, u16)>)>,
    try_meta: Vec<TryMeta>,
    defer_meta: Vec<(usize, usize)>,
}

// Compile-time flow context. Mirrors where the interpreter's `Flow` values are
// consumed: loops catch Break/Continue; an expression block swallows every flow
// (Return yields the block's value skipping the tail; Break/Continue/Throw fall
// through to the tail); a Region is a runtime sub-run boundary (try body/catch/
// finally/defer body) that flows must cross as Flow ops, not jumps.
enum FlowCtx {
    Loop { breaks: Vec<usize>, continues: Vec<usize>, body_depth: u16 },
    Block { tail_jumps: Vec<usize>, value_jumps: Vec<usize>, body_depth: u16 },
    Region,
}

// Where a flow statement lands, resolved innermost-first at compile time.
enum Boundary { Loop(usize), Block(usize), Region, Function }

enum StubKind { Break, Continue, Return, Throw }

fn loop_jumps(ctx: FlowCtx) -> (Vec<usize>, Vec<usize>) {
    match ctx {
        FlowCtx::Loop { breaks, continues, .. } => (breaks, continues),
        _ => unreachable!("loop context expected"),
    }
}

impl<'a, 'i> FnCompiler<'a, 'i> {
    fn new(ctx: &'a mut Ctx<'i>) -> Self {
        FnCompiler {
            code: Vec::new(), consts: Vec::new(),
            scopes: vec![HashMap::new()], barriers: Vec::new(), n_locals: 0,
            ctx, flows: Vec::new(), cur_depth: 0, any_defers: false,
            del_exprs: Vec::new(), match_tests: Vec::new(),
            try_meta: Vec::new(), defer_meta: Vec::new(),
        }
    }

    // function / lambda-block body: run statements, then an implicit `null` return
    fn compile_body(&mut self, body: &[Stmt]) -> Result<(), String> {
        self.block(body)?;
        self.code.push(Op::ConstNull);
        self.code.push(Op::Return);
        Ok(())
    }

    // a lambda's body: an expression returns its value; a block behaves like a fn body
    fn compile_lambda_body(&mut self, body: &LambdaBody) -> Result<(), String> {
        match body {
            LambdaBody::Expr(e) => { self.expr(e)?; self.code.push(Op::Return); }
            LambdaBody::Block(stmts) => { self.compile_body(stmts)?; }
        }
        Ok(())
    }

    fn finish(self, name: String, n_params: usize, optimize_flag: bool, captures: Vec<String>) -> Chunk {
        // try/defer side tables hold raw code ranges; the optimizer would move them
        let has_regions = !self.try_meta.is_empty() || !self.defer_meta.is_empty();
        let code = if optimize_flag && !has_regions { optimize(self.code) } else { self.code };
        Chunk {
            name,
            n_params,
            n_locals: self.n_locals as usize,
            consts: self.consts,
            code,
            del_exprs: self.del_exprs,
            match_tests: self.match_tests,
            captures,
            try_meta: self.try_meta,
            defer_meta: self.defer_meta,
        }
    }

    // ---- locals / scopes ----
    fn enter(&mut self) { self.scopes.push(HashMap::new()); }
    fn exit(&mut self) { self.scopes.pop(); }
    fn define(&mut self, name: &str) -> u16 {
        let slot = self.n_locals;
        self.n_locals += 1;
        self.scopes.last_mut().unwrap().insert(name.to_string(), slot);
        slot
    }
    fn fresh(&mut self) -> u16 { let s = self.n_locals; self.n_locals += 1; s }
    fn resolve(&self, name: &str) -> Option<u16> {
        for sc in self.scopes.iter().rev() {
            if let Some(s) = sc.get(name) { return Some(*s); }
        }
        None
    }
    // Slot for a WRITE: only layers inside the innermost expression block are
    // eligible (the interp writes into that block's cloned scope), everything
    // below must be shadowed by a fresh slot instead of mutated.
    fn resolve_write(&self, name: &str) -> Option<u16> {
        let floor = *self.barriers.last().unwrap_or(&0);
        for (i, sc) in self.scopes.iter().enumerate().rev() {
            if i < floor { break; }
            if let Some(s) = sc.get(name) { return Some(*s); }
        }
        None
    }
    fn write_slot(&mut self, name: &str) -> u16 {
        self.resolve_write(name).unwrap_or_else(|| self.define(name))
    }
    // innermost boundary for break/continue (loops catch them, blocks swallow them)
    fn break_boundary(&self) -> Boundary {
        for (i, f) in self.flows.iter().enumerate().rev() {
            match f {
                FlowCtx::Loop { .. } => return Boundary::Loop(i),
                FlowCtx::Block { .. } => return Boundary::Block(i),
                FlowCtx::Region => return Boundary::Region,
            }
        }
        Boundary::Function
    }
    // innermost boundary for return/throw (loops pass them through)
    fn exit_boundary(&self) -> Boundary {
        for (i, f) in self.flows.iter().enumerate().rev() {
            match f {
                FlowCtx::Block { .. } => return Boundary::Block(i),
                FlowCtx::Region => return Boundary::Region,
                FlowCtx::Loop { .. } => {}
            }
        }
        Boundary::Function
    }
    fn boundary_depth(&self, i: usize) -> u16 {
        match &self.flows[i] {
            FlowCtx::Loop { body_depth, .. } | FlowCtx::Block { body_depth, .. } => *body_depth,
            FlowCtx::Region => 0,
        }
    }
    fn emit_run_defers(&mut self, depth: u16) {
        if self.any_defers { self.emit(Op::RunDefers(depth)); }
    }
    // flatten the visible name->slot bindings (innermost wins) for a delegated node
    fn snapshot(&self) -> Vec<(String, u16)> {
        let mut m: HashMap<String, u16> = HashMap::new();
        for sc in &self.scopes {
            for (k, v) in sc { m.insert(k.clone(), *v); }
        }
        m.into_iter().collect()
    }
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(i) = self.ctx.names.iter().position(|b| b == name) { return i as u32; }
        self.ctx.names.push(name.to_string());
        (self.ctx.names.len() - 1) as u32
    }
    fn konst(&mut self, v: Value) -> u32 { self.consts.push(v); (self.consts.len() - 1) as u32 }
    fn here(&self) -> usize { self.code.len() }
    fn emit(&mut self, op: Op) -> usize { self.code.push(op); self.code.len() - 1 }
    fn patch(&mut self, at: usize, target: usize) {
        match &mut self.code[at] {
            Op::Jump(t) | Op::JumpIfFalse(t) | Op::JumpIfTrue(t)
            | Op::IterStep(_, _, _, t) | Op::MatchTest(_, _, t) => *t = target,
            _ => unreachable!("patching a non-jump op"),
        }
    }
    fn delegate_expr(&mut self, e: &Expr) {
        let snap = self.snapshot();
        let i = self.del_exprs.len() as u32;
        self.del_exprs.push((e.clone(), snap));
        self.emit(Op::EvalAst(i));
    }

    // ---- statement blocks ----
    // Compile one nesting level. Mirrors the interpreter's exec_block: `defer`
    // statements are collected and their bodies run at block exit (LIFO); we
    // register them at runtime via PushDefer and compile each body as a
    // jumped-over region AT THE END of the block, so names the block defines
    // after the defer statement still resolve (the interp looks them up late).
    fn block(&mut self, stmts: &[Stmt]) -> Result<(), String> {
        self.block_opt(stmts, true)
    }

    fn block_opt(&mut self, stmts: &[Stmt], run_own_defers: bool) -> Result<(), String> {
        self.cur_depth += 1;
        let depth = self.cur_depth;
        let mut pending: Vec<(&[Stmt], u32)> = Vec::new();
        for s in stmts {
            if let Stmt::Defer(body) = s {
                let mi = self.defer_meta.len() as u32;
                self.defer_meta.push((0, 0));
                self.any_defers = true;
                self.emit(Op::PushDefer(depth, mi));
                pending.push((body, mi));
                continue;
            }
            self.stmt(s)?;
        }
        let had_defers = !pending.is_empty();
        for (body, mi) in pending {
            let skip = self.emit(Op::Jump(0));
            let start = self.here();
            self.flows.push(FlowCtx::Region);
            // a defer body's own top-level defers are registered into a scratch
            // list and dropped, exactly like the interp's `tmp` vec
            self.block_opt(body, false)?;
            self.flows.pop();
            let end = self.here();
            self.defer_meta[mi as usize] = (start, end);
            let target = self.here();
            self.patch(skip, target);
        }
        if had_defers && run_own_defers { self.emit(Op::RunDefers(depth)); }
        self.cur_depth -= 1;
        Ok(())
    }

    // ---- statements (leave nothing on the stack) ----
    fn stmt(&mut self, s: &Stmt) -> Result<(), String> {
        match s {
            Stmt::Let { name, value, .. } => {
                self.expr(value)?;
                let slot = self.write_slot(name);
                self.emit(Op::StoreLocal(slot));
            }
            Stmt::Assign { name, value } => {
                self.expr(value)?;
                let slot = self.write_slot(name);
                self.emit(Op::StoreLocal(slot));
            }
            Stmt::Expr(e) => { self.expr(e)?; self.emit(Op::Pop); }
            Stmt::Return(opt) => {
                match opt { Some(e) => self.expr(e)?, None => { self.emit(Op::ConstNull); } }
                match self.exit_boundary() {
                    // `return` inside an expression block yields the block's
                    // value, skipping the tail (the interp consumes Flow::Return
                    // at the Expr::Block boundary)
                    Boundary::Block(i) => {
                        let d = self.boundary_depth(i);
                        self.emit_run_defers(d);
                        let j = self.emit(Op::Jump(0));
                        if let FlowCtx::Block { value_jumps, .. } = &mut self.flows[i] {
                            value_jumps.push(j);
                        }
                    }
                    // inside a try region or at function level: a real return;
                    // pending defers run at the region/frame boundary
                    _ => { self.emit(Op::Return); }
                }
            }
            Stmt::IndexAssign { base, index, value } => {
                // base[index] = value, mutating the heap value in place
                self.expr(base)?;
                self.expr(index)?;
                self.expr(value)?;
                self.emit(Op::IndexSet);
            }
            Stmt::FieldAssign { base, field, value } => {
                self.expr(base)?;
                self.expr(value)?;
                let n = self.intern(field);
                self.emit(Op::SetField(n));
            }
            Stmt::If { cond, then, els } => {
                self.expr(cond)?;
                let jf = self.emit(Op::JumpIfFalse(0));
                self.block(then)?;
                if let Some(els) = els {
                    let jend = self.emit(Op::Jump(0));
                    let lelse = self.here();
                    self.patch(jf, lelse);
                    self.block(els)?;
                    let lend = self.here();
                    self.patch(jend, lend);
                } else {
                    let lend = self.here();
                    self.patch(jf, lend);
                }
            }
            Stmt::While { cond, body } => {
                let lstart = self.here();
                self.expr(cond)?;
                let jf = self.emit(Op::JumpIfFalse(0));
                self.flows.push(FlowCtx::Loop {
                    breaks: Vec::new(), continues: Vec::new(), body_depth: self.cur_depth + 1,
                });
                self.block(body)?;
                let ctx = self.flows.pop().unwrap();
                let (breaks, continues) = loop_jumps(ctx);
                for c in continues { self.patch(c, lstart); }
                self.emit(Op::Jump(lstart));
                let lend = self.here();
                self.patch(jf, lend);
                for b in breaks { self.patch(b, lend); }
            }
            Stmt::ForRange { var, start, end, inclusive, body } => {
                self.expr(start)?;
                let cnt = self.fresh();
                self.emit(Op::StoreLocal(cnt));
                self.expr(end)?;
                let lim = self.fresh();
                self.emit(Op::StoreLocal(lim));
                let lstart = self.here();
                self.emit(Op::LoadLocal(cnt));
                self.emit(Op::LoadLocal(lim));
                self.emit(Op::Bin(if *inclusive { BinOp::Le } else { BinOp::Lt }));
                let jf = self.emit(Op::JumpIfFalse(0));
                self.flows.push(FlowCtx::Loop {
                    breaks: Vec::new(), continues: Vec::new(), body_depth: self.cur_depth + 1,
                });
                // the loop var lives in the enclosing (function-flat) scope and
                // persists after the loop, exactly like the interpreter's insert
                let vslot = self.write_slot(var);
                self.emit(Op::LoadLocal(cnt));
                self.emit(Op::StoreLocal(vslot));
                self.block(body)?;
                let ctx = self.flows.pop().unwrap();
                let (breaks, continues) = loop_jumps(ctx);
                let linc = self.here();
                for c in continues { self.patch(c, linc); }
                self.emit(Op::LoadLocal(cnt));
                let one = self.konst(Value::Int(1));
                self.emit(Op::Const(one));
                self.emit(Op::Bin(BinOp::Add));
                self.emit(Op::StoreLocal(cnt));
                self.emit(Op::Jump(lstart));
                let lend = self.here();
                self.patch(jf, lend);
                for b in breaks { self.patch(b, lend); }
            }
            Stmt::ForEach { var, iter, body } => {
                // it = iter; idx = 0; loop { var, idx = next(it, idx) or break; body }
                // IterStep fetches lazily, so even infinite generators work with `break`.
                self.expr(iter)?;
                let it = self.fresh();
                self.emit(Op::StoreLocal(it));
                let idx = self.fresh();
                let zero = self.konst(Value::Int(0));
                self.emit(Op::Const(zero));
                self.emit(Op::StoreLocal(idx));
                self.flows.push(FlowCtx::Loop {
                    breaks: Vec::new(), continues: Vec::new(), body_depth: self.cur_depth + 1,
                });
                let vslot = self.write_slot(var);
                let lstart = self.here();
                let step = self.emit(Op::IterStep(it, idx, vslot, 0));
                self.block(body)?;
                self.emit(Op::Jump(lstart));
                let ctx = self.flows.pop().unwrap();
                let (breaks, continues) = loop_jumps(ctx);
                let lend = self.here();
                self.patch(step, lend);
                for b in breaks { self.patch(b, lend); }
                for c in continues { self.patch(c, lstart); }
            }
            Stmt::Break(opt) => {
                // A loop catches it (discarding any value); an expression block
                // swallows it and falls through to its tail; a try region turns
                // it into a runtime Flow; at function level it acts like the
                // interpreter's call boundary: return the value.
                match self.break_boundary() {
                    Boundary::Loop(i) => {
                        if let Some(e) = opt { self.expr(e)?; self.emit(Op::Pop); }
                        let d = self.boundary_depth(i);
                        self.emit_run_defers(d);
                        let j = self.emit(Op::Jump(0));
                        if let FlowCtx::Loop { breaks, .. } = &mut self.flows[i] { breaks.push(j); }
                    }
                    Boundary::Block(i) => {
                        if let Some(e) = opt { self.expr(e)?; self.emit(Op::Pop); }
                        let d = self.boundary_depth(i);
                        self.emit_run_defers(d);
                        let j = self.emit(Op::Jump(0));
                        if let FlowCtx::Block { tail_jumps, .. } = &mut self.flows[i] { tail_jumps.push(j); }
                    }
                    Boundary::Region => {
                        match opt { Some(e) => self.expr(e)?, None => { self.emit(Op::ConstNull); } }
                        self.emit(Op::FlowBreak);
                    }
                    Boundary::Function => {
                        match opt { Some(e) => self.expr(e)?, None => { self.emit(Op::ConstNull); } }
                        self.emit(Op::Return);
                    }
                }
            }
            Stmt::Continue => {
                match self.break_boundary() {
                    Boundary::Loop(i) => {
                        let d = self.boundary_depth(i);
                        self.emit_run_defers(d);
                        let j = self.emit(Op::Jump(0));
                        if let FlowCtx::Loop { continues, .. } = &mut self.flows[i] { continues.push(j); }
                    }
                    Boundary::Block(i) => {
                        let d = self.boundary_depth(i);
                        self.emit_run_defers(d);
                        let j = self.emit(Op::Jump(0));
                        if let FlowCtx::Block { tail_jumps, .. } = &mut self.flows[i] { tail_jumps.push(j); }
                    }
                    Boundary::Region => { self.emit(Op::FlowContinue); }
                    Boundary::Function => {
                        // interp: Flow::Continue at the call boundary -> null
                        self.emit(Op::ConstNull);
                        self.emit(Op::Return);
                    }
                }
            }
            Stmt::Throw(e) => {
                match self.exit_boundary() {
                    // inside an expression block with no try in between, the
                    // interp DISCARDS the Flow::Throw at the block boundary and
                    // evaluates the tail — so no unwinding happens at all
                    Boundary::Block(i) => {
                        self.expr(e)?;
                        self.emit(Op::Pop);
                        let d = self.boundary_depth(i);
                        self.emit_run_defers(d);
                        let j = self.emit(Op::Jump(0));
                        if let FlowCtx::Block { tail_jumps, .. } = &mut self.flows[i] { tail_jumps.push(j); }
                    }
                    _ => {
                        self.expr(e)?;
                        self.emit(Op::Throw);
                    }
                }
            }
            Stmt::TryCatch { body, catch_var, catch_body, finally_body } => {
                // resolve where flows that survive the try must continue,
                // BEFORE the region masks the enclosing contexts
                let brk_b = self.break_boundary();
                let exit_b = self.exit_boundary();
                let mi = self.try_meta.len();
                self.try_meta.push(TryMeta {
                    body: (0, 0), catch: None, finally: None, catch_slot: None,
                    has_catch: catch_body.is_some(), body_depth: self.cur_depth + 1,
                    after: 0, brk: None, cont: None, ret: None, thr: None,
                });
                let op_at = self.emit(Op::Try(mi as u32));
                let _ = op_at;
                // catch var binds into the enclosing scope and persists, like
                // the interp's scope.insert
                let catch_slot = catch_var.as_ref().map(|v| self.write_slot(v));
                self.flows.push(FlowCtx::Region);
                let bs = self.here();
                self.block(body)?;
                let be = self.here();
                let catch = match catch_body {
                    Some(cb) => { let cs = self.here(); self.block(cb)?; Some((cs, self.here())) }
                    None => None,
                };
                let finally = match finally_body {
                    Some(fb) => { let fs = self.here(); self.block(fb)?; Some((fs, self.here())) }
                    None => None,
                };
                self.flows.pop();
                // continuation stubs (compiled in the ENCLOSING contexts)
                let brk = self.try_stub(&brk_b, StubKind::Break);
                let cont = self.try_stub(&brk_b, StubKind::Continue);
                let ret = self.try_stub(&exit_b, StubKind::Return);
                let thr = self.try_stub(&exit_b, StubKind::Throw);
                let after = self.here();
                let m = &mut self.try_meta[mi];
                m.body = (bs, be);
                m.catch = catch;
                m.finally = finally;
                m.catch_slot = catch_slot;
                m.after = after;
                m.brk = brk;
                m.cont = cont;
                m.ret = ret;
                m.thr = thr;
            }
            Stmt::Defer(body) => {
                // only reachable when a defer is executed outside a block list;
                // the interp fallback runs it immediately
                self.block(body)?;
            }
            _ => return Err("vm: unsupported statement (internal: should be interp-only)".into()),
        }
        Ok(())
    }

    // Emit the continuation stub for one flow kind leaving a try. The Op::Try
    // handler pushes the flow's value (Break/Return/Throw) and jumps here.
    fn try_stub(&mut self, b: &Boundary, kind: StubKind) -> Option<usize> {
        match b {
            Boundary::Loop(i) => {
                let i = *i;
                match kind {
                    StubKind::Break => {
                        let at = self.here();
                        self.emit(Op::Pop); // loops discard the break value
                        let d = self.boundary_depth(i);
                        self.emit_run_defers(d);
                        let j = self.emit(Op::Jump(0));
                        if let FlowCtx::Loop { breaks, .. } = &mut self.flows[i] { breaks.push(j); }
                        Some(at)
                    }
                    StubKind::Continue => {
                        let at = self.here();
                        let d = self.boundary_depth(i);
                        self.emit_run_defers(d);
                        let j = self.emit(Op::Jump(0));
                        if let FlowCtx::Loop { continues, .. } = &mut self.flows[i] { continues.push(j); }
                        Some(at)
                    }
                    // loops pass return/throw through; resolved by exit_boundary
                    StubKind::Return | StubKind::Throw => None,
                }
            }
            Boundary::Block(i) => {
                let i = *i;
                let at = self.here();
                let d = self.boundary_depth(i);
                match kind {
                    StubKind::Break | StubKind::Throw => {
                        self.emit(Op::Pop); // value discarded, fall through to tail
                        self.emit_run_defers(d);
                        let j = self.emit(Op::Jump(0));
                        if let FlowCtx::Block { tail_jumps, .. } = &mut self.flows[i] { tail_jumps.push(j); }
                    }
                    StubKind::Continue => {
                        self.emit_run_defers(d);
                        let j = self.emit(Op::Jump(0));
                        if let FlowCtx::Block { tail_jumps, .. } = &mut self.flows[i] { tail_jumps.push(j); }
                    }
                    StubKind::Return => {
                        // value stays: the block yields it, skipping the tail
                        self.emit_run_defers(d);
                        let j = self.emit(Op::Jump(0));
                        if let FlowCtx::Block { value_jumps, .. } = &mut self.flows[i] { value_jumps.push(j); }
                    }
                }
                Some(at)
            }
            // propagate the Flow out of this run (an outer try or the frame
            // boundary picks it up)
            Boundary::Region | Boundary::Function => None,
        }
    }

    // ---- expressions (leave exactly one value on the stack) ----
    fn expr(&mut self, e: &Expr) -> Result<(), String> {
        match e {
            Expr::At { pos, expr } => {
                // keep the interp's cur_pos in sync so located runtime errors
                // ("line X, col Y: ...") come out byte-identical to `nova run`
                self.emit(Op::Pos(pos.0, pos.1));
                self.expr(expr)?;
            }
            Expr::Int(n) => { let k = self.konst(Value::Int(*n)); self.emit(Op::Const(k)); }
            Expr::Float(x) => { let k = self.konst(Value::Float(*x)); self.emit(Op::Const(k)); }
            Expr::Str(s) => { let k = self.konst(Value::Str(s.clone())); self.emit(Op::Const(k)); }
            Expr::Bool(b) => { self.emit(Op::ConstBool(*b)); }
            Expr::Null => { self.emit(Op::ConstNull); }
            Expr::BigIntLit(s) => {
                use std::str::FromStr;
                let b = num_bigint::BigInt::from_str(s)
                    .map_err(|_| format!("vm: bad big-integer literal {}", s))?;
                let k = self.konst(norm_big(b));
                self.emit(Op::Const(k));
            }
            Expr::Ident(name) => {
                // a local resolves to a slot; anything else (global const, ...) is
                // delegated so the interpreter resolves it.
                match self.resolve(name) {
                    Some(slot) => { self.emit(Op::LoadLocal(slot)); }
                    None => self.delegate_expr(e),
                }
            }
            Expr::Unary { op, expr } => { self.expr(expr)?; self.emit(Op::Un(*op)); }
            Expr::Binary { op, lhs, rhs } => match op {
                BinOp::And => {
                    self.expr(lhs)?;
                    let jf = self.emit(Op::JumpIfFalse(0));
                    self.expr(rhs)?;
                    self.emit(Op::Truthy);
                    let jend = self.emit(Op::Jump(0));
                    let lf = self.here(); self.patch(jf, lf);
                    self.emit(Op::ConstBool(false));
                    let le = self.here(); self.patch(jend, le);
                }
                BinOp::Or => {
                    self.expr(lhs)?;
                    let jt = self.emit(Op::JumpIfTrue(0));
                    self.expr(rhs)?;
                    self.emit(Op::Truthy);
                    let jend = self.emit(Op::Jump(0));
                    let lt = self.here(); self.patch(jt, lt);
                    self.emit(Op::ConstBool(true));
                    let le = self.here(); self.patch(jend, le);
                }
                _ => { self.expr(lhs)?; self.expr(rhs)?; self.emit(Op::Bin(*op)); }
            },
            Expr::If { cond, then, els } => {
                self.expr(cond)?;
                let jf = self.emit(Op::JumpIfFalse(0));
                self.expr(then)?;
                let jend = self.emit(Op::Jump(0));
                let lelse = self.here(); self.patch(jf, lelse);
                self.expr(els)?;
                let lend = self.here(); self.patch(jend, lend);
            }
            Expr::Block { stmts, tail } => {
                // The interp evaluates a block expression over a CLONE of the
                // scope and consumes every Flow at this boundary: Return yields
                // the block's value (tail skipped); Break/Continue/Throw are
                // discarded and the tail still runs. Writes shadow, reads pass.
                self.enter();
                self.barriers.push(self.scopes.len() - 1);
                self.flows.push(FlowCtx::Block {
                    tail_jumps: Vec::new(), value_jumps: Vec::new(),
                    body_depth: self.cur_depth + 1,
                });
                self.block(stmts)?;
                let (tails, values) = match self.flows.pop() {
                    Some(FlowCtx::Block { tail_jumps, value_jumps, .. }) => (tail_jumps, value_jumps),
                    _ => unreachable!("block context expected"),
                };
                let ltail = self.here();
                for j in tails { self.patch(j, ltail); }
                match tail { Some(t) => self.expr(t)?, None => { self.emit(Op::ConstNull); } }
                let lend = self.here();
                for j in values { self.patch(j, lend); }
                self.barriers.pop();
                self.exit();
            }
            Expr::Call { callee, args } => {
                if args.len() > u8::MAX as usize {
                    return Err("vm: too many call arguments".into());
                }
                let argc = args.len() as u8;
                // A call whose callee is a local holds a closure value: load it and
                // call it natively as a value.
                if let Some(slot) = self.resolve(callee) {
                    self.emit(Op::LoadLocal(slot));
                    for a in args { self.expr(a)?; }
                    self.emit(Op::CallValue(argc));
                    return Ok(());
                }
                // Otherwise evaluate the arguments natively and dispatch: a compiled
                // top-level function goes through the fast `Call`; everything else
                // (builtins, stdlib, enum variants, struct/machine constructors,
                // interp-only functions) goes through `CallDyn` -> call_named.
                for a in args { self.expr(a)?; }
                if let Some(idx) = self.ctx.index.get(callee).copied() {
                    self.emit(Op::Call(idx as u32, argc));
                } else {
                    let n = self.intern(callee);
                    self.emit(Op::CallDyn(n, argc));
                }
            }
            // --- native heap literals & reads (Phase 3A) ---
            Expr::Array(elems) => {
                if elems.len() > u32::MAX as usize { return Err("vm: array literal too large".into()); }
                for el in elems { self.expr(el)?; }
                self.emit(Op::MakeArray(elems.len() as u32));
            }
            Expr::MapLit(entries) => {
                for (k, v) in entries { self.expr(k)?; self.expr(v)?; }
                self.emit(Op::MakeMap(entries.len() as u32));
            }
            Expr::SetLit(elems) => {
                for el in elems { self.expr(el)?; }
                self.emit(Op::MakeSet(elems.len() as u32));
            }
            Expr::RangeLit { lo: Some(lo), hi: Some(hi), inclusive } => {
                // concrete-bounds range materializes to an array natively;
                // open-ended ranges (used only as slice indices) are delegated.
                self.expr(lo)?;
                self.expr(hi)?;
                self.emit(Op::MakeRange(*inclusive));
            }
            Expr::Index { base, index } => {
                // a range index slices; a plain index reads an element/char/key.
                if let Expr::RangeLit { lo, hi, inclusive } = &**index {
                    self.expr(base)?;
                    if let Some(lo) = lo { self.expr(lo)?; }
                    if let Some(hi) = hi { self.expr(hi)?; }
                    self.emit(Op::Slice(lo.is_some(), hi.is_some(), *inclusive));
                } else {
                    self.expr(base)?;
                    self.expr(index)?;
                    self.emit(Op::Index);
                }
            }
            Expr::Field { base, field } => {
                self.expr(base)?;
                let n = self.intern(field);
                self.emit(Op::GetField(n));
            }
            Expr::SafeField { base, field } => {
                self.expr(base)?;
                let n = self.intern(field);
                self.emit(Op::SafeField(n));
            }
            Expr::StructLit { name, fields } => {
                let mut field_idxs = Vec::with_capacity(fields.len());
                for (fname, fexpr) in fields {
                    self.expr(fexpr)?;
                    field_idxs.push(self.intern(fname));
                }
                let n = self.intern(name);
                self.emit(Op::MakeStruct(n, field_idxs));
            }
            Expr::FmtStr(parts) => {
                if parts.len() > u32::MAX as usize { return Err("vm: f-string too large".into()); }
                for part in parts {
                    match part {
                        FmtPart::Lit(t) => { let k = self.konst(Value::Str(t.clone())); self.emit(Op::Const(k)); }
                        FmtPart::Expr(ex) => self.expr(ex)?,
                    }
                }
                self.emit(Op::Fmt(parts.len() as u32));
            }
            Expr::MethodCall { base, method, args } => {
                // `m.sqrt(..)` where `m` is a module alias (not a local) is not a
                // real receiver call — let the interpreter resolve it. Otherwise
                // evaluate receiver + args natively and dispatch.
                let module_alias = matches!(&**base, Expr::Ident(n) if self.resolve(n).is_none());
                if module_alias || args.len() > u8::MAX as usize {
                    self.delegate_expr(e);
                } else {
                    self.expr(base)?;
                    for a in args { self.expr(a)?; }
                    let n = self.intern(method);
                    self.emit(Op::Method(n, args.len() as u8));
                }
            }
            Expr::Match { scrutinee, arms } => {
                // scrutinee -> a slot; each arm: test pattern (binding into slots),
                // optional guard, then body; fall through to the next arm on miss.
                self.expr(scrutinee)?;
                let scrut = self.fresh();
                self.emit(Op::StoreLocal(scrut));
                let mut ends: Vec<usize> = Vec::new();
                for arm in arms {
                    self.enter();
                    let vars = pattern_vars(&arm.pattern);
                    let map: Vec<(String, u16)> =
                        vars.iter().map(|n| (n.clone(), self.define(n))).collect();
                    let ti = self.match_tests.len() as u32;
                    self.match_tests.push((arm.pattern.clone(), map));
                    let test = self.emit(Op::MatchTest(ti, scrut, 0));
                    if let Some(g) = &arm.guard {
                        self.expr(g)?;
                        let jf = self.emit(Op::JumpIfFalse(0));
                        self.expr(&arm.body)?;
                        ends.push(self.emit(Op::Jump(0)));
                        let next = self.here();
                        self.patch(test, next);
                        self.patch(jf, next);
                    } else {
                        self.expr(&arm.body)?;
                        ends.push(self.emit(Op::Jump(0)));
                        let next = self.here();
                        self.patch(test, next);
                    }
                    self.exit();
                }
                self.emit(Op::NoMatch);
                let end = self.here();
                for j in ends { self.patch(j, end); }
            }
            Expr::Lambda { params, body } => {
                // Compile the lambda body to its own chunk when it is VM-native;
                // otherwise delegate the whole lambda to the interpreter.
                if !lambda_compilable(body) {
                    self.delegate_expr(e);
                    return Ok(());
                }
                // capture the whole enclosing frame by name (mirrors the
                // interpreter's `captured: scope.clone()`); the child chunk gives
                // these names slots n_params..n_params+E.
                let captures = self.snapshot();
                let capture_names: Vec<String> = captures.iter().map(|(n, _)| n.clone()).collect();
                let chunk_id = self.ctx.next_chunk_id;
                self.ctx.next_chunk_id += 1;
                self.ctx.jobs.push_back(LambdaJob {
                    chunk_id,
                    params: params.clone(),
                    body: (**body).clone(),
                    capture_names,
                });
                let ti = self.ctx.lambdas.len() as u32;
                self.ctx.lambdas.push(LambdaInfo {
                    chunk_id,
                    params: params.clone(),
                    body: (**body).clone(),
                    captures,
                });
                self.emit(Op::MakeClosure(ti));
            }
            Expr::CallValue { callee, args } => {
                if args.len() > u8::MAX as usize {
                    return Err("vm: too many call arguments".into());
                }
                self.expr(callee)?;
                for a in args { self.expr(a)?; }
                self.emit(Op::CallValue(args.len() as u8));
            }
            // everything else (comprehensions, channels, async, ...) is delegated
            // to the interpreter — sound because eval is read-only.
            _ => self.delegate_expr(e),
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VM
// ---------------------------------------------------------------------------

pub fn run(c: &Compiled, interp: &Interp) -> Result<(), String> {
    eval_main(c, interp).map(|_| ())
}

pub fn run_jit(c: &Compiled, interp: &Interp, jit: &crate::jit::Jit) -> Result<(), String> {
    eval_main_jit(c, interp, Some(jit)).map(|_| ())
}

pub fn run_tiered(c: &Compiled, interp: &Interp, t: &crate::jit::TieredJit) -> Result<(), String> {
    eval_main_tiered(c, interp, t).map(|_| ())
}

pub fn eval_main(c: &Compiled, interp: &Interp) -> Result<Value, String> {
    eval_main_opts(c, interp, None, None)
}

pub fn eval_main_jit(c: &Compiled, interp: &Interp, jit: Option<&crate::jit::Jit>) -> Result<Value, String> {
    eval_main_opts(c, interp, jit, None)
}

pub fn eval_main_tiered(c: &Compiled, interp: &Interp, t: &crate::jit::TieredJit) -> Result<Value, String> {
    eval_main_opts(c, interp, None, Some(t))
}

fn eval_main_opts(c: &Compiled, interp: &Interp, jit: Option<&crate::jit::Jit>,
                  tiered: Option<&crate::jit::TieredJit>) -> Result<Value, String> {
    interp.init_consts()?;
    let vm = Vm::new(c, interp, jit, tiered);
    let main = &c.chunks[c.main];
    let result = vm.exec(main, vec![Value::Null; main.n_locals]);
    // Drive any still-queued fire-and-forget tasks to completion, exactly as the
    // tree-walking interpreter does after `main` returns.
    interp.drain_tasks()?;
    // top-level conversion, byte-identical to Interp::run
    match result {
        Err(e) if e == THROW_SENTINEL => {
            let v = interp.take_pending_throw();
            Err(interp.locate(format!("uncaught exception: {}", v)))
        }
        Err(e) => Err(interp.locate(e)),
        ok => ok,
    }
}

// ---------------------------------------------------------------------------
// Optimizer: jump-threading + reachability-based dead-code elimination, then
// compaction. Length-changing transforms write `Nop`s first; `compact` drops
// them and remaps every jump target, so the result is verifiably equivalent
// (the test suite compares optimized vs unoptimized output, and the example
// sweep checks both against `nova run`).
// ---------------------------------------------------------------------------

// Names a pattern binds, in first-seen order (no duplicates). Must match exactly
// what `Interp::match_pattern` inserts into its bindings scope.
fn pattern_vars(p: &Pattern) -> Vec<String> {
    let mut out = Vec::new();
    collect_pattern_vars(p, &mut out);
    out
}

fn collect_pattern_vars(p: &Pattern, out: &mut Vec<String>) {
    let add = |name: &str, out: &mut Vec<String>| {
        if !out.iter().any(|n| n == name) { out.push(name.to_string()); }
    };
    match p {
        Pattern::Binding(name) => add(name, out),
        Pattern::EnumVariant { sub, .. } | Pattern::Tuple(sub) | Pattern::Or(sub) => {
            for s in sub { collect_pattern_vars(s, out); }
        }
        Pattern::Struct { fields, .. } => {
            for (_, sp) in fields { collect_pattern_vars(sp, out); }
        }
        Pattern::Slice { prefix, rest, suffix } => {
            for s in prefix { collect_pattern_vars(s, out); }
            if let Some(Some(name)) = rest { add(name, out); }
            for s in suffix { collect_pattern_vars(s, out); }
        }
        _ => {}
    }
}

fn jump_target(op: &Op) -> Option<usize> {
    match op {
        Op::Jump(t) | Op::JumpIfFalse(t) | Op::JumpIfTrue(t)
        | Op::IterStep(_, _, _, t) | Op::MatchTest(_, _, t)
        | Op::LocalsBinJf(_, _, _, t) => Some(*t),
        _ => None,
    }
}

fn jump_target_mut(op: &mut Op) -> Option<&mut usize> {
    match op {
        Op::Jump(t) | Op::JumpIfFalse(t) | Op::JumpIfTrue(t)
        | Op::IterStep(_, _, _, t) | Op::MatchTest(_, _, t)
        | Op::LocalsBinJf(_, _, _, t) => Some(t),
        _ => None,
    }
}

fn optimize(code: Vec<Op>) -> Vec<Op> {
    fuse(compact(dead_code_to_nops(thread_jumps(code))))
}

// Peephole superinstructions over the compacted stream. Each rewrite collapses
// an exact op sequence into one op with identical semantics (same eval_binop,
// same error text, same store timing), so the differential suite can't tell
// them apart. A window is only fused when no jump lands inside it.
fn fuse(code: Vec<Op>) -> Vec<Op> {
    use std::collections::HashSet;
    let targets: HashSet<usize> = code.iter().filter_map(jump_target).collect();
    let mut out = code;
    let n = out.len();
    let mut i = 0;
    while i < n {
        // Pos a; Pos b  ->  Pos b (nothing observable between two markers)
        if i + 1 < n && !targets.contains(&(i + 1)) {
            if let (Op::Pos(..), Op::Pos(..)) = (&out[i], &out[i + 1]) {
                out[i] = Op::Nop;
                i += 1;
                continue;
            }
        }
        if i + 3 < n
            && !targets.contains(&(i + 1))
            && !targets.contains(&(i + 2))
            && !targets.contains(&(i + 3))
        {
            // LoadLocal s; Const k; Bin op; StoreLocal s  (e.g. `i = i + 1`)
            if let (Op::LoadLocal(a), Op::Const(k), Op::Bin(op), Op::StoreLocal(b)) =
                (&out[i], &out[i + 1], &out[i + 2], &out[i + 3])
            {
                if a == b {
                    out[i] = Op::IncLocal(*a, *k, *op);
                    out[i + 1] = Op::Nop;
                    out[i + 2] = Op::Nop;
                    out[i + 3] = Op::Nop;
                    i += 4;
                    continue;
                }
            }
            // LoadLocal a; LoadLocal b; Bin op; JumpIfFalse t  (loop headers)
            if let (Op::LoadLocal(a), Op::LoadLocal(b), Op::Bin(op), Op::JumpIfFalse(t)) =
                (&out[i], &out[i + 1], &out[i + 2], &out[i + 3])
            {
                out[i] = Op::LocalsBinJf(*a, *b, *op, *t);
                out[i + 1] = Op::Nop;
                out[i + 2] = Op::Nop;
                out[i + 3] = Op::Nop;
                i += 4;
                continue;
            }
        }
        // 3-op operand fusions (tried after the 4-op patterns above, so those win
        // when they apply). These evaluate a binary op straight from locals/consts
        // with no intermediate stack traffic — the bulk of arithmetic in loops.
        if i + 2 < n && !targets.contains(&(i + 1)) && !targets.contains(&(i + 2)) {
            if let (Op::LoadLocal(a), Op::LoadLocal(b), Op::Bin(op)) =
                (&out[i], &out[i + 1], &out[i + 2])
            {
                out[i] = Op::BinLL(*a, *b, *op);
                out[i + 1] = Op::Nop;
                out[i + 2] = Op::Nop;
                i += 3;
                continue;
            }
            if let (Op::LoadLocal(a), Op::Const(k), Op::Bin(op)) =
                (&out[i], &out[i + 1], &out[i + 2])
            {
                out[i] = Op::BinLC(*a, *k, *op);
                out[i + 1] = Op::Nop;
                out[i + 2] = Op::Nop;
                i += 3;
                continue;
            }
        }
        i += 1;
    }
    compact(out)
}

// Point every jump that lands on an unconditional `Jump` straight at its final
// destination (bounded to avoid cycles).
fn thread_jumps(mut code: Vec<Op>) -> Vec<Op> {
    let n = code.len();
    let resolved: Vec<Option<usize>> = (0..n).map(|i| {
        jump_target(&code[i]).map(|mut t| {
            let mut steps = 0;
            while steps < n {
                match &code[t] {
                    Op::Jump(t2) if *t2 != t => { t = *t2; steps += 1; }
                    _ => break,
                }
            }
            t
        })
    }).collect();
    for i in 0..n {
        if let (Some(slot), Some(t)) = (jump_target_mut(&mut code[i]), resolved[i]) { *slot = t; }
    }
    code
}

// Replace every instruction not reachable from ip 0 with `Nop`.
fn dead_code_to_nops(mut code: Vec<Op>) -> Vec<Op> {
    let mut seen = vec![false; code.len()];
    let mut work = vec![0usize];
    while let Some(ip) = work.pop() {
        if ip >= code.len() || seen[ip] { continue; }
        seen[ip] = true;
        match &code[ip] {
            Op::Return | Op::NoMatch => {}
            Op::Jump(t) => work.push(*t),
            Op::JumpIfFalse(t) | Op::JumpIfTrue(t)
            | Op::IterStep(_, _, _, t) | Op::MatchTest(_, _, t) => {
                work.push(*t); work.push(ip + 1);
            }
            _ => work.push(ip + 1),
        }
    }
    for (i, reachable) in seen.into_iter().enumerate() {
        if !reachable { code[i] = Op::Nop; }
    }
    code
}

// Drop `Nop`s and remap all jump targets to the new indices.
fn compact(code: Vec<Op>) -> Vec<Op> {
    let mut map = vec![0usize; code.len() + 1];
    let mut n = 0;
    for (i, op) in code.iter().enumerate() {
        map[i] = n;
        if !matches!(op, Op::Nop) { n += 1; }
    }
    map[code.len()] = n;
    let mut out = Vec::with_capacity(n);
    for mut op in code {
        if matches!(op, Op::Nop) { continue; }
        if let Some(slot) = jump_target_mut(&mut op) { *slot = map[*slot]; }
        out.push(op);
    }
    out
}

// ---------------------------------------------------------------------------
// Disassembler (`nova disasm`) and a bytecode verifier (used in tests).
// ---------------------------------------------------------------------------

pub fn disassemble(c: &Compiled) -> String {
    let mut out = String::new();
    for (ci, ch) in c.chunks.iter().enumerate() {
        let marker = if ci == c.main { "  (main)" } else { "" };
        out.push_str(&format!(
            "chunk {} `{}`{}   params={} locals={}\n", ci, ch.name, marker, ch.n_params, ch.n_locals));
        for (i, op) in ch.code.iter().enumerate() {
            out.push_str(&format!("  {:>4}  {}\n", i, fmt_op(op, ch, c)));
        }
        if !ch.consts.is_empty() {
            out.push_str("  consts:");
            for (k, v) in ch.consts.iter().enumerate() {
                out.push_str(&format!(" [{}]={}", k, v));
            }
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

fn fmt_op(op: &Op, ch: &Chunk, c: &Compiled) -> String {
    match op {
        Op::Const(i) => format!("Const {} = {}", i, ch.consts[*i as usize]),
        Op::LoadLocal(s) => format!("LoadLocal {}", s),
        Op::StoreLocal(s) => format!("StoreLocal {}", s),
        Op::Bin(o) => format!("Bin {:?}", o),
        Op::Un(o) => format!("Un {:?}", o),
        Op::MakeArray(n) => format!("MakeArray {}", n),
        Op::MakeMap(n) => format!("MakeMap {}", n),
        Op::MakeSet(n) => format!("MakeSet {}", n),
        Op::MakeRange(inc) => format!("MakeRange inclusive={}", inc),
        Op::Slice(lo, hi, inc) => format!("Slice lo={} hi={} inclusive={}", lo, hi, inc),
        Op::GetField(n) => format!("GetField .{}", c.names[*n as usize]),
        Op::SafeField(n) => format!("SafeField ?.{}", c.names[*n as usize]),
        Op::SetField(n) => format!("SetField .{}", c.names[*n as usize]),
        Op::MakeStruct(n, fs) => {
            let fields: Vec<&str> = fs.iter().map(|i| c.names[*i as usize].as_str()).collect();
            format!("MakeStruct {} {{{}}}", c.names[*n as usize], fields.join(", "))
        }
        Op::Fmt(n) => format!("Fmt {}", n),
        Op::IterStep(it, idx, var, end) => format!("IterStep it={} idx={} var={} -> {}", it, idx, var, end),
        Op::Jump(t) => format!("Jump -> {}", t),
        Op::JumpIfFalse(t) => format!("JumpIfFalse -> {}", t),
        Op::JumpIfTrue(t) => format!("JumpIfTrue -> {}", t),
        Op::Call(idx, argc) => format!("Call {} `{}` argc={}", idx, c.chunks[*idx as usize].name, argc),
        Op::CallDyn(idx, argc) => format!("CallDyn `{}` argc={}", c.names[*idx as usize], argc),
        Op::EvalAst(i) => format!("EvalAst #{}", i),
        other => format!("{:?}", other),
    }
}

// Sanity-check a compiled program: jump targets, local slots, and try/defer
// code ranges in range. Catches compiler/optimizer bugs; used by the test suite.
#[cfg(test)]
pub fn verify(c: &Compiled) -> Result<(), String> {
    for ch in &c.chunks {
        let len = ch.code.len();
        for (i, op) in ch.code.iter().enumerate() {
            if let Some(t) = jump_target(op) {
                if t > len { return Err(format!("`{}`@{}: jump target {} out of range {}", ch.name, i, t, len)); }
            }
            let slot = match op {
                Op::LoadLocal(s) | Op::StoreLocal(s) | Op::IncLocal(s, _, _)
                | Op::BinLC(s, _, _) => Some(*s as usize),
                Op::IterStep(it, idx, var, _) => Some((*it).max(*idx).max(*var) as usize),
                Op::LocalsBinJf(a, b, _, _) | Op::BinLL(a, b, _) => Some((*a).max(*b) as usize),
                _ => None,
            };
            if let Some(s) = slot {
                if s >= ch.n_locals { return Err(format!("`{}`@{}: local slot {} >= {}", ch.name, i, s, ch.n_locals)); }
            }
            if let Op::PushDefer(_, mi) = op {
                if *mi as usize >= ch.defer_meta.len() {
                    return Err(format!("`{}`@{}: defer meta {} out of range", ch.name, i, mi));
                }
            }
            if let Op::Try(mi) = op {
                if *mi as usize >= ch.try_meta.len() {
                    return Err(format!("`{}`@{}: try meta {} out of range", ch.name, i, mi));
                }
            }
        }
        let check_range = |what: &str, (s, e): (usize, usize)| -> Result<(), String> {
            if s > e || e > len {
                return Err(format!("`{}`: {} range {}..{} out of range {}", ch.name, what, s, e, len));
            }
            Ok(())
        };
        for m in &ch.try_meta {
            check_range("try body", m.body)?;
            if let Some(r) = m.catch { check_range("catch", r)?; }
            if let Some(r) = m.finally { check_range("finally", r)?; }
            if m.after > len { return Err(format!("`{}`: try after {} out of range", ch.name, m.after)); }
            if let Some(slot) = m.catch_slot {
                if slot as usize >= ch.n_locals {
                    return Err(format!("`{}`: catch slot {} >= {}", ch.name, slot, ch.n_locals));
                }
            }
        }
        for r in &ch.defer_meta { check_range("defer body", *r)?; }
    }
    Ok(())
}

// Pop an integer off the value stack, matching the interpreter's `eval_int`
// error text for non-integers (range bounds must be integers).
fn pop_int(stack: &mut Vec<Value>) -> Result<i64, String> {
    match stack.pop().unwrap() {
        Value::Int(n) => Ok(n),
        other => Err(format!("expected integer, got {}", other.type_name())),
    }
}

struct Vm<'a> {
    c: &'a Compiled,
    interp: &'a Interp,
    jit: Option<&'a crate::jit::Jit>,               // eager JIT (--jit)
    tiered: Option<&'a crate::jit::TieredJit<'a>>,  // tiered JIT (default)
    call_counts: RefCell<Vec<u64>>,                 // per-chunk, for tiering
    // free-lists of reusable buffers so calls don't allocate a fresh operand
    // stack / locals frame each time (big win for recursion-heavy code).
    stack_pool: RefCell<Vec<Vec<Value>>>,
    locals_pool: RefCell<Vec<Vec<Value>>>,
}

impl<'a> Vm<'a> {
    fn new(c: &'a Compiled, interp: &'a Interp,
           jit: Option<&'a crate::jit::Jit>,
           tiered: Option<&'a crate::jit::TieredJit<'a>>) -> Self {
        Vm {
            c, interp, jit, tiered,
            call_counts: RefCell::new(vec![0; c.chunks.len()]),
            stack_pool: RefCell::new(Vec::new()),
            locals_pool: RefCell::new(Vec::new()),
        }
    }

    // build an interpreter Scope from the current locals using a delegation snapshot
    fn scope_from(&self, snap: &[(String, u16)], locals: &[Value]) -> Scope {
        let mut s = Scope::with_capacity(snap.len());
        for (name, slot) in snap { s.insert(name.clone(), locals[*slot as usize].clone()); }
        s
    }

    // Run a chunk over an owned locals frame, returning both buffers to their
    // pools afterwards (on success or error) so they can be reused. This is the
    // frame boundary: it converts the escaping Flow exactly like the tree-
    // walker's call_function (Break/Return -> value, Continue -> null, Throw ->
    // park + sentinel), and runs any still-pending defers first.
    fn exec(&self, chunk: &Chunk, mut locals: Vec<Value>) -> Result<Value, String> {
        let mut stack = self.stack_pool.borrow_mut().pop().unwrap_or_default();
        stack.clear();
        let mut defers: Vec<DeferEntry> = Vec::new();
        let flow = self.run_flow(chunk, &mut locals, &mut stack, &mut defers, 0, chunk.code.len());
        if !defers.is_empty() {
            self.run_defers(chunk, &mut locals, &mut stack, &mut defers, 0);
        }
        let r = match flow {
            Ok(Flow::Normal) => Ok(stack.pop().unwrap_or(Value::Null)),
            Ok(Flow::Return(v)) | Ok(Flow::Break(v)) => Ok(v),
            Ok(Flow::Continue) => Ok(Value::Null),
            Ok(Flow::Throw(v)) => {
                self.interp.park_throw(v);
                Err(THROW_SENTINEL.to_string())
            }
            Err(e) => Err(e),
        };
        stack.clear();
        self.stack_pool.borrow_mut().push(stack);
        locals.clear();
        self.locals_pool.borrow_mut().push(locals);
        r
    }

    // Pop and run pending defers registered at depth >= min_depth, most recent
    // first — the interpreter's LIFO block-exit order. A defer body's outcome
    // (flows, errors) is deliberately swallowed, and defers it registers itself
    // go into a scratch list that is dropped, both matching exec_block.
    fn run_defers(&self, chunk: &Chunk, locals: &mut Vec<Value>, stack: &mut Vec<Value>,
                  defers: &mut Vec<DeferEntry>, min_depth: u16)
    {
        while defers.last().map_or(false, |d| d.depth >= min_depth) {
            let d = defers.pop().unwrap();
            let (s, e) = chunk.defer_meta[d.meta];
            let base = stack.len();
            let mut scratch: Vec<DeferEntry> = Vec::new();
            let _ = self.run_flow(chunk, locals, stack, &mut scratch, s, e);
            stack.truncate(base);
        }
    }

    // Call a closure value: run its compiled chunk if the VM built one (filling
    // capture slots from the closure's captured environment), else fall back to
    // the interpreter (closures it created, or builtins that returned one).
    fn call_value(&self, callee: Value, args: Vec<Value>) -> Result<Value, String> {
        if let Value::Closure(c) = &callee {
            if let Some(chunk_id) = c.vm_chunk {
                let chunk = &self.c.chunks[chunk_id];
                if args.len() != chunk.n_params {
                    return Err(format!("closure expects {} args, got {}", chunk.n_params, args.len()));
                }
                let mut fl = self.locals_pool.borrow_mut().pop().unwrap_or_default();
                fl.clear();
                fl.extend(args);
                for name in &chunk.captures {
                    fl.push(c.captured.get(name).cloned().unwrap_or(Value::Null));
                }
                fl.resize(chunk.n_locals, Value::Null);
                return self.exec(chunk, fl);
            }
        }
        self.interp.call_closure(&callee, args)
    }

    // Execute ops in [start, end): the whole chunk for a call frame, or a
    // body/catch/finally/defer region as a sub-run sharing the same frame.
    // Falling off the end is Flow::Normal; Return/Throw/FlowBreak/FlowContinue
    // surface as the corresponding Flow for the caller (Op::Try or exec) to
    // consume, mirroring the tree-walker's exec_block contract.
    fn run_flow(&self, chunk: &Chunk, locals: &mut Vec<Value>, stack: &mut Vec<Value>,
                defers: &mut Vec<DeferEntry>, start: usize, end: usize)
        -> Result<Flow, String>
    {
        let mut ip = start;
        while ip < end {
            match &chunk.code[ip] {
                Op::Const(i) => stack.push(chunk.consts[*i as usize].clone()),
                Op::Nop => {}
                Op::ConstNull => stack.push(Value::Null),
                Op::ConstBool(b) => stack.push(Value::Bool(*b)),
                Op::LoadLocal(s) => stack.push(locals[*s as usize].clone()),
                Op::StoreLocal(s) => { let v = stack.pop().unwrap(); locals[*s as usize] = v; }
                Op::Pop => { stack.pop(); }
                Op::Bin(op) => {
                    let r = stack.pop().unwrap(); let l = stack.pop().unwrap();
                    stack.push(eval_binop(*op, l, r)?);
                }
                Op::Un(op) => { let v = stack.pop().unwrap(); stack.push(eval_unop(*op, v)?); }
                Op::Truthy => { let v = stack.pop().unwrap(); stack.push(Value::Bool(v.is_truthy())); }
                Op::MakeArray(n) => {
                    let vals = stack.split_off(stack.len() - *n as usize);
                    stack.push(Value::Array(Rc::new(RefCell::new(vals))));
                }
                Op::MakeMap(n) => {
                    let flat = stack.split_off(stack.len() - 2 * *n as usize);
                    let mut entries = Vec::with_capacity(*n as usize);
                    let mut it = flat.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) { entries.push((k, v)); }
                    stack.push(build_map(entries));
                }
                Op::MakeSet(n) => {
                    let vals = stack.split_off(stack.len() - *n as usize);
                    stack.push(build_set(vals));
                }
                Op::MakeRange(inclusive) => {
                    let hi = pop_int(stack)?;
                    let lo = pop_int(stack)?;
                    stack.push(build_range(lo, hi, *inclusive));
                }
                Op::Index => {
                    let idx = stack.pop().unwrap();
                    let base = stack.pop().unwrap();
                    stack.push(index_get(&base, &idx)?);
                }
                Op::Slice(has_lo, has_hi, inclusive) => {
                    let hi = if *has_hi { Some(pop_int(stack)?) } else { None };
                    let lo = if *has_lo { Some(pop_int(stack)?) } else { None };
                    let base = stack.pop().unwrap();
                    stack.push(do_slice(&base, lo, hi, *inclusive)?);
                }
                Op::GetField(n) => {
                    let base = stack.pop().unwrap();
                    stack.push(field_get(&base, &self.c.names[*n as usize])?);
                }
                Op::SafeField(n) => {
                    let base = stack.pop().unwrap();
                    stack.push(safe_field_get(&base, &self.c.names[*n as usize])?);
                }
                Op::IterStep(it, idx, var, end) => {
                    let k = match &locals[*idx as usize] {
                        Value::Int(n) => *n as usize,
                        _ => 0,
                    };
                    match self.interp.vm_iter_next(&locals[*it as usize], k)? {
                        Some(item) => {
                            locals[*var as usize] = item;
                            locals[*idx as usize] = Value::Int((k + 1) as i64);
                        }
                        None => { ip = *end; continue; }
                    }
                }
                Op::Jump(t) => { ip = *t; continue; }
                Op::JumpIfFalse(t) => { let v = stack.pop().unwrap(); if !v.is_truthy() { ip = *t; continue; } }
                Op::JumpIfTrue(t) => { let v = stack.pop().unwrap(); if v.is_truthy() { ip = *t; continue; } }
                Op::Call(idx, argc) => {
                    let n = *argc as usize;
                    let callee = &self.c.chunks[*idx as usize];
                    if n != callee.n_params {
                        return Err(format!("vm: `{}` expects {} args, got {}", callee.name, callee.n_params, n));
                    }
                    // Tiered JIT (default): count calls per chunk; once a function
                    // crosses the threshold, compile its callee closure and take
                    // the native path from then on. Cold functions never compile.
                    if let Some(t) = self.tiered {
                        if !t.is_compiled(&callee.name) && t.is_eligible(&callee.name) {
                            let mut counts = self.call_counts.borrow_mut();
                            counts[*idx as usize] += 1;
                            let crossed = counts[*idx as usize] == t.threshold;
                            drop(counts);
                            // compile exactly once at the crossing; if the backend
                            // fails, TieredJit stops trying and the VM runs it all
                            if crossed { t.compile_closure(&callee.name); }
                        }
                        if t.is_compiled_f64(&callee.name) {
                            let base = stack.len() - n;
                            if stack[base..].iter().all(|v| matches!(v, Value::Float(_))) {
                                let fa: Vec<f64> = stack[base..].iter()
                                    .map(|v| if let Value::Float(x) = v { *x } else { 0.0 }).collect();
                                stack.truncate(base);
                                stack.push(Value::Float(t.raw_call_f64(&callee.name, &fa)));
                                ip += 1;
                                continue;
                            }
                        }
                        if t.is_compiled(&callee.name) {
                            let base = stack.len() - n;
                            if stack[base..].iter().all(|v| matches!(v, Value::Int(_))) {
                                let ia: Vec<i64> = stack[base..].iter()
                                    .map(|v| if let Value::Int(k) = v { *k } else { 0 }).collect();
                                stack.truncate(base);
                                let (raw, deopt) = t.raw_call(&callee.name, &ia);
                                if deopt {
                                    let mut fl = self.locals_pool.borrow_mut().pop().unwrap_or_default();
                                    fl.clear();
                                    fl.extend(ia.into_iter().map(Value::Int));
                                    fl.resize(callee.n_locals, Value::Null);
                                    stack.push(self.exec(callee, fl)?);
                                } else {
                                    stack.push(Value::Int(raw));
                                }
                                ip += 1;
                                continue;
                            }
                        }
                        // Numeric (mixed int/float) track: all-Int args like the i64
                        // track, but the raw i64 result is an integer OR f64 bits.
                        if t.is_compiled_num(&callee.name) {
                            let base = stack.len() - n;
                            if stack[base..].iter().all(|v| matches!(v, Value::Int(_))) {
                                let ia: Vec<i64> = stack[base..].iter()
                                    .map(|v| if let Value::Int(k) = v { *k } else { 0 }).collect();
                                stack.truncate(base);
                                let (raw, deopt) = t.raw_call_num(&callee.name, &ia);
                                if deopt {
                                    let mut fl = self.locals_pool.borrow_mut().pop().unwrap_or_default();
                                    fl.clear();
                                    fl.extend(ia.into_iter().map(Value::Int));
                                    fl.resize(callee.n_locals, Value::Null);
                                    stack.push(self.exec(callee, fl)?);
                                } else if t.num_ret_is_float(&callee.name) {
                                    stack.push(Value::Float(f64::from_bits(raw as u64)));
                                } else {
                                    stack.push(Value::Int(raw));
                                }
                                ip += 1;
                                continue;
                            }
                        }
                    }
                    // Eager JIT (--jit): a compiled integer-pure function called with
                    // all-integer args runs as native code; a deopt re-runs it on
                    // the VM (safe — eligible functions are pure).
                    if let Some(jit) = self.jit {
                        let base = stack.len() - n;
                        if jit.is_compiled_f64(&callee.name)
                            && stack[base..].iter().all(|v| matches!(v, Value::Float(_)))
                        {
                            let fa: Vec<f64> = stack[base..].iter()
                                .map(|v| if let Value::Float(x) = v { *x } else { 0.0 }).collect();
                            stack.truncate(base);
                            stack.push(Value::Float(jit.raw_call_f64(&callee.name, &fa)));
                            ip += 1;
                            continue;
                        }
                        if jit.is_compiled(&callee.name)
                            && stack[base..].iter().all(|v| matches!(v, Value::Int(_)))
                        {
                            let ia: Vec<i64> = stack[base..].iter()
                                .map(|v| if let Value::Int(k) = v { *k } else { 0 }).collect();
                            stack.truncate(base);
                            let (raw, deopt) = jit.raw_call(&callee.name, &ia);
                            if deopt {
                                let mut fl = self.locals_pool.borrow_mut().pop().unwrap_or_default();
                                fl.clear();
                                fl.extend(ia.into_iter().map(Value::Int));
                                fl.resize(callee.n_locals, Value::Null);
                                stack.push(self.exec(callee, fl)?);
                            } else {
                                stack.push(Value::Int(raw));
                            }
                            ip += 1;
                            continue;
                        }
                    }
                    let mut fl = self.locals_pool.borrow_mut().pop().unwrap_or_default();
                    fl.clear();
                    fl.extend(stack.drain(stack.len() - n..));
                    fl.resize(callee.n_locals, Value::Null);
                    let rv = self.exec(callee, fl)?;
                    stack.push(rv);
                }
                Op::CallDyn(idx, argc) => {
                    let n = *argc as usize;
                    let args = stack.split_off(stack.len() - n);
                    let name = &self.c.names[*idx as usize];
                    stack.push(self.interp.call_named(name, args)?);
                }
                Op::Method(idx, argc) => {
                    let n = *argc as usize;
                    let args = stack.split_off(stack.len() - n);
                    let receiver = stack.pop().unwrap();
                    let name = &self.c.names[*idx as usize];
                    stack.push(self.interp.call_method_vals(receiver, name, args)?);
                }
                Op::MakeClosure(i) => {
                    let info = &self.c.lambdas[*i as usize];
                    let mut captured = Scope::with_capacity(info.captures.len());
                    for (name, slot) in &info.captures {
                        captured.insert(name.clone(), locals[*slot as usize].clone());
                    }
                    stack.push(Value::Closure(Rc::new(crate::interp::ClosureVal {
                        params: info.params.clone(),
                        body: info.body.clone(),
                        captured,
                        vm_chunk: Some(info.chunk_id),
                    })));
                }
                Op::CallValue(argc) => {
                    let n = *argc as usize;
                    let args = stack.split_off(stack.len() - n);
                    let callee = stack.pop().unwrap();
                    stack.push(self.call_value(callee, args)?);
                }
                Op::MakeStruct(name, field_idxs) => {
                    let vals = stack.split_off(stack.len() - field_idxs.len());
                    let fields: Vec<(String, Value)> = field_idxs.iter()
                        .map(|i| self.c.names[*i as usize].clone())
                        .zip(vals.into_iter())
                        .collect();
                    stack.push(self.interp.make_struct(&self.c.names[*name as usize], fields)?);
                }
                Op::Fmt(n) => {
                    let parts = stack.split_off(stack.len() - *n as usize);
                    let mut s = String::new();
                    for p in &parts { s.push_str(&p.to_string()); }
                    stack.push(Value::Str(s));
                }
                Op::IndexSet => {
                    let v = stack.pop().unwrap();
                    let idx = stack.pop().unwrap();
                    let base = stack.pop().unwrap();
                    index_set(&base, &idx, v)?;
                }
                Op::SetField(n) => {
                    let v = stack.pop().unwrap();
                    let base = stack.pop().unwrap();
                    field_set(&base, &self.c.names[*n as usize], v)?;
                }
                Op::MatchTest(i, scrut, fail) => {
                    let (pat, binds) = &chunk.match_tests[*i as usize];
                    let mut b: Scope = Scope::new();
                    if self.interp.match_pattern(pat, &locals[*scrut as usize], &mut b)? {
                        for (name, slot) in binds {
                            locals[*slot as usize] = b.get(name).cloned().unwrap_or(Value::Null);
                        }
                    } else {
                        ip = *fail; continue;
                    }
                }
                Op::NoMatch => return Err("no match arm matched (non-exhaustive match)".into()),
                Op::EvalAst(i) => {
                    let (expr, snap) = &chunk.del_exprs[*i as usize];
                    let scope = self.scope_from(snap, &locals);
                    stack.push(self.interp.eval(expr, &scope)?);
                }
                Op::Return => return Ok(Flow::Return(stack.pop().unwrap_or(Value::Null))),
                Op::Pos(l, c) => self.interp.set_pos((*l, *c)),
                Op::Throw => return Ok(Flow::Throw(stack.pop().unwrap())),
                Op::FlowBreak => return Ok(Flow::Break(stack.pop().unwrap())),
                Op::FlowContinue => return Ok(Flow::Continue),
                Op::PushDefer(d, mi) => defers.push(DeferEntry { depth: *d, meta: *mi as usize }),
                Op::RunDefers(d) => self.run_defers(chunk, locals, stack, defers, *d),
                Op::IncLocal(s, k, op) => {
                    let cur = locals[*s as usize].clone();
                    locals[*s as usize] = eval_binop(*op, cur, chunk.consts[*k as usize].clone())?;
                }
                Op::LocalsBinJf(a, b, op, t) => {
                    let v = eval_binop(*op, locals[*a as usize].clone(), locals[*b as usize].clone())?;
                    if !v.is_truthy() { ip = *t; continue; }
                }
                Op::BinLL(a, b, op) => {
                    let v = eval_binop(*op, locals[*a as usize].clone(), locals[*b as usize].clone())?;
                    stack.push(v);
                }
                Op::BinLC(a, k, op) => {
                    let v = eval_binop(*op, locals[*a as usize].clone(), chunk.consts[*k as usize].clone())?;
                    stack.push(v);
                }
                Op::Try(mi) => {
                    // A statement-level transcription of the interpreter's
                    // Stmt::TryCatch arm, over sub-runs of this frame.
                    let m = &chunk.try_meta[*mi as usize];
                    let base = stack.len();
                    let body_res = self.run_flow(chunk, locals, stack, defers, m.body.0, m.body.1);
                    let outcome = match body_res {
                        Ok(f) => {
                            if !matches!(f, Flow::Normal) {
                                self.run_defers(chunk, locals, stack, defers, m.body_depth);
                            }
                            f
                        }
                        Err(e) => {
                            self.run_defers(chunk, locals, stack, defers, m.body_depth);
                            if e == YIELD_STOP {
                                return Err(e); // generator unwinding passes through
                            } else if e == THROW_SENTINEL {
                                Flow::Throw(self.interp.take_pending_throw())
                            } else if m.has_catch {
                                // a runtime error is catchable when a handler exists
                                Flow::Throw(Value::Str(e))
                            } else {
                                if let Some((fs, fe)) = m.finally {
                                    // interp: exec_block(fin)? — flows discarded,
                                    // errors propagate
                                    self.run_flow(chunk, locals, stack, defers, fs, fe)?;
                                }
                                return Err(e);
                            }
                        }
                    };
                    stack.truncate(base);
                    let result = match outcome {
                        Flow::Throw(err) => {
                            if m.has_catch {
                                if let Some(slot) = m.catch_slot {
                                    locals[slot as usize] = err;
                                }
                                let (cs, ce) = m.catch.unwrap();
                                match self.run_flow(chunk, locals, stack, defers, cs, ce) {
                                    Ok(f) => f,
                                    Err(e) => {
                                        self.run_defers(chunk, locals, stack, defers, m.body_depth);
                                        if let Some((fs, fe)) = m.finally {
                                            self.run_flow(chunk, locals, stack, defers, fs, fe)?;
                                        }
                                        return Err(e);
                                    }
                                }
                            } else {
                                Flow::Normal // no handler — swallow; finally still runs
                            }
                        }
                        other => other,
                    };
                    stack.truncate(base);
                    let mut result = result;
                    if let Some((fs, fe)) = m.finally {
                        match self.run_flow(chunk, locals, stack, defers, fs, fe) {
                            Ok(Flow::Normal) => {}
                            Ok(f) => result = f, // return/throw in finally wins
                            Err(e) => {
                                self.run_defers(chunk, locals, stack, defers, m.body_depth);
                                return Err(e);
                            }
                        }
                        stack.truncate(base);
                    }
                    match result {
                        Flow::Normal => { ip = m.after; continue; }
                        Flow::Return(v) => match m.ret {
                            Some(t) => { stack.push(v); ip = t; continue; }
                            None => return Ok(Flow::Return(v)),
                        },
                        Flow::Break(v) => match m.brk {
                            Some(t) => { stack.push(v); ip = t; continue; }
                            None => return Ok(Flow::Break(v)),
                        },
                        Flow::Continue => match m.cont {
                            Some(t) => { ip = t; continue; }
                            None => return Ok(Flow::Continue),
                        },
                        Flow::Throw(v) => match m.thr {
                            Some(t) => { stack.push(v); ip = t; continue; }
                            None => return Ok(Flow::Throw(v)),
                        },
                    }
                }
            }
            ip += 1;
        }
        Ok(Flow::Normal)
    }
}

struct DeferEntry { depth: u16, meta: usize }

#[cfg(test)]
mod vm_tests {
    use super::{compile_program, compile_program_opt, eval_main, verify};
    use crate::parser::parse_program;
    use crate::interp::{Interp, Value};

    fn same(src: &str) -> Value {
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        let compiled = compile_program(&prog).expect("compile");
        verify(&compiled).expect("verify opt");
        let vm_val = eval_main(&compiled, &interp).expect("vm");
        let interp_val = interp.run().expect("interp run");
        assert_eq!(format!("{}", vm_val), format!("{}", interp_val), "VM != interp for: {}", src);
        // the optimizer must not change observable behavior
        let i2 = Interp::new(&prog).expect("interp");
        let unopt = eval_main(&compile_program_opt(&prog, false).expect("compile"), &i2).expect("vm-unopt");
        assert_eq!(format!("{}", vm_val), format!("{}", unopt), "opt != unopt for: {}", src);
        vm_val
    }

    #[test] fn arithmetic() { assert!(matches!(same("fn main(){ 2 + 3 * 4 }"), Value::Int(14))); }
    #[test] fn if_value() { assert!(matches!(same("fn main(){ if 1<2 {10} else {20} }"), Value::Int(10))); }
    #[test] fn loops() { same("fn main(){ total=0; for i in 1..=100 { total = total + i }; total }"); }
    #[test] fn recursion() { same("fn f(n){ if n<2 {n} else {f(n-1)+f(n-2)} } fn main(){ f(15) }"); }
    #[test] fn short_circuit() { same("fn main(){ (1<2) && (3>1) }"); same("fn main(){ false || (2==2) }"); }
    // Phase 2: delegated heap features must match the interpreter exactly
    #[test] fn arrays_and_foreach() {
        same("fn main(){ a=[1,2,3,4]; s=0; for x in a { s=s+x }; s }");
    }
    #[test] fn strings_foreach() {
        same("fn main(){ n=0; for c in \"hello\" { n=n+1 }; n }");
    }
    #[test] fn structs_and_match() {
        same("enum Opt { None, Some(i32) }\nfn main(){ x=Some(7); match x { Some(v)=>v*2, None=>0 } }");
    }
    #[test] fn closures() {
        same("fn main(){ add=(a,b)=>a+b; add(3,4) }");
    }
    // Phase 3A: native heap opcodes must match the interpreter exactly
    #[test] fn native_array_index() {
        same("fn main(){ a=[10,20,30]; a[0]+a[1]+a[2] }");
    }
    #[test] fn native_array_index_set() {
        same("fn main(){ a=[1,2,3]; a[1]=99; a[0]+a[1]+a[2] }");
    }
    #[test] fn native_map() {
        same("fn main(){ m=#{\"a\": 1, \"b\": 2, \"a\": 3}; m[\"a\"]+m[\"b\"] }");
    }
    #[test] fn native_map_set() {
        same("fn main(){ m=#{\"x\": 1}; m[\"y\"]=2; m[\"x\"]+m[\"y\"] }");
    }
    #[test] fn native_set_dedup() {
        // a set is a map elem->null; build it natively and read a member back
        same("fn main(){ s=#(1,2,2,3,3,3); m=#{1: \"a\", 1: \"b\"}; m[1] }");
    }
    #[test] fn native_range_value() {
        same("fn main(){ r=1..=5; s=0; for x in r { s=s+x }; s }");
    }
    #[test] fn native_struct_field() {
        same("data Point(x, y)\nfn main(){ p=Point { x: 3, y: 4 }; p.x*10 + p.y }");
    }
    #[test] fn native_field_set() {
        same("data Point(x, y)\nfn main(){ p=Point { x: 1, y: 2 }; p.x = 7; p.x + p.y }");
    }
    #[test] fn native_fmtstr() {
        same("fn main(){ n=42; name=\"Nova\"; f\"{name} is {n}\" }");
    }
    #[test] fn native_slice() {
        same("fn main(){ a=[0,1,2,3,4,5]; b=a[1..4]; c=a[..2]; d=a[3..]; e=a[1..=3]; len(b)+len(c)+len(d)+len(e) }");
    }
    #[test] fn native_str_slice() {
        same("fn main(){ s=\"hello world\"; s[0..5] }");
    }
    #[test] fn native_nested_struct_array() {
        same("data Box(items)\nfn main(){ b=Box { items: [1,2,3] }; b.items[1] }");
    }
    #[test] fn native_loop_break_continue_with_array() {
        same("fn main(){ a=[1,2,3,4,5,6]; s=0; for x in a { if x==2 {continue}; if x==5 {break}; s=s+x }; s }");
    }
    #[test] fn native_while_index_mutation() {
        same("fn main(){ a=[0,0,0,0]; i=0; while i<4 { a[i]=i*i; i=i+1 }; a[0]+a[1]+a[2]+a[3] }");
    }
    #[test] fn native_struct_in_array_mutation() {
        same("data P(x)\nfn main(){ ps=[P{x:1}, P{x:2}]; ps[0].x = 100; ps[0].x + ps[1].x }");
    }
    #[test] fn native_nested_map() {
        same("fn main(){ m=#{\"a\": [1,2], \"b\": [3,4]}; m[\"a\"][1] + m[\"b\"][0] }");
    }
    // Phase 4A: native match
    #[test] fn native_match_enum() {
        same("enum Opt { None, Some(i32) }\nfn main(){ x=Some(7); match x { Some(v)=>v*2, None=>0 } }");
    }
    #[test] fn native_match_guard() {
        same("fn main(){ n=15; match n { x if x>10 => \"big\", _ => \"small\" } }");
    }
    #[test] fn native_match_literals_and_range() {
        same("fn main(){ s=0; for n in 0..6 { s = s + match n { 0=>10, 1|2=>20, 3..=4=>30, _=>40 } }; s }");
    }
    #[test] fn native_match_tuple_and_wildcard() {
        same("fn main(){ p=[1,2]; match p { [a, b] => a+b, _ => 0 } }");
    }
    #[test] fn native_match_struct_pattern() {
        same("data Point(x, y)\nfn main(){ p=Point{x:3,y:4}; match p { Point{x: a, y: b} => a*10+b } }");
    }
    #[test] fn native_match_nested() {
        same("enum T { Leaf(i32), Node(i32) }\nfn main(){ t=Node(5); match t { Leaf(n) => n, Node(n) if n>3 => n*100, Node(n) => n } }");
    }
    // Phase 4B: native method dispatch
    #[test] fn native_array_methods() {
        same("fn main(){ a=[1,2,3]; a.push(4); a.push(5); x=a.pop(); a.len()*100 + x + a.get(0) }");
    }
    #[test] fn native_string_methods() {
        same("fn main(){ s=\"Hello\"; f\"{s.upper()} {s.lower()} {s.len()}\" }");
    }
    #[test] fn native_user_method() {
        same("data Counter(n)\nimpl Counter { fn bump(self, by) { self.n + by } }\nfn main(){ c=Counter{n:10}; c.bump(5) }");
    }
    // Phase 4C: native closures (compiled lambda bodies)
    #[test] fn native_closure_capture() {
        same("fn main(){ x=10; f=(y)=>x+y; f(5) + f(100) }");
    }
    #[test] fn native_closure_returned() {
        same("fn make(n){ (x)=>x+n } fn main(){ add5=make(5); add5(3) + make(100)(1) }");
    }
    #[test] fn native_closure_nested() {
        same("fn main(){ a=1; f=(x)=> { g=(y)=>x+y+a; g(10) }; f(100) }");
    }
    #[test] fn native_closure_higher_order() {
        same("fn apply(f, v){ f(v) } fn main(){ apply((x)=>x*x, 7) }");
    }
    #[test] fn native_closure_in_array() {
        same("fn main(){ fs=[(x)=>x+1, (x)=>x*2]; fs[0](10) + fs[1](10) }");
    }
    #[test] fn native_closure_block_body() {
        same("fn main(){ acc=(a, b)=> { let s = a + b; s * 2 }; acc(3, 4) }");
    }
    // Phase 5: native exceptions, defers, and flow-boundary parity
    #[test] fn try_catch_thrown_value() {
        same("fn main(){ r=\"\"; try { throw \"boom\" } catch e { r = e }; r }");
    }
    #[test] fn try_catch_runtime_error() {
        same("fn main(){ try { 1/0 } catch e { e } }");
    }
    #[test] fn try_catch_across_call() {
        same("fn f(){ throw \"deep\" } fn main(){ try { f() } catch e { \"got \" + e } }");
    }
    #[test] fn try_finally_runs_on_all_paths() {
        same("fn main(){ log=[]; try { push(log,1); throw \"x\" } catch e { push(log,2) } finally { push(log,3) }; log[0]*100+log[1]*10+log[2] }");
    }
    #[test] fn try_finally_wins_over_return() {
        same("fn f(){ try { return 1 } finally { return 2 } } fn main(){ f() }");
    }
    #[test] fn try_no_catch_swallows_throw() {
        same("fn main(){ r=0; try { throw \"x\"; r=1 } finally { r=r+10 }; r }");
    }
    #[test] fn nested_try_rethrow() {
        same("fn main(){ try { try { throw \"in\" } catch e { throw e + \"ner\" } } catch e { e } }");
    }
    #[test] fn catch_var_persists_after_try() {
        same("fn main(){ try { throw 7 } catch e {}; e }");
    }
    #[test] fn try_break_crossing_to_loop() {
        same("fn main(){ log=[]; i=0; while i<5 { try { if i==2 { break } push(log,i) } finally { push(log,99) }; i=i+1 }; len(log)*1000 + log[0]*100 + log[1]*10 + i }");
    }
    #[test] fn try_return_through_finally() {
        same("fn f(){ log=[]; try { return 42 } finally { push(log, 1) } } fn main(){ f() }");
    }
    #[test] fn defer_lifo_order() {
        same("fn main(){ log=[]; { defer push(log,1); defer push(log,2); push(log,0) }; log[0]*100+log[1]*10+log[2] }");
    }
    #[test] fn defer_runs_per_iteration() {
        same("fn main(){ log=[]; for i in 0..3 { defer push(log,i) }; len(log)*100 + log[0]*10 + log[2] }");
    }
    #[test] fn defer_sees_later_definition() {
        same("fn main(){ r=[]; { defer push(r, x); x = 5 }; r[0] }");
    }
    #[test] fn defer_runs_on_throw_unwind() {
        same("fn main(){ log=[]; try { defer push(log,1); throw \"x\" } catch e { push(log,2) }; log[0]*10+log[1] }");
    }
    #[test] fn break_with_value_inside_loop() {
        same("fn main(){ i=0; while true { i=i+1; if i==3 { break i*10 } }; i }");
    }
    #[test] fn break_outside_loop_returns() {
        same("fn f(){ break 9 } fn main(){ f() }");
    }
    // expression-block flow boundary: the interp consumes every Flow there
    #[test] fn expr_block_swallows_break() {
        same("fn main(){ i=0; r=0; while i<3 { r = { if i==1 { break }; i*10 }; i=i+1 }; r*10 + i }");
    }
    #[test] fn expr_block_return_yields_value() {
        same("fn main(){ x = { return 5; 3 }; x }");
    }
    #[test] fn expr_block_swallows_throw() {
        same("fn main(){ x = { throw \"e\"; 3 }; x }");
    }
    #[test] fn expr_block_writes_shadow() {
        same("fn main(){ x=1; z = { x = 2; x + 10 }; x*100 + z }");
    }
    // statement blocks share the function's flat scope, like the interpreter
    #[test] fn if_binding_persists() {
        same("fn main(){ if 1<2 { y = 41 }; y + 1 }");
    }
    #[test] fn loop_var_persists() {
        same("fn main(){ for i in 0..3 {}; i }");
    }
    #[test] fn uncaught_throw_matches_interp() {
        let src = "fn f(){ throw \"boom\" } fn main(){ f() }";
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        let compiled = compile_program(&prog).expect("compile");
        let vm_err = eval_main(&compiled, &interp).expect_err("vm should throw");
        let i2 = Interp::new(&prog).expect("interp2");
        let interp_err = i2.run().expect_err("interp should throw");
        assert_eq!(vm_err, interp_err);
    }
    #[test] fn fused_ops_preserve_bigint_promotion() {
        // IncLocal must promote to BigInt on overflow exactly like Bin(Add)
        same("fn main(){ i = 9223372036854775806; n = 0; while n < 3 { i = i + 1; n = n + 1 }; str(i) }");
    }
}
