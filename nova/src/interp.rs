// Tree-walking interpreter for Nova Core.
// Walks the AST directly, maintaining a call stack of scopes.

use std::collections::HashMap;
use std::cell::RefCell;
use std::rc::Rc;
use std::fmt;
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero, Signed};

use crate::ast::*;

// A fast, deterministic hasher (FNV-1a) for the interpreter's name→definition
// tables. `funcs`/`generators` are looked up on every single call, where the
// standard library's SipHash — built for DoS resistance, not speed — dominates
// the per-call cost. FNV is a few instructions per byte for these short keys.
#[derive(Default)]
pub struct FnvHasher(u64);
impl std::hash::Hasher for FnvHasher {
    #[inline] fn finish(&self) -> u64 { self.0 }
    #[inline] fn write(&mut self, bytes: &[u8]) {
        let mut h = if self.0 == 0 { 0xcbf2_9ce4_8422_2325 } else { self.0 };
        for &b in bytes { h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3); }
        self.0 = h;
    }
}
type FnvBuild = std::hash::BuildHasherDefault<FnvHasher>;
type FastMap<K, V> = HashMap<K, V, FnvBuild>;
type FastSet<K> = std::collections::HashSet<K, FnvBuild>;

// A lexical scope: an insertion-ordered association list with last-wins
// semantics (identical to the old `HashMap<String, Value>` for every operation
// the interpreter performs, since names are unique per scope). The win is in
// the hot paths: cloning a scope (done per call/block/match-arm/closure) is now
// a `Vec` copy with cheap `Rc<str>` key bumps instead of rehashing and
// re-allocating every key string, and small scopes resolve variables by a short
// linear scan instead of hashing. Insert overwrites an existing binding in
// place, so a loop that rebinds its variable each iteration never grows.
#[derive(Clone, Default, Debug)]
pub struct Scope {
    entries: Vec<(Rc<str>, Value)>,
}

impl Scope {
    #[inline]
    pub fn new() -> Self { Scope { entries: Vec::new() } }
    #[inline]
    pub fn with_capacity(n: usize) -> Self { Scope { entries: Vec::with_capacity(n) } }

    #[inline]
    pub fn insert(&mut self, key: impl Into<Rc<str>>, value: Value) {
        let key = key.into();
        // reverse scan: a just-inserted binding (e.g. a loop variable) is found
        // immediately, and shadowing rebinds hit the most recent entry first
        for (k, v) in self.entries.iter_mut().rev() {
            if **k == *key { *v = value; return; }
        }
        self.entries.push((key, value));
    }

    #[inline]
    pub fn get(&self, key: &str) -> Option<&Value> {
        for (k, v) in self.entries.iter().rev() {
            if **k == *key { return Some(v); }
        }
        None
    }

    #[inline]
    pub fn contains_key(&self, key: &str) -> bool { self.get(key).is_some() }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, (Rc<str>, Value)> { self.entries.iter() }

    // Build a scope over a recycled backing buffer (from the interpreter's pool).
    #[inline]
    pub fn from_backing(mut entries: Vec<(Rc<str>, Value)>) -> Self {
        entries.clear();
        Scope { entries }
    }
    // Reclaim the backing buffer for reuse.
    #[inline]
    pub fn into_backing(self) -> Vec<(Rc<str>, Value)> { self.entries }
}

// Internal marker: an Err carrying this string means "a Nova `throw` is unwinding";
// the real thrown Value lives in Interp::pending_throw. Distinguishes user throws
// from genuine runtime errors so try/catch only catches the former.
pub(crate) const THROW_SENTINEL: &str = "\u{0}__nova_throw__";
// Internal control signal: unwinds a generator body to the `produce` boundary
// once the requested yield is reached. Never caught by try/catch.
pub(crate) const YIELD_STOP: &str = "\u{0}__nova_yield_stop__";

thread_local! {
    // xorshift64* RNG state for the `rand` module. Seeded from the wall clock on
    // first use; reseedable via rand.seed(n) for reproducible runs.
    static RNG_STATE: RefCell<u64> = RefCell::new(0);
}

fn rng_seed_from_clock() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    nanos ^ 0x2545F4914F6CDD1D
}

fn rng_next() -> u64 {
    RNG_STATE.with(|s| {
        let mut x = *s.borrow();
        if x == 0 { x = rng_seed_from_clock(); }
        // xorshift64*
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *s.borrow_mut() = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    })
}

fn rng_seed(n: u64) {
    RNG_STATE.with(|s| *s.borrow_mut() = if n == 0 { 1 } else { n });
}

// uniform float in [0, 1)
fn rng_float() -> f64 {
    (rng_next() >> 11) as f64 / (1u64 << 53) as f64
}

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    BigInt(num_bigint::BigInt),
    Float(f64),
    Str(String),
    Bool(bool),
    Null,
    Array(Rc<RefCell<Vec<Value>>>),
    Struct(Rc<RefCell<StructInstance>>),
    Closure(Rc<ClosureVal>),
    Enum(Rc<EnumVal>),
    Map(Rc<RefCell<Vec<(Value, Value)>>>),
    // A lazy async computation: holds the body + captured scope until awaited.
    Future(Rc<RefCell<FutureVal>>),
    // A handle to a spawned task; awaiting it drives the scheduler until done.
    Task(Rc<RefCell<TaskVal>>),
    // A buffered channel for passing values between tasks.
    Channel(Rc<RefCell<ChannelVal>>),
    // A lazy generator produced by calling a generator function (one with `yield`).
    Generator(Rc<GenVal>),
}

// A lazy generator: its body + captured arguments, plus a cursor for the next
// value to produce. Laziness is achieved by re-running the body up to the cursor-th
// `yield` on demand (so even infinite sequences work — we stop at the value asked
// for). Generator bodies are expected to be pure: side effects re-run on each pull.
#[derive(Debug)]
pub struct GenVal {
    pub body: Vec<Stmt>,
    pub scope: Scope,
    pub cursor: RefCell<usize>,
}

// Transient state for one `produce` of a generator: which yield index we want,
// how many we have passed so far, and the captured value once reached.
struct GenState {
    target: usize,
    count: usize,
    value: Option<Value>,
}

// State of an async-fn future. Created Pending; awaiting runs the body once
// and caches the result, so a future can be awaited at most meaningfully once.
#[derive(Debug, Clone)]
pub enum FutureState {
    Pending,
    Running,   // guards against awaiting a future from within itself
    Done(Value),
    Failed(Value), // a throw escaped the future body
}

pub struct FutureVal {
    pub body: Vec<Stmt>,
    pub scope: Scope,
    pub state: FutureState,
}

impl std::fmt::Debug for FutureVal {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "FutureVal({:?})", self.state)
    }
}

// A spawned task. Like a future but identified for the scheduler queue.
#[derive(Debug)]
pub struct TaskVal {
    pub id: u64,
    pub state: FutureState,
}

impl TaskVal {
    fn state_clone(&self) -> FutureState {
        self.state.clone()
    }
}

// A buffered FIFO channel. `closed` lets receivers learn the sender is gone.
#[derive(Debug)]
pub struct ChannelVal {
    pub buffer: std::collections::VecDeque<Value>,
    // retained for introspection/back-pressure; not consumed by the scheduler yet
    #[allow(dead_code)]
    pub capacity: usize, // 0 means unbounded
    pub closed: bool,
}

#[derive(Debug)]
pub struct EnumVal {
    pub enum_name: String,
    pub variant: String,
    pub data: Vec<Value>,
}

#[derive(Debug)]
pub struct ClosureVal {
    pub params: Vec<String>,
    pub body: LambdaBody,
    // captured environment: a snapshot of the defining scope
    pub captured: Scope,
    // index of a compiled bytecode chunk for this lambda's body, when the VM
    // built it; `None` for closures created by the tree-walking interpreter.
    // The interpreter ignores this; only the VM's `CallValue` uses it.
    pub vm_chunk: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct StructInstance {
    pub type_name: String,
    pub fields: HashMap<String, Value>,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        values_eq(self, other)
    }
}

// Normalize a BigInt back to a machine Int when it fits, keeping the fast path hot.
pub(crate) fn norm_big(b: BigInt) -> Value {
    match b.to_i64() {
        Some(n) => Value::Int(n),
        None => Value::BigInt(b),
    }
}

// View any integer value as a BigInt for mixed arithmetic.
fn as_big(v: &Value) -> Option<BigInt> {
    match v {
        Value::Int(n) => Some(BigInt::from(*n)),
        Value::BigInt(b) => Some(b.clone()),
        _ => None,
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::BigInt(b) => write!(f, "{}", b),
            Value::Float(x) => {
                if x.fract() == 0.0 && x.is_finite() {
                    write!(f, "{:.1}", x)
                } else {
                    write!(f, "{}", x)
                }
            }
            Value::Str(s) => write!(f, "{}", s),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Null => write!(f, "null"),
            Value::Array(items) => {
                let inner = items.borrow();
                let parts: Vec<String> = inner.iter().map(|v| match v {
                    Value::Str(s) => format!("\"{}\"", s),
                    other => other.to_string(),
                }).collect();
                write!(f, "[{}]", parts.join(", "))
            }
            Value::Struct(inst) => {
                let inst = inst.borrow();
                // Print fields in a stable order isn't guaranteed by HashMap; sort for readability.
                let mut keys: Vec<&String> = inst.fields.keys().collect();
                keys.sort();
                let parts: Vec<String> = keys.iter().map(|k| {
                    let v = &inst.fields[*k];
                    let vs = match v {
                        Value::Str(s) => format!("\"{}\"", s),
                        other => other.to_string(),
                    };
                    format!("{}: {}", k, vs)
                }).collect();
                write!(f, "{} {{ {} }}", inst.type_name, parts.join(", "))
            }
            Value::Closure(c) => write!(f, "<closure/{}>", c.params.len()),
            Value::Enum(e) => {
                if e.data.is_empty() {
                    write!(f, "{}", e.variant)
                } else {
                    let parts: Vec<String> = e.data.iter().map(|v| match v {
                        Value::Str(s) => format!("\"{}\"", s),
                        other => other.to_string(),
                    }).collect();
                    write!(f, "{}({})", e.variant, parts.join(", "))
                }
            }
            Value::Map(m) => {
                let inner = m.borrow();
                let parts: Vec<String> = inner.iter().map(|(k, v)| {
                    let ks = match k {
                        Value::Str(s) => format!("\"{}\"", s),
                        other => other.to_string(),
                    };
                    let vs = match v {
                        Value::Str(s) => format!("\"{}\"", s),
                        other => other.to_string(),
                    };
                    format!("{}: {}", ks, vs)
                }).collect();
                write!(f, "{{{}}}", parts.join(", "))
            }
            Value::Future(fut) => {
                match &fut.borrow().state {
                    FutureState::Done(v) => write!(f, "<future={}>", v),
                    FutureState::Failed(_) => write!(f, "<future:failed>"),
                    _ => write!(f, "<future:pending>"),
                }
            }
            Value::Task(t) => write!(f, "<task#{}>", t.borrow().id),
            Value::Channel(ch) => {
                let c = ch.borrow();
                write!(f, "<channel:{} buffered>", c.buffer.len())
            }
            Value::Generator(g) => write!(f, "<generator@{}>", *g.cursor.borrow()),
        }
    }
}

impl Value {
    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "int",
            Value::BigInt(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "string",
            Value::Bool(_) => "bool",
            Value::Null => "null",
            Value::Array(_) => "array",
            Value::Struct(_) => "struct",
            Value::Closure(_) => "closure",
            Value::Enum(_) => "enum",
            Value::Map(_) => "map",
            Value::Future(_) => "future",
            Value::Task(_) => "task",
            Value::Channel(_) => "channel",
            Value::Generator(_) => "generator",
        }
    }
    pub(crate) fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Null => false,
            Value::Int(n) => *n != 0,
            Value::BigInt(b) => !b.is_zero(),
            _ => true,
        }
    }
}

// Control-flow signal used to unwind out of a function on `return`.
pub(crate) enum Flow {
    Normal,
    Return(Value),
    Throw(Value),
    Break(Value),
    Continue,
}

pub struct Interp {
    funcs: FastMap<String, Func>,
    structs: HashMap<String, StructDef>,
    // methods, nested type_name -> method_name -> Rc<Func>. Nesting lets dispatch
    // look up with &str keys (no per-call key allocation), and the Rc means the
    // hot method-call path clones a refcount instead of the whole body AST.
    methods: HashMap<String, HashMap<String, Rc<Func>>>,
    // variant_name -> (enum_name, arity)
    variants: FastMap<String, (String, usize)>,
    // module alias -> real module root (e.g. "m" -> "math")
    module_aliases: HashMap<String, String>,
    // carries a thrown value across function-call boundaries
    pending_throw: RefCell<Option<Value>>,
    // CLI arguments visible to the program via the args() builtin
    cli_args: RefCell<Vec<String>>,
    // collected test blocks, in source order
    tests: Vec<crate::ast::TestBlock>,
    // state machines: name -> (initial_state, transitions[(from,to,event)])
    machines: FastMap<String, (String, Vec<(String, String, String)>)>,
    // global constants, evaluated at load time
    consts: RefCell<HashMap<String, Value>>,
    // pending const expressions to evaluate lazily on first run
    const_exprs: Vec<(String, crate::ast::Expr)>,
    // trait name -> (required method names, default method funcs)
    traits: HashMap<String, (Vec<String>, Vec<crate::ast::Func>)>,
    // names declared in `extern` blocks; calling one errors (no FFI yet)
    extern_funcs: FastSet<String>,
    // names of generator functions (their body contains `yield`)
    generators: FastSet<String>,
    // stack of active generator productions (supports nested generators)
    gen_ctx: RefCell<Vec<GenState>>,
    // source position (line, col) of the statement currently executing, for errors
    cur_pos: std::cell::Cell<(u32, u32)>,
    // refined type alias name -> predicate over the bound value `it`
    refinements: HashMap<String, Expr>,
    // --- async scheduler ---
    // ready queue of spawned tasks waiting to run to completion
    ready_queue: RefCell<std::collections::VecDeque<Rc<RefCell<TaskVal>>>>,
    // the body+scope for each spawned task, keyed by task id
    task_bodies: RefCell<HashMap<u64, (Vec<Stmt>, Scope)>>,
    // monotonic id source for tasks
    next_task_id: std::cell::Cell<u64>,
    // free-list of scope backing buffers, so ordinary (non-escaping) calls reuse
    // an allocation instead of heap-allocating a fresh frame every time
    scope_pool: RefCell<Vec<Vec<(Rc<str>, Value)>>>,
    // free-list of argument vectors, so evaluating a call's arguments doesn't
    // heap-allocate a fresh Vec on every single call
    arg_pool: RefCell<Vec<Vec<Value>>>,
    // runtime-replaced function bodies for `#[hot_swap]` functions: name -> closure
    // installed via the `hot_swap(name, closure)` builtin; consulted before the
    // original body runs.
    swapped: RefCell<HashMap<String, Value>>,
    // `#[memo]` result cache, keyed by "name(args)"; `#[profile]` call counts;
    // one-shot `#[deprecate]` warning set.
    memo: RefCell<HashMap<String, Value>>,
    profile: RefCell<HashMap<String, i64>>,
    deprecated_warned: RefCell<std::collections::HashSet<String>>,
    // `#[time_travel(depth: N)]` per-function ring buffer of the last N results,
    // queryable via `history_of(name)` for rollback/inspection.
    history: RefCell<HashMap<String, std::collections::VecDeque<Value>>>,
    // `#[anti_tamper]` baseline body hashes recorded at first call.
    tamper_base: RefCell<HashMap<String, i64>>,
    // `#[instrument]` call counts (queryable via instrument_of); `#[budget(n)]` /
    // `#[cost]` remaining-call allowance (throws when exhausted, budget_of reads
    // it); `#[cache(ttl: n)]` remaining reuses before a memo entry is recomputed;
    // one-shot `#[experimental]` warning set; `snapshot`/`rollback` named values.
    instrument: RefCell<HashMap<String, i64>>,
    budget: RefCell<HashMap<String, i64>>,
    cache_ttl: RefCell<HashMap<String, i64>>,
    experimental_warned: RefCell<std::collections::HashSet<String>>,
    snapshots: RefCell<HashMap<String, Value>>,
    // state migrations keyed by the source ("from") type name: (to, body).
    migrations: HashMap<String, (String, Vec<Stmt>)>,
    // results of `#[comptime]` functions, evaluated once before main.
    comptime: RefCell<HashMap<String, Value>>,
    // Open TCP sockets (listeners and connections) addressed by an integer handle
    // returned to Nova as a plain `Int` — so networking needs no new Value variant
    // and stays byte-identical across tiers (net programs use these builtins via
    // `call_named`, exactly like file I/O). `next_sock` hands out fresh handles.
    sockets: RefCell<HashMap<i64, Sock>>,
    next_sock: std::cell::Cell<i64>,
}

// A live socket behind a Nova integer handle. `&TcpStream` implements Read+Write,
// so connections are used through a shared borrow of the registry.
enum Sock {
    Listener(std::net::TcpListener),
    Stream(std::net::TcpStream),
}

// --- crypto / encoding primitives (dependency-free) -------------------------
// Standard base64 (RFC 4648) and SHA-1, used by base64_*/sha1_hex/ws_accept.
const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | (b2 as u32);
        out.push(B64[((n >> 18) & 63) as usize] as char);
        out.push(B64[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { B64[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { B64[(n & 63) as usize] as char } else { '=' });
    }
    out
}

fn b64_decode(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62), b'/' => Some(63),
            _ => None,
        }
    }
    let (mut acc, mut nbits, mut out) = (0u32, 0u32, Vec::new());
    for c in s.bytes() {
        if c == b'=' || c.is_ascii_whitespace() { continue; }
        acc = (acc << 6) | val(c)?;
        nbits += 6;
        if nbits >= 8 { nbits -= 8; out.push(((acc >> nbits) & 0xFF) as u8); }
    }
    Some(out)
}

fn sha1_digest(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let ml = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&ml.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]);
        }
        for i in 16..80 { w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1); }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let tmp = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(wi);
            e = d; d = c; c = b.rotate_left(30); b = a; a = tmp;
        }
        h[0] = h[0].wrapping_add(a); h[1] = h[1].wrapping_add(b); h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d); h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for i in 0..5 { out[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_be_bytes()); }
    out
}

fn hex_of(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes { s.push_str(&format!("{:02x}", b)); }
    s
}

impl Interp {
    #[inline]
    fn take_args(&self) -> Vec<Value> {
        self.arg_pool.borrow_mut().pop().map(|mut v| { v.clear(); v }).unwrap_or_default()
    }
    #[inline]
    fn give_args(&self, mut v: Vec<Value>) {
        v.clear();
        let mut pool = self.arg_pool.borrow_mut();
        if pool.len() < 64 { pool.push(v); }
    }

    pub fn new(program: &Program) -> Result<Self, String> {
        let mut funcs: FastMap<String, Func> = FastMap::default();
        let mut structs = HashMap::new();
        let mut methods: HashMap<String, HashMap<String, Rc<Func>>> = HashMap::new();
        let mut variants: FastMap<String, (String, usize)> = FastMap::default();
        let mut module_aliases = HashMap::new();
        let mut tests = Vec::new();
        let mut machines: FastMap<String, (String, Vec<(String, String, String)>)> = FastMap::default();
        let mut const_exprs = Vec::new();
        let mut traits: HashMap<String, (Vec<String>, Vec<crate::ast::Func>)> = HashMap::new();
        let mut extern_funcs: FastSet<String> = FastSet::default();
        let mut generators: FastSet<String> = FastSet::default();
        let mut refinements: HashMap<String, Expr> = HashMap::new();
        let mut migrations: HashMap<String, (String, Vec<Stmt>)> = HashMap::new();
        // PASS 0: collect trait definitions first, so `impl Trait for Type` can
        // pull in default methods regardless of declaration order.
        for item in &program.items {
            if let Item::Trait(t) = item {
                traits.insert(t.name.clone(), (t.required.clone(), t.defaults.clone()));
            }
        }
        for item in &program.items {
            match item {
                Item::Func(f) => {
                    if funcs.contains_key(&f.name) {
                        return Err(format!("duplicate function: {}", f.name));
                    }
                    if body_has_yield(&f.body) { generators.insert(f.name.clone()); }
                    funcs.insert(f.name.clone(), f.clone());
                }
                Item::Struct(s) => {
                    if structs.contains_key(&s.name) {
                        return Err(format!("duplicate struct: {}", s.name));
                    }
                    structs.insert(s.name.clone(), s.clone());
                }
                Item::Impl(imp) => {
                    // implemented method names in this block
                    let mut provided: Vec<String> = Vec::new();
                    for m in &imp.methods {
                        methods.entry(imp.type_name.clone()).or_default()
                            .insert(m.name.clone(), Rc::new(m.clone()));
                        provided.push(m.name.clone());
                    }
                    // `impl Trait for Type`: pull in default methods + verify contract
                    if let Some(trait_name) = &imp.trait_name {
                        let (required, defaults) = traits.get(trait_name)
                            .ok_or_else(|| format!("unknown trait: {}", trait_name))?
                            .clone();
                        // add default methods that the impl didn't override
                        for d in &defaults {
                            if !provided.contains(&d.name) {
                                methods.entry(imp.type_name.clone()).or_default()
                                    .insert(d.name.clone(), Rc::new(d.clone()));
                                provided.push(d.name.clone());
                            }
                        }
                        // every required method must be provided (directly or by default)
                        for req in &required {
                            if !provided.contains(req) {
                                return Err(format!(
                                    "type `{}` does not implement required method `{}` of trait `{}`",
                                    imp.type_name, req, trait_name
                                ));
                            }
                        }
                    }
                }
                Item::Enum(e) => {
                    for v in &e.variants {
                        if variants.contains_key(&v.name) {
                            return Err(format!("duplicate enum variant: {}", v.name));
                        }
                        variants.insert(v.name.clone(), (e.name.clone(), v.arity));
                    }
                }
                Item::Use(u) => {
                    // stdlib functions are always available as `module.fn` and bare `fn`;
                    // `use` validates the module exists and registers any alias.
                    if !is_known_module(&u.module_root()) {
                        return Err(format!("unknown module: {}", u.module));
                    }
                    if let Some(alias) = &u.alias {
                        module_aliases.insert(alias.clone(), u.module_root());
                    }
                }
                Item::Test(t) => tests.push(t.clone()),
                Item::Machine(m) => {
                    machines.insert(m.name.clone(), (m.initial.clone(), m.transitions.clone()));
                }
                Item::Const { name, value } => {
                    const_exprs.push((name.clone(), value.clone()));
                }
                Item::Trait(_) => { /* collected in PASS 0 */ }
                Item::Macro(_) => { /* expanded at parse time */ }
                Item::TypeAlias { name, refinement, .. } => {
                    if let Some(pred) = refinement {
                        refinements.insert(name.clone(), pred.clone());
                    }
                }
                Item::Extern(fns) => {
                    for f in fns { extern_funcs.insert(f.name.clone()); }
                }
                Item::Import { .. } => { /* resolved by the module loader pre-execution */ }
                Item::Migration { from, to, body } => {
                    migrations.insert(from.clone(), (to.clone(), body.clone()));
                }
            }
        }
        Ok(Interp {
            funcs, structs, methods, variants, module_aliases,
            pending_throw: RefCell::new(None), tests, machines,
            cli_args: RefCell::new(Vec::new()),
            consts: RefCell::new(HashMap::new()), const_exprs, traits, extern_funcs,
            generators, gen_ctx: RefCell::new(Vec::new()),
            cur_pos: std::cell::Cell::new((0, 0)), refinements,
            ready_queue: RefCell::new(std::collections::VecDeque::new()),
            task_bodies: RefCell::new(HashMap::new()),
            next_task_id: std::cell::Cell::new(1),
            scope_pool: RefCell::new(Vec::new()),
            arg_pool: RefCell::new(Vec::new()),
            swapped: RefCell::new(HashMap::new()),
            memo: RefCell::new(HashMap::new()),
            profile: RefCell::new(HashMap::new()),
            deprecated_warned: RefCell::new(std::collections::HashSet::new()),
            history: RefCell::new(HashMap::new()),
            tamper_base: RefCell::new(HashMap::new()),
            instrument: RefCell::new(HashMap::new()),
            budget: RefCell::new(HashMap::new()),
            cache_ttl: RefCell::new(HashMap::new()),
            experimental_warned: RefCell::new(std::collections::HashSet::new()),
            snapshots: RefCell::new(HashMap::new()),
            migrations,
            comptime: RefCell::new(HashMap::new()),
            sockets: RefCell::new(HashMap::new()),
            next_sock: std::cell::Cell::new(1),
        })
    }

    // Register a socket and hand back its integer handle.
    fn sock_register(&self, s: Sock) -> i64 {
        let h = self.next_sock.get();
        self.next_sock.set(h + 1);
        self.sockets.borrow_mut().insert(h, s);
        h
    }

    // Evaluate all global constants into the consts map (idempotent).
    pub(crate) fn init_consts(&self) -> Result<(), String> {
        self.init_comptime();
        if !self.consts.borrow().is_empty() || self.const_exprs.is_empty() { return Ok(()); }
        let empty: Scope = Scope::new();
        for (name, expr) in &self.const_exprs {
            let v = self.eval(expr, &empty)?;
            self.consts.borrow_mut().insert(name.clone(), v);
        }
        Ok(())
    }

    // Evaluate every no-argument `#[comptime]` function exactly once, before main,
    // caching its result. Calls then return the precomputed constant — real
    // compile-time evaluation (e.g. a lookup table built once at startup).
    fn init_comptime(&self) {
        if !self.comptime.borrow().is_empty() { return; }
        for (name, f) in self.funcs.iter() {
            if f.params.is_empty() && f.attrs.iter().any(|a| a.name == "comptime") {
                let mut scope = Scope::new();
                if let Ok(Flow::Return(v)) | Ok(Flow::Break(v)) = self.exec_block(&f.body, &mut scope) {
                    self.comptime.borrow_mut().insert(name.clone(), v);
                }
            }
        }
    }

    // Run all `test "..." { ... }` blocks and report pass/fail. Returns the
    // number of failures (0 = all passed). Each block runs in a fresh scope.
    pub fn run_tests(&self) -> i32 {
        let _ = self.init_consts();
        if self.tests.is_empty() {
            println!("no tests found");
            return 0;
        }
        let mut passed = 0;
        let mut failed = 0;
        println!("running {} test(s)\n", self.tests.len());
        for t in &self.tests {
            let mut scope: Scope = Scope::new();
            match self.exec_block(&t.body, &mut scope) {
                Ok(_) => {
                    println!("  PASS  {}", t.name);
                    passed += 1;
                }
                Err(e) => {
                    let msg = if e == THROW_SENTINEL {
                        self.pending_throw.borrow_mut().take()
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "assertion failed".to_string())
                    } else {
                        self.locate(e)
                    };
                    println!("  FAIL  {}  ({})", t.name, msg);
                    failed += 1;
                }
            }
        }
        println!("\n{} passed, {} failed", passed, failed);
        failed
    }

    pub fn set_args(&self, args: Vec<String>) {
        *self.cli_args.borrow_mut() = args;
    }

    pub fn run(&self) -> Result<Value, String> {
        if !self.funcs.contains_key("main") {
            return Err("no `main` function found".into());
        }
        if self.generators.contains("main") {
            return Err("`main` cannot be a generator (it must not contain `yield`)".into());
        }
        self.init_consts()?;
        let result = self.call("main", vec![]);
        // Drive any still-queued fire-and-forget tasks to completion.
        self.drain_tasks()?;
        match result {
            Err(e) if e == THROW_SENTINEL => {
                let v = self.pending_throw.borrow_mut().take().unwrap_or(Value::Null);
                Err(self.locate(format!("uncaught exception: {}", v)))
            }
            Err(e) => Err(self.locate(e)),
            ok => ok,
        }
    }

    // Raise an assertion failure as a throw the test harness / try-catch can catch.
    fn fail_assert(&self, msg: String) -> Result<Value, String> {
        *self.pending_throw.borrow_mut() = Some(Value::Str(msg));
        Err(THROW_SENTINEL.to_string())
    }

    // The VM shares the interpreter's throw channel: these expose it so the
    // bytecode tier reproduces `call_function`'s sentinel protocol exactly.
    pub(crate) fn take_pending_throw(&self) -> Value {
        self.pending_throw.borrow_mut().take().unwrap_or(Value::Null)
    }
    pub(crate) fn park_throw(&self, v: Value) {
        *self.pending_throw.borrow_mut() = Some(v);
    }
    pub(crate) fn set_pos(&self, pos: (u32, u32)) {
        self.cur_pos.set(pos);
    }

    // Prefix a message with the current source position, when known.
    pub(crate) fn locate(&self, msg: String) -> String {
        let (line, col) = self.cur_pos.get();
        if line == 0 { msg } else { format!("line {}, col {}: {}", line, col, msg) }
    }

    // Run every task left in the ready queue until it is empty. Used after main
    // so spawned tasks that were never awaited still get to run.
    pub(crate) fn drain_tasks(&self) -> Result<(), String> {
        let mut guard = 0;
        loop {
            let next = self.ready_queue.borrow_mut().pop_front();
            match next {
                Some(task) => {
                    let id = task.borrow().id;
                    self.run_task(id, &task)?;
                }
                None => return Ok(()),
            }
            guard += 1;
            if guard > 1_000_000 {
                return Err("scheduler: too many tasks (possible runaway spawn)".into());
            }
        }
    }

    // Register the items (functions, structs, impls, enums, uses) of a freshly
    // parsed program into this interpreter, mutating its tables. Used by the REPL
    // so definitions persist across input lines.
    pub fn register_items(&mut self, program: &Program) -> Result<(), String> {
        for item in &program.items {
            match item {
                Item::Func(f) => {
                    if body_has_yield(&f.body) { self.generators.insert(f.name.clone()); }
                    self.funcs.insert(f.name.clone(), f.clone());
                }
                Item::Struct(s) => { self.structs.insert(s.name.clone(), s.clone()); }
                Item::Impl(imp) => {
                    for m in &imp.methods {
                        self.methods.entry(imp.type_name.clone()).or_default()
                            .insert(m.name.clone(), Rc::new(m.clone()));
                    }
                }
                Item::Enum(e) => {
                    for v in &e.variants {
                        self.variants.insert(v.name.clone(), (e.name.clone(), v.arity));
                    }
                }
                Item::Use(u) => {
                    if !is_known_module(&u.module_root()) {
                        return Err(format!("unknown module: {}", u.module));
                    }
                    if let Some(alias) = &u.alias {
                        self.module_aliases.insert(alias.clone(), u.module_root());
                    }
                }
                Item::Test(_) => {} // tests are ignored in REPL mode
                Item::Machine(m) => {
                    self.machines.insert(m.name.clone(), (m.initial.clone(), m.transitions.clone()));
                }
                Item::Const { name, value } => {
                    let empty: Scope = Scope::new();
                    let v = self.eval(value, &empty)?;
                    self.consts.borrow_mut().insert(name.clone(), v);
                }
                Item::Trait(t) => {
                    self.traits.insert(t.name.clone(), (t.required.clone(), t.defaults.clone()));
                }
                Item::Macro(_) => {}
                Item::TypeAlias { name, refinement, .. } => {
                    if let Some(pred) = refinement {
                        self.refinements.insert(name.clone(), pred.clone());
                    }
                }
                Item::Extern(fns) => {
                    for f in fns { self.extern_funcs.insert(f.name.clone()); }
                }
                Item::Import { .. } => { /* resolved by the module loader pre-execution */ }
                Item::Migration { from, to, body } => {
                    self.migrations.insert(from.clone(), (to.clone(), body.clone()));
                }
            }
        }
        Ok(())
    }

    // Execute a single REPL statement/expression list in a persistent scope.
    // Returns the value of the last expression statement, if any, for printing.
    pub fn eval_repl(&self, stmts: &[Stmt], scope: &mut Scope) -> Result<Option<Value>, String> {
        let mut last_value: Option<Value> = None;
        for stmt in stmts {
            // capture the value of a bare expression so the REPL can echo it
            if let Stmt::Expr(e) = stmt {
                let v = self.eval(e, scope)?;
                last_value = Some(v);
                continue;
            }
            match self.exec_stmt(stmt, scope) {
                Ok(Flow::Normal) => { last_value = None; }
                Ok(Flow::Return(v)) => { last_value = Some(v); }
                Ok(Flow::Break(v)) => { last_value = Some(v); }
                Ok(Flow::Continue) => { last_value = None; }
                Ok(Flow::Throw(v)) => return Err(format!("uncaught exception: {}", v)),
                Err(e) if e == THROW_SENTINEL => {
                    let v = self.pending_throw.borrow_mut().take().unwrap_or(Value::Null);
                    return Err(format!("uncaught exception: {}", v));
                }
                Err(e) => return Err(e),
            }
        }
        Ok(last_value)
    }

    // Dispatch a call by name with pre-evaluated arguments: state-machine and
    // enum-variant constructors first, then everything `call` handles (builtins,
    // stdlib, user functions). Shared by the tree-walker's `Expr::Call` and the
    // bytecode VM's `CallDyn`, so both resolve names identically. (Closures held
    // in locals are handled by the caller, which has the scope.)
    pub(crate) fn call_named(&self, callee: &str, vals: Vec<Value>) -> Result<Value, String> {
        // Ordinary function calls dominate; state-machine and enum-variant
        // constructors are comparatively rare. When a program declares no machines
        // (resp. no enum variants) the corresponding map is empty, so the `is_empty`
        // guard skips the wasted string-hash lookup on every call and falls straight
        // through to `self.call` (which resolves user functions in one lookup). The
        // resolution order is unchanged when the maps are non-empty, so this stays
        // byte-identical to the previous behaviour.
        // state-machine constructor: TrafficLight() -> struct with `state` field
        if !self.machines.is_empty() {
            if let Some((initial, _)) = self.machines.get(callee) {
                if !vals.is_empty() {
                    return Err(format!("machine {} takes no constructor args", callee));
                }
                let mut fields = HashMap::new();
                fields.insert("state".to_string(), Value::Str(initial.clone()));
                return Ok(Value::Struct(Rc::new(RefCell::new(StructInstance {
                    type_name: callee.to_string(),
                    fields,
                }))));
            }
        }
        // enum variant constructor: Some(x), Cons(h, t), ...
        if !self.variants.is_empty() {
            if let Some((enum_name, arity)) = self.variants.get(callee) {
                if vals.len() != *arity {
                    return Err(format!(
                        "enum variant {} expects {} args, got {}",
                        callee, arity, vals.len()
                    ));
                }
                return Ok(Value::Enum(Rc::new(EnumVal {
                    enum_name: enum_name.clone(),
                    variant: callee.to_string(),
                    data: vals,
                })));
            }
        }
        self.call(callee, vals)
    }

    // Construct a struct instance from evaluated (name, value) fields, validating
    // against the declaration. Shared by `Expr::StructLit` and the VM's MakeStruct.
    pub(crate) fn make_struct(&self, name: &str, fields: Vec<(String, Value)>)
        -> Result<Value, String>
    {
        let def = self.structs.get(name)
            .ok_or_else(|| format!("unknown struct type: {}", name))?
            .clone();
        let mut field_map = HashMap::new();
        for (fname, v) in fields {
            field_map.insert(fname, v);
        }
        for declared in &def.fields {
            if !field_map.contains_key(declared) {
                return Err(format!("struct {} missing field: {}", name, declared));
            }
        }
        for given in field_map.keys() {
            if !def.fields.contains(given) {
                return Err(format!("struct {} has no field: {}", name, given));
            }
        }
        Ok(Value::Struct(Rc::new(RefCell::new(StructInstance {
            type_name: name.to_string(),
            fields: field_map,
        }))))
    }

    // The full attribute-aware call path. Applies (in order): hot_swap replacement,
    // memo cache lookup, requires/assumes entry contracts, retry/self_healing around
    // the body, ensures exit contract, trace/log/audit printing, profile counting,
    // deprecate warning, and memo store. Attributed functions are interp-only in the
    // VM compiler, so this behaviour is identical on every tier.
    fn call_attributed(&self, name: &str, func: &Func, args: Vec<Value>) -> Result<Value, String> {
        let has = |n: &str| func.attrs.iter().any(|a| a.name == n);

        // hot_swap: an installed replacement body wins entirely
        if has("hot_swap") {
            if let Some(swap) = self.swapped.borrow().get(name).cloned() {
                return self.call_closure(&swap, args);
            }
        }

        // deprecate / deprecated: warn once per function
        if has("deprecate") || has("deprecated") {
            if self.deprecated_warned.borrow_mut().insert(name.to_string()) {
                let note = func.attrs.iter().find(|a| a.name.starts_with("deprecat"))
                    .and_then(|a| a.args.iter().find(|(_, _)| true).map(|(_, v)| v.clone()))
                    .unwrap_or_default();
                eprintln!("warning: `{}` is deprecated{}", name,
                    if note.is_empty() { String::new() } else { format!(": {}", note) });
            }
        }

        // experimental / since: warn once per function, like deprecate
        if has("experimental") || has("since") {
            if self.experimental_warned.borrow_mut().insert(name.to_string()) {
                eprintln!("warning: `{}` is experimental", name);
            }
        }

        // budget(n) / cost: a call allowance — throw once it is exhausted. The
        // remaining count is queryable via budget_of(name).
        if let Some(att) = func.attrs.iter().find(|a| a.name == "budget" || a.name == "cost") {
            if let Some(n) = att.args.iter().find_map(|(_, v)| v.parse::<i64>().ok()) {
                let mut b = self.budget.borrow_mut();
                let rem = b.entry(name.to_string()).or_insert(n);
                if *rem <= 0 {
                    drop(b);
                    return self.fail_assert(format!("budget exceeded in `{}`", name));
                }
                *rem -= 1;
            }
        }

        // instrument: count calls (queryable via instrument_of)
        if has("instrument") {
            *self.instrument.borrow_mut().entry(name.to_string()).or_insert(0) += 1;
        }

        // memo / memoize: return a cached result if present. cache(ttl: n) is a
        // memo whose entry is reused at most n times before being recomputed.
        let ttl = func.attrs.iter().find(|a| a.name == "cache")
            .and_then(|a| a.int_arg("ttl").or_else(|| a.int_arg("")));
        let memo_key = if has("memo") || has("memoize") || ttl.is_some() {
            Some(format!("{}|{}", name, args.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")))
        } else { None };
        if let Some(k) = &memo_key {
            let fresh = match ttl {
                Some(_) => {
                    let mut c = self.cache_ttl.borrow_mut();
                    match c.get_mut(k) { Some(r) if *r > 0 => { *r -= 1; true } _ => false }
                }
                None => true,
            };
            if fresh {
                if let Some(v) = self.memo.borrow().get(k) { return Ok(v.clone()); }
            }
        }

        // anti_debug: refuse to run under a debugger (best-effort)
        if has("anti_debug") && detect_debugger() {
            return self.fail_assert(format!("`{}`: debugger detected (#[anti_debug])", name));
        }
        // anti_tamper: verify the function's body hasn't been altered vs a baseline
        // hash recorded at first call — detects live-patched code.
        if has("anti_tamper") {
            let h = body_hash(func);
            let mut base = self.tamper_base.borrow_mut();
            match base.get(name) {
                Some(&b) if b != h =>
                    return self.fail_assert(format!("`{}`: integrity check failed (#[anti_tamper])", name)),
                Some(_) => {}
                None => { base.insert(name.to_string(), h); }
            }
        }

        // requires / assumes: entry contracts, evaluated with params bound to args
        for att in func.attrs.iter().filter(|a| a.name == "requires" || a.name == "assumes") {
            let mut scope = Scope::with_capacity(func.params.len());
            for (p, v) in func.params.iter().zip(args.iter()) { scope.insert(p.clone(), v.clone()); }
            for pred in &att.exprs {
                if !self.eval(pred, &scope)?.is_truthy() {
                    return self.fail_assert(format!("{} contract violated in `{}`", att.name, name));
                }
            }
        }

        // retry / self_healing: run the body, retrying on error up to N attempts
        let attempts = func.attrs.iter()
            .filter(|a| a.name == "self_healing" || a.name == "retry")
            .filter_map(|a| a.int_arg("attempts")).max().unwrap_or(1).max(1);
        let mut result = Ok(Value::Null);
        for _ in 0..attempts {
            result = self.run_user_body(func, args.clone());
            if result.is_ok() { break; }
        }

        // ensures: exit contract, evaluated with params + `result` bound
        if let Ok(ret) = &result {
            for att in func.attrs.iter().filter(|a| a.name == "ensures") {
                let mut scope = Scope::with_capacity(func.params.len() + 1);
                for (p, v) in func.params.iter().zip(args.iter()) { scope.insert(p.clone(), v.clone()); }
                scope.insert("result".to_string(), ret.clone());
                for pred in &att.exprs {
                    if !self.eval(pred, &scope)?.is_truthy() {
                        return self.fail_assert(format!("ensures contract violated in `{}`", name));
                    }
                }
            }
        }

        // trace / log / audit: print a deterministic call record
        if has("trace") || has("log") || has("audit") {
            let a = args.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
            match &result {
                Ok(v) => println!("trace: {}({}) -> {}", name, a, v),
                Err(_) => println!("trace: {}({}) -> <error>", name, a),
            }
        }

        // profile: count calls (queryable via profile_of)
        if has("profile") {
            *self.profile.borrow_mut().entry(name.to_string()).or_insert(0) += 1;
        }

        // time_travel: record the last N results in a ring buffer (history_of)
        if let Some(att) = func.attrs.iter().find(|a| a.name == "time_travel") {
            if let Ok(v) = &result {
                let depth = att.int_arg("depth").unwrap_or(8).max(1) as usize;
                let mut hist = self.history.borrow_mut();
                let ring = hist.entry(name.to_string()).or_default();
                ring.push_back(v.clone());
                while ring.len() > depth { ring.pop_front(); }
            }
        }

        // memo: store a successful result (cache(ttl) also seeds the reuse count)
        if let (Some(k), Ok(v)) = (&memo_key, &result) {
            self.memo.borrow_mut().insert(k.clone(), v.clone());
            if let Some(n) = ttl {
                self.cache_ttl.borrow_mut().insert(k.clone(), n.max(0));
            }
        }
        result
    }

    // If `tn` names a refinement type, verify `v` satisfies its predicate (over the
    // bound name `it`). This is the single enforcement point shared by refined
    // `let`s, function parameters and returns, so a value of a refined type always
    // satisfies its invariant wherever it is introduced — a soundness guarantee the
    // interpreter is the oracle for (refined-typed functions run on the interpreter
    // tier; see `uses_refinements` in bytecode.rs, which keeps every tier identical).
    pub(crate) fn refine_check(&self, tn: &str, v: &Value, scope: &Scope) -> Result<(), String> {
        if let Some(pred) = self.refinements.get(tn).cloned() {
            let mut pscope = scope.clone();
            pscope.insert("it".to_string(), v.clone());
            if !self.eval(&pred, &pscope)?.is_truthy() {
                return Err(format!("refinement `{}` violated by value {}", tn, v));
            }
        }
        Ok(())
    }

    // Compile-time refinement checking: a refined `let x: T = <constant>` whose
    // constant value violates T's predicate is rejected before the program ever
    // runs (reported by `nova check`), rather than crashing at runtime. Only fires
    // for decidable constant values, so it never produces a false positive.
    pub fn static_refinement_errors(&self, program: &Program) -> Vec<String> {
        let mut errs = Vec::new();
        if self.refinements.is_empty() { return errs; }
        for item in &program.items {
            if let Item::Func(f) = item { self.scan_refined_consts(&f.body, &mut errs); }
        }
        errs
    }

    // Only the direct (top-level) statements of a function body are scanned: a
    // top-level `let x: T = <bad constant>` unconditionally violates its invariant,
    // so it is safe to reject at compile time. Bindings nested inside `try`
    // (intentionally caught), or `if`/loop branches (conditionally reached) are
    // left to the runtime check, avoiding any false compile-time rejection.
    fn scan_refined_consts(&self, body: &[Stmt], errs: &mut Vec<String>) {
        for s in body {
            if let Stmt::Let { ty: Some(tn), value, .. } = s {
                if self.refinements.contains_key(tn) {
                    if let Some(v) = Self::const_literal(value) {
                        if self.refine_check(tn, &v, &Scope::new()).is_err() {
                            errs.push(format!(
                                "refinement `{}` violated at compile time by constant {}", tn, v));
                        }
                    }
                }
            }
        }
    }

    // A compile-time-known value (literal, or a negated numeric literal). `None`
    // for anything not decidably constant — those defer to the runtime check.
    fn const_literal(e: &Expr) -> Option<Value> {
        match e {
            Expr::At { expr, .. } => Self::const_literal(expr),
            Expr::Int(n) => Some(Value::Int(*n)),
            Expr::Float(x) => Some(Value::Float(*x)),
            Expr::Bool(b) => Some(Value::Bool(*b)),
            Expr::Str(s) => Some(Value::Str(s.clone())),
            Expr::Null => Some(Value::Null),
            Expr::Unary { op: UnOp::Neg, expr } => match Self::const_literal(expr)? {
                Value::Int(n) => Some(Value::Int(-n)),
                Value::Float(x) => Some(Value::Float(-x)),
                _ => None,
            },
            _ => None,
        }
    }

    // Run a non-generator, non-async user function body over a pooled scope frame,
    // converting the resulting Flow to a value exactly as the call boundary does.
    fn run_user_body(&self, func: &Func, mut args: Vec<Value>) -> Result<Value, String> {
        let backing = self.scope_pool.borrow_mut().pop().unwrap_or_default();
        let mut scope = Scope::from_backing(backing);
        for (p, v) in func.params.iter().zip(args.drain(..)) { scope.insert(p.clone(), v); }
        self.give_args(args);
        // Enforce refinement types on parameters and the return value (cheap
        // `is_empty` guard skips this entirely for programs that declare none).
        let refined = !self.refinements.is_empty();
        let outcome = (|| -> Result<Value, String> {
            if refined {
                for (i, p) in func.params.iter().enumerate() {
                    if let Some(Some(tn)) = func.param_types.get(i) {
                        let v = scope.get(p).cloned().unwrap_or(Value::Null);
                        self.refine_check(tn, &v, &scope)?;
                    }
                }
            }
            let flow = self.exec_block(&func.body, &mut scope)?;
            let v = match flow {
                Flow::Return(v) | Flow::Break(v) => v,
                Flow::Continue | Flow::Normal => Value::Null,
                Flow::Throw(e) => {
                    *self.pending_throw.borrow_mut() = Some(e);
                    return Err(THROW_SENTINEL.to_string());
                }
            };
            if refined {
                if let Some(tn) = &func.ret_type {
                    self.refine_check(tn, &v, &scope)?;
                }
            }
            Ok(v)
        })();
        let mut backing = scope.into_backing();
        backing.clear();
        self.scope_pool.borrow_mut().push(backing);
        outcome
    }

    pub(crate) fn call(&self, name: &str, args: Vec<Value>) -> Result<Value, String> {
        // A user-defined function shadows any same-named builtin or stdlib
        // function (so projects can define their own map/reduce/contains/...).
        // The VM behaves identically: compiled user chunks resolve first there.
        //
        // Hot path: resolve a user function in a SINGLE map lookup and run it
        // directly, bypassing the ~150-arm builtin match entirely. This is the
        // per-call cost of ordinary recursion, so it matters most.
        if let Some(func) = self.funcs.get(name) {
            if args.len() != func.params.len() {
                return Err(format!(
                    "function `{}` expects {} args, got {}",
                    name, func.params.len(), args.len()
                ));
            }
            // `#[comptime]` no-arg functions are const-evaluated once before main;
            // every call returns the precomputed constant instead of re-running.
            if args.is_empty() {
                if let Some(v) = self.comptime.borrow().get(name) {
                    return Ok(v.clone());
                }
            }
            // Functions with behavioural attributes take a slower, explicit path
            // (rare — the common case skips this in one branch). Pure hint/metadata
            // attributes don't change behaviour and use the fast path.
            if func.attrs.iter().any(|a| crate::bytecode::is_behavioural_attr(&a.name)) {
                return self.call_attributed(name, func, args);
            }
            // generators/async capture their scope and outlive this call, so they
            // build a plain frame; ordinary calls borrow one from the pool.
            if self.generators.contains(name) {
                let mut scope = Scope::with_capacity(func.params.len());
                for (p, v) in func.params.iter().zip(args.into_iter()) { scope.insert(p.clone(), v); }
                return Ok(Value::Generator(Rc::new(GenVal {
                    body: func.body.clone(), scope, cursor: RefCell::new(0),
                })));
            }
            if func.is_async {
                let mut scope = Scope::with_capacity(func.params.len());
                for (p, v) in func.params.iter().zip(args.into_iter()) { scope.insert(p.clone(), v); }
                return Ok(Value::Future(Rc::new(RefCell::new(FutureVal {
                    body: func.body.clone(), scope, state: FutureState::Pending,
                }))));
            }
            return self.run_user_body(func, args);
        }
        {
        // built-in functions
        match name {
            "type_of" => {
                let v = args.get(0).ok_or("type_of expects 1 argument")?;
                return Ok(Value::Str(v.type_name().to_string()));
            }
            // `#[hot_swap]` support: install a runtime replacement body for a
            // function declared hot-swappable. Errors if the target isn't marked.
            "hot_swap" => {
                let tgt = match args.get(0) { Some(Value::Str(s)) => s.clone(),
                    _ => return Err("hot_swap expects (name: string, closure)".into()) };
                let clo = args.get(1).cloned().ok_or("hot_swap expects (name, closure)")?;
                if !matches!(clo, Value::Closure(_)) {
                    return Err("hot_swap: second argument must be a closure".into());
                }
                match self.funcs.get(&tgt) {
                    Some(f) if f.attrs.iter().any(|a| a.name == "hot_swap") => {
                        self.swapped.borrow_mut().insert(tgt, clo);
                        return Ok(Value::Null);
                    }
                    Some(_) => return Err(format!("hot_swap: `{}` is not #[hot_swap]", tgt)),
                    None => return Err(format!("hot_swap: no function `{}`", tgt)),
                }
            }
            // `#[encrypt]` support: a real (obfuscation-grade, not cryptographic)
            // keyed XOR cipher, hex-encoded so the result is a printable string.
            // encrypt then decrypt with the same key round-trips.
            "encrypt" => {
                let s = str_arg(&args, 0, "encrypt")?;
                let key = str_arg(&args, 1, "encrypt")?;
                return Ok(Value::Str(xor_hex_encrypt(&s, &key)));
            }
            "decrypt" => {
                let s = str_arg(&args, 0, "decrypt")?;
                let key = str_arg(&args, 1, "decrypt")?;
                return Ok(Value::Str(xor_hex_decrypt(&s, &key)?));
            }
            // `#[anti_debug]` support: best-effort debugger detection (Linux reads
            // /proc/self/status TracerPid). Honest and documented as best-effort.
            "is_debugged" => return Ok(Value::Bool(detect_debugger())),
            // State Migration: transform a value of the old struct shape into the
            // new one using the matching `migrate from Old to New { ... }` block.
            // The block runs with `old` bound to the value and the old struct's
            // fields also in scope, and returns the new value.
            "migrate" => {
                let val = args.get(0).cloned().ok_or("migrate expects a value")?;
                let from = match &val {
                    Value::Struct(s) => s.borrow().type_name.clone(),
                    other => return Err(format!("migrate expects a struct value, got {}", other.type_name())),
                };
                let (_, body) = self.migrations.get(&from)
                    .ok_or_else(|| format!("no migration defined from `{}`", from))?;
                let mut scope = Scope::new();
                if let Value::Struct(s) = &val {
                    for (k, v) in s.borrow().fields.iter() { scope.insert(k.clone(), v.clone()); }
                }
                scope.insert("old".to_string(), val);
                return match self.exec_block(body, &mut scope)? {
                    Flow::Return(v) | Flow::Break(v) => Ok(v),
                    Flow::Normal | Flow::Continue => Ok(Value::Null),
                    Flow::Throw(e) => { *self.pending_throw.borrow_mut() = Some(e); Err(THROW_SENTINEL.to_string()) }
                };
            }
            // metadata attributes (version/since/intent/throws/deps/...): read an
            // attribute's argument value, making all metadata real and queryable.
            "meta_of" => {
                let tgt = str_arg(&args, 0, "meta_of")?;
                let key = str_arg(&args, 1, "meta_of")?;
                let f = self.funcs.get(&tgt)
                    .ok_or_else(|| format!("meta_of: no function `{}`", tgt))?;
                let val = f.attrs.iter().find(|a| a.name == key)
                    .map(|a| a.args.iter().map(|(k, v)| if k.is_empty() { v.clone() } else { format!("{}: {}", k, v) })
                        .collect::<Vec<_>>().join(", "))
                    .unwrap_or_default();
                return Ok(Value::Str(val));
            }
            // introspection: the attribute names on a function (every attribute is
            // captured, so even not-yet-behavioural ones are visible and usable).
            "attrs_of" => {
                let tgt = match args.get(0) { Some(Value::Str(s)) => s.clone(),
                    _ => return Err("attrs_of expects a function name".into()) };
                let f = self.funcs.get(&tgt)
                    .ok_or_else(|| format!("attrs_of: no function `{}`", tgt))?;
                let names: Vec<Value> = f.attrs.iter().map(|a| Value::Str(a.name.clone())).collect();
                return Ok(Value::Array(Rc::new(RefCell::new(names))));
            }
            // `#[time_travel]` support: the recorded past results (oldest first),
            // for rollback / inspection.
            "history_of" => {
                let tgt = match args.get(0) { Some(Value::Str(s)) => s.clone(),
                    _ => return Err("history_of expects a function name".into()) };
                let hist = self.history.borrow();
                let vals: Vec<Value> = hist.get(&tgt).map(|r| r.iter().cloned().collect()).unwrap_or_default();
                return Ok(Value::Array(Rc::new(RefCell::new(vals))));
            }
            // `#[profile]` support: how many times a profiled function was called.
            "profile_of" => {
                let tgt = match args.get(0) { Some(Value::Str(s)) => s.clone(),
                    _ => return Err("profile_of expects a function name".into()) };
                return Ok(Value::Int(*self.profile.borrow().get(&tgt).unwrap_or(&0)));
            }
            // `#[instrument]` support: call count of an instrumented function.
            "instrument_of" => {
                let tgt = match args.get(0) { Some(Value::Str(s)) => s.clone(),
                    _ => return Err("instrument_of expects a function name".into()) };
                return Ok(Value::Int(*self.instrument.borrow().get(&tgt).unwrap_or(&0)));
            }
            // `#[budget(n)]` support: the remaining call allowance (0 once spent).
            "budget_of" => {
                let tgt = match args.get(0) { Some(Value::Str(s)) => s.clone(),
                    _ => return Err("budget_of expects a function name".into()) };
                return Ok(Value::Int(*self.budget.borrow().get(&tgt).unwrap_or(&0)));
            }
            // `#[snapshot]` / `#[rollback]` support: capture and restore a named
            // value. `snapshot(id, value)` stores it; `rollback(id)` returns the
            // last snapshot (or null).
            "snapshot" => {
                let id = match args.get(0) { Some(Value::Str(s)) => s.clone(),
                    _ => return Err("snapshot expects (id: string, value)".into()) };
                let v = args.get(1).cloned().unwrap_or(Value::Null);
                self.snapshots.borrow_mut().insert(id, v.clone());
                return Ok(v);
            }
            "rollback" => {
                let id = match args.get(0) { Some(Value::Str(s)) => s.clone(),
                    _ => return Err("rollback expects an id string".into()) };
                return Ok(self.snapshots.borrow().get(&id).cloned().unwrap_or(Value::Null));
            }
            // `#[integrity]` support: a stable content hash of a function's body,
            // so a program can verify its own code hasn't been altered.
            "integrity_of" => {
                let tgt = match args.get(0) { Some(Value::Str(s)) => s.clone(),
                    _ => return Err("integrity_of expects a function name".into()) };
                let f = self.funcs.get(&tgt)
                    .ok_or_else(|| format!("integrity_of: no function `{}`", tgt))?;
                return Ok(Value::Int(body_hash(f)));
            }
            // ---- system interface (Phase 9): argv, env, files, stdio ----
            "args" => {
                let vals: Vec<Value> = self.cli_args.borrow().iter()
                    .map(|s| Value::Str(s.clone())).collect();
                return Ok(Value::Array(Rc::new(RefCell::new(vals))));
            }
            "env" => {
                let key = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("env expects a string name".into()),
                };
                return Ok(match std::env::var(&key) {
                    Ok(v) => Value::Str(v),
                    Err(_) => Value::Null,
                });
            }
            // --- TCP networking -------------------------------------------------
            // Blocking sockets addressed by integer handles. Enough to write real
            // servers and clients (and, on top, HTTP) directly in Nova.
            "tcp_listen" => {
                let addr = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("tcp_listen expects an address string like \"127.0.0.1:8080\"".into()),
                };
                return match std::net::TcpListener::bind(&addr) {
                    Ok(l) => Ok(Value::Int(self.sock_register(Sock::Listener(l)))),
                    Err(e) => self.fail_assert(format!("tcp_listen {}: {}", addr, e)),
                };
            }
            "tcp_accept" => {
                let h = match args.get(0) {
                    Some(Value::Int(n)) => *n,
                    _ => return Err("tcp_accept expects a listener handle".into()),
                };
                let accepted = {
                    let socks = self.sockets.borrow();
                    match socks.get(&h) {
                        Some(Sock::Listener(l)) => l.accept(),
                        _ => return Err(format!("tcp_accept: handle {} is not a listener", h)),
                    }
                };
                return match accepted {
                    Ok((stream, _)) => Ok(Value::Int(self.sock_register(Sock::Stream(stream)))),
                    Err(e) => self.fail_assert(format!("tcp_accept: {}", e)),
                };
            }
            "tcp_connect" => {
                let addr = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("tcp_connect expects an address string".into()),
                };
                return match std::net::TcpStream::connect(&addr) {
                    Ok(s) => Ok(Value::Int(self.sock_register(Sock::Stream(s)))),
                    Err(e) => self.fail_assert(format!("tcp_connect {}: {}", addr, e)),
                };
            }
            "tcp_read" => {
                use std::io::Read as _;
                let h = match args.get(0) { Some(Value::Int(n)) => *n, _ => return Err("tcp_read expects a connection handle".into()) };
                let n = match args.get(1) { Some(Value::Int(n)) => (*n).max(0) as usize, None => 65536, _ => return Err("tcp_read(conn, max_bytes)".into()) };
                let mut buf = vec![0u8; n];
                let got = {
                    let socks = self.sockets.borrow();
                    match socks.get(&h) {
                        Some(Sock::Stream(s)) => { let mut sr = s; sr.read(&mut buf) }
                        _ => return Err(format!("tcp_read: handle {} is not a connection", h)),
                    }
                };
                return match got {
                    Ok(k) => Ok(Value::Str(String::from_utf8_lossy(&buf[..k]).into_owned())),
                    Err(e) => self.fail_assert(format!("tcp_read: {}", e)),
                };
            }
            "tcp_write" => {
                use std::io::Write as _;
                let h = match args.get(0) { Some(Value::Int(n)) => *n, _ => return Err("tcp_write expects a connection handle".into()) };
                let data = match args.get(1) { Some(Value::Str(s)) => s.clone(), Some(v) => v.to_string(), None => return Err("tcp_write(conn, text)".into()) };
                let wrote = {
                    let socks = self.sockets.borrow();
                    match socks.get(&h) {
                        Some(Sock::Stream(s)) => { let mut sr = s; sr.write_all(data.as_bytes()).map(|_| data.len()) }
                        _ => return Err(format!("tcp_write: handle {} is not a connection", h)),
                    }
                };
                return match wrote {
                    Ok(k) => Ok(Value::Int(k as i64)),
                    Err(e) => self.fail_assert(format!("tcp_write: {}", e)),
                };
            }
            "tcp_close" => {
                let h = match args.get(0) { Some(Value::Int(n)) => *n, _ => return Err("tcp_close expects a handle".into()) };
                self.sockets.borrow_mut().remove(&h);
                return Ok(Value::Null);
            }
            // Binary-safe socket I/O: bytes as an array of ints 0..255, so binary
            // protocols (e.g. WebSocket frames) survive intact (unlike tcp_read,
            // which is UTF-8-lossy and meant for text).
            "tcp_read_bytes" => {
                use std::io::Read as _;
                let h = match args.get(0) { Some(Value::Int(n)) => *n, _ => return Err("tcp_read_bytes expects a connection handle".into()) };
                let n = match args.get(1) { Some(Value::Int(n)) => (*n).max(0) as usize, None => 65536, _ => return Err("tcp_read_bytes(conn, max_bytes)".into()) };
                let mut buf = vec![0u8; n];
                let got = {
                    let socks = self.sockets.borrow();
                    match socks.get(&h) {
                        Some(Sock::Stream(s)) => { let mut sr = s; sr.read(&mut buf) }
                        _ => return Err(format!("tcp_read_bytes: handle {} is not a connection", h)),
                    }
                };
                return match got {
                    Ok(k) => Ok(Value::Array(Rc::new(RefCell::new(
                        buf[..k].iter().map(|b| Value::Int(*b as i64)).collect())))),
                    Err(e) => self.fail_assert(format!("tcp_read_bytes: {}", e)),
                };
            }
            "tcp_write_bytes" => {
                use std::io::Write as _;
                let h = match args.get(0) { Some(Value::Int(n)) => *n, _ => return Err("tcp_write_bytes expects a connection handle".into()) };
                let bytes: Vec<u8> = match args.get(1) {
                    Some(Value::Array(a)) => a.borrow().iter().map(|v| match v { Value::Int(n) => (*n & 0xFF) as u8, _ => 0 }).collect(),
                    _ => return Err("tcp_write_bytes(conn, [int]) expects a byte array".into()),
                };
                let wrote = {
                    let socks = self.sockets.borrow();
                    match socks.get(&h) {
                        Some(Sock::Stream(s)) => { let mut sr = s; sr.write_all(&bytes).map(|_| bytes.len()) }
                        _ => return Err(format!("tcp_write_bytes: handle {} is not a connection", h)),
                    }
                };
                return match wrote {
                    Ok(k) => Ok(Value::Int(k as i64)),
                    Err(e) => self.fail_assert(format!("tcp_write_bytes: {}", e)),
                };
            }
            // DNS / hostfile: `resolve` uses the OS resolver (which honours
            // /etc/hosts), `hostname` reports this machine's name.
            "resolve" => {
                use std::net::ToSocketAddrs as _;
                let host = match args.get(0) { Some(Value::Str(s)) => s.clone(), _ => return Err("resolve expects a \"host\" or \"host:port\" string".into()) };
                let query = if host.contains(':') { host.clone() } else { format!("{}:0", host) };
                return match query.to_socket_addrs() {
                    Ok(addrs) => Ok(Value::Array(Rc::new(RefCell::new(
                        addrs.map(|a| Value::Str(a.ip().to_string())).collect())))),
                    Err(e) => self.fail_assert(format!("resolve {}: {}", host, e)),
                };
            }
            "hostname" => {
                let name = std::fs::read_to_string("/etc/hostname").ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .or_else(|| std::env::var("HOSTNAME").ok())
                    .unwrap_or_else(|| "localhost".to_string());
                return Ok(Value::Str(name));
            }
            // Encoding + hashing (dependency-free), enough for HTTP auth, data URIs
            // and the WebSocket upgrade handshake.
            "base64_encode" => {
                let s = match args.get(0) { Some(Value::Str(s)) => s.clone(), Some(v) => v.to_string(), None => return Err("base64_encode(text)".into()) };
                return Ok(Value::Str(b64_encode(s.as_bytes())));
            }
            "base64_decode" => {
                let s = match args.get(0) { Some(Value::Str(s)) => s.clone(), _ => return Err("base64_decode(text)".into()) };
                return match b64_decode(&s) {
                    Some(bytes) => Ok(Value::Str(String::from_utf8_lossy(&bytes).into_owned())),
                    None => self.fail_assert("base64_decode: invalid base64".to_string()),
                };
            }
            "sha1_hex" => {
                let s = match args.get(0) { Some(Value::Str(s)) => s.clone(), Some(v) => v.to_string(), None => return Err("sha1_hex(text)".into()) };
                return Ok(Value::Str(hex_of(&sha1_digest(s.as_bytes()))));
            }
            // WebSocket upgrade token: base64(sha1(key + GUID)) per RFC 6455.
            "ws_accept" => {
                let key = match args.get(0) { Some(Value::Str(s)) => s.clone(), _ => return Err("ws_accept(sec_websocket_key)".into()) };
                let mut data = key.into_bytes();
                data.extend_from_slice(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
                return Ok(Value::Str(b64_encode(&sha1_digest(&data))));
            }
            "read_file" => {
                let path = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("read_file expects a string path".into()),
                };
                return match std::fs::read_to_string(&path) {
                    Ok(s) => Ok(Value::Str(s)),
                    Err(e) => self.fail_assert(format!("cannot read {}: {}", path, e)),
                };
            }
            "write_file" | "append_file" => {
                let path = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err(format!("{} expects a string path", name)),
                };
                let data = match args.get(1) {
                    Some(Value::Str(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => return Err(format!("{} expects (path, text)", name)),
                };
                let res = if name == "write_file" {
                    std::fs::write(&path, &data)
                } else {
                    use std::io::Write as _;
                    std::fs::OpenOptions::new().create(true).append(true).open(&path)
                        .and_then(|mut f| f.write_all(data.as_bytes()))
                };
                return match res {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => self.fail_assert(format!("cannot write {}: {}", path, e)),
                };
            }
            "file_exists" => {
                let path = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("file_exists expects a string path".into()),
                };
                return Ok(Value::Bool(std::path::Path::new(&path).exists()));
            }
            "remove_file" => {
                let path = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("remove_file expects a string path".into()),
                };
                return match std::fs::remove_file(&path) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => self.fail_assert(format!("cannot remove {}: {}", path, e)),
                };
            }
            "read_line" | "input" => {
                if name == "input" {
                    if let Some(p) = args.get(0) {
                        use std::io::Write as _;
                        print!("{}", p);
                        let _ = std::io::stdout().flush();
                    }
                }
                let mut line = String::new();
                use std::io::BufRead as _;
                return match std::io::stdin().lock().read_line(&mut line) {
                    Ok(0) => Ok(Value::Null),
                    Ok(_) => {
                        if line.ends_with('\n') { line.pop(); }
                        if line.ends_with('\r') { line.pop(); }
                        Ok(Value::Str(line))
                    }
                    Err(e) => self.fail_assert(format!("cannot read stdin: {}", e)),
                };
            }
            "eprint" => {
                let line = args.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ");
                eprintln!("{}", line);
                return Ok(Value::Null);
            }
            "exit" => {
                let code = match args.get(0) {
                    Some(Value::Int(n)) => *n as i32,
                    None => 0,
                    _ => return Err("exit expects an integer code".into()),
                };
                std::process::exit(code);
            }
            "to_int" => {
                return Ok(match args.get(0) {
                    Some(Value::Str(s)) => match s.trim().parse::<i64>() {
                        Ok(n) => Value::Int(n),
                        Err(_) => Value::Null,
                    },
                    Some(Value::Int(n)) => Value::Int(*n),
                    Some(Value::Float(x)) => Value::Int(*x as i64),
                    _ => return Err("to_int expects a string or number".into()),
                });
            }
            "to_float" => {
                return Ok(match args.get(0) {
                    Some(Value::Str(s)) => match s.trim().parse::<f64>() {
                        Ok(x) => Value::Float(x),
                        Err(_) => Value::Null,
                    },
                    Some(Value::Int(n)) => Value::Float(*n as f64),
                    Some(Value::Float(x)) => Value::Float(*x),
                    _ => return Err("to_float expects a string or number".into()),
                });
            }
            "chr" => {
                let n = match args.get(0) {
                    Some(Value::Int(n)) => *n,
                    _ => return Err("chr expects an integer".into()),
                };
                return match u32::try_from(n).ok().and_then(char::from_u32) {
                    Some(c) => Ok(Value::Str(c.to_string())),
                    None => self.fail_assert(format!("chr: {} is not a valid code point", n)),
                };
            }
            "ord" => {
                let s = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("ord expects a string".into()),
                };
                return match s.chars().next() {
                    Some(c) => Ok(Value::Int(c as i64)),
                    None => self.fail_assert("ord: empty string".into()),
                };
            }
            // ---- system interface: processes, filesystem, time, env ----
            "exec" => {
                // exec(cmd) or exec(cmd, [arg, ...]) -> { code, stdout, stderr }
                let cmd = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("exec expects a command string".into()),
                };
                let mut command = std::process::Command::new(&cmd);
                if let Some(Value::Array(a)) = args.get(1) {
                    for item in a.borrow().iter() {
                        match item {
                            Value::Str(s) => { command.arg(s); }
                            other => { command.arg(other.to_string()); }
                        }
                    }
                } else if args.len() > 1 {
                    return Err("exec expects (command, [args])".into());
                }
                return match command.output() {
                    Ok(out) => {
                        let code = out.status.code().unwrap_or(-1) as i64;
                        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
                        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                        let map = vec![
                            (Value::Str("code".into()), Value::Int(code)),
                            (Value::Str("stdout".into()), Value::Str(stdout)),
                            (Value::Str("stderr".into()), Value::Str(stderr)),
                        ];
                        Ok(Value::Map(Rc::new(RefCell::new(map))))
                    }
                    Err(e) => self.fail_assert(format!("cannot exec {}: {}", cmd, e)),
                };
            }
            "list_dir" => {
                let path = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("list_dir expects a string path".into()),
                };
                return match std::fs::read_dir(&path) {
                    Ok(rd) => {
                        let mut names: Vec<Value> = rd
                            .filter_map(|e| e.ok())
                            .map(|e| Value::Str(e.file_name().to_string_lossy().into_owned()))
                            .collect();
                        // deterministic order (read_dir order is OS-dependent)
                        names.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
                        Ok(Value::Array(Rc::new(RefCell::new(names))))
                    }
                    Err(e) => self.fail_assert(format!("cannot list {}: {}", path, e)),
                };
            }
            "mkdir" => {
                let path = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("mkdir expects a string path".into()),
                };
                return match std::fs::create_dir_all(&path) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => self.fail_assert(format!("cannot mkdir {}: {}", path, e)),
                };
            }
            "cwd" => {
                return match std::env::current_dir() {
                    Ok(p) => Ok(Value::Str(p.to_string_lossy().into_owned())),
                    Err(e) => self.fail_assert(format!("cannot get cwd: {}", e)),
                };
            }
            "chdir" => {
                let path = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("chdir expects a string path".into()),
                };
                return match std::env::set_current_dir(&path) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => self.fail_assert(format!("cannot chdir {}: {}", path, e)),
                };
            }
            "now_ms" => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let ms = SystemTime::now().duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as i64).unwrap_or(0);
                return Ok(Value::Int(ms));
            }
            "sleep_ms" => {
                let n = match args.get(0) {
                    Some(Value::Int(n)) if *n >= 0 => *n as u64,
                    _ => return Err("sleep_ms expects a non-negative integer".into()),
                };
                std::thread::sleep(std::time::Duration::from_millis(n));
                return Ok(Value::Null);
            }
            "setenv" => {
                let key = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("setenv expects (name, value) strings".into()),
                };
                let val = match args.get(1) {
                    Some(Value::Str(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => return Err("setenv expects (name, value)".into()),
                };
                std::env::set_var(&key, &val);
                return Ok(Value::Null);
            }
            "print" => {
                print_values(&args);
                return Ok(Value::Null);
            }
            "send" => {
                let inst = match args.get(0) {
                    Some(Value::Struct(s)) => s.clone(),
                    _ => return Err("send expects a state machine as first argument".into()),
                };
                let event = match args.get(1) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("send expects an event string as second argument".into()),
                };
                let type_name = inst.borrow().type_name.clone();
                let (_, transitions) = self.machines.get(&type_name)
                    .ok_or_else(|| format!("{} is not a state machine", type_name))?;
                let current = match inst.borrow().fields.get("state") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => return Err("machine has no state".into()),
                };
                for (from, to, ev) in transitions {
                    if from == &current && ev == &event {
                        inst.borrow_mut().fields.insert("state".to_string(), Value::Str(to.clone()));
                        return Ok(Value::Str(to.clone()));
                    }
                }
                *self.pending_throw.borrow_mut() = Some(Value::Str(
                    format!("invalid transition: {} cannot handle '{}' in state {}", type_name, event, current)
                ));
                return Err(THROW_SENTINEL.to_string());
            }
            "state_of" => {
                let inst = match args.get(0) {
                    Some(Value::Struct(s)) => s.clone(),
                    _ => return Err("state_of expects a state machine".into()),
                };
                let st = inst.borrow().fields.get("state").cloned().unwrap_or(Value::Null);
                return Ok(st);
            }
            "assert" => {
                let cond = args.get(0).ok_or("assert expects a condition")?;
                if !cond.is_truthy() {
                    let msg = args.get(1).map(|v| v.to_string())
                        .unwrap_or_else(|| "assertion failed".to_string());
                    // throw so a test harness (or try/catch) can capture it
                    *self.pending_throw.borrow_mut() = Some(Value::Str(msg));
                    return Err(THROW_SENTINEL.to_string());
                }
                return Ok(Value::Null);
            }
            "assert_eq" => {
                let a = args.get(0).ok_or("assert_eq expects two values")?;
                let b = args.get(1).ok_or("assert_eq expects two values")?;
                if !values_eq(a, b) {
                    let msg = format!("assertion failed: {} != {}", a, b);
                    *self.pending_throw.borrow_mut() = Some(Value::Str(msg));
                    return Err(THROW_SENTINEL.to_string());
                }
                return Ok(Value::Null);
            }
            "assert_ne" => {
                let a = args.get(0).ok_or("assert_ne expects two values")?;
                let b = args.get(1).ok_or("assert_ne expects two values")?;
                if values_eq(a, b) {
                    return self.fail_assert(format!("assertion failed: {} == {}", a, b));
                }
                return Ok(Value::Null);
            }
            "assert_true" | "assert_false" => {
                let v = args.get(0).ok_or("assert_true/false expects a value")?;
                let want = name == "assert_true";
                if v.is_truthy() != want {
                    return self.fail_assert(format!("assertion failed: expected {}, got {}", want, v));
                }
                return Ok(Value::Null);
            }
            "assert_gt" | "assert_lt" => {
                let a = args.get(0).cloned().ok_or("assert_gt/lt expects two values")?;
                let b = args.get(1).cloned().ok_or("assert_gt/lt expects two values")?;
                let op = if name == "assert_gt" { BinOp::Gt } else { BinOp::Lt };
                let ok = eval_binop(op, a.clone(), b.clone()).map(|v| v.is_truthy()).unwrap_or(false);
                if !ok {
                    let sym = if name == "assert_gt" { ">" } else { "<" };
                    return self.fail_assert(format!("assertion failed: {} {} {} is false", a, sym, b));
                }
                return Ok(Value::Null);
            }
            "assert_contains" => {
                let hay = args.get(0).ok_or("assert_contains expects (collection, item)")?;
                let needle = args.get(1).ok_or("assert_contains expects (collection, item)")?;
                let found = match hay {
                    Value::Array(a) => a.borrow().iter().any(|x| values_eq(x, needle)),
                    Value::Str(s) => matches!(needle, Value::Str(n) if s.contains(n.as_str())),
                    other => return Err(format!("assert_contains: cannot search {}", other.type_name())),
                };
                if !found {
                    return self.fail_assert(format!("assertion failed: {} does not contain {}", hay, needle));
                }
                return Ok(Value::Null);
            }
            "len" => {
                let v = args.get(0).ok_or("len expects 1 argument")?;
                return match v {
                    Value::Array(a) => Ok(Value::Int(a.borrow().len() as i64)),
                    Value::Str(s) => Ok(Value::Int(s.chars().count() as i64)),
                    other => Err(format!("len expects array or string, got {}", other.type_name())),
                };
            }
            "push" => {
                let arr = args.get(0).ok_or("push expects (array, value)")?;
                let val = args.get(1).ok_or("push expects (array, value)")?.clone();
                return match arr {
                    Value::Array(a) => { a.borrow_mut().push(val); Ok(Value::Null) }
                    other => Err(format!("push expects array, got {}", other.type_name())),
                };
            }
            "pop" => {
                let arr = args.get(0).ok_or("pop expects (array)")?;
                return match arr {
                    Value::Array(a) => Ok(a.borrow_mut().pop().unwrap_or(Value::Null)),
                    other => Err(format!("pop expects array, got {}", other.type_name())),
                };
            }
            "str" => {
                let v = args.get(0).ok_or("str expects 1 argument")?;
                return Ok(Value::Str(v.to_string()));
            }
            "int" => {
                let v = args.get(0).ok_or("int expects 1 argument")?;
                return match v {
                    Value::Int(n) => Ok(Value::Int(*n)),
                    Value::BigInt(b) => Ok(Value::BigInt(b.clone())),
                    Value::Float(x) => Ok(Value::Int(*x as i64)),
                    Value::Str(s) => s.trim().parse::<i64>().map(Value::Int)
                        .map_err(|_| format!("cannot parse '{}' as int", s)),
                    Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
                    other => Err(format!("cannot convert {} to int", other.type_name())),
                };
            }
            "float" => {
                let v = args.get(0).ok_or("float expects 1 argument")?;
                return match v {
                    Value::Int(n) => Ok(Value::Float(*n as f64)),
                    Value::Float(x) => Ok(Value::Float(*x)),
                    Value::Str(s) => s.trim().parse::<f64>().map(Value::Float)
                        .map_err(|_| format!("cannot parse '{}' as float", s)),
                    other => Err(format!("cannot convert {} to float", other.type_name())),
                };
            }
            "abs" => {
                let v = args.get(0).ok_or("abs expects 1 argument")?;
                return match v {
                    Value::Int(n) => Ok(Value::Int(n.abs())),
                    Value::BigInt(b) => Ok(norm_big(b.abs())),
                    Value::Float(x) => Ok(Value::Float(x.abs())),
                    other => Err(format!("abs expects number, got {}", other.type_name())),
                };
            }
            "sqrt" => {
                let v = args.get(0).ok_or("sqrt expects 1 argument")?;
                let f = match v {
                    Value::Int(n) => *n as f64,
                    Value::Float(x) => *x,
                    other => return Err(format!("sqrt expects number, got {}", other.type_name())),
                };
                return Ok(Value::Float(f.sqrt()));
            }
            "array_fill" => {
                // array_fill(value, count) backs the [v; n] literal
                let fill = args.get(0).cloned().unwrap_or(Value::Null);
                let n = match args.get(1) {
                    Some(Value::Int(n)) if *n >= 0 => *n as usize,
                    _ => return Err("[v; n] expects a non-negative integer count".into()),
                };
                return Ok(Value::Array(Rc::new(RefCell::new(vec![fill; n]))));
            }
            "array" => {
                // array(n, fill) -> new array of n copies, or array() -> empty
                if args.is_empty() {
                    return Ok(Value::Array(Rc::new(RefCell::new(Vec::new()))));
                }
                let n = match args.get(0) {
                    Some(Value::Int(n)) if *n >= 0 => *n as usize,
                    _ => return Err("array(n, fill) expects a non-negative int size".into()),
                };
                let fill = args.get(1).cloned().unwrap_or(Value::Null);
                return Ok(Value::Array(Rc::new(RefCell::new(vec![fill; n]))));
            }
            "map" => {
                let arr = expect_array(args.get(0), "map")?;
                let f = args.get(1).cloned().ok_or("map(array, fn) expects 2 args")?;
                let items: Vec<Value> = arr.borrow().clone();
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.call_closure(&f, vec![item])?);
                }
                return Ok(Value::Array(Rc::new(RefCell::new(out))));
            }
            "filter" => {
                let arr = expect_array(args.get(0), "filter")?;
                let f = args.get(1).cloned().ok_or("filter(array, fn) expects 2 args")?;
                let items: Vec<Value> = arr.borrow().clone();
                let mut out = Vec::new();
                for item in items {
                    if self.call_closure(&f, vec![item.clone()])?.is_truthy() {
                        out.push(item);
                    }
                }
                return Ok(Value::Array(Rc::new(RefCell::new(out))));
            }
            "reduce" => {
                let arr = expect_array(args.get(0), "reduce")?;
                let f = args.get(1).cloned().ok_or("reduce(array, fn, init) expects 3 args")?;
                let init = args.get(2).cloned().ok_or("reduce(array, fn, init) expects 3 args")?;
                let items: Vec<Value> = arr.borrow().clone();
                let mut acc = init;
                for item in items {
                    acc = self.call_closure(&f, vec![acc, item])?;
                }
                return Ok(acc);
            }
            "range" => {
                // range(n) -> [0, 1, ..., n-1] ; range(a, b) -> [a..b)
                let (start, end) = match (args.get(0), args.get(1)) {
                    (Some(Value::Int(n)), None) => (0, *n),
                    (Some(Value::Int(a)), Some(Value::Int(b))) => (*a, *b),
                    _ => return Err("range(n) or range(a, b) expects integers".into()),
                };
                let mut v = Vec::new();
                let mut i = start;
                while i < end { v.push(Value::Int(i)); i += 1; }
                return Ok(Value::Array(Rc::new(RefCell::new(v))));
            }
            "dict" => {
                // dict() -> new empty key-value map
                if !args.is_empty() {
                    return Err("dict() takes no arguments".into());
                }
                return Ok(Value::Map(Rc::new(RefCell::new(Vec::new()))));
            }
            "chan" => {
                // chan() -> unbounded channel; chan(n) -> capacity hint (advisory)
                let cap = match args.get(0) {
                    Some(Value::Int(n)) if *n >= 0 => *n as usize,
                    None => 0,
                    _ => return Err("chan(cap) expects a non-negative integer".into()),
                };
                let c = ChannelVal { buffer: std::collections::VecDeque::new(), capacity: cap, closed: false };
                return Ok(Value::Channel(Rc::new(RefCell::new(c))));
            }
            "recv" => {
                let ch = args.get(0).ok_or("recv expects a channel")?.clone();
                return self.channel_recv(ch);
            }
            "send_to" => {
                // send_to(ch, v): function form of `ch <- v`
                let ch = args.get(0).ok_or("send_to expects (channel, value)")?.clone();
                let v = args.get(1).ok_or("send_to expects (channel, value)")?.clone();
                return match ch {
                    Value::Channel(c) => {
                        if c.borrow().closed {
                            *self.pending_throw.borrow_mut() =
                                Some(Value::Str("send on closed channel".into()));
                            return Err(THROW_SENTINEL.to_string());
                        }
                        c.borrow_mut().buffer.push_back(v);
                        Ok(Value::Null)
                    }
                    other => Err(format!("send_to expects a channel, got {}", other.type_name())),
                };
            }
            "close" => {
                let ch = args.get(0).ok_or("close expects a channel")?;
                return match ch {
                    Value::Channel(c) => { c.borrow_mut().closed = true; Ok(Value::Null) }
                    other => Err(format!("close expects a channel, got {}", other.type_name())),
                };
            }
            "chan_len" => {
                let ch = args.get(0).ok_or("chan_len expects a channel")?;
                return match ch {
                    Value::Channel(c) => Ok(Value::Int(c.borrow().buffer.len() as i64)),
                    other => Err(format!("chan_len expects a channel, got {}", other.type_name())),
                };
            }
            "await" => {
                // await(x): function form of `x.await`, drives a future/task
                let v = args.get(0).ok_or("await expects 1 argument")?.clone();
                return self.drive_to_value(v);
            }
            "is_pending" => {
                // introspection: true if a future/task hasn't resolved yet
                let v = args.get(0).ok_or("is_pending expects 1 argument")?;
                let pending = match v {
                    Value::Future(f) => matches!(f.borrow().state, FutureState::Pending | FutureState::Running),
                    Value::Task(t) => matches!(t.borrow().state, FutureState::Pending | FutureState::Running),
                    _ => false,
                };
                return Ok(Value::Bool(pending));
            }
            "map_set" => {
                let m = expect_map(args.get(0), "map_set")?;
                let key = args.get(1).cloned().ok_or("map_set(m, key, value) expects 3 args")?;
                let val = args.get(2).cloned().ok_or("map_set(m, key, value) expects 3 args")?;
                let mut entries = m.borrow_mut();
                if let Some(slot) = entries.iter_mut().find(|(k, _)| values_eq(k, &key)) {
                    slot.1 = val;
                } else {
                    entries.push((key, val));
                }
                return Ok(Value::Null);
            }
            "map_get" => {
                let m = expect_map(args.get(0), "map_get")?;
                let key = args.get(1).ok_or("map_get(m, key) expects 2 args")?;
                let entries = m.borrow();
                for (k, v) in entries.iter() {
                    if values_eq(k, key) { return Ok(v.clone()); }
                }
                // optional 3rd arg is a default
                return Ok(args.get(2).cloned().unwrap_or(Value::Null));
            }
            "map_has" => {
                let m = expect_map(args.get(0), "map_has")?;
                let key = args.get(1).ok_or("map_has(m, key) expects 2 args")?;
                let found = m.borrow().iter().any(|(k, _)| values_eq(k, key));
                return Ok(Value::Bool(found));
            }
            "map_del" => {
                let m = expect_map(args.get(0), "map_del")?;
                let key = args.get(1).ok_or("map_del(m, key) expects 2 args")?;
                let mut entries = m.borrow_mut();
                let before = entries.len();
                entries.retain(|(k, _)| !values_eq(k, key));
                return Ok(Value::Bool(entries.len() != before));
            }
            "map_keys" => {
                let m = expect_map(args.get(0), "map_keys")?;
                let keys: Vec<Value> = m.borrow().iter().map(|(k, _)| k.clone()).collect();
                return Ok(Value::Array(Rc::new(RefCell::new(keys))));
            }
            "map_values" => {
                let m = expect_map(args.get(0), "map_values")?;
                let vals: Vec<Value> = m.borrow().iter().map(|(_, v)| v.clone()).collect();
                return Ok(Value::Array(Rc::new(RefCell::new(vals))));
            }
            "map_len" => {
                let m = expect_map(args.get(0), "map_len")?;
                let n = m.borrow().len() as i64;
                return Ok(Value::Int(n));
            }
            _ => {}
        }

        // standard library (math.*, strings.*, arrays.*, or bare after `use`)
        if let Some(v) = call_stdlib(name, &args)? {
            return Ok(v);
        }
        } // end: only when no user function shadows the name

        if self.extern_funcs.contains(name) {
            return Err(format!(
                "extern function `{}` has no implementation: Nova does not support FFI yet",
                name
            ));
        }

        Err(format!("call to undefined function: {}", name))
    }

    // Produce the `k`-th (0-based) value of a generator by replaying its body and
    // stopping at the k-th `yield`. Returns Ok(None) when the body finishes before
    // reaching that yield (the generator is exhausted). Pure bodies give stable,
    // correct results; this is what makes even infinite generators usable.
    fn gen_produce(&self, gen: &GenVal, k: usize) -> Result<Option<Value>, String> {
        self.gen_ctx.borrow_mut().push(GenState { target: k, count: 0, value: None });
        let mut scope = gen.scope.clone();
        let result = self.exec_block(&gen.body, &mut scope);
        let ctx = self.gen_ctx.borrow_mut().pop().expect("gen_ctx underflow");
        match result {
            Err(e) if e == YIELD_STOP => Ok(ctx.value),
            Ok(_) => Ok(None), // body ran to completion: no k-th value
            Err(e) => Err(e),  // a real error or throw propagates
        }
    }

    // Fetch the k-th item a value yields under `for ... in`, matching the
    // ForEach statement's semantics (array element, string char, generator
    // yield). Returns None when the iterable is exhausted. Lazy, so the VM can
    // iterate even infinite generators with `break`.
    pub(crate) fn vm_iter_next(&self, v: &Value, k: usize) -> Result<Option<Value>, String> {
        match v {
            Value::Array(a) => {
                let arr = a.borrow();
                Ok(if k < arr.len() { Some(arr[k].clone()) } else { None })
            }
            Value::Str(s) => Ok(s.chars().nth(k).map(|c| Value::Str(c.to_string()))),
            Value::Generator(g) => self.gen_produce(g, k),
            other => Err(format!("cannot iterate over {}", other.type_name())),
        }
    }

    pub(crate) fn match_pattern(&self, pat: &Pattern, val: &Value, bindings: &mut Scope)
        -> Result<bool, String>
    {
        match pat {
            Pattern::Wildcard => Ok(true),
            Pattern::Binding(name) => {
                bindings.insert(name.clone(), val.clone());
                Ok(true)
            }
            Pattern::Int(n) => Ok(matches!(val, Value::Int(m) if m == n)),
            Pattern::Float(x) => Ok(matches!(val, Value::Float(y) if y == x)),
            Pattern::Str(s) => Ok(matches!(val, Value::Str(t) if t == s)),
            Pattern::Bool(b) => Ok(matches!(val, Value::Bool(c) if c == b)),
            Pattern::Null => Ok(matches!(val, Value::Null)),
            Pattern::Range { lo, hi, inclusive } => {
                match val {
                    Value::Int(n) => {
                        let ok = if *inclusive { *n >= *lo && *n <= *hi }
                                 else { *n >= *lo && *n < *hi };
                        Ok(ok)
                    }
                    _ => Ok(false),
                }
            }
            Pattern::Or(alts) => {
                for alt in alts {
                    let mut sub = bindings.clone();
                    if self.match_pattern(alt, val, &mut sub)? {
                        *bindings = sub;
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Pattern::Tuple(subs) => {
                match val {
                    Value::Array(a) => {
                        let arr = a.borrow();
                        if arr.len() != subs.len() { return Ok(false); }
                        for (p, v) in subs.iter().zip(arr.iter()) {
                            if !self.match_pattern(p, v, bindings)? { return Ok(false); }
                        }
                        Ok(true)
                    }
                    _ => Ok(false),
                }
            }
            Pattern::Struct { name, fields } => {
                match val {
                    Value::Struct(s) => {
                        let inst = s.borrow();
                        if !name.is_empty() && &inst.type_name != name { return Ok(false); }
                        for (fname, fpat) in fields {
                            match inst.fields.get(fname) {
                                Some(fv) => {
                                    if !self.match_pattern(fpat, fv, bindings)? { return Ok(false); }
                                }
                                None => return Ok(false),
                            }
                        }
                        Ok(true)
                    }
                    _ => Ok(false),
                }
            }
            Pattern::Slice { prefix, rest, suffix } => {
                let arr = match val {
                    Value::Array(a) => a.borrow(),
                    _ => return Ok(false),
                };
                let n = arr.len();
                match rest {
                    // exact-length match: every element pairs with a sub-pattern
                    None => {
                        if n != prefix.len() { return Ok(false); }
                        for (p, v) in prefix.iter().zip(arr.iter()) {
                            if !self.match_pattern(p, v, bindings)? { return Ok(false); }
                        }
                        Ok(true)
                    }
                    // open match: prefix from the front, suffix from the back, `...` soaks the middle
                    Some(rest_name) => {
                        if n < prefix.len() + suffix.len() { return Ok(false); }
                        for (p, v) in prefix.iter().zip(arr.iter()) {
                            if !self.match_pattern(p, v, bindings)? { return Ok(false); }
                        }
                        let suffix_start = n - suffix.len();
                        for (p, v) in suffix.iter().zip(arr[suffix_start..].iter()) {
                            if !self.match_pattern(p, v, bindings)? { return Ok(false); }
                        }
                        if let Some(name) = rest_name {
                            let mid: Vec<Value> = arr[prefix.len()..suffix_start].to_vec();
                            bindings.insert(name.clone(), Value::Array(Rc::new(RefCell::new(mid))));
                        }
                        Ok(true)
                    }
                }
            }
            Pattern::EnumVariant { name, sub } => {
                match val {
                    Value::Enum(e) => {
                        if &e.variant != name {
                            return Ok(false);
                        }
                        if sub.len() != e.data.len() {
                            return Ok(false);
                        }
                        for (p, v) in sub.iter().zip(e.data.iter()) {
                            if !self.match_pattern(p, v, bindings)? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    }
                    _ => Ok(false),
                }
            }
        }
    }

    pub(crate) fn call_closure(&self, f: &Value, args: Vec<Value>) -> Result<Value, String> {
        let c = match f {
            Value::Closure(c) => c.clone(),
            other => return Err(format!("cannot call {} as a function", other.type_name())),
        };
        if args.len() != c.params.len() {
            return Err(format!(
                "closure expects {} args, got {}",
                c.params.len(), args.len()
            ));
        }
        // start from captured environment, then bind parameters
        let mut scope: Scope = c.captured.clone();
        for (p, v) in c.params.iter().zip(args.into_iter()) {
            scope.insert(p.clone(), v);
        }
        match &c.body {
            LambdaBody::Expr(e) => self.eval(e, &scope),
            LambdaBody::Block(stmts) => {
                match self.exec_block(stmts, &mut scope)? {
                    Flow::Return(v) => Ok(v),
                    Flow::Break(v) => Ok(v),
                    Flow::Continue => Ok(Value::Null),
                    Flow::Throw(e) => {
                *self.pending_throw.borrow_mut() = Some(e);
                Err(THROW_SENTINEL.to_string())
            }
                    Flow::Normal => Ok(Value::Null),
                }
            }
        }
    }

    fn call_method(&self, receiver: Value, method: &str, args: &[Expr], scope: &Scope)
        -> Result<Value, String>
    {
        // evaluate args first
        let mut argvals = Vec::with_capacity(args.len());
        for a in args { argvals.push(self.eval(a, scope)?); }
        self.call_method_vals(receiver, method, argvals)
    }

    // Dispatch a method call on an already-evaluated receiver + arguments.
    // Shared by the tree-walker's `Expr::MethodCall` and the VM's `Method` op.
    pub(crate) fn call_method_vals(&self, receiver: Value, method: &str, argvals: Vec<Value>)
        -> Result<Value, String>
    {
        // built-in array methods
        if let Value::Array(ref a) = receiver {
            match method {
                "len" => return Ok(Value::Int(a.borrow().len() as i64)),
                "push" => {
                    let v = argvals.get(0).cloned().ok_or("push expects 1 argument")?;
                    a.borrow_mut().push(v);
                    return Ok(Value::Null);
                }
                "pop" => return Ok(a.borrow_mut().pop().unwrap_or(Value::Null)),
                "get" => {
                    let idx = as_int(argvals.get(0))?;
                    let arr = a.borrow();
                    if idx < 0 || idx as usize >= arr.len() {
                        return Ok(Value::Null);
                    }
                    return Ok(arr[idx as usize].clone());
                }
                _ => {}
            }
        }
        // built-in string methods
        if let Value::Str(ref s) = receiver {
            match method {
                "len" => return Ok(Value::Int(s.chars().count() as i64)),
                "upper" => return Ok(Value::Str(s.to_uppercase())),
                "lower" => return Ok(Value::Str(s.to_lowercase())),
                _ => {}
            }
        }
        // lazy generator methods
        if let Value::Generator(ref g) = receiver {
            match method {
                // advance one step: the next value, or null when exhausted
                "next" => {
                    let k = *g.cursor.borrow();
                    return match self.gen_produce(g, k)? {
                        Some(v) => { *g.cursor.borrow_mut() = k + 1; Ok(v) }
                        None => Ok(Value::Null),
                    };
                }
                // collect up to n values into an array (safe for infinite generators)
                "take" => {
                    let n = as_int(argvals.get(0))?;
                    let mut out = Vec::new();
                    let mut k = *g.cursor.borrow();
                    while (out.len() as i64) < n {
                        match self.gen_produce(g, k)? {
                            Some(v) => { out.push(v); k += 1; }
                            None => break,
                        }
                    }
                    *g.cursor.borrow_mut() = k;
                    return Ok(Value::Array(Rc::new(RefCell::new(out))));
                }
                _ => {}
            }
        }

        // user-defined struct methods — nested-map lookup with &str keys (no
        // per-call key allocation) and an Rc clone (no body-AST deep copy)
        if let Value::Struct(ref inst) = receiver {
            let type_name = inst.borrow().type_name.clone();
            let func = self.methods.get(type_name.as_str())
                .and_then(|m| m.get(method))
                .ok_or_else(|| format!("type {} has no method: {}", type_name, method))?
                .clone();
            // bind self + params
            let mut mscope: Scope = Scope::new();
            let mut pi = 0;
            // first param is `self` by convention if present
            if func.params.get(0).map(|p| p == "self").unwrap_or(false) {
                mscope.insert("self".to_string(), receiver.clone());
                pi = 1;
            }
            let expected = func.params.len() - pi;
            if argvals.len() != expected {
                return Err(format!(
                    "method {}.{} expects {} args, got {}",
                    type_name, method, expected, argvals.len()
                ));
            }
            for (p, v) in func.params[pi..].iter().zip(argvals.into_iter()) {
                mscope.insert(p.clone(), v);
            }
            return match self.exec_block(&func.body, &mut mscope)? {
                Flow::Return(v) => Ok(v),
                Flow::Break(v) => Ok(v),
                Flow::Continue => Ok(Value::Null),
                Flow::Throw(e) => {
                *self.pending_throw.borrow_mut() = Some(e);
                Err(THROW_SENTINEL.to_string())
            }
                Flow::Normal => Ok(Value::Null),
            };
        }

        Err(format!("cannot call method '{}' on {}", method, receiver.type_name()))
    }

    fn exec_block(&self, stmts: &[Stmt], scope: &mut Scope) -> Result<Flow, String> {
        // Collect deferred blocks and run them in reverse order on the way out,
        // regardless of how the block exits (normal, return, throw, or error).
        let mut deferred: Vec<&[Stmt]> = Vec::new();
        let outcome = self.exec_block_inner(stmts, scope, &mut deferred);
        // run defers LIFO; their own flow/errors are best-effort and don't mask the outcome
        for d in deferred.iter().rev() {
            let mut tmp: Vec<&[Stmt]> = Vec::new();
            let _ = self.exec_block_inner(d, scope, &mut tmp);
        }
        outcome
    }

    fn exec_block_inner<'a>(&self, stmts: &'a [Stmt], scope: &mut Scope, deferred: &mut Vec<&'a [Stmt]>)
        -> Result<Flow, String>
    {
        for stmt in stmts {
            if let Stmt::Defer(body) = stmt {
                deferred.push(body);
                continue;
            }
            match self.exec_stmt(stmt, scope)? {
                Flow::Return(v) => return Ok(Flow::Return(v)),
                Flow::Throw(v) => return Ok(Flow::Throw(v)),
                Flow::Break(v) => return Ok(Flow::Break(v)),
                Flow::Continue => return Ok(Flow::Continue),
                Flow::Normal => {}
            }
        }
        Ok(Flow::Normal)
    }

    pub(crate) fn exec_stmt(&self, stmt: &Stmt, scope: &mut Scope) -> Result<Flow, String> {
        match stmt {
            Stmt::Let { name, ty, value } => {
                let v = self.eval(value, scope)?;
                // if annotated with a refined type, the value must satisfy its predicate
                if let Some(tn) = ty {
                    if let Some(pred) = self.refinements.get(tn).cloned() {
                        let mut pscope = scope.clone();
                        pscope.insert("it".to_string(), v.clone());
                        if !self.eval(&pred, &pscope)?.is_truthy() {
                            return Err(format!("refinement `{}` violated by value {}", tn, v));
                        }
                    }
                }
                scope.insert(name.clone(), v);
                Ok(Flow::Normal)
            }
            Stmt::Assign { name, value } => {
                // `let` is optional: assigning to an unknown name defines it.
                let v = self.eval(value, scope)?;
                scope.insert(name.clone(), v);
                Ok(Flow::Normal)
            }
            Stmt::IndexAssign { base, index, value } => {
                let target = self.eval(base, scope)?;
                let idx = match target {
                    Value::Map(_) => self.eval(index, scope)?,
                    _ => Value::Int(self.eval_int(index, scope)?),
                };
                let v = self.eval(value, scope)?;
                index_set(&target, &idx, v)?;
                Ok(Flow::Normal)
            }
            Stmt::FieldAssign { base, field, value } => {
                let target = self.eval(base, scope)?;
                let v = self.eval(value, scope)?;
                field_set(&target, field, v)?;
                Ok(Flow::Normal)
            }
            Stmt::Expr(e) => {
                self.eval(e, scope)?;
                Ok(Flow::Normal)
            }
            Stmt::Return(e) => {
                let v = match e {
                    Some(e) => self.eval(e, scope)?,
                    None => Value::Null,
                };
                Ok(Flow::Return(v))
            }
            Stmt::If { cond, then, els } => {
                if self.eval(cond, scope)?.is_truthy() {
                    self.exec_block(then, scope)
                } else if let Some(els) = els {
                    self.exec_block(els, scope)
                } else {
                    Ok(Flow::Normal)
                }
            }
            Stmt::While { cond, body } => {
                let mut guard = 0u64;
                while self.eval(cond, scope)?.is_truthy() {
                    match self.exec_block(body, scope)? {
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        Flow::Throw(v) => return Ok(Flow::Throw(v)),
                        Flow::Break(_) => return Ok(Flow::Normal),
                        Flow::Continue | Flow::Normal => {}
                    }
                    guard += 1;
                    if guard > 100_000_000 {
                        return Err("while loop exceeded 100M iterations (likely infinite)".into());
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::ForRange { var, start, end, inclusive, body } => {
                let s = self.eval_int(start, scope)?;
                let e = self.eval_int(end, scope)?;
                let last = if *inclusive { e } else { e - 1 };
                let mut i = s;
                while i <= last {
                    scope.insert(var.clone(), Value::Int(i));
                    match self.exec_block(body, scope)? {
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        Flow::Throw(v) => return Ok(Flow::Throw(v)),
                        Flow::Break(_) => return Ok(Flow::Normal),
                        Flow::Continue | Flow::Normal => {}
                    }
                    i += 1;
                }
                Ok(Flow::Normal)
            }
            Stmt::ForEach { var, iter, body } => {
                let collection = self.eval(iter, scope)?;
                match collection {
                    Value::Array(a) => {
                        // snapshot to a Vec so mutation during iteration is well-defined
                        let items: Vec<Value> = a.borrow().clone();
                        for item in items {
                            scope.insert(var.clone(), item);
                            match self.exec_block(body, scope)? {
                                Flow::Return(v) => return Ok(Flow::Return(v)),
                                Flow::Throw(v) => return Ok(Flow::Throw(v)),
                                Flow::Break(_) => return Ok(Flow::Normal),
                                Flow::Continue | Flow::Normal => {}
                            }
                        }
                        Ok(Flow::Normal)
                    }
                    Value::Str(s) => {
                        for ch in s.chars() {
                            scope.insert(var.clone(), Value::Str(ch.to_string()));
                            match self.exec_block(body, scope)? {
                                Flow::Return(v) => return Ok(Flow::Return(v)),
                                Flow::Throw(v) => return Ok(Flow::Throw(v)),
                                Flow::Break(_) => return Ok(Flow::Normal),
                                Flow::Continue | Flow::Normal => {}
                            }
                        }
                        Ok(Flow::Normal)
                    }
                    Value::Generator(g) => {
                        // pull values lazily, one yield at a time
                        let mut k = 0;
                        while let Some(item) = self.gen_produce(&g, k)? {
                            k += 1;
                            scope.insert(var.clone(), item);
                            match self.exec_block(body, scope)? {
                                Flow::Return(v) => return Ok(Flow::Return(v)),
                                Flow::Throw(v) => return Ok(Flow::Throw(v)),
                                Flow::Break(_) => return Ok(Flow::Normal),
                                Flow::Continue | Flow::Normal => {}
                            }
                        }
                        Ok(Flow::Normal)
                    }
                    other => Err(format!("cannot iterate over {}", other.type_name())),
                }
            }
            Stmt::Throw(e) => {
                let v = self.eval(e, scope)?;
                Ok(Flow::Throw(v))
            }
            Stmt::Yield(e) => {
                let v = match e { Some(ex) => self.eval(ex, scope)?, None => Value::Null };
                let mut stack = self.gen_ctx.borrow_mut();
                let ctx = stack.last_mut()
                    .ok_or("`yield` used outside a generator function")?;
                if ctx.count == ctx.target {
                    // this is the value the current `produce` wants: capture and unwind
                    ctx.value = Some(v);
                    drop(stack);
                    Err(YIELD_STOP.to_string())
                } else {
                    // an earlier yield (already produced on a previous pull): skip it
                    ctx.count += 1;
                    Ok(Flow::Normal)
                }
            }
            Stmt::Break(e) => {
                let v = match e { Some(ex) => self.eval(ex, scope)?, None => Value::Null };
                Ok(Flow::Break(v))
            }
            Stmt::Continue => Ok(Flow::Continue),
            Stmt::Defer(body) => {
                // fallback: if a defer reaches exec_stmt directly, run it immediately
                self.exec_block(body, scope)
            }
            Stmt::TryCatch { body, catch_var, catch_body, finally_body } => {
                // Run the body. A throw can surface two ways:
                //  - Flow::Throw  (threw directly in this block)
                //  - Err(SENTINEL) with the value parked in pending_throw
                //    (threw deeper inside a called function)
                let body_result = self.exec_block(body, scope);
                let outcome = match body_result {
                    Ok(flow) => flow,
                    Err(e) => {
                        if e == YIELD_STOP {
                            // generator unwinding passes straight through try/catch
                            return Err(e);
                        } else if e == THROW_SENTINEL {
                            let v = self.pending_throw.borrow_mut().take().unwrap_or(Value::Null);
                            Flow::Throw(v)
                        } else if catch_body.is_some() {
                            // a runtime error (e.g. division by zero, index out of
                            // range) is a catchable exception when a handler exists;
                            // the catch variable receives the message as a string.
                            Flow::Throw(Value::Str(e))
                        } else {
                            // no handler: run finally, then propagate as before
                            if let Some(fin) = finally_body {
                                self.exec_block(fin, scope)?;
                            }
                            return Err(e);
                        }
                    }
                };

                let result = match outcome {
                    Flow::Throw(err) => {
                        if let Some(catch) = catch_body {
                            if let Some(var) = catch_var {
                                scope.insert(var.clone(), err);
                            }
                            // catch body itself may throw/return
                            match self.exec_block(catch, scope) {
                                Ok(flow) => flow,
                                Err(e) => {
                                    if let Some(fin) = finally_body {
                                        self.exec_block(fin, scope)?;
                                    }
                                    return Err(e);
                                }
                            }
                        } else {
                            Flow::Normal // no handler — swallow; finally still runs
                        }
                    }
                    other => other, // Normal / Return propagate after finally
                };

                if let Some(fin) = finally_body {
                    match self.exec_block(fin, scope)? {
                        Flow::Normal => {}
                        other => return Ok(other), // return/throw in finally wins
                    }
                }
                Ok(result)
            }
        }
    }

    // Slice an array or string by a (possibly open, possibly inclusive) range.
    // Negative bounds count from the end: a[-2..] is the last two elements.
    fn eval_slice(&self, base: &Expr, lo: &Option<Box<Expr>>, hi: &Option<Box<Expr>>,
                  inclusive: bool, scope: &Scope) -> Result<Value, String> {
        let target = self.eval(base, scope)?;
        let lo = match lo { Some(e) => Some(self.eval_int(e, scope)?), None => None };
        let hi = match hi { Some(e) => Some(self.eval_int(e, scope)?), None => None };
        do_slice(&target, lo, hi, inclusive)
    }

    fn eval_int(&self, e: &Expr, scope: &Scope) -> Result<i64, String> {
        match self.eval(e, scope)? {
            Value::Int(n) => Ok(n),
            other => Err(format!("expected integer, got {}", other.type_name())),
        }
    }

    pub(crate) fn eval(&self, e: &Expr, scope: &Scope) -> Result<Value, String> {
        match e {
            Expr::Int(n) => Ok(Value::Int(*n)),
            Expr::BigIntLit(s) => {
                use std::str::FromStr;
                BigInt::from_str(s).map(norm_big)
                    .map_err(|_| format!("bad big integer literal: {}", s))
            }
            Expr::Float(x) => Ok(Value::Float(*x)),
            Expr::Str(s) => Ok(Value::Str(s.clone())),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Null => Ok(Value::Null),
            Expr::At { pos, expr } => {
                // record the source position, then evaluate transparently
                self.cur_pos.set(*pos);
                self.eval(expr, scope)
            }
            Expr::Array(elems) => {
                let mut vals = Vec::with_capacity(elems.len());
                for e in elems {
                    vals.push(self.eval(e, scope)?);
                }
                Ok(Value::Array(Rc::new(RefCell::new(vals))))
            }
            Expr::MapLit(entries) => {
                let mut pairs = Vec::with_capacity(entries.len());
                for (k, v) in entries {
                    let kv = self.eval(k, scope)?;
                    let vv = self.eval(v, scope)?;
                    pairs.push((kv, vv));
                }
                Ok(build_map(pairs))
            }
            Expr::SetLit(elems) => {
                let mut vals = Vec::with_capacity(elems.len());
                for e in elems { vals.push(self.eval(e, scope)?); }
                Ok(build_set(vals))
            }
            Expr::Comprehension { body, var, iter, cond } => {
                let src = self.eval(iter, scope)?;
                let items: Vec<Value> = match src {
                    Value::Array(a) => a.borrow().clone(),
                    Value::Map(m) => m.borrow().iter().map(|(k, _)| k.clone()).collect(),
                    other => return Err(format!("cannot iterate over {} in comprehension", other.type_name())),
                };
                let mut out = Vec::new();
                let mut local = scope.clone();
                for item in items {
                    local.insert(var.clone(), item);
                    let keep = match cond {
                        Some(c) => self.eval(c, &local)?.is_truthy(),
                        None => true,
                    };
                    if keep {
                        out.push(self.eval(body, &local)?);
                    }
                }
                Ok(Value::Array(Rc::new(RefCell::new(out))))
            }
            Expr::FmtStr(parts) => {
                let mut s = String::new();
                for part in parts {
                    match part {
                        FmtPart::Lit(t) => s.push_str(t),
                        FmtPart::Expr(e) => s.push_str(&self.eval(e, scope)?.to_string()),
                    }
                }
                Ok(Value::Str(s))
            }
            Expr::Index { base, index } => {
                // a[lo..hi] — indexing with a range produces a slice
                if let Expr::RangeLit { lo, hi, inclusive } = &**index {
                    return self.eval_slice(base, lo, hi, *inclusive, scope);
                }
                let target = self.eval(base, scope)?;
                // for a Map the index is a key (any value); otherwise an integer
                let idx = match target {
                    Value::Map(_) => self.eval(index, scope)?,
                    _ => Value::Int(self.eval_int(index, scope)?),
                };
                index_get(&target, &idx)
            }
            Expr::RangeLit { lo, hi, inclusive } => {
                // a standalone range materializes into an array of integers,
                // e.g. 1..4 -> [1, 2, 3], 1..=4 -> [1, 2, 3, 4]
                let start = match lo { Some(e) => self.eval_int(e, scope)?, None => 0 };
                let end = match hi {
                    Some(e) => self.eval_int(e, scope)?,
                    None => return Err("open-ended range has no concrete length".into()),
                };
                Ok(build_range(start, end, *inclusive))
            }
            Expr::StructLit { name, fields } => {
                let mut field_vals = Vec::with_capacity(fields.len());
                for (fname, fexpr) in fields {
                    field_vals.push((fname.clone(), self.eval(fexpr, scope)?));
                }
                self.make_struct(name, field_vals)
            }
            Expr::Field { base, field } => {
                let target = self.eval(base, scope)?;
                field_get(&target, field)
            }
            Expr::SafeField { base, field } => {
                // a?.b short-circuits to Null when a is Null
                let target = self.eval(base, scope)?;
                safe_field_get(&target, field)
            }
            Expr::MethodCall { base, method, args } => {
                // module-alias call: `m.sqrt(...)` where m is an alias for a stdlib module
                if let Expr::Ident(name) = &**base {
                    if !scope.contains_key(name) {
                        if let Some(root) = self.module_aliases.get(name) {
                            let mut argvals = Vec::with_capacity(args.len());
                            for a in args { argvals.push(self.eval(a, scope)?); }
                            let qualified = format!("{}.{}", root, method);
                            if let Some(v) = call_stdlib(&qualified, &argvals)? {
                                return Ok(v);
                            }
                            return Err(format!("module {} has no function {}", root, method));
                        }
                    }
                }
                let receiver = self.eval(base, scope)?;
                self.call_method(receiver, method, args, scope)
            }
            Expr::Lambda { params, body } => {
                // capture the current scope by snapshot (closures see values at creation time;
                // shared mutable state still works because arrays/structs are Rc-backed)
                Ok(Value::Closure(Rc::new(ClosureVal {
                    params: params.clone(),
                    body: (**body).clone(),
                    captured: scope.clone(),
                    vm_chunk: None,
                })))
            }
            Expr::CallValue { callee, args } => {
                let f = self.eval(callee, scope)?;
                let mut argvals = Vec::with_capacity(args.len());
                for a in args { argvals.push(self.eval(a, scope)?); }
                self.call_closure(&f, argvals)
            }
            Expr::Ident(name) => {
                // A bare unit enum variant (`None`/`Nil`) keeps its priority, but
                // we only hash the variant table when the program actually has
                // enums — so the common no-enum case resolves a local in one short
                // scan instead of a SipHash miss on every variable read.
                if !self.variants.is_empty() {
                    if let Some((enum_name, arity)) = self.variants.get(name) {
                        if *arity == 0 {
                            return Ok(Value::Enum(Rc::new(EnumVal {
                                enum_name: enum_name.clone(),
                                variant: name.clone(),
                                data: vec![],
                            })));
                        }
                    }
                }
                if let Some(v) = scope.get(name) {
                    return Ok(v.clone());
                }
                self.consts.borrow().get(name).cloned()
                    .ok_or_else(|| format!("undefined variable: {}", name))
            }
            Expr::Unary { op, expr } => {
                let v = self.eval(expr, scope)?;
                eval_unop(*op, v)
            }
            Expr::Binary { op, lhs, rhs } => {
                // short-circuit for && and ||
                match op {
                    BinOp::And => {
                        let l = self.eval(lhs, scope)?;
                        if !l.is_truthy() { return Ok(Value::Bool(false)); }
                        return Ok(Value::Bool(self.eval(rhs, scope)?.is_truthy()));
                    }
                    BinOp::Or => {
                        let l = self.eval(lhs, scope)?;
                        if l.is_truthy() { return Ok(Value::Bool(true)); }
                        return Ok(Value::Bool(self.eval(rhs, scope)?.is_truthy()));
                    }
                    _ => {}
                }
                let l = self.eval(lhs, scope)?;
                let r = self.eval(rhs, scope)?;
                // Fast path for the overwhelmingly common Int-op-Int case: skip
                // the big `eval_binop` dispatch and go straight to a checked op
                // (falling back to `eval_binop` on overflow so BigInt promotion
                // and every other combination stays byte-identical).
                if let (Value::Int(a), Value::Int(b)) = (&l, &r) {
                    let (a, b) = (*a, *b);
                    match op {
                        BinOp::Add => if let Some(v) = a.checked_add(b) { return Ok(Value::Int(v)); },
                        BinOp::Sub => if let Some(v) = a.checked_sub(b) { return Ok(Value::Int(v)); },
                        BinOp::Mul => if let Some(v) = a.checked_mul(b) { return Ok(Value::Int(v)); },
                        BinOp::Lt => return Ok(Value::Bool(a < b)),
                        BinOp::Le => return Ok(Value::Bool(a <= b)),
                        BinOp::Gt => return Ok(Value::Bool(a > b)),
                        BinOp::Ge => return Ok(Value::Bool(a >= b)),
                        BinOp::Eq => return Ok(Value::Bool(a == b)),
                        BinOp::Ne => return Ok(Value::Bool(a != b)),
                        _ => {}
                    }
                }
                eval_binop(*op, l, r)
            }
            Expr::Call { callee, args } => {
                let mut vals = self.take_args();
                for a in args {
                    vals.push(self.eval(a, scope)?);
                }
                // a local variable bound to a closure shadows a top-level function
                if let Some(v) = scope.get(callee) {
                    if matches!(v, Value::Closure(_)) {
                        let f = v.clone();
                        return self.call_closure(&f, vals);
                    }
                }
                self.call_named(callee, vals)
            }
            Expr::If { cond, then, els } => {
                if self.eval(cond, scope)?.is_truthy() {
                    self.eval(then, scope)
                } else {
                    self.eval(els, scope)
                }
            }
            Expr::Match { scrutinee, arms } => {
                let val = self.eval(scrutinee, scope)?;
                for arm in arms {
                    let mut bindings: Scope = Scope::new();
                    if self.match_pattern(&arm.pattern, &val, &mut bindings)? {
                        // evaluate in a scope extended with the pattern bindings
                        let mut arm_scope = scope.clone();
                        for (k, v) in bindings.iter() {
                            arm_scope.insert(k.clone(), v.clone());
                        }
                        // check guard
                        if let Some(guard) = &arm.guard {
                            if !self.eval(guard, &arm_scope)?.is_truthy() {
                                continue;
                            }
                        }
                        return self.eval(&arm.body, &arm_scope);
                    }
                }
                Err("no match arm matched (non-exhaustive match)".into())
            }
            Expr::Block { stmts, tail } => {
                let mut local = scope.clone();
                if let Flow::Return(v) = self.exec_block(stmts, &mut local)? {
                    return Ok(v);
                }
                match tail {
                    Some(t) => self.eval(t, &local),
                    None => Ok(Value::Null),
                }
            }
            Expr::Await(inner) => {
                let v = self.eval(inner, scope)?;
                self.drive_to_value(v)
            }
            Expr::Spawn(body) => {
                // Queue the body as a task; hand back a JoinHandle (Value::Task).
                let id = self.next_task_id.get();
                self.next_task_id.set(id + 1);
                let task = Rc::new(RefCell::new(TaskVal { id, state: FutureState::Pending }));
                self.task_bodies.borrow_mut().insert(id, (body.clone(), scope.clone()));
                self.ready_queue.borrow_mut().push_back(task.clone());
                Ok(Value::Task(task))
            }
            Expr::Send { chan, value } => {
                let ch = self.eval(chan, scope)?;
                let v = self.eval(value, scope)?;
                match ch {
                    Value::Channel(c) => {
                        {
                            let b = c.borrow();
                            if b.closed {
                                *self.pending_throw.borrow_mut() =
                                    Some(Value::Str("send on closed channel".into()));
                                return Err(THROW_SENTINEL.to_string());
                            }
                        }
                        c.borrow_mut().buffer.push_back(v);
                        Ok(Value::Null)
                    }
                    other => Err(format!("`<-` send expects a channel, got {}", other.type_name())),
                }
            }
            Expr::Recv(chan) => {
                let ch = self.eval(chan, scope)?;
                self.channel_recv(ch)
            }
            Expr::Select(arms) => self.eval_select(arms, scope),
        }
    }

    // Drive a Future or Task to a concrete value, running the scheduler as needed.
    fn drive_to_value(&self, v: Value) -> Result<Value, String> {
        match v {
            Value::Future(fut) => {
                // Inspect current state.
                let state = std::mem::replace(&mut fut.borrow_mut().state, FutureState::Running);
                match state {
                    FutureState::Done(val) => {
                        fut.borrow_mut().state = FutureState::Done(val.clone());
                        Ok(val)
                    }
                    FutureState::Failed(e) => {
                        *self.pending_throw.borrow_mut() = Some(e);
                        Err(THROW_SENTINEL.to_string())
                    }
                    FutureState::Running => {
                        Err("await on a future that is already running (cyclic await)".into())
                    }
                    FutureState::Pending => {
                        // Run the body now, in its captured scope.
                        let (body, mut sc) = {
                            let b = fut.borrow();
                            (b.body.clone(), b.scope.clone())
                        };
                        let result = self.run_async_body(&body, &mut sc);
                        match result {
                            Ok(val) => {
                                fut.borrow_mut().state = FutureState::Done(val.clone());
                                Ok(val)
                            }
                            Err(e) => {
                                if e == THROW_SENTINEL {
                                    let tv = self.pending_throw.borrow().clone().unwrap_or(Value::Null);
                                    fut.borrow_mut().state = FutureState::Failed(tv);
                                }
                                Err(e)
                            }
                        }
                    }
                }
            }
            Value::Task(task) => {
                let id = task.borrow().id;
                // If already done, return cached value.
                {
                    let st = &task.borrow().state;
                    if let FutureState::Done(val) = st { return Ok(val.clone()); }
                    if let FutureState::Failed(e) = st {
                        *self.pending_throw.borrow_mut() = Some(e.clone());
                        return Err(THROW_SENTINEL.to_string());
                    }
                }
                // Otherwise run it (and let other queued tasks make progress first).
                self.run_task(id, &task)?;
                let st = task.borrow().state_clone();
                match st {
                    FutureState::Done(val) => Ok(val),
                    FutureState::Failed(e) => {
                        *self.pending_throw.borrow_mut() = Some(e);
                        Err(THROW_SENTINEL.to_string())
                    }
                    _ => Ok(Value::Null),
                }
            }
            // Awaiting a plain value is a no-op: it's already resolved.
            other => Ok(other),
        }
    }

    // Run a single spawned task body to completion, recording its result.
    fn run_task(&self, id: u64, task: &Rc<RefCell<TaskVal>>) -> Result<(), String> {
        let entry = self.task_bodies.borrow_mut().remove(&id);
        let (body, mut sc) = match entry {
            Some(x) => x,
            None => return Ok(()), // already consumed
        };
        task.borrow_mut().state = FutureState::Running;
        match self.run_async_body(&body, &mut sc) {
            Ok(val) => { task.borrow_mut().state = FutureState::Done(val); Ok(()) }
            Err(e) => {
                if e == THROW_SENTINEL {
                    let tv = self.pending_throw.borrow().clone().unwrap_or(Value::Null);
                    task.borrow_mut().state = FutureState::Failed(tv);
                    // a throw in a spawned task is contained: it surfaces only on await
                    return Ok(());
                }
                Err(e)
            }
        }
    }

    // Execute an async body (function/spawn) and extract its return value,
    // honoring `return`, implicit tail value, and throw.
    fn run_async_body(&self, body: &[Stmt], scope: &mut Scope) -> Result<Value, String> {
        match self.exec_block(body, scope)? {
            Flow::Return(v) => Ok(v),
            Flow::Break(v) => Ok(v),
            Flow::Continue => Ok(Value::Null),
            Flow::Throw(e) => {
                *self.pending_throw.borrow_mut() = Some(e);
                Err(THROW_SENTINEL.to_string())
            }
            Flow::Normal => Ok(Value::Null),
        }
    }

    // Receive one value from a channel. If empty, drive queued tasks (which may
    // produce a value via `ch <- v`), then retry; error if nothing ever arrives.
    fn channel_recv(&self, ch: Value) -> Result<Value, String> {
        let c = match ch {
            Value::Channel(c) => c,
            other => return Err(format!("recv expects a channel, got {}", other.type_name())),
        };
        // Fast path: a value is already buffered.
        if let Some(v) = c.borrow_mut().buffer.pop_front() {
            return Ok(v);
        }
        // Otherwise, let spawned tasks run to try to fill the channel.
        let mut guard = 0;
        loop {
            let next = self.ready_queue.borrow_mut().pop_front();
            match next {
                Some(task) => {
                    let id = task.borrow().id;
                    self.run_task(id, &task)?;
                }
                None => break,
            }
            if let Some(v) = c.borrow_mut().buffer.pop_front() {
                return Ok(v);
            }
            guard += 1;
            if guard > 1_000_000 { break; }
        }
        if let Some(v) = c.borrow_mut().buffer.pop_front() {
            return Ok(v);
        }
        if c.borrow().closed {
            return Ok(Value::Null);
        }
        *self.pending_throw.borrow_mut() =
            Some(Value::Str("receive on empty channel with no pending senders (deadlock)".into()));
        Err(THROW_SENTINEL.to_string())
    }

    // select { <- ch => body, ... }: pick the first arm whose channel has a value.
    // If none is ready, drive queued tasks once and retry, like channel_recv.
    fn eval_select(&self, arms: &[crate::ast::SelectArm], scope: &Scope) -> Result<Value, String> {
        let mut guard = 0;
        loop {
            // Evaluate each arm's channel and test readiness.
            for arm in arms {
                let chv = self.eval(&arm.chan, scope)?;
                if let Value::Channel(c) = &chv {
                    let ready = !c.borrow().buffer.is_empty();
                    if ready {
                        let v = c.borrow_mut().buffer.pop_front().unwrap();
                        let mut arm_scope = scope.clone();
                        if let Some(name) = &arm.binding {
                            arm_scope.insert(name.clone(), v);
                        } else {
                            arm_scope.insert("_recv".to_string(), v);
                        }
                        return self.eval(&arm.body, &arm_scope);
                    }
                }
            }
            // Nothing ready: run a queued task to make progress.
            let next = self.ready_queue.borrow_mut().pop_front();
            match next {
                Some(task) => {
                    let id = task.borrow().id;
                    self.run_task(id, &task)?;
                }
                None => {
                    *self.pending_throw.borrow_mut() =
                        Some(Value::Str("select: all channels empty and no tasks to run".into()));
                    return Err(THROW_SENTINEL.to_string());
                }
            }
            guard += 1;
            if guard > 1_000_000 {
                return Err("select: scheduler exceeded step budget".into());
            }
        }
    }
}

// Exact arithmetic/comparison on two BigInts (already coerced).
fn big_binop(op: BinOp, a: BigInt, b: BigInt) -> Result<Value, String> {
    use BinOp::*;
    use num_traits::Pow;
    let v = match op {
        Add => norm_big(a + b),
        Sub => norm_big(a - b),
        Mul => norm_big(a * b),
        Div => { if b.is_zero() { return Err("division by zero".into()); } norm_big(a / b) }
        Rem => { if b.is_zero() { return Err("modulo by zero".into()); } norm_big(a % b) }
        Pow => {
            if b.is_negative() {
                Value::Float(a.to_f64().unwrap_or(f64::NAN).powf(b.to_f64().unwrap_or(0.0)))
            } else {
                match b.to_u32() { Some(e) => norm_big(a.pow(e)), None => return Err("exponent too large".into()) }
            }
        }
        BitAnd => norm_big(a & b), BitOr => norm_big(a | b), BitXor => norm_big(a ^ b),
        Shl => { let s = b.to_u32().ok_or("shift too large")?; norm_big(a << s) }
        Shr => { let s = b.to_u32().ok_or("shift too large")?; norm_big(a >> s) }
        Eq => Value::Bool(a == b), Ne => Value::Bool(a != b),
        Lt => Value::Bool(a < b), Le => Value::Bool(a <= b),
        Gt => Value::Bool(a > b), Ge => Value::Bool(a >= b),
        And | Or => return Err("logical op on integers".into()),
    };
    Ok(v)
}

// ---------------------------------------------------------------------------
// Heap-value helpers shared by the tree-walker and the bytecode VM, so native
// VM opcodes produce results byte-identical to `nova run`. Each takes
// already-evaluated `Value`s (the VM evaluates sub-expressions itself).
// ---------------------------------------------------------------------------

// Build a Nova map from key/value pairs, last-write-wins on duplicate keys and
// preserving first-occurrence order (matching `Expr::MapLit`).
pub(crate) fn build_map(entries: Vec<(Value, Value)>) -> Value {
    let mut pairs: Vec<(Value, Value)> = Vec::with_capacity(entries.len());
    for (kv, vv) in entries {
        if let Some(slot) = pairs.iter_mut().find(|(ek, _)| values_eq(ek, &kv)) {
            slot.1 = vv;
        } else {
            pairs.push((kv, vv));
        }
    }
    Value::Map(Rc::new(RefCell::new(pairs)))
}

// Build a Nova set (a map from element -> null) preserving uniqueness and
// first-occurrence order (matching `Expr::SetLit`).
pub(crate) fn build_set(elems: Vec<Value>) -> Value {
    let mut pairs: Vec<(Value, Value)> = Vec::with_capacity(elems.len());
    for ev in elems {
        if !pairs.iter().any(|(k, _)| values_eq(k, &ev)) {
            pairs.push((ev, Value::Null));
        }
    }
    Value::Map(Rc::new(RefCell::new(pairs)))
}

// Materialize `lo..hi` / `lo..=hi` into an array of integers (matching the
// concrete-bounds case of `Expr::RangeLit`).
pub(crate) fn build_range(lo: i64, hi: i64, inclusive: bool) -> Value {
    let last = if inclusive { hi } else { hi - 1 };
    let mut out = Vec::new();
    let mut i = lo;
    while i <= last { out.push(Value::Int(i)); i += 1; }
    Value::Array(Rc::new(RefCell::new(out)))
}

// Element/char/key read: `base[idx]` for a non-range index (matching the
// non-slice path of `Expr::Index`).
pub(crate) fn index_get(target: &Value, idx: &Value) -> Result<Value, String> {
    if let Value::Map(m) = target {
        for (k, v) in m.borrow().iter() {
            if values_eq(k, idx) { return Ok(v.clone()); }
        }
        return Ok(Value::Null);
    }
    let i = match idx {
        Value::Int(n) => *n,
        other => return Err(format!("expected integer, got {}", other.type_name())),
    };
    match target {
        Value::Array(a) => {
            let arr = a.borrow();
            if i < 0 || i as usize >= arr.len() {
                return Err(format!("index {} out of bounds (len {})", i, arr.len()));
            }
            Ok(arr[i as usize].clone())
        }
        Value::Str(s) => {
            let chars: Vec<char> = s.chars().collect();
            if i < 0 || i as usize >= chars.len() {
                return Err(format!("string index {} out of bounds (len {})", i, chars.len()));
            }
            Ok(Value::Str(chars[i as usize].to_string()))
        }
        other => Err(format!("cannot index into {}", other.type_name())),
    }
}

// Field read `base.field` (matching `Expr::Field`).
pub(crate) fn field_get(target: &Value, field: &str) -> Result<Value, String> {
    match target {
        Value::Struct(inst) => {
            let inst = inst.borrow();
            inst.fields.get(field).cloned()
                .ok_or_else(|| format!("struct {} has no field: {}", inst.type_name, field))
        }
        other => Err(format!("cannot access field '{}' on {}", field, other.type_name())),
    }
}

// Optional-chained field read `base?.field` (matching `Expr::SafeField`).
pub(crate) fn safe_field_get(target: &Value, field: &str) -> Result<Value, String> {
    match target {
        Value::Null => Ok(Value::Null),
        Value::Struct(inst) => {
            let inst = inst.borrow();
            Ok(inst.fields.get(field).cloned().unwrap_or(Value::Null))
        }
        other => Err(format!("cannot access field '{}' on {}", field, other.type_name())),
    }
}

// Slice `base[lo..hi]` / `base[lo..=hi]` over an array or string, with negative
// indices counting from the end and bounds clamped (matching `eval_slice`).
pub(crate) fn do_slice(target: &Value, lo: Option<i64>, hi: Option<i64>, inclusive: bool)
    -> Result<Value, String>
{
    let len = match target {
        Value::Array(a) => a.borrow().len(),
        Value::Str(s) => s.chars().count(),
        other => return Err(format!("cannot slice {}", other.type_name())),
    } as i64;
    let start = match lo {
        Some(v) => if v < 0 { (len + v).max(0) } else { v.min(len) },
        None => 0,
    };
    let mut end = match hi {
        Some(raw) => {
            let mut x = if raw < 0 { len + raw } else { raw };
            if inclusive { x += 1; }
            x.clamp(0, len)
        }
        None => len,
    };
    if end < start { end = start; }
    let (s, e) = (start as usize, end as usize);
    match target {
        Value::Array(a) => Ok(Value::Array(Rc::new(RefCell::new(a.borrow()[s..e].to_vec())))),
        Value::Str(string) => {
            let chars: Vec<char> = string.chars().collect();
            Ok(Value::Str(chars[s..e].iter().collect()))
        }
        _ => unreachable!(),
    }
}

// Element/key write `base[idx] = v` (matching `Stmt::IndexAssign`).
pub(crate) fn index_set(target: &Value, idx: &Value, v: Value) -> Result<(), String> {
    if let Value::Map(m) = target {
        let mut entries = m.borrow_mut();
        if let Some(slot) = entries.iter_mut().find(|(k, _)| values_eq(k, idx)) {
            slot.1 = v;
        } else {
            entries.push((idx.clone(), v));
        }
        return Ok(());
    }
    let i = match idx {
        Value::Int(n) => *n,
        other => return Err(format!("expected integer, got {}", other.type_name())),
    };
    match target {
        Value::Array(a) => {
            let mut arr = a.borrow_mut();
            if i < 0 || i as usize >= arr.len() {
                return Err(format!("index {} out of bounds (len {})", i, arr.len()));
            }
            arr[i as usize] = v;
            Ok(())
        }
        other => Err(format!("cannot index-assign into {}", other.type_name())),
    }
}

// Field write `base.field = v` (matching `Stmt::FieldAssign`).
pub(crate) fn field_set(target: &Value, field: &str, v: Value) -> Result<(), String> {
    match target {
        Value::Struct(inst) => {
            let mut inst = inst.borrow_mut();
            if !inst.fields.contains_key(field) {
                return Err(format!("struct {} has no field: {}", inst.type_name, field));
            }
            inst.fields.insert(field.to_string(), v);
            Ok(())
        }
        other => Err(format!("cannot assign field '{}' on {}", field, other.type_name())),
    }
}

// The `print` builtin's output, shared by the interpreter and the VM.
pub(crate) fn print_values(args: &[Value]) {
    let line = args.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ");
    println!("{}", line);
}

// Apply a unary operator. Shared by the tree-walker and the bytecode VM so both
// have identical semantics.
pub(crate) fn eval_unop(op: UnOp, v: Value) -> Result<Value, String> {
    match (op, v) {
        (UnOp::Neg, Value::Int(n)) => Ok(Value::Int(-n)),
        (UnOp::Neg, Value::Float(x)) => Ok(Value::Float(-x)),
        (UnOp::Neg, Value::BigInt(b)) => Ok(norm_big(-b.clone())),
        (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
        (UnOp::Not, v) => Ok(Value::Bool(!v.is_truthy())),
        (UnOp::BitNot, Value::Int(n)) => Ok(Value::Int(!n)),
        (op, v) => Err(format!("cannot apply {:?} to {}", op, v.type_name())),
    }
}

// Keyed XOR cipher, hex-encoded (for `#[encrypt]`). Obfuscation-grade, not
// cryptographic — documented as such. Round-trips with the same key.
fn xor_hex_encrypt(s: &str, key: &str) -> String {
    let kb = key.as_bytes();
    if kb.is_empty() { return hex_encode(s.as_bytes()); }
    let out: Vec<u8> = s.bytes().enumerate().map(|(i, b)| b ^ kb[i % kb.len()]).collect();
    hex_encode(&out)
}
fn xor_hex_decrypt(s: &str, key: &str) -> Result<String, String> {
    let bytes = hex_decode(s).ok_or("decrypt: input is not valid hex")?;
    let kb = key.as_bytes();
    let out: Vec<u8> = if kb.is_empty() { bytes }
        else { bytes.iter().enumerate().map(|(i, b)| b ^ kb[i % kb.len()]).collect() };
    String::from_utf8(out).map_err(|_| "decrypt: result is not valid UTF-8 (wrong key?)".into())
}
fn hex_encode(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b { s.push_str(&format!("{:02x}", x)); }
    s
}
fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 { return None; }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i+2], 16).ok()).collect()
}

// Best-effort debugger detection (for `#[anti_debug]`). On Linux, a non-zero
// TracerPid in /proc/self/status means a debugger is attached. Honest and
// documented as best-effort (other platforms return false).
fn detect_debugger() -> bool {
    std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("TracerPid:")).map(|l| l.to_string()))
        .map(|l| l.split_whitespace().nth(1).map_or(false, |n| n != "0"))
        .unwrap_or(false)
}

// A stable FNV-1a content hash of a function's body (for `#[integrity]`). Based on
// the Debug rendering of the AST, so it changes iff the code changes; masked to a
// non-negative i64 so it prints cleanly.
fn body_hash(f: &Func) -> i64 {
    let text = format!("{:?}{:?}", f.params, f.body);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() { h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3); }
    (h >> 1) as i64
}

pub(crate) fn eval_binop(op: BinOp, l: Value, r: Value) -> Result<Value, String> {
    use Value::*;
    // Fast path for Int-op-Int — the dominant case in loops and arithmetic. This
    // is also reached from the bytecode VM's `Op::Bin`, so the VM benefits too.
    // On overflow we fall through to the general arms (BigInt promotion).
    if let (Int(a), Int(b)) = (&l, &r) {
        let (a, b) = (*a, *b);
        match op {
            BinOp::Add => if let Some(v) = a.checked_add(b) { return Ok(Int(v)); },
            BinOp::Sub => if let Some(v) = a.checked_sub(b) { return Ok(Int(v)); },
            BinOp::Mul => if let Some(v) = a.checked_mul(b) { return Ok(Int(v)); },
            BinOp::Lt => return Ok(Bool(a < b)),
            BinOp::Le => return Ok(Bool(a <= b)),
            BinOp::Gt => return Ok(Bool(a > b)),
            BinOp::Ge => return Ok(Bool(a >= b)),
            BinOp::Eq => return Ok(Bool(a == b)),
            BinOp::Ne => return Ok(Bool(a != b)),
            _ => {}
        }
    }
    // Fast path for Float-op-Float — the common case in numeric/float code and
    // in the bytecode VM's float loops. Matches the general `is_num` arms.
    if let (Float(a), Float(b)) = (&l, &r) {
        let (a, b) = (*a, *b);
        match op {
            BinOp::Add => return Ok(Float(a + b)),
            BinOp::Sub => return Ok(Float(a - b)),
            BinOp::Mul => return Ok(Float(a * b)),
            BinOp::Div => return Ok(Float(a / b)),
            BinOp::Lt => return Ok(Bool(a < b)),
            BinOp::Le => return Ok(Bool(a <= b)),
            BinOp::Gt => return Ok(Bool(a > b)),
            BinOp::Ge => return Ok(Bool(a >= b)),
            _ => {}
        }
    }
    // BigInt path: if either side is a BigInt (or an Int that must promote),
    // do exact integer math. Floats still win when present (handled below).
    if matches!((&l, &r), (BigInt(_), _) | (_, BigInt(_)))
        && !matches!(&l, Float(_)) && !matches!(&r, Float(_)) {
        if let (Some(a), Some(b)) = (as_big(&l), as_big(&r)) {
            return big_binop(op, a, b);
        }
    }
    match (op, &l, &r) {
        // integer arithmetic — promote to BigInt on overflow
        (BinOp::Add, Int(a), Int(b)) => Ok(match a.checked_add(*b) {
            Some(v) => Int(v), None => norm_big(num_bigint::BigInt::from(*a) + num_bigint::BigInt::from(*b)),
        }),
        (BinOp::Sub, Int(a), Int(b)) => Ok(match a.checked_sub(*b) {
            Some(v) => Int(v), None => norm_big(num_bigint::BigInt::from(*a) - num_bigint::BigInt::from(*b)),
        }),
        (BinOp::Mul, Int(a), Int(b)) => Ok(match a.checked_mul(*b) {
            Some(v) => Int(v), None => norm_big(num_bigint::BigInt::from(*a) * num_bigint::BigInt::from(*b)),
        }),
        (BinOp::Div, Int(a), Int(b)) => {
            if *b == 0 { Err("division by zero".into()) } else { Ok(Int(a / b)) }
        }
        (BinOp::Rem, Int(a), Int(b)) => {
            if *b == 0 { Err("modulo by zero".into()) } else { Ok(Int(a % b)) }
        }
        (BinOp::Pow, Int(a), Int(b)) => {
            if *b < 0 {
                Ok(Float((*a as f64).powi(*b as i32)))
            } else if *b <= u32::MAX as i64 {
                match a.checked_pow(*b as u32) {
                    Some(v) => Ok(Int(v)),
                    None => {
                        use num_traits::Pow;
                        Ok(norm_big(num_bigint::BigInt::from(*a).pow(*b as u32)))
                    }
                }
            } else {
                Ok(Float((*a as f64).powf(*b as f64)))
            }
        }
        (BinOp::BitOr, Int(a), Int(b)) => Ok(Int(a | b)),
        (BinOp::BitXor, Int(a), Int(b)) => Ok(Int(a ^ b)),
        (BinOp::BitAnd, Int(a), Int(b)) => Ok(Int(a & b)),
        (BinOp::Shl, Int(a), Int(b)) => Ok(Int(a.wrapping_shl(*b as u32))),
        (BinOp::Shr, Int(a), Int(b)) => Ok(Int(a.wrapping_shr(*b as u32))),
        (BinOp::BitOr, _, _) | (BinOp::BitXor, _, _) | (BinOp::BitAnd, _, _)
        | (BinOp::Shl, _, _) | (BinOp::Shr, _, _) =>
            Err("bitwise operators require integer operands".into()),
        // float arithmetic (with int promotion)
        (BinOp::Add, _, _) if is_num(&l) && is_num(&r) => Ok(Float(as_f(&l) + as_f(&r))),
        (BinOp::Sub, _, _) if is_num(&l) && is_num(&r) => Ok(Float(as_f(&l) - as_f(&r))),
        (BinOp::Mul, _, _) if is_num(&l) && is_num(&r) => Ok(Float(as_f(&l) * as_f(&r))),
        (BinOp::Div, _, _) if is_num(&l) && is_num(&r) => Ok(Float(as_f(&l) / as_f(&r))),
        (BinOp::Rem, _, _) if is_num(&l) && is_num(&r) => Ok(Float(as_f(&l) % as_f(&r))),
        (BinOp::Pow, _, _) if is_num(&l) && is_num(&r) => Ok(Float(as_f(&l).powf(as_f(&r)))),
        // string concat
        (BinOp::Add, Str(a), Str(b)) => Ok(Str(format!("{}{}", a, b))),
        (BinOp::Add, Str(a), _) => Ok(Str(format!("{}{}", a, r))),
        (BinOp::Add, _, Str(b)) => Ok(Str(format!("{}{}", l, b))),
        // string ordering (lexicographic, like Rust's str ordering)
        (BinOp::Lt, Str(a), Str(b)) => Ok(Bool(a < b)),
        (BinOp::Le, Str(a), Str(b)) => Ok(Bool(a <= b)),
        (BinOp::Gt, Str(a), Str(b)) => Ok(Bool(a > b)),
        (BinOp::Ge, Str(a), Str(b)) => Ok(Bool(a >= b)),
        // comparisons
        (BinOp::Eq, _, _) => Ok(Bool(values_eq(&l, &r))),
        (BinOp::Ne, _, _) => Ok(Bool(!values_eq(&l, &r))),
        (BinOp::Lt, _, _) if is_num(&l) && is_num(&r) => Ok(Bool(as_f(&l) < as_f(&r))),
        (BinOp::Le, _, _) if is_num(&l) && is_num(&r) => Ok(Bool(as_f(&l) <= as_f(&r))),
        (BinOp::Gt, _, _) if is_num(&l) && is_num(&r) => Ok(Bool(as_f(&l) > as_f(&r))),
        (BinOp::Ge, _, _) if is_num(&l) && is_num(&r) => Ok(Bool(as_f(&l) >= as_f(&r))),
        _ => Err(format!(
            "cannot apply {:?} to {} and {}",
            op, l.type_name(), r.type_name()
        )),
    }
}

fn is_known_module(name: &str) -> bool {
    matches!(name, "math" | "strings" | "arrays" | "collections" | "iter" | "rand" | "time" | "json" | "serialize" | "io" | "std")
}

// Dispatch a standard-library function by (optional module) + name.
// Returns Ok(Some(value)) if handled, Ok(None) if the name isn't a stdlib fn.
pub(crate) fn call_stdlib(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    // strip an optional module prefix: "math.sqrt" -> "sqrt"
    let short = name.rsplit('.').next().unwrap_or(name);
    let r = match short {
        // ---- math ----
        "pi" => Some(Value::Float(std::f64::consts::PI)),
        "e" => Some(Value::Float(std::f64::consts::E)),
        "sqrt" => Some(Value::Float(num_arg(args, 0, "sqrt")?.sqrt())),
        "abs" => {
            match args.get(0) {
                Some(Value::Int(n)) => Some(Value::Int(n.abs())),
                Some(Value::BigInt(b)) => Some(norm_big(b.abs())),
                Some(Value::Float(x)) => Some(Value::Float(x.abs())),
                _ => return Err("abs expects a number".into()),
            }
        }
        "pow" => {
            let b = num_arg(args, 0, "pow")?;
            let e = num_arg(args, 1, "pow")?;
            Some(Value::Float(b.powf(e)))
        }
        "floor" => Some(Value::Int(num_arg(args, 0, "floor")?.floor() as i64)),
        "ceil" => Some(Value::Int(num_arg(args, 0, "ceil")?.ceil() as i64)),
        "round" => Some(Value::Int(num_arg(args, 0, "round")?.round() as i64)),
        "sin" => Some(Value::Float(num_arg(args, 0, "sin")?.sin())),
        "cos" => Some(Value::Float(num_arg(args, 0, "cos")?.cos())),
        "tan" => Some(Value::Float(num_arg(args, 0, "tan")?.tan())),
        "log" => Some(Value::Float(num_arg(args, 0, "log")?.ln())),
        "log10" => Some(Value::Float(num_arg(args, 0, "log10")?.log10())),
        "exp" => Some(Value::Float(num_arg(args, 0, "exp")?.exp())),
        "min" => {
            let a = num_arg(args, 0, "min")?;
            let b = num_arg(args, 1, "min")?;
            Some(num_back(args, a.min(b)))
        }
        "max" => {
            let a = num_arg(args, 0, "max")?;
            let b = num_arg(args, 1, "max")?;
            Some(num_back(args, a.max(b)))
        }
        "gcd" => {
            let mut a = int_arg(args, 0, "gcd")?.abs();
            let mut b = int_arg(args, 1, "gcd")?.abs();
            while b != 0 { let t = b; b = a % b; a = t; }
            Some(Value::Int(a))
        }
        "lcm" => {
            let a = int_arg(args, 0, "lcm")?.abs();
            let b = int_arg(args, 1, "lcm")?.abs();
            if a == 0 || b == 0 { Some(Value::Int(0)) } else {
                let (mut x, mut y) = (a, b);
                while y != 0 { let t = y; y = x % y; x = t; }
                Some(Value::Int(a / x * b))
            }
        }
        "tau" => Some(Value::Float(std::f64::consts::TAU)),
        "phi" => Some(Value::Float(1.618_033_988_749_895_f64)),
        "cbrt" => Some(Value::Float(num_arg(args, 0, "cbrt")?.cbrt())),
        "sign" => {
            let n = num_arg(args, 0, "sign")?;
            Some(Value::Int(if n > 0.0 { 1 } else if n < 0.0 { -1 } else { 0 }))
        }
        "trunc" => Some(Value::Float(num_arg(args, 0, "trunc")?.trunc())),
        "fract" => Some(Value::Float(num_arg(args, 0, "fract")?.fract())),
        "hypot" => Some(Value::Float(num_arg(args, 0, "hypot")?.hypot(num_arg(args, 1, "hypot")?))),
        "clamp" => {
            let (x, lo, hi) = (num_arg(args, 0, "clamp")?, num_arg(args, 1, "clamp")?, num_arg(args, 2, "clamp")?);
            Some(Value::Float(x.max(lo).min(hi)))
        }
        "lerp" => {
            let (a, b, t) = (num_arg(args, 0, "lerp")?, num_arg(args, 1, "lerp")?, num_arg(args, 2, "lerp")?);
            Some(Value::Float(a + (b - a) * t))
        }
        "deg2rad" => Some(Value::Float(num_arg(args, 0, "deg2rad")?.to_radians())),
        "rad2deg" => Some(Value::Float(num_arg(args, 0, "rad2deg")?.to_degrees())),
        "asin" => Some(Value::Float(num_arg(args, 0, "asin")?.asin())),
        "acos" => Some(Value::Float(num_arg(args, 0, "acos")?.acos())),
        "atan" => Some(Value::Float(num_arg(args, 0, "atan")?.atan())),
        "atan2" => Some(Value::Float(num_arg(args, 0, "atan2")?.atan2(num_arg(args, 1, "atan2")?))),
        "sinh" => Some(Value::Float(num_arg(args, 0, "sinh")?.sinh())),
        "cosh" => Some(Value::Float(num_arg(args, 0, "cosh")?.cosh())),
        "tanh" => Some(Value::Float(num_arg(args, 0, "tanh")?.tanh())),
        "log2" => Some(Value::Float(num_arg(args, 0, "log2")?.log2())),
        "ln" => Some(Value::Float(num_arg(args, 0, "ln")?.ln())),
        "factorial" => {
            let n = int_arg(args, 0, "factorial")?;
            if n < 0 { return Err("factorial of negative number".into()); }
            let mut acc: i64 = 1;
            for i in 2..=n { acc = acc.saturating_mul(i); }
            Some(Value::Int(acc))
        }
        // ---- strings ----
        "upper" => Some(Value::Str(str_arg(args, 0, "upper")?.to_uppercase())),
        "lower" => Some(Value::Str(str_arg(args, 0, "lower")?.to_lowercase())),
        "trim" => Some(Value::Str(str_arg(args, 0, "trim")?.trim().to_string())),
        "split" => {
            let s = str_arg(args, 0, "split")?;
            let sep = str_arg(args, 1, "split")?;
            let parts: Vec<Value> = if sep.is_empty() {
                s.chars().map(|c| Value::Str(c.to_string())).collect()
            } else {
                s.split(&sep).map(|p| Value::Str(p.to_string())).collect()
            };
            Some(Value::Array(Rc::new(RefCell::new(parts))))
        }
        "join" => {
            let arr = match args.get(0) {
                Some(Value::Array(a)) => a.clone(),
                _ => return Err("join expects (array, separator)".into()),
            };
            let sep = str_arg(args, 1, "join")?;
            let joined = arr.borrow().iter().map(|v| v.to_string()).collect::<Vec<_>>().join(&sep);
            Some(Value::Str(joined))
        }
        "contains" => {
            let s = str_arg(args, 0, "contains")?;
            let sub = str_arg(args, 1, "contains")?;
            Some(Value::Bool(s.contains(&sub)))
        }
        "replace" => {
            let s = str_arg(args, 0, "replace")?;
            let from = str_arg(args, 1, "replace")?;
            let to = str_arg(args, 2, "replace")?;
            Some(Value::Str(s.replace(&from, &to)))
        }
        "starts_with" => {
            let s = str_arg(args, 0, "starts_with")?;
            let p = str_arg(args, 1, "starts_with")?;
            Some(Value::Bool(s.starts_with(&p)))
        }
        "ends_with" => {
            let s = str_arg(args, 0, "ends_with")?;
            let p = str_arg(args, 1, "ends_with")?;
            Some(Value::Bool(s.ends_with(&p)))
        }
        "trim_start" => Some(Value::Str(str_arg(args, 0, "trim_start")?.trim_start().to_string())),
        "trim_end" => Some(Value::Str(str_arg(args, 0, "trim_end")?.trim_end().to_string())),
        "repeat" => {
            let s = str_arg(args, 0, "repeat")?;
            let n = int_arg(args, 1, "repeat")?.max(0) as usize;
            Some(Value::Str(s.repeat(n)))
        }
        "reverse_str" => Some(Value::Str(str_arg(args, 0, "reverse_str")?.chars().rev().collect())),
        "capitalize" => {
            let s = str_arg(args, 0, "capitalize")?;
            let mut c = s.chars();
            Some(Value::Str(match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                None => String::new(),
            }))
        }
        "title" => {
            let s = str_arg(args, 0, "title")?;
            let titled = s.split(' ').map(|w| {
                let mut c = w.chars();
                match c.next() {
                    Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                    None => String::new(),
                }
            }).collect::<Vec<_>>().join(" ");
            Some(Value::Str(titled))
        }
        "pad_left" => {
            let s = str_arg(args, 0, "pad_left")?;
            let width = int_arg(args, 1, "pad_left")?.max(0) as usize;
            let pad = args.get(2).map(|v| v.to_string()).unwrap_or_else(|| " ".to_string());
            let pad_char = pad.chars().next().unwrap_or(' ');
            let cur = s.chars().count();
            Some(Value::Str(if cur >= width { s } else {
                pad_char.to_string().repeat(width - cur) + &s
            }))
        }
        "pad_right" => {
            let s = str_arg(args, 0, "pad_right")?;
            let width = int_arg(args, 1, "pad_right")?.max(0) as usize;
            let pad = args.get(2).map(|v| v.to_string()).unwrap_or_else(|| " ".to_string());
            let pad_char = pad.chars().next().unwrap_or(' ');
            let cur = s.chars().count();
            Some(Value::Str(if cur >= width { s } else {
                s + &pad_char.to_string().repeat(width - cur)
            }))
        }
        "find" => {
            let s = str_arg(args, 0, "find")?;
            let sub = str_arg(args, 1, "find")?;
            Some(match s.find(&sub) {
                Some(idx) => Value::Int(s[..idx].chars().count() as i64),
                None => Value::Int(-1),
            })
        }
        "substring" => {
            let s = str_arg(args, 0, "substring")?;
            let start = int_arg(args, 1, "substring")?.max(0) as usize;
            let chars: Vec<char> = s.chars().collect();
            let end = match args.get(2) {
                Some(Value::Int(e)) => (*e).max(0) as usize,
                _ => chars.len(),
            }.min(chars.len());
            let start = start.min(end);
            Some(Value::Str(chars[start..end].iter().collect()))
        }
        "levenshtein" => {
            let a = str_arg(args, 0, "levenshtein")?;
            let b = str_arg(args, 1, "levenshtein")?;
            let (ac, bc): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
            let mut prev: Vec<usize> = (0..=bc.len()).collect();
            let mut cur = vec![0usize; bc.len() + 1];
            for i in 1..=ac.len() {
                cur[0] = i;
                for j in 1..=bc.len() {
                    let cost = if ac[i - 1] == bc[j - 1] { 0 } else { 1 };
                    cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
                }
                std::mem::swap(&mut prev, &mut cur);
            }
            Some(Value::Int(prev[bc.len()] as i64))
        }
        // ---- arrays / collections ----
        "sort" => {
            let arr = match args.get(0) {
                Some(Value::Array(a)) => a.clone(),
                _ => return Err("sort expects an array".into()),
            };
            let mut items = arr.borrow().clone();
            items.sort_by(|a, b| {
                let (x, y) = (as_f(a), as_f(b));
                x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
            });
            Some(Value::Array(Rc::new(RefCell::new(items))))
        }
        "reverse" => {
            let arr = match args.get(0) {
                Some(Value::Array(a)) => a.clone(),
                _ => return Err("reverse expects an array".into()),
            };
            let mut items = arr.borrow().clone();
            items.reverse();
            Some(Value::Array(Rc::new(RefCell::new(items))))
        }
        "sum" => {
            let arr = match args.get(0) {
                Some(Value::Array(a)) => a.clone(),
                _ => return Err("sum expects an array".into()),
            };
            let mut all_int = true;
            let mut acc = 0.0;
            for v in arr.borrow().iter() {
                match v {
                    Value::Int(n) => acc += *n as f64,
                    Value::Float(x) => { acc += x; all_int = false; }
                    other => return Err(format!("sum expects numbers, got {}", other.type_name())),
                }
            }
            Some(if all_int { Value::Int(acc as i64) } else { Value::Float(acc) })
        }
        "contains_elem" => {
            let arr = match args.get(0) {
                Some(Value::Array(a)) => a.clone(),
                _ => return Err("contains_elem expects (array, value)".into()),
            };
            let needle = args.get(1).ok_or("contains_elem expects (array, value)")?;
            let found = arr.borrow().iter().any(|v| values_eq(v, needle));
            Some(Value::Bool(found))
        }
        // ---- collections ----
        "first" => {
            let arr = expect_array(args.get(0), "first")?;
            let v = arr.borrow().first().cloned().unwrap_or(Value::Null);
            Some(v)
        }
        "last" => {
            let arr = expect_array(args.get(0), "last")?;
            let v = arr.borrow().last().cloned().unwrap_or(Value::Null);
            Some(v)
        }
        "min_of" => {
            let arr = expect_array(args.get(0), "min_of")?;
            let b = arr.borrow();
            if b.is_empty() { return Ok(Some(Value::Null)); }
            let mut m = b[0].clone();
            for v in b.iter().skip(1) {
                if value_lt(v, &m) { m = v.clone(); }
            }
            Some(m)
        }
        "max_of" => {
            let arr = expect_array(args.get(0), "max_of")?;
            let b = arr.borrow();
            if b.is_empty() { return Ok(Some(Value::Null)); }
            let mut m = b[0].clone();
            for v in b.iter().skip(1) {
                if value_lt(&m, v) { m = v.clone(); }
            }
            Some(m)
        }
        "unique" => {
            let arr = expect_array(args.get(0), "unique")?;
            let mut out: Vec<Value> = Vec::new();
            for v in arr.borrow().iter() {
                if !out.iter().any(|x| values_eq(x, v)) { out.push(v.clone()); }
            }
            Some(Value::Array(Rc::new(RefCell::new(out))))
        }
        "slice" => {
            let arr = expect_array(args.get(0), "slice")?;
            let start = int_arg(args, 1, "slice")?.max(0) as usize;
            let out: Vec<Value> = {
                let b = arr.borrow();
                let end = match args.get(2) {
                    Some(Value::Int(e)) => (*e).max(0) as usize,
                    _ => b.len(),
                }.min(b.len());
                let start = start.min(end);
                b[start..end].to_vec()
            };
            Some(Value::Array(Rc::new(RefCell::new(out))))
        }
        "take" => {
            let arr = expect_array(args.get(0), "take")?;
            let n = int_arg(args, 1, "take")?.max(0) as usize;
            let out: Vec<Value> = arr.borrow().iter().take(n).cloned().collect();
            Some(Value::Array(Rc::new(RefCell::new(out))))
        }
        "skip" => {
            let arr = expect_array(args.get(0), "skip")?;
            let n = int_arg(args, 1, "skip")?.max(0) as usize;
            let out: Vec<Value> = arr.borrow().iter().skip(n).cloned().collect();
            Some(Value::Array(Rc::new(RefCell::new(out))))
        }
        "flatten" => {
            let arr = expect_array(args.get(0), "flatten")?;
            let mut out: Vec<Value> = Vec::new();
            for v in arr.borrow().iter() {
                match v {
                    Value::Array(inner) => out.extend(inner.borrow().iter().cloned()),
                    other => out.push(other.clone()),
                }
            }
            Some(Value::Array(Rc::new(RefCell::new(out))))
        }
        "zip" => {
            let a = expect_array(args.get(0), "zip")?;
            let b = expect_array(args.get(1), "zip")?;
            let out: Vec<Value> = {
                let (ab, bb) = (a.borrow(), b.borrow());
                ab.iter().zip(bb.iter()).map(|(x, y)| {
                    Value::Array(Rc::new(RefCell::new(vec![x.clone(), y.clone()])))
                }).collect()
            };
            Some(Value::Array(Rc::new(RefCell::new(out))))
        }
        "enumerate" => {
            let arr = expect_array(args.get(0), "enumerate")?;
            let out: Vec<Value> = arr.borrow().iter().enumerate().map(|(i, v)| {
                Value::Array(Rc::new(RefCell::new(vec![Value::Int(i as i64), v.clone()])))
            }).collect();
            Some(Value::Array(Rc::new(RefCell::new(out))))
        }
        "chunk" => {
            let arr = expect_array(args.get(0), "chunk")?;
            let size = int_arg(args, 1, "chunk")?.max(1) as usize;
            let out: Vec<Value> = {
                let b = arr.borrow();
                b.chunks(size).map(|c| {
                    Value::Array(Rc::new(RefCell::new(c.to_vec())))
                }).collect()
            };
            Some(Value::Array(Rc::new(RefCell::new(out))))
        }
        "concat" => {
            let a = expect_array(args.get(0), "concat")?;
            let b = expect_array(args.get(1), "concat")?;
            let mut out = a.borrow().clone();
            out.extend(b.borrow().iter().cloned());
            Some(Value::Array(Rc::new(RefCell::new(out))))
        }
        // ---- iter ----
        "count" => {
            let arr = expect_array(args.get(0), "count")?;
            let n = arr.borrow().len() as i64;
            Some(Value::Int(n))
        }
        "position" => {
            let arr = expect_array(args.get(0), "position")?;
            let needle = args.get(1).ok_or("position expects (array, value)")?;
            let p = arr.borrow().iter().position(|v| values_eq(v, needle));
            Some(match p {
                Some(i) => Value::Int(i as i64),
                None => Value::Int(-1),
            })
        }
        "all_true" => {
            let arr = expect_array(args.get(0), "all_true")?;
            let r = arr.borrow().iter().all(|v| v.is_truthy());
            Some(Value::Bool(r))
        }
        "any_true" => {
            let arr = expect_array(args.get(0), "any_true")?;
            let r = arr.borrow().iter().any(|v| v.is_truthy());
            Some(Value::Bool(r))
        }
        // ---- rand ----
        "seed" => {
            let n = int_arg(args, 0, "seed")?;
            rng_seed(n as u64);
            Some(Value::Null)
        }
        "random" => Some(Value::Float(rng_float())),
        "rand_int" => {
            // rand_int(lo, hi) -> integer in [lo, hi] inclusive
            let lo = int_arg(args, 0, "rand_int")?;
            let hi = int_arg(args, 1, "rand_int")?;
            if hi < lo { return Err("rand_int: hi must be >= lo".into()); }
            let span = (hi - lo + 1) as u64;
            Some(Value::Int(lo + (rng_next() % span) as i64))
        }
        "rand_float" => {
            // rand_float(lo, hi) -> float in [lo, hi)
            let lo = num_arg(args, 0, "rand_float")?;
            let hi = num_arg(args, 1, "rand_float")?;
            Some(Value::Float(lo + rng_float() * (hi - lo)))
        }
        "rand_bool" => Some(Value::Bool(rng_next() & 1 == 1)),
        "choice" => {
            let arr = expect_array(args.get(0), "choice")?;
            let b = arr.borrow();
            if b.is_empty() { return Ok(Some(Value::Null)); }
            let idx = (rng_next() % b.len() as u64) as usize;
            Some(b[idx].clone())
        }
        "shuffle" => {
            // Fisher-Yates, returns a new shuffled array
            let arr = expect_array(args.get(0), "shuffle")?;
            let mut out = arr.borrow().clone();
            let n = out.len();
            for i in (1..n).rev() {
                let j = (rng_next() % (i as u64 + 1)) as usize;
                out.swap(i, j);
            }
            Some(Value::Array(Rc::new(RefCell::new(out))))
        }
        // ---- time ----
        "now" => {
            // unix timestamp in seconds (float, sub-second precision)
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            Some(Value::Float(secs))
        }
        "now_millis" => {
            let ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            Some(Value::Int(ms))
        }
        "now_nanos" => {
            let ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0);
            Some(Value::Int(ns))
        }
        // ---- json / serialize ----
        "json_parse" => {
            let s = str_arg(args, 0, "json_parse")?;
            Some(json_parse(&s)?)
        }
        "json_stringify" => {
            let v = args.get(0).ok_or("json_stringify expects a value")?;
            Some(Value::Str(json_stringify(v, false, 0)))
        }
        "json_pretty" => {
            let v = args.get(0).ok_or("json_pretty expects a value")?;
            Some(Value::Str(json_stringify(v, true, 0)))
        }
        _ => None,
    };
    Ok(r)
}

fn num_arg(args: &[Value], i: usize, f: &str) -> Result<f64, String> {
    match args.get(i) {
        Some(Value::Int(n)) => Ok(*n as f64),
        Some(Value::Float(x)) => Ok(*x),
        _ => Err(format!("{} expects a number at position {}", f, i + 1)),
    }
}
fn int_arg(args: &[Value], i: usize, f: &str) -> Result<i64, String> {
    match args.get(i) {
        Some(Value::Int(n)) => Ok(*n),
        _ => Err(format!("{} expects an integer at position {}", f, i + 1)),
    }
}
fn str_arg(args: &[Value], i: usize, f: &str) -> Result<String, String> {
    match args.get(i) {
        Some(Value::Str(s)) => Ok(s.clone()),
        _ => Err(format!("{} expects a string at position {}", f, i + 1)),
    }
}
// preserve int-ness for min/max when both inputs are ints
fn num_back(args: &[Value], result: f64) -> Value {
    let both_int = matches!(args.get(0), Some(Value::Int(_)))
        && matches!(args.get(1), Some(Value::Int(_)));
    if both_int { Value::Int(result as i64) } else { Value::Float(result) }
}

fn is_num(v: &Value) -> bool { matches!(v, Value::Int(_) | Value::Float(_) | Value::BigInt(_)) }
fn as_f(v: &Value) -> f64 {
    match v { Value::Int(n) => *n as f64, Value::Float(x) => *x,
              Value::BigInt(b) => b.to_f64().unwrap_or(f64::NAN), _ => 0.0 }
}
fn as_int(v: Option<&Value>) -> Result<i64, String> {
    match v {
        Some(Value::Int(n)) => Ok(*n),
        Some(other) => Err(format!("expected int, got {}", other.type_name())),
        None => Err("missing argument".into()),
    }
}
fn expect_array(v: Option<&Value>, fname: &str) -> Result<Rc<RefCell<Vec<Value>>>, String> {
    match v {
        Some(Value::Array(a)) => Ok(a.clone()),
        Some(other) => Err(format!("{} expects an array, got {}", fname, other.type_name())),
        None => Err(format!("{} missing array argument", fname)),
    }
}
fn expect_map(v: Option<&Value>, fname: &str) -> Result<Rc<RefCell<Vec<(Value, Value)>>>, String> {
    match v {
        Some(Value::Map(m)) => Ok(m.clone()),
        Some(other) => Err(format!("{} expects a map, got {}", fname, other.type_name())),
        None => Err(format!("{} missing map argument", fname)),
    }
}
fn value_lt(a: &Value, b: &Value) -> bool {
    use Value::*;
    match (a, b) {
        (Int(x), Int(y)) => x < y,
        (BigInt(x), BigInt(y)) => x < y,
        (BigInt(x), Int(y)) => *x < num_bigint::BigInt::from(*y),
        (Int(x), BigInt(y)) => num_bigint::BigInt::from(*x) < *y,
        (Float(x), Float(y)) => x < y,
        (Int(x), Float(y)) => (*x as f64) < *y,
        (Float(x), Int(y)) => *x < (*y as f64),
        (Str(x), Str(y)) => x < y,
        (Bool(x), Bool(y)) => !x & y,
        _ => false,
    }
}

fn values_eq(a: &Value, b: &Value) -> bool {
    use Value::*;
    match (a, b) {
        (Int(x), Int(y)) => x == y,
        (BigInt(x), BigInt(y)) => x == y,
        (BigInt(x), Int(y)) | (Int(y), BigInt(x)) => *x == num_bigint::BigInt::from(*y),
        (Float(x), Float(y)) => x == y,
        (Int(x), Float(y)) | (Float(y), Int(x)) => (*x as f64) == *y,
        (Str(x), Str(y)) => x == y,
        (Bool(x), Bool(y)) => x == y,
        (Null, Null) => true,
        (Array(x), Array(y)) => {
            let (xb, yb) = (x.borrow(), y.borrow());
            xb.len() == yb.len() && xb.iter().zip(yb.iter()).all(|(a, b)| values_eq(a, b))
        }
        (Struct(x), Struct(y)) => {
            // structural equality: same type and all fields equal
            let (xb, yb) = (x.borrow(), y.borrow());
            xb.type_name == yb.type_name
                && xb.fields.len() == yb.fields.len()
                && xb.fields.iter().all(|(k, v)| yb.fields.get(k).map_or(false, |w| values_eq(v, w)))
        }
        (Closure(x), Closure(y)) => Rc::ptr_eq(x, y),
        (Enum(x), Enum(y)) => {
            x.enum_name == y.enum_name
                && x.variant == y.variant
                && x.data.len() == y.data.len()
                && x.data.iter().zip(y.data.iter()).all(|(a, b)| values_eq(a, b))
        }
        (Map(x), Map(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

// ---- JSON: a real recursive-descent parser and serializer ----

struct JsonParser<'a> {
    chars: Vec<char>,
    pos: usize,
    _src: &'a str,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        JsonParser { chars: s.chars().collect(), pos: 0, _src: s }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() { self.pos += 1; }
        c
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\n' || c == '\r' { self.pos += 1; } else { break; }
        }
    }

    fn parse_value(&mut self) -> Result<Value, String> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => Ok(Value::Str(self.parse_string()?)),
            Some('t') | Some('f') => self.parse_bool(),
            Some('n') => self.parse_null(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(format!("unexpected character '{}' in JSON", c)),
            None => Err("unexpected end of JSON input".into()),
        }
    }

    fn parse_object(&mut self) -> Result<Value, String> {
        self.bump(); // consume '{'
        let mut pairs: Vec<(Value, Value)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some('}') { self.bump(); return Ok(Value::Map(Rc::new(RefCell::new(pairs)))); }
        loop {
            self.skip_ws();
            if self.peek() != Some('"') { return Err("expected string key in JSON object".into()); }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.bump() != Some(':') { return Err("expected ':' in JSON object".into()); }
            let val = self.parse_value()?;
            pairs.push((Value::Str(key), val));
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some('}') => break,
                _ => return Err("expected ',' or '}' in JSON object".into()),
            }
        }
        Ok(Value::Map(Rc::new(RefCell::new(pairs))))
    }

    fn parse_array(&mut self) -> Result<Value, String> {
        self.bump(); // consume '['
        let mut items: Vec<Value> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') { self.bump(); return Ok(Value::Array(Rc::new(RefCell::new(items)))); }
        loop {
            let v = self.parse_value()?;
            items.push(v);
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some(']') => break,
                _ => return Err("expected ',' or ']' in JSON array".into()),
            }
        }
        Ok(Value::Array(Rc::new(RefCell::new(items))))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.bump(); // consume opening quote
        let mut out = String::new();
        loop {
            match self.bump() {
                Some('"') => break,
                Some('\\') => match self.bump() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('b') => out.push('\u{0008}'),
                    Some('f') => out.push('\u{000C}'),
                    Some('u') => {
                        let mut code = 0u32;
                        for _ in 0..4 {
                            let c = self.bump().ok_or("incomplete \\u escape in JSON")?;
                            code = code * 16 + c.to_digit(16).ok_or("bad hex in \\u escape")?;
                        }
                        out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                    }
                    _ => return Err("invalid escape in JSON string".into()),
                },
                Some(c) => out.push(c),
                None => return Err("unterminated JSON string".into()),
            }
        }
        Ok(out)
    }

    fn parse_number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        let mut is_float = false;
        if self.peek() == Some('-') { self.bump(); }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() { self.bump(); }
            else if c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' { is_float = true; self.bump(); }
            else { break; }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        if is_float {
            text.parse::<f64>().map(Value::Float).map_err(|_| format!("bad JSON number: {}", text))
        } else {
            match text.parse::<i64>() {
                Ok(n) => Ok(Value::Int(n)),
                Err(_) => text.parse::<f64>().map(Value::Float).map_err(|_| format!("bad JSON number: {}", text)),
            }
        }
    }

    fn parse_bool(&mut self) -> Result<Value, String> {
        if self.chars[self.pos..].starts_with(&['t','r','u','e']) {
            self.pos += 4; Ok(Value::Bool(true))
        } else if self.chars[self.pos..].starts_with(&['f','a','l','s','e']) {
            self.pos += 5; Ok(Value::Bool(false))
        } else {
            Err("invalid JSON literal".into())
        }
    }

    fn parse_null(&mut self) -> Result<Value, String> {
        if self.chars[self.pos..].starts_with(&['n','u','l','l']) {
            self.pos += 4; Ok(Value::Null)
        } else {
            Err("invalid JSON literal".into())
        }
    }
}

fn json_parse(s: &str) -> Result<Value, String> {
    let mut p = JsonParser::new(s);
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.chars.len() {
        return Err("trailing characters after JSON value".into());
    }
    Ok(v)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_stringify(v: &Value, pretty: bool, indent: usize) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Str(s) => json_escape(s),
        Value::Array(a) => {
            let b = a.borrow();
            if b.is_empty() { return "[]".to_string(); }
            if pretty {
                let pad = "  ".repeat(indent + 1);
                let close = "  ".repeat(indent);
                let items: Vec<String> = b.iter().map(|x| format!("{}{}", pad, json_stringify(x, true, indent + 1))).collect();
                format!("[\n{}\n{}]", items.join(",\n"), close)
            } else {
                let items: Vec<String> = b.iter().map(|x| json_stringify(x, false, 0)).collect();
                format!("[{}]", items.join(","))
            }
        }
        Value::Map(m) => {
            let b = m.borrow();
            if b.is_empty() { return "{}".to_string(); }
            if pretty {
                let pad = "  ".repeat(indent + 1);
                let close = "  ".repeat(indent);
                let items: Vec<String> = b.iter().map(|(k, val)| {
                    format!("{}{}: {}", pad, json_escape(&k.to_string()), json_stringify(val, true, indent + 1))
                }).collect();
                format!("{{\n{}\n{}}}", items.join(",\n"), close)
            } else {
                let items: Vec<String> = b.iter().map(|(k, val)| {
                    format!("{}:{}", json_escape(&k.to_string()), json_stringify(val, false, 0))
                }).collect();
                format!("{{{}}}", items.join(","))
            }
        }
        _ => "null".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Constant folding: a semantics-preserving pre-pass run before execution.
//
// It replaces fully-constant `Unary`/`Binary` subexpressions with their literal
// result. Correctness is guaranteed by construction: the result is computed with
// the *same* `eval_binop`/unary logic the interpreter uses at runtime, and a node
// is only rewritten when that evaluation succeeds and yields a scalar literal
// (Int/Float/Str/Bool/Null). Errors (e.g. `1/0`) and big-integer results are left
// untouched so runtime behaviour — including catchable exceptions — is unchanged.
// ---------------------------------------------------------------------------

pub fn fold_program(p: &mut Program) {
    for item in &mut p.items {
        match item {
            Item::Func(f) => fold_block(&mut f.body),
            Item::Impl(im) => { for m in &mut im.methods { fold_block(&mut m.body); } }
            Item::Trait(t) => { for d in &mut t.defaults { fold_block(&mut d.body); } }
            Item::Test(t) => fold_block(&mut t.body),
            Item::Const { value, .. } => fold_expr(value),
            Item::Migration { body, .. } => fold_block(body),
            Item::Struct(_) | Item::Enum(_) | Item::Use(_) | Item::Machine(_)
            | Item::Macro(_) | Item::TypeAlias { .. } | Item::Extern(_)
            | Item::Import { .. } => {}
        }
    }
}

fn fold_block(stmts: &mut Vec<Stmt>) {
    for s in stmts { fold_stmt(s); }
}

fn fold_stmt(s: &mut Stmt) {
    match s {
        Stmt::Let { value, .. } => fold_expr(value),
        Stmt::Assign { value, .. } => fold_expr(value),
        Stmt::IndexAssign { base, index, value } => { fold_expr(base); fold_expr(index); fold_expr(value); }
        Stmt::FieldAssign { base, value, .. } => { fold_expr(base); fold_expr(value); }
        Stmt::Expr(e) => fold_expr(e),
        Stmt::Return(opt) => { if let Some(e) = opt { fold_expr(e); } }
        Stmt::If { cond, then, els } => {
            fold_expr(cond); fold_block(then);
            if let Some(e) = els { fold_block(e); }
        }
        Stmt::While { cond, body } => { fold_expr(cond); fold_block(body); }
        Stmt::ForRange { start, end, body, .. } => { fold_expr(start); fold_expr(end); fold_block(body); }
        Stmt::ForEach { iter, body, .. } => { fold_expr(iter); fold_block(body); }
        Stmt::Throw(e) => fold_expr(e),
        Stmt::Yield(opt) => { if let Some(e) = opt { fold_expr(e); } }
        Stmt::Break(opt) => { if let Some(e) = opt { fold_expr(e); } }
        Stmt::Continue => {}
        Stmt::Defer(b) => fold_block(b),
        Stmt::TryCatch { body, catch_body, finally_body, .. } => {
            fold_block(body);
            if let Some(c) = catch_body { fold_block(c); }
            if let Some(f) = finally_body { fold_block(f); }
        }
    }
}

fn fold_expr(e: &mut Expr) {
    match e {
        // leaves: nothing to fold
        Expr::Int(_) | Expr::BigIntLit(_) | Expr::Float(_) | Expr::Str(_)
        | Expr::Bool(_) | Expr::Null | Expr::Ident(_) => {}
        Expr::At { expr, .. } => fold_expr(expr), // fold inside, keep the position marker
        Expr::Array(xs) | Expr::SetLit(xs) => { for x in xs { fold_expr(x); } }
        Expr::MapLit(pairs) => { for (k, v) in pairs { fold_expr(k); fold_expr(v); } }
        Expr::Comprehension { body, iter, cond, .. } => {
            fold_expr(body); fold_expr(iter);
            if let Some(c) = cond { fold_expr(c); }
        }
        Expr::FmtStr(parts) => {
            for p in parts { if let FmtPart::Expr(ex) = p { fold_expr(ex); } }
        }
        Expr::Index { base, index } => { fold_expr(base); fold_expr(index); }
        Expr::RangeLit { lo, hi, .. } => {
            if let Some(l) = lo { fold_expr(l); }
            if let Some(h) = hi { fold_expr(h); }
        }
        Expr::StructLit { fields, .. } => { for (_, v) in fields { fold_expr(v); } }
        Expr::Field { base, .. } | Expr::SafeField { base, .. } => fold_expr(base),
        Expr::MethodCall { base, args, .. } => { fold_expr(base); for a in args { fold_expr(a); } }
        Expr::Lambda { body, .. } => match body.as_mut() {
            LambdaBody::Expr(ex) => fold_expr(ex),
            LambdaBody::Block(b) => fold_block(b),
        },
        Expr::CallValue { callee, args } => { fold_expr(callee); for a in args { fold_expr(a); } }
        Expr::Call { args, .. } => { for a in args { fold_expr(a); } }
        Expr::Block { stmts, tail } => {
            fold_block(stmts);
            if let Some(t) = tail { fold_expr(t); }
        }
        Expr::If { cond, then, els } => { fold_expr(cond); fold_expr(then); fold_expr(els); }
        Expr::Match { scrutinee, arms } => {
            fold_expr(scrutinee);
            for a in arms {
                if let Some(g) = &mut a.guard { fold_expr(g); }
                fold_expr(&mut a.body);
            }
        }
        Expr::Await(x) | Expr::Recv(x) => fold_expr(x),
        Expr::Spawn(b) => fold_block(b),
        Expr::Send { chan, value } => { fold_expr(chan); fold_expr(value); }
        Expr::Select(arms) => {
            for a in arms { fold_expr(&mut a.chan); fold_expr(&mut a.body); }
        }
        Expr::Unary { op, expr } => {
            fold_expr(expr);
            if let Some(v) = const_value(expr) {
                if let Some(res) = fold_unary(*op, &v) {
                    if let Some(lit) = value_to_lit(&res) { *e = lit; }
                }
            }
        }
        Expr::Binary { op, lhs, rhs } => {
            fold_expr(lhs); fold_expr(rhs);
            if let (Some(a), Some(b)) = (const_value(lhs), const_value(rhs)) {
                if let Ok(res) = eval_binop(*op, a, b) {
                    if let Some(lit) = value_to_lit(&res) { *e = lit; }
                }
            }
        }
    }
}

// A literal expression's runtime value (scalars only; BigInt literals are left
// dynamic to avoid any divergence in how the interpreter materializes them).
fn const_value(e: &Expr) -> Option<Value> {
    match e {
        Expr::Int(n) => Some(Value::Int(*n)),
        Expr::Float(x) => Some(Value::Float(*x)),
        Expr::Str(s) => Some(Value::Str(s.clone())),
        Expr::Bool(b) => Some(Value::Bool(*b)),
        Expr::Null => Some(Value::Null),
        _ => None,
    }
}

// Turn a folded scalar value back into a literal expression. Non-scalars (BigInt,
// arrays, ...) return None so the original expression is kept.
fn value_to_lit(v: &Value) -> Option<Expr> {
    match v {
        Value::Int(n) => Some(Expr::Int(*n)),
        Value::Float(x) => Some(Expr::Float(*x)),
        Value::Str(s) => Some(Expr::Str(s.clone())),
        Value::Bool(b) => Some(Expr::Bool(*b)),
        Value::Null => Some(Expr::Null),
        _ => None,
    }
}

// Unary on a literal scalar, mirroring the interpreter exactly for these cases.
fn fold_unary(op: UnOp, v: &Value) -> Option<Value> {
    match (op, v) {
        (UnOp::Neg, Value::Int(n)) if *n != i64::MIN => Some(Value::Int(-n)),
        (UnOp::Neg, Value::Float(x)) => Some(Value::Float(-x)),
        (UnOp::Not, Value::Bool(b)) => Some(Value::Bool(!b)),
        (UnOp::BitNot, Value::Int(n)) => Some(Value::Int(!n)),
        _ => None,
    }
}

#[cfg(test)]
mod attr_tests {
    use super::*;
    use crate::parser::parse_program;

    #[test]
    fn self_healing_retries_until_success() {
        // shared array state persists across retries (Rc-backed), so this
        // succeeds on the third attempt with attempts: 5
        let src = "#[self_healing(attempts: 5)]\n\
                   fn f(s){ s[0]=s[0]+1; if s[0]<3 { throw \"x\" } s[0] }\n\
                   fn main(){ 0 }";
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        let arr = Value::Array(Rc::new(RefCell::new(vec![Value::Int(0)])));
        let r = interp.call("f", vec![arr]);
        assert_eq!(format!("{:?}", r), format!("{:?}", Ok::<_, String>(Value::Int(3))));
    }

    #[test]
    fn self_healing_exhausts_then_errors() {
        let src = "#[self_healing(attempts: 3)]\n\
                   fn f(c){ c[0]=c[0]+1; throw \"nope\" }\nfn main(){ 0 }";
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        let arr = Value::Array(Rc::new(RefCell::new(vec![Value::Int(0)])));
        let r = interp.call("f", vec![arr.clone()]);
        assert!(r.is_err(), "exhausted retries must error");
        if let Value::Array(a) = &arr { assert_eq!(a.borrow()[0], Value::Int(3), "must try exactly 3 times"); }
    }

    #[test]
    fn integrity_hash_is_stable_and_distinct() {
        let src = "#[integrity] fn a(){ 1 + 2 }\n#[integrity] fn b(){ 1 + 3 }\nfn main(){ 0 }";
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        let h1 = interp.call("integrity_of", vec![Value::Str("a".into())]).unwrap();
        let h2 = interp.call("integrity_of", vec![Value::Str("a".into())]).unwrap();
        let h3 = interp.call("integrity_of", vec![Value::Str("b".into())]).unwrap();
        assert_eq!(format!("{:?}", h1), format!("{:?}", h2), "same fn -> same hash");
        assert_ne!(format!("{:?}", h1), format!("{:?}", h3), "different fn -> different hash");
    }

    #[test]
    fn zero_alloc_flags_allocation_and_passes_pure() {
        let src = "#[zero_alloc] fn ok(a,b){ a*b + a }\n\
                   #[zero_alloc] fn bad(n){ xs = [n]; xs }\nfn main(){ 0 }";
        let prog = parse_program(src).expect("parse");
        let (errors, _) = crate::types::Checker::new(&prog).check(&prog);
        assert!(errors.iter().any(|e| e.contains("bad") && e.contains("zero_alloc")),
            "must flag the allocating function: {:?}", errors);
        assert!(!errors.iter().any(|e| e.contains("`ok`")), "pure fn must pass: {:?}", errors);
    }

    #[test]
    fn memo_caches_and_contracts_enforce() {
        let src = "#[memo] fn sq(n){ n*n }\n\
                   #[requires(x > 0)] #[ensures(result > x)] fn inc(x){ x+1 }\nfn main(){0}";
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        // memo returns the same value; contract passes for valid input
        assert_eq!(format!("{:?}", interp.call("sq", vec![Value::Int(5)])),
                   format!("{:?}", Ok::<_,String>(Value::Int(25))));
        assert_eq!(format!("{:?}", interp.call("sq", vec![Value::Int(5)])),
                   format!("{:?}", Ok::<_,String>(Value::Int(25))));
        assert!(interp.call("inc", vec![Value::Int(3)]).is_ok());
        // requires(x > 0) violated -> error (throw sentinel)
        assert!(interp.call("inc", vec![Value::Int(0)]).is_err());
    }

    #[test]
    fn encrypt_round_trips_and_metadata_reads() {
        let src = "#[version(v: \"2.0\")] fn f(){ 1 }\nfn main(){0}";
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        let enc = interp.call("encrypt", vec![Value::Str("hello".into()), Value::Str("key".into())]).unwrap();
        let enc_s = if let Value::Str(s) = &enc { s.clone() } else { panic!() };
        assert_ne!(enc_s, "hello");
        let dec = interp.call("decrypt", vec![Value::Str(enc_s.clone()), Value::Str("key".into())]).unwrap();
        assert_eq!(format!("{:?}", dec), format!("{:?}", Value::Str("hello".into())));
        // wrong key must not reproduce the plaintext
        let bad = interp.call("decrypt", vec![Value::Str(enc_s), Value::Str("nope".into())]);
        assert!(bad.is_err() || format!("{:?}", bad.unwrap()) != format!("{:?}", Value::Str("hello".into())));
        // metadata is captured and queryable (the version string round-trips)
        let m = interp.call("meta_of", vec![Value::Str("f".into()), Value::Str("version".into())]).unwrap();
        assert_eq!(format!("{:?}", m), format!("{:?}", Value::Str("2.0".into())));
    }

    #[test]
    fn anti_tamper_passes_when_unchanged() {
        let src = "#[anti_tamper] fn f(){ 1 + 2 }\nfn main(){0}";
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        // repeated calls with the same body pass (baseline matches)
        assert!(interp.call("f", vec![]).is_ok());
        assert!(interp.call("f", vec![]).is_ok());
    }

    #[test]
    fn time_travel_records_bounded_history() {
        let src = "#[time_travel(depth: 3)] fn s(n){ n*n }\nfn main(){0}";
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        for n in 1..=5 { let _ = interp.call("s", vec![Value::Int(n)]); }
        let h = interp.call("history_of", vec![Value::Str("s".into())]).unwrap();
        if let Value::Array(a) = h {
            let got: Vec<String> = a.borrow().iter().map(|v| v.to_string()).collect();
            assert_eq!(got, vec!["9", "16", "25"], "only the last 3 results kept");
        } else { panic!("expected array"); }
    }

    #[test]
    fn profile_counts_and_attrs_of_lists() {
        let src = "#[profile] fn p(n){ n }\n#[memo] #[trace] fn q(n){ n }\nfn main(){0}";
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        for _ in 0..4 { let _ = interp.call("p", vec![Value::Int(1)]); }
        assert_eq!(format!("{:?}", interp.call("profile_of", vec![Value::Str("p".into())])),
                   format!("{:?}", Ok::<_,String>(Value::Int(4))));
        let a = interp.call("attrs_of", vec![Value::Str("q".into())]).unwrap();
        if let Value::Array(arr) = a { assert_eq!(arr.borrow().len(), 2); } else { panic!("expected array"); }
    }

    #[test]
    fn budget_instrument_cache_snapshot() {
        let src = "#[budget(calls: \"2\")] fn lim(x){ x }\n\
                   #[instrument] fn tr(n){ n }\n\
                   #[cache(ttl: 1)] fn sq(n){ n * n }\nfn main(){ 0 }";
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        // budget: two calls succeed, budget_of drops to 0, the third throws
        assert!(interp.call("lim", vec![Value::Int(1)]).is_ok());
        assert!(interp.call("lim", vec![Value::Int(2)]).is_ok());
        assert_eq!(format!("{:?}", interp.call("budget_of", vec![Value::Str("lim".into())])),
                   format!("{:?}", Ok::<_,String>(Value::Int(0))));
        assert!(interp.call("lim", vec![Value::Int(3)]).is_err());
        // instrument: counts calls
        for _ in 0..3 { let _ = interp.call("tr", vec![Value::Int(1)]); }
        assert_eq!(format!("{:?}", interp.call("instrument_of", vec![Value::Str("tr".into())])),
                   format!("{:?}", Ok::<_,String>(Value::Int(3))));
        // cache(ttl:1): reused once then recomputed — value stays correct
        assert_eq!(format!("{:?}", interp.call("sq", vec![Value::Int(5)])),
                   format!("{:?}", Ok::<_,String>(Value::Int(25))));
        assert_eq!(format!("{:?}", interp.call("sq", vec![Value::Int(5)])),
                   format!("{:?}", Ok::<_,String>(Value::Int(25))));
        // snapshot / rollback round-trips a value; missing id -> null
        interp.call("snapshot", vec![Value::Str("k".into()), Value::Int(42)]).unwrap();
        assert_eq!(format!("{:?}", interp.call("rollback", vec![Value::Str("k".into())])),
                   format!("{:?}", Ok::<_,String>(Value::Int(42))));
        assert_eq!(format!("{:?}", interp.call("rollback", vec![Value::Str("absent".into())])),
                   format!("{:?}", Ok::<_,String>(Value::Null)));
    }

    #[test]
    fn hot_swap_replaces_body() {
        let src = "#[hot_swap(scope: function)] fn g(x){ x + 1 }\nfn main(){ 0 }";
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        assert_eq!(format!("{:?}", interp.call("g", vec![Value::Int(10)])),
                   format!("{:?}", Ok::<_,String>(Value::Int(11))));
        // install a replacement: x => x * 100
        let clo = Value::Closure(Rc::new(ClosureVal {
            params: vec!["x".into()],
            body: LambdaBody::Expr(Expr::Binary {
                op: BinOp::Mul, lhs: Box::new(Expr::Ident("x".into())), rhs: Box::new(Expr::Int(100)) }),
            captured: Scope::new(), vm_chunk: None,
        }));
        interp.call("hot_swap", vec![Value::Str("g".into()), clo]).unwrap();
        assert_eq!(format!("{:?}", interp.call("g", vec![Value::Int(10)])),
                   format!("{:?}", Ok::<_,String>(Value::Int(1000))));
    }

    #[test]
    fn comptime_evaluates_once_and_caches() {
        // The body sums 1..=10 into a global-free accumulator; the result (55)
        // is precomputed at init and returned by every call. `init_consts`
        // triggers the const-eval; calls then hit the cache.
        let src = "#[comptime] fn total(){ let mut s = 0; let mut i = 1; while i <= 10 { s = s + i; i = i + 1 } s }\n\
                   fn main(){ 0 }";
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        interp.init_consts().expect("init");
        assert_eq!(format!("{:?}", interp.call("total", vec![])),
                   format!("{:?}", Ok::<_,String>(Value::Int(55))));
        assert_eq!(format!("{:?}", interp.call("total", vec![])),
                   format!("{:?}", Ok::<_,String>(Value::Int(55))));
    }

    #[test]
    fn migrate_transforms_old_struct_to_new() {
        // migrate binds `old` and the old struct's fields; the body produces the
        // new-shape value (age bumped, `active` field added and defaulted).
        let src = "struct A { name: Str, age: Int }\n\
                   struct B { name: Str, age: Int, active: Bool }\n\
                   migrate from A to B { B { name: name, age: age + 1, active: true } }\n\
                   fn main(){ let o = A { name: \"x\", age: 4 }; let n = migrate(o); n.age }";
        let prog = parse_program(src).expect("parse");
        let interp = Interp::new(&prog).expect("interp");
        assert_eq!(format!("{:?}", interp.call("main", vec![])),
                   format!("{:?}", Ok::<_,String>(Value::Int(5))));
    }
}

#[cfg(test)]
mod fold_tests {
    use super::*;

    fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::Binary { op, lhs: Box::new(l), rhs: Box::new(r) }
    }

    #[test]
    fn folds_nested_arithmetic() {
        // 2 + 3 * 4  ->  14
        let mut e = bin(BinOp::Add, Expr::Int(2), bin(BinOp::Mul, Expr::Int(3), Expr::Int(4)));
        fold_expr(&mut e);
        assert!(matches!(e, Expr::Int(14)), "expected Int(14), got {:?}", e);
    }

    #[test]
    fn does_not_fold_with_variable() {
        // x + 1  stays a Binary (x is not constant)
        let mut e = bin(BinOp::Add, Expr::Ident("x".into()), Expr::Int(1));
        fold_expr(&mut e);
        assert!(matches!(e, Expr::Binary { .. }), "must not fold non-constant");
    }

    #[test]
    fn does_not_fold_division_by_zero() {
        // 6 / 0  is left for the interpreter (so it stays catchable)
        let mut e = bin(BinOp::Div, Expr::Int(6), Expr::Int(0));
        fold_expr(&mut e);
        assert!(matches!(e, Expr::Binary { .. }), "div-by-zero must not fold");
    }

    #[test]
    fn does_not_fold_bigint_result() {
        // 2 ** 100 overflows i64 -> BigInt result, left unfolded
        let mut e = bin(BinOp::Pow, Expr::Int(2), Expr::Int(100));
        fold_expr(&mut e);
        assert!(matches!(e, Expr::Binary { .. }), "bigint result must not fold");
    }

    #[test]
    fn folds_string_concat_and_unary() {
        let mut e = bin(BinOp::Add, Expr::Str("a".into()), Expr::Str("b".into()));
        fold_expr(&mut e);
        assert!(matches!(&e, Expr::Str(s) if s == "ab"));

        let mut u = Expr::Unary { op: UnOp::Neg, expr: Box::new(Expr::Int(7)) };
        fold_expr(&mut u);
        assert!(matches!(u, Expr::Int(-7)));
    }
}

// ---------------------------------------------------------------------------
// Generator detection: a function is a generator iff its body contains a `yield`
// statement in its own control flow. We deliberately do not descend into nested
// lambdas/spawns — a `yield` there belongs to a different scope.
// ---------------------------------------------------------------------------

fn body_has_yield(stmts: &[Stmt]) -> bool {
    stmts.iter().any(stmt_has_yield)
}

fn stmt_has_yield(s: &Stmt) -> bool {
    match s {
        Stmt::Yield(_) => true,
        Stmt::If { cond, then, els } =>
            expr_has_yield(cond) || body_has_yield(then)
            || els.as_ref().map_or(false, |e| body_has_yield(e)),
        Stmt::While { cond, body } => expr_has_yield(cond) || body_has_yield(body),
        Stmt::ForRange { start, end, body, .. } =>
            expr_has_yield(start) || expr_has_yield(end) || body_has_yield(body),
        Stmt::ForEach { iter, body, .. } => expr_has_yield(iter) || body_has_yield(body),
        Stmt::Defer(b) => body_has_yield(b),
        Stmt::TryCatch { body, catch_body, finally_body, .. } =>
            body_has_yield(body)
            || catch_body.as_ref().map_or(false, |b| body_has_yield(b))
            || finally_body.as_ref().map_or(false, |b| body_has_yield(b)),
        Stmt::Let { value, .. } | Stmt::Assign { value, .. } => expr_has_yield(value),
        Stmt::IndexAssign { value, .. } | Stmt::FieldAssign { value, .. } => expr_has_yield(value),
        Stmt::Expr(e) | Stmt::Throw(e) => expr_has_yield(e),
        Stmt::Return(opt) | Stmt::Break(opt) => opt.as_ref().map_or(false, expr_has_yield),
        Stmt::Continue => false,
    }
}

// `yield` is a statement, so we only need to reach statement lists nested inside
// block / if / match expressions. We stop at lambda and spawn boundaries.
fn expr_has_yield(e: &Expr) -> bool {
    match e {
        Expr::At { expr, .. } => expr_has_yield(expr),
        Expr::Block { stmts, tail } =>
            body_has_yield(stmts) || tail.as_ref().map_or(false, |t| expr_has_yield(t)),
        Expr::If { cond, then, els } =>
            expr_has_yield(cond) || expr_has_yield(then) || expr_has_yield(els),
        Expr::Match { scrutinee, arms } =>
            expr_has_yield(scrutinee)
            || arms.iter().any(|a| a.guard.as_ref().map_or(false, expr_has_yield) || expr_has_yield(&a.body)),
        _ => false,
    }
}

#[cfg(test)]
mod net_tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;

    fn interp() -> Interp {
        let prog = crate::parser::parse_program("fn main(){ 0 }").expect("parse");
        Interp::new(&prog).expect("interp")
    }

    // A full loopback exercise of the TCP builtins: a server thread binds, accepts,
    // reads and echoes; the client connects, writes and reads the reply back. The
    // mpsc handshake removes any bind/connect race.
    #[test]
    fn tcp_echo_loopback() {
        let (tx, rx) = mpsc::channel::<()>();
        let server = thread::spawn(move || {
            let it = interp();
            let ln = it.call("tcp_listen", vec![Value::Str("127.0.0.1:19099".into())]).expect("listen");
            tx.send(()).unwrap();
            let conn = it.call("tcp_accept", vec![ln]).expect("accept");
            let msg = it.call("tcp_read", vec![conn.clone(), Value::Int(1024)]).expect("read");
            let m = match msg { Value::Str(s) => s, _ => String::new() };
            it.call("tcp_write", vec![conn, Value::Str(format!("echo:{}", m))]).expect("write");
        });
        rx.recv().unwrap();
        let it = interp();
        let conn = it.call("tcp_connect", vec![Value::Str("127.0.0.1:19099".into())]).expect("connect");
        it.call("tcp_write", vec![conn.clone(), Value::Str("ping".into())]).expect("write");
        let reply = it.call("tcp_read", vec![conn, Value::Int(1024)]).expect("read");
        server.join().unwrap();
        match reply { Value::Str(s) => assert_eq!(s, "echo:ping"), other => panic!("got {:?}", other) }
    }
}
