// Nova AST — the typed tree the interpreter walks.
// Covers Nova Core: literals, vars, arithmetic/logic/comparison,
// if/else, while, for-range, functions (incl. recursion), calls, return.

#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Item {
    Func(Func),
    Struct(StructDef),
    Impl(ImplBlock),
    Enum(EnumDef),
    Use(UseDecl),
    Test(TestBlock),
    Machine(MachineDef),
    Const { name: String, value: Expr },
    Trait(TraitDef),
    // macros are fully expanded at parse time, so the stored def is never read again
    #[allow(dead_code)]
    Macro(MacroDef),
    // `type Name = Target;` — a type alias. `target` is the aliased type's head
    // name (e.g. "Int"); the type checker resolves it through this map.
    // `refinement` holds a predicate for `type Pos = Int if it > 0;` — the value
    // is bound to `it` and the predicate is checked at annotated-`let` time.
    TypeAlias { name: String, target: String, refinement: Option<Expr> },
    // `extern { fn c(a, b); ... }` — declares foreign functions. Nova has no FFI,
    // so these are known to the checker (name + arity) but error if actually called.
    Extern(Vec<ExternFn>),
    // `use "path.nova";` — import every item from another Nova file. Resolved and
    // inlined by the module loader before type-checking/execution.
    Import { path: String },
    // `migrate from Old to New { <body producing a New value> }` — state migration:
    // transforms a value of the old struct shape into the new one (for preserving
    // state across a code/schema update). The body runs with `old` bound to the
    // incoming value; the `migrate(value)` builtin applies the matching migration.
    Migration { from: String, to: String, body: Vec<Stmt> },
}

#[derive(Debug, Clone)]
pub struct ExternFn {
    pub name: String,
    pub arity: usize,
    // true when the signature ends with `...` (variadic); arity is then a minimum
    pub variadic: bool,
}

#[derive(Debug, Clone)]
pub struct MacroDef {
    pub name: String,
    // parameter names captured by the matcher, e.g. ["x", "y"] from ($x:expr, $y:expr)
    pub params: Vec<String>,
    // body template text with $param placeholders, e.g. "$x * $x"
    pub body: String,
}

#[derive(Debug, Clone)]
pub struct MachineDef {
    pub name: String,
    pub initial: String,
    // (from_state, to_state, event)
    pub transitions: Vec<(String, String, String)>,
}

#[derive(Debug, Clone)]
pub enum FmtPart {
    Lit(String),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct TestBlock {
    pub name: String,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub struct UseDecl {
    pub module: String,            // e.g. "math"
    #[allow(dead_code)]
    pub names: Vec<String>,        // specific names, or empty for wildcard/whole-module
    #[allow(dead_code)]
    pub wildcard: bool,            // use math.*
    pub alias: Option<String>,     // use math as m
}

impl UseDecl {
    // the first path segment, e.g. "math" from "math.linalg"
    pub fn module_root(&self) -> String {
        self.module.split('.').next().unwrap_or(&self.module).to_string()
    }
}

#[derive(Debug, Clone)]
pub struct EnumDef {
    pub name: String,
    pub variants: Vec<VariantDef>,
}

#[derive(Debug, Clone)]
pub struct VariantDef {
    pub name: String,
    pub arity: usize, // number of tuple payload slots; 0 = unit variant
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ImplBlock {
    pub type_name: String,
    // Some("Describe") when this is `impl Describe for Type`, else None
    pub trait_name: Option<String>,
    pub methods: Vec<Func>,
}

#[derive(Debug, Clone)]
pub struct TraitDef {
    pub name: String,
    // required method names (signatures only, no body)
    pub required: Vec<String>,
    // default method implementations (name + full function)
    pub defaults: Vec<Func>,
}

#[derive(Debug, Clone)]
pub struct Func {
    pub name: String,
    pub params: Vec<String>,
    // optional declared type annotation per parameter (parallel to params);
    // None where the source omitted it. Strings are raw type names like "Int", "T".
    pub param_types: Vec<Option<String>>,
    // ownership mode per parameter (parallel to params): Some("linear"/"affine")
    // from a `linear T` / `affine T` annotation, else None.
    pub param_modes: Vec<Option<String>>,
    // optional declared return type, e.g. Some("Int") or Some("T")
    pub ret_type: Option<String>,
    // generic type parameter names, e.g. ["T", "U"] from fn f[T, U](...)
    pub type_params: Vec<String>,
    // trait bounds on generic params, from `[T: Trait]` and `where` clauses:
    // each entry is (generic_param_name, [trait names it must implement]).
    pub where_bounds: Vec<(String, Vec<String>)>,
    // declared effects from `![IO, ...]`: None = unannotated (effect-unchecked),
    // Some([]) = pure, Some([..]) = may only perform the listed effects.
    pub effects: Option<Vec<String>>,
    pub body: Vec<Stmt>,
    // true when declared `async fn`: calling it yields a Future instead of
    // running the body eagerly. Defaults to false for every existing path.
    pub is_async: bool,
    // `#[...]` attributes attached to this function (empty for most). These carry
    // real semantics — zero_alloc (a static allocation ban), self_healing (retry
    // on error), hot_swap (runtime body replacement), integrity (tamper hash), …
    pub attrs: Vec<Attr>,
}

// A parsed `#[name(k: v, ...)]` attribute. `args` holds the `(key, value)` pairs
// (positional args use the value with an empty key), values kept as raw strings.
// `exprs` holds expression arguments for contract attributes (`requires`,
// `ensures`, `assumes`) which take real predicates rather than literals.
#[derive(Debug, Clone)]
pub struct Attr {
    pub name: String,
    pub args: Vec<(String, String)>,
    pub exprs: Vec<Expr>,
    // the attribute's raw source text (without the surrounding `#[ ]`), e.g.
    // `self_healing(attempts: 5)`. Kept so the formatter can re-emit the exact
    // syntax — the `compiler_attr` grammar has fixed keyword forms that the
    // parsed `args` don't fully capture. Contract attrs (requires/ensures/
    // assumes) are re-serialized from `exprs` instead, so their predicates
    // track identifier renames (e.g. under `nova obfuscate`).
    pub raw: String,
}

impl Attr {
    // integer-valued argument by key (or the first positional), if present
    pub fn int_arg(&self, key: &str) -> Option<i64> {
        self.args.iter()
            .find(|(k, _)| k == key || k.is_empty())
            .and_then(|(_, v)| v.parse().ok())
    }
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let { name: String, ty: Option<String>, value: Expr },
    Assign { name: String, value: Expr },
    IndexAssign { base: Expr, index: Expr, value: Expr },
    FieldAssign { base: Expr, field: String, value: Expr },
    Expr(Expr),
    Return(Option<Expr>),
    If { cond: Expr, then: Vec<Stmt>, els: Option<Vec<Stmt>> },
    While { cond: Expr, body: Vec<Stmt> },
    ForRange { var: String, start: Expr, end: Expr, inclusive: bool, body: Vec<Stmt> },
    ForEach { var: String, iter: Expr, body: Vec<Stmt> },
    Throw(Expr),
    // `yield expr` inside a generator function: produces the next value lazily.
    Yield(Option<Expr>),
    Break(Option<Expr>),
    Continue,
    Defer(Vec<Stmt>),
    TryCatch {
        body: Vec<Stmt>,
        catch_var: Option<String>,
        catch_body: Option<Vec<Stmt>>,
        finally_body: Option<Vec<Stmt>>,
    },
}

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    BigIntLit(String),
    Float(f64),
    Str(String),
    Bool(bool),
    Null,
    Ident(String),
    Array(Vec<Expr>),
    MapLit(Vec<(Expr, Expr)>),
    SetLit(Vec<Expr>),
    Comprehension { body: Box<Expr>, var: String, iter: Box<Expr>, cond: Option<Box<Expr>> },
    FmtStr(Vec<FmtPart>),
    Index { base: Box<Expr>, index: Box<Expr> },
    // a range value: lo..hi or lo..=hi (bounds optional). Used both as a
    // standalone value and as an array/string index (producing a slice).
    RangeLit { lo: Option<Box<Expr>>, hi: Option<Box<Expr>>, inclusive: bool },
    StructLit { name: String, fields: Vec<(String, Expr)> },
    Field { base: Box<Expr>, field: String },
    SafeField { base: Box<Expr>, field: String },
    MethodCall { base: Box<Expr>, method: String, args: Vec<Expr> },
    Lambda { params: Vec<String>, body: Box<LambdaBody> },
    // calling a value that is itself a closure/function value: f(args)
    CallValue { callee: Box<Expr>, args: Vec<Expr> },
    Unary { op: UnOp, expr: Box<Expr> },
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Call { callee: String, args: Vec<Expr> },
    // a tail block that yields a value: { stmts...; final_expr }
    Block { stmts: Vec<Stmt>, tail: Option<Box<Expr>> },
    If { cond: Box<Expr>, then: Box<Expr>, els: Box<Expr> },
    Match { scrutinee: Box<Expr>, arms: Vec<MatchArm> },
    // --- async / concurrency ---
    // `expr.await` or prefix `await expr`: drive a Future/JoinHandle to a value
    Await(Box<Expr>),
    // `spawn { ... }`: queue a task on the scheduler, yield a JoinHandle
    Spawn(Vec<Stmt>),
    // `<- ch`: receive from a channel (reserved; receive is via recv()/select today)
    #[allow(dead_code)]
    Recv(Box<Expr>),
    // `ch <- v`: send a value into a channel
    Send { chan: Box<Expr>, value: Box<Expr> },
    // `select { chan ch => arm, ... }`: wait on the first ready channel
    Select(Vec<SelectArm>),
    // Source-position marker wrapping a statement-level expression. The parser
    // inserts it so the interpreter and type checker can report `line, col` in
    // error messages. Transparent: it evaluates to its inner expression's value.
    At { pos: (u32, u32), expr: Box<Expr> },
}

#[derive(Debug, Clone)]
pub struct SelectArm {
    // the channel expression to receive from
    pub chan: Expr,
    // optional binding for the received value, usable inside `body`
    pub binding: Option<String>,
    // expression evaluated when this arm fires
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard,
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Null,
    Binding(String),                       // x  -> binds the value
    // EnumVariant: Name(sub_patterns...) e.g. Some(x), Cons(h, t), None
    EnumVariant { name: String, sub: Vec<Pattern> },
    Or(Vec<Pattern>),                      // p1 | p2 | p3
    Range { lo: i64, hi: i64, inclusive: bool },
    // (p1, p2, ...) — matches an array of exactly that length, binding positionally
    Tuple(Vec<Pattern>),
    // Name { field: pat, ... } — matches a struct, binding named fields
    Struct { name: String, fields: Vec<(String, Pattern)> },
    // [p, p, ...] / [p, ...rest, p] — matches an array. `prefix` matches from the
    // front, `suffix` from the back. `rest` is None for an exact-length match, or
    // Some(name?) for a `...` rest that soaks up the middle (binding it when named).
    Slice { prefix: Vec<Pattern>, rest: Option<Option<String>>, suffix: Vec<Pattern> },
}

#[derive(Debug, Clone)]
pub enum LambdaBody {
    Expr(Expr),
    Block(Vec<Stmt>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp { Neg, Not, BitNot }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Rem, Pow,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
    BitOr, BitXor, BitAnd, Shl, Shr,
}
