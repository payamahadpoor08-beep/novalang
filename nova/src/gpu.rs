// GPU kernel emission for `#[gpu]` data-parallel kernels.
//
// A `#[gpu]` kernel is a single-parameter numeric function whose body is a
// numeric expression in that parameter — the elementwise map applied by
// `gpu_map(array, kernel)`. This module emits the GLSL compute-shader source a
// Vulkan pipeline would compile to SPIR-V and dispatch over the array buffer
// (one invocation per element). The runtime chooses the GPU path when a GPU is
// present (`gpu_available()`); with no GPU it runs the identical computation on
// the CPU, so `gpu_map` is byte-identical to `map` either way.
//
// Honest scope: this emits the on-device *source*; actually compiling it to
// SPIR-V and dispatching it needs a GPU + driver. The CPU fallback is the
// verified path in a GPU-less environment.

use crate::ast::*;

// Emit GLSL compute source for every `#[gpu]` kernel in the program, in order.
// Each entry is (kernel name, Ok(glsl) | Err(reason it can't be offloaded)).
pub fn emit_kernels(prog: &Program) -> Vec<(String, Result<String, String>)> {
    let mut out = Vec::new();
    for item in &prog.items {
        if let Item::Func(f) = item {
            if f.attrs.iter().any(|a| a.name == "gpu") {
                out.push((f.name.clone(), emit_one(f)));
            }
        }
    }
    out
}

fn strip(e: &Expr) -> &Expr {
    match e { Expr::At { expr, .. } => strip(expr), other => other }
}

// the kernel's single numeric expression: a body of exactly one expression /
// return statement.
fn kernel_expr(body: &[Stmt]) -> Option<&Expr> {
    if body.len() != 1 { return None; }
    match &body[0] {
        Stmt::Expr(e) | Stmt::Return(Some(e)) => Some(e),
        _ => None,
    }
}

fn emit_one(f: &Func) -> Result<String, String> {
    if f.params.len() != 1 {
        return Err("a #[gpu] kernel must take exactly one element parameter".into());
    }
    let param = &f.params[0];
    let body = kernel_expr(&f.body)
        .ok_or("a #[gpu] kernel body must be a single numeric expression")?;
    let float = expr_is_float(strip(body), param);
    let ty = if float { "float" } else { "int" };
    let expr = glsl_expr(strip(body), param, float)?;
    Ok(format!(
"#version 450
// GPU compute kernel generated from Nova `#[gpu] fn {name}`.
// One invocation per array element: data[i] = {name}(data[i]).
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Buf {{ {ty} data[]; }};
void main() {{
    uint i = gl_GlobalInvocationID.x;
    if (i >= data.length()) return;
    {ty} {param} = data[i];
    data[i] = {expr};
}}
", name = f.name, ty = ty, param = param, expr = expr))
}

// does this kernel expression operate on floats? (any float literal or float
// intrinsic makes the whole buffer float.)
fn expr_is_float(e: &Expr, param: &str) -> bool {
    match e {
        Expr::At { expr, .. } => expr_is_float(expr, param),
        Expr::Float(_) => true,
        Expr::Unary { expr, .. } => expr_is_float(expr, param),
        Expr::Binary { lhs, rhs, .. } => expr_is_float(lhs, param) || expr_is_float(rhs, param),
        Expr::Call { callee, args } =>
            callee == "to_float" || callee == "sqrt" || callee == "sin" || callee == "cos"
            || args.iter().any(|a| expr_is_float(a, param)),
        _ => false,
    }
}

// translate a numeric kernel expression to GLSL. Bounded to arithmetic on the
// element parameter + a few intrinsics; anything else -> Err (kernel is reported
// CPU-only, never emitted wrong).
fn glsl_expr(e: &Expr, param: &str, float: bool) -> Result<String, String> {
    let num = |x: f64| if float { format!("{:?}", x) } else { format!("{}", x as i64) };
    Ok(match e {
        Expr::At { expr, .. } => glsl_expr(expr, param, float)?,
        Expr::Int(n) => if float { format!("{:?}", *n as f64) } else { n.to_string() },
        Expr::Float(x) => num(*x),
        Expr::Ident(n) if n == param => param.to_string(),
        Expr::Ident(n) => return Err(format!("kernel refers to a non-element variable `{}`", n)),
        Expr::Unary { op: UnOp::Neg, expr } => format!("(-{})", glsl_expr(expr, param, float)?),
        Expr::Binary { op, lhs, rhs } => {
            let o = match op {
                BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*",
                BinOp::Div => "/", BinOp::Rem => "%",
                _ => return Err("only + - * / % are supported in a #[gpu] kernel".into()),
            };
            format!("({} {} {})", glsl_expr(lhs, param, float)?, o, glsl_expr(rhs, param, float)?)
        }
        Expr::Call { callee, args } => {
            let a: Result<Vec<String>, String> =
                args.iter().map(|x| glsl_expr(x, param, float)).collect();
            let a = a?;
            match (callee.as_str(), a.len()) {
                ("to_float", 1) => format!("float({})", a[0]),
                ("to_int", 1) => format!("int({})", a[0]),
                ("abs", 1) => format!("abs({})", a[0]),
                ("sqrt", 1) => format!("sqrt({})", a[0]),
                ("sin", 1) => format!("sin({})", a[0]),
                ("cos", 1) => format!("cos({})", a[0]),
                ("min", 2) => format!("min({}, {})", a[0], a[1]),
                ("max", 2) => format!("max({}, {})", a[0], a[1]),
                _ => return Err(format!("call `{}` is not supported in a #[gpu] kernel", callee)),
            }
        }
        _ => return Err("unsupported expression in a #[gpu] kernel".into()),
    })
}

#[cfg(test)]
mod tests {
    use crate::parser::parse_program;

    #[test]
    fn emits_glsl_for_numeric_kernels() {
        let src = "#[gpu] fn square(x){ x*x }\n#[gpu] fn scale(x){ to_float(x)*1.5+2.0 }\nfn main(){ print(1) }";
        let prog = parse_program(src).expect("parse");
        let ks = super::emit_kernels(&prog);
        assert_eq!(ks.len(), 2);
        let sq = ks[0].1.as_ref().expect("square emits");
        assert!(sq.contains("int data[]"), "int buffer: {}", sq);
        assert!(sq.contains("data[i] = (x * x);"), "square body: {}", sq);
        let sc = ks[1].1.as_ref().expect("scale emits");
        assert!(sc.contains("float data[]"), "float buffer");
        assert!(sc.contains("float(x)"), "to_float -> float()");
    }

    #[test]
    fn declines_non_numeric_kernel() {
        // a kernel referring to a free variable / unsupported call is CPU-only.
        let src = "#[gpu] fn k(x){ x + y }\nfn main(){ print(1) }";
        let prog = parse_program(src).expect("parse");
        let ks = super::emit_kernels(&prog);
        assert_eq!(ks.len(), 1);
        assert!(ks[0].1.is_err(), "free-variable kernel must be reported CPU-only");
    }

    #[test]
    fn ignores_unannotated_functions() {
        let src = "fn plain(x){ x*x }\nfn main(){ print(1) }";
        let prog = parse_program(src).expect("parse");
        assert!(super::emit_kernels(&prog).is_empty(), "only #[gpu] fns are kernels");
    }
}
