fn main() {
    let (w, h, maxi) = (600i32, 600i32, 200i32);
    let mut total = 0i64;
    for py in 0..h { for px in 0..w {
        let x0 = px as f64 / w as f64 * 3.5 - 2.5;
        let y0 = py as f64 / h as f64 * 2.0 - 1.0;
        let (mut x, mut y, mut it) = (0.0f64, 0.0f64, 0i32);
        while x*x + y*y <= 4.0 && it < maxi { let xt = x*x - y*y + x0; y = 2.0*x*y + y0; x = xt; it += 1; }
        if it == maxi { total += 1; }
    }}
    println!("{}", total);
}
