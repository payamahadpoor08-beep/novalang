fn main() {
    let n = 2000000usize;
    let mut s = vec![1u8; n];
    s[0] = 0; s[1] = 0;
    let mut i = 2;
    while i * i < n { if s[i] == 1 { let mut j = i*i; while j < n { s[j] = 0; j += i; } } i += 1; }
    let c: i64 = s.iter().map(|&x| x as i64).sum();
    println!("{}", c);
}
