/* nova_native_rt.c — the native-object AOT backend's runtime *primitives*.
 *
 * This file provides NO program logic. It supplies exactly the C-linkage
 * symbols that Nova's Cranelift codegen (src/jit.rs) calls into — the same
 * primitives the in-process JIT resolves against Rust functions — with a raw
 * i64/f64 ABI that matches those Rust helpers byte-for-byte:
 *   - arena arrays (nova_arr_*): the local-array track's backing store;
 *   - nova_fmod / nova_fpow: f64 `%` and `**` (no Cranelift instruction);
 *   - nova_print_f64: float printing, byte-identical to the interpreter's
 *     `impl Display for Value` via the same fmt_f64 used by runtime/nova_rt.c.
 *
 * Program logic still comes entirely from the Cranelift-emitted object; this is
 * linked alongside purely as the runtime, exactly as the JIT links Rust helpers.
 * The build's oracle gate (byte-diff vs `nova run`) covers every emitted binary,
 * so any ABI drift here shows up as a fallback, never as wrong output.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int64_t i64;

/* ---- arena arrays: handle = pointer to a growable i64 vector -------------- */
/* Matches src/jit.rs nova_arr_* exactly, including the deopt pointer being the
 * FIRST argument of get/set/pop (OOB/empty sets *dp = 1). One process = one run,
 * so allocations live until exit (no free needed), like the JIT's per-call arena. */

typedef struct { i64 len, cap; i64 *data; } NArr;

static NArr *arr_new_impl(i64 cap) {
    NArr *a = (NArr *)malloc(sizeof(NArr));
    a->len = 0;
    a->cap = cap > 0 ? cap : 4;
    a->data = (i64 *)malloc((size_t)a->cap * sizeof(i64));
    return a;
}

i64 nova_arr_new(void) { return (i64)(intptr_t)arr_new_impl(4); }

i64 nova_arr_fill(i64 n, i64 v) {
    i64 cnt = n <= 0 ? 0 : n;
    NArr *a = arr_new_impl(cnt > 0 ? cnt : 4);
    for (i64 i = 0; i < cnt; i++) a->data[i] = v;
    a->len = cnt;
    return (i64)(intptr_t)a;
}

void nova_arr_push(i64 h, i64 v) {
    NArr *a = (NArr *)(intptr_t)h;
    if (a->len == a->cap) {
        a->cap *= 2;
        a->data = (i64 *)realloc(a->data, (size_t)a->cap * sizeof(i64));
    }
    a->data[a->len++] = v;
}

i64 nova_arr_len(i64 h) { return ((NArr *)(intptr_t)h)->len; }

i64 nova_arr_get(i64 *dp, i64 h, i64 i) {
    NArr *a = (NArr *)(intptr_t)h;
    if (i < 0 || i >= a->len) { *dp = 1; return 0; }
    return a->data[i];
}

void nova_arr_set(i64 *dp, i64 h, i64 i, i64 v) {
    NArr *a = (NArr *)(intptr_t)h;
    if (i < 0 || i >= a->len) { *dp = 1; return; }
    a->data[i] = v;
}

i64 nova_arr_pop(i64 *dp, i64 h) {
    NArr *a = (NArr *)(intptr_t)h;
    if (a->len == 0) { *dp = 1; return 0; }
    return a->data[--a->len];
}

/* ---- f64 % and ** (call back like the JIT's Rust nova_fmod/nova_fpow) ----- */
double fmod(double, double); /* libm; no math.h to keep the TU minimal */
double pow(double, double);
double nova_fmod(double a, double b) { return fmod(a, b); }
double nova_fpow(double a, double b) { return pow(a, b); }

/* ---- float printing: byte-identical to interp.rs `impl Display` ----------- */
/* fmt_f64 below is copied VERBATIM from runtime/nova_rt.c (the oracle-verified
 * formatter): integral & finite -> Rust {:.1}; else the shortest round-tripping
 * digit string in plain decimal. Kept in sync with that file. */

typedef struct SB { char *buf; i64 len, cap; } SB;
static void sb_init(SB *sb) { sb->cap = 64; sb->len = 0; sb->buf = malloc(64); }
static void sb_put(SB *sb, const char *s, i64 n) {
    while (sb->len + n + 1 > sb->cap) { sb->cap *= 2; sb->buf = realloc(sb->buf, sb->cap); }
    memcpy(sb->buf + sb->len, s, n);
    sb->len += n;
    sb->buf[sb->len] = 0;
}
static void sb_puts(SB *sb, const char *s) { sb_put(sb, s, (i64)strlen(s)); }

static void fmt_f64(SB *sb, double x) {
    if (x != x) { sb_puts(sb, "NaN"); return; }
    if (x > 1.7e308 && x / 2 == x) { sb_puts(sb, "inf"); return; }
    if (x < -1.7e308 && x / 2 == x) { sb_puts(sb, "-inf"); return; }
    int integral = (x >= 4503599627370496.0 || x <= -4503599627370496.0)
                || ((double)(i64)x == x);
    char big[352];
    if (integral) {
        snprintf(big, sizeof big, "%.1f", x); /* == Rust {:.1}, incl. "-0.0" */
        sb_puts(sb, big);
        return;
    }
    char tmp[64];
    int prec = 1;
    for (; prec <= 17; prec++) {
        snprintf(tmp, sizeof tmp, "%.*e", prec - 1, x);
        if (strtod(tmp, NULL) == x) break;
    }
    char digits[32]; int nd = 0; int exp10 = 0; int neg = 0;
    const char *p = tmp;
    if (*p == '-') { neg = 1; p++; }
    for (; *p && *p != 'e'; p++) {
        if (*p != '.') digits[nd++] = *p;
    }
    if (*p == 'e') exp10 = (int)strtol(p + 1, NULL, 10);
    {
        char full[800];
        snprintf(full, sizeof full, "%.770e", x);
        char fd[784]; int fn = 0; int fexp = 0;
        const char *q = full;
        if (*q == '-') q++;
        for (; *q && *q != 'e'; q++) { if (*q != '.') fd[fn++] = *q; }
        if (*q == 'e') fexp = (int)strtol(q + 1, NULL, 10);
        if (fn > nd && fd[nd] == '5') {
            int tie = 1;
            for (int i = nd + 1; i < fn; i++) { if (fd[i] != '0') { tie = 0; break; } }
            if (tie) {
                memcpy(digits, fd, (size_t)nd);
                exp10 = fexp;
                int i = nd - 1;
                for (; i >= 0; i--) {
                    if (digits[i] != '9') { digits[i]++; break; }
                    digits[i] = '0';
                }
                if (i < 0) { digits[0] = '1'; nd = 1; exp10++; }
            }
        }
    }
    char out[400]; int o = 0;
    if (neg) out[o++] = '-';
    if (exp10 < 0) {
        out[o++] = '0'; out[o++] = '.';
        for (int i = 0; i < -exp10 - 1; i++) out[o++] = '0';
        for (int i = 0; i < nd; i++) out[o++] = digits[i];
    } else {
        for (int i = 0; i <= exp10; i++) out[o++] = i < nd ? digits[i] : '0';
        out[o++] = '.';
        for (int i = exp10 + 1; i < nd; i++) out[o++] = digits[i];
        if (o > 0 && out[o - 1] == '.') out[o++] = '0';
    }
    out[o] = 0;
    sb_puts(sb, out);
}

/* print a Nova float value: fmt_f64 + trailing newline, one write(2). */
void nova_print_f64(double x) {
    SB sb; sb_init(&sb);
    fmt_f64(&sb, x);
    sb_put(&sb, "\n", 1);
    ssize_t _ = write(1, sb.buf, (size_t)sb.len);
    (void)_;
    free(sb.buf);
}
