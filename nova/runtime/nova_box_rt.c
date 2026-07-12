/* Nova native BOXED runtime — the general "compile any scalar/string program"
 * tier for the Cranelift object backend (src/jit.rs `BoxGen`). It is the pointer-
 * ABI, arena-allocated counterpart of runtime/nova_rt.c's tagged `NV` value: a
 * Nova value is an i64 HANDLE (a pointer to a heap `BV`), so Cranelift never has
 * to pass the tagged struct by value. One process = one program run, so values
 * are allocate-and-leak (no refcounting) — exactly like the arena arrays in
 * nova_native_rt.c. Every operator's SEMANTICS are copied byte-for-byte from
 * nova_rt.c (which mirrors src/interp.rs); the build-time oracle gate (byte-diff
 * vs `nova run`) enforces the match, so output is never wrong. i64 arithmetic
 * wraps: an overflowing program diverges from the interpreter's BigInt and the
 * gate falls back to the C/embed build, identical to nova_rt.c's int tier.
 *
 * Written from scratch for Nova. Phase 1 surface: int/float/bool/null/string +
 * arithmetic/comparison/logic/concat/bitwise, f-strings, `len`, and `print`.
 * Container types (arrays/maps/structs) are NOT here — those programs stay on the
 * C boxed tier via the fallback. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef int64_t i64;

enum { BV_INT, BV_FLOAT, BV_BOOL, BV_NULL, BV_STR };

typedef struct BStr {
    i64 nchars;      /* Nova indexes strings by Unicode char */
    i64 nbytes;
    char *utf8;
} BStr;

typedef struct BV {
    uint8_t tag;
    union { i64 i; double f; BStr *s; };
} BV;

/* handle <-> pointer. Allocate-and-leak: a boxed value lives for the whole run. */
static BV *bv_h(i64 h) { return (BV *)(intptr_t)h; }
static i64 bv_mk(BV v) { BV *p = malloc(sizeof(BV)); *p = v; return (i64)(intptr_t)p; }

static void bv_die(const char *msg) {
    fprintf(stderr, "runtime error: %s\n", msg);
    exit(1);
}

/* ---- constructors (all return a handle) ---- */
i64 nv_int(i64 v)    { BV x; x.tag = BV_INT;   x.i = v;      return bv_mk(x); }
i64 nv_float(double v){ BV x; x.tag = BV_FLOAT; x.f = v;      return bv_mk(x); }
i64 nv_bool(i64 v)   { BV x; x.tag = BV_BOOL;  x.i = v != 0; return bv_mk(x); }
i64 nv_null(void)    { BV x; x.tag = BV_NULL;  x.i = 0;      return bv_mk(x); }

/* build a string value from `len` UTF-8 bytes at `ptr`, counting Unicode chars */
static BStr *bstr_new(const char *bytes, i64 nbytes) {
    BStr *s = malloc(sizeof(BStr));
    s->nbytes = nbytes;
    s->utf8 = malloc(nbytes + 1);
    memcpy(s->utf8, bytes, nbytes);
    s->utf8[nbytes] = 0;
    i64 nchars = 0;
    for (i64 i = 0; i < nbytes; i++)
        if (((unsigned char)bytes[i] & 0xC0) != 0x80) nchars++;
    s->nchars = nchars;
    return s;
}
i64 nv_str(i64 ptr, i64 len) {
    BV x; x.tag = BV_STR; x.s = bstr_new((const char *)(intptr_t)ptr, len);
    return bv_mk(x);
}

static const char *bv_type_name(BV *v) {
    switch (v->tag) {
        case BV_INT: return "int";
        case BV_FLOAT: return "float";
        case BV_BOOL: return "bool";
        case BV_NULL: return "null";
        default: return "string";
    }
}

/* ---- truthiness / length / equality (mirror nova_rt.c) ---- */
i64 nv_truthy(i64 h) {
    BV *v = bv_h(h);
    switch (v->tag) {
        case BV_BOOL: case BV_INT: return v->i != 0;
        case BV_NULL: return 0;
        default: return 1;
    }
}

i64 nv_len(i64 h) {
    BV *v = bv_h(h);
    if (v->tag == BV_STR) return nv_int(v->s->nchars);
    fprintf(stderr, "runtime error: len expects array or string, got %s\n", bv_type_name(v));
    exit(1);
}

static i64 bv_eq(BV *a, BV *b) {
    if (a->tag == BV_INT && b->tag == BV_INT) return a->i == b->i;
    if (a->tag == BV_FLOAT && b->tag == BV_FLOAT) return a->f == b->f;
    if (a->tag == BV_INT && b->tag == BV_FLOAT) return (double)a->i == b->f;
    if (a->tag == BV_FLOAT && b->tag == BV_INT) return a->f == (double)b->i;
    if (a->tag == BV_BOOL && b->tag == BV_BOOL) return a->i == b->i;
    if (a->tag == BV_NULL && b->tag == BV_NULL) return 1;
    if (a->tag == BV_STR && b->tag == BV_STR)
        return a->s->nbytes == b->s->nbytes && memcmp(a->s->utf8, b->s->utf8, a->s->nbytes) == 0;
    return 0;
}
i64 nv_eq(i64 a, i64 b) { return nv_bool(bv_eq(bv_h(a), bv_h(b))); }
i64 nv_ne(i64 a, i64 b) { return nv_bool(!bv_eq(bv_h(a), bv_h(b))); }

/* ---- display: byte-identical to interp.rs `impl Display for Value` ---- */
typedef struct SB { char *buf; i64 len, cap; } SB;
static void sb_init(SB *sb) { sb->cap = 64; sb->len = 0; sb->buf = malloc(64); }
static void sb_put(SB *sb, const char *s, i64 n) {
    while (sb->len + n + 1 > sb->cap) { sb->cap *= 2; sb->buf = realloc(sb->buf, sb->cap); }
    memcpy(sb->buf + sb->len, s, n);
    sb->len += n;
    sb->buf[sb->len] = 0;
}
static void sb_puts(SB *sb, const char *s) { sb_put(sb, s, (i64)strlen(s)); }

/* Nova floats print exactly like interp.rs (see runtime/nova_rt.c:fmt_f64 for the
 * full derivation): integral & finite -> Rust {:.1}; anything else -> the shortest
 * round-tripping digit string in plain decimal (no e-notation). Copied verbatim. */
static void fmt_f64(SB *sb, double x) {
    if (x != x) { sb_puts(sb, "NaN"); return; }
    if (x > 1.7e308 && x / 2 == x) { sb_puts(sb, "inf"); return; }
    if (x < -1.7e308 && x / 2 == x) { sb_puts(sb, "-inf"); return; }
    int integral = (x >= 4503599627370496.0 || x <= -4503599627370496.0)
                || ((double)(i64)x == x);
    char big[352];
    if (integral) {
        snprintf(big, sizeof big, "%.1f", x);
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

static void bv_fmt(SB *sb, BV *v) {
    char tmp[32];
    switch (v->tag) {
        case BV_INT:
            snprintf(tmp, sizeof tmp, "%lld", (long long)v->i);
            sb_puts(sb, tmp);
            break;
        case BV_FLOAT: fmt_f64(sb, v->f); break;
        case BV_BOOL: sb_puts(sb, v->i ? "true" : "false"); break;
        case BV_NULL: sb_puts(sb, "null"); break;
        case BV_STR: sb_put(sb, v->s->utf8, v->s->nbytes); break;
    }
}

void nv_print(i64 h) {
    SB sb; sb_init(&sb);
    bv_fmt(&sb, bv_h(h));
    fwrite(sb.buf, 1, sb.len, stdout);
    fputc('\n', stdout);
    free(sb.buf);
}

/* owned string of h's display form (for f-strings / string +) */
i64 nv_tostr(i64 h) {
    BV *v = bv_h(h);
    if (v->tag == BV_STR) return h; /* already a string: same handle */
    SB sb; sb_init(&sb);
    bv_fmt(&sb, v);
    i64 r = nv_str((i64)(intptr_t)sb.buf, sb.len);
    free(sb.buf);
    return r;
}

/* ---- operators (mirror eval_binop; args + result are handles) ---- */
static double bv_as_f(BV *v) {
    if (v->tag == BV_INT) return (double)v->i;
    if (v->tag == BV_FLOAT) return v->f;
    bv_die("expected a number");
    return 0;
}
static int bv_is_num(BV *v) { return v->tag == BV_INT || v->tag == BV_FLOAT; }

i64 nv_concat2(i64 ah, i64 bh) { /* string + value / value + string */
    i64 sa = nv_tostr(ah);
    i64 sb2 = nv_tostr(bh);
    BStr *x = bv_h(sa)->s, *y = bv_h(sb2)->s;
    i64 nbytes = x->nbytes + y->nbytes;
    char *buf = malloc(nbytes);
    memcpy(buf, x->utf8, x->nbytes);
    memcpy(buf + x->nbytes, y->utf8, y->nbytes);
    i64 r = nv_str((i64)(intptr_t)buf, nbytes);
    free(buf);
    return r;
}

i64 nv_add(i64 ah, i64 bh) {
    BV *a = bv_h(ah), *b = bv_h(bh);
    if (a->tag == BV_INT && b->tag == BV_INT) return nv_int(a->i + b->i); /* overflow -> gate */
    if (a->tag == BV_STR || b->tag == BV_STR) return nv_concat2(ah, bh);
    if (bv_is_num(a) && bv_is_num(b)) return nv_float(bv_as_f(a) + bv_as_f(b));
    fprintf(stderr, "runtime error: cannot apply Add to %s and %s\n", bv_type_name(a), bv_type_name(b));
    exit(1);
}
i64 nv_sub(i64 ah, i64 bh) {
    BV *a = bv_h(ah), *b = bv_h(bh);
    if (a->tag == BV_INT && b->tag == BV_INT) return nv_int(a->i - b->i);
    if (bv_is_num(a) && bv_is_num(b)) return nv_float(bv_as_f(a) - bv_as_f(b));
    fprintf(stderr, "runtime error: cannot apply Sub to %s and %s\n", bv_type_name(a), bv_type_name(b));
    exit(1);
}
i64 nv_mul(i64 ah, i64 bh) {
    BV *a = bv_h(ah), *b = bv_h(bh);
    if (a->tag == BV_INT && b->tag == BV_INT) return nv_int(a->i * b->i);
    if (bv_is_num(a) && bv_is_num(b)) return nv_float(bv_as_f(a) * bv_as_f(b));
    fprintf(stderr, "runtime error: cannot apply Mul to %s and %s\n", bv_type_name(a), bv_type_name(b));
    exit(1);
}
i64 nv_div(i64 ah, i64 bh) {
    BV *a = bv_h(ah), *b = bv_h(bh);
    if (a->tag == BV_INT && b->tag == BV_INT) {
        if (b->i == 0) bv_die("division by zero");
        return nv_int(a->i / b->i);
    }
    if (bv_is_num(a) && bv_is_num(b)) return nv_float(bv_as_f(a) / bv_as_f(b));
    fprintf(stderr, "runtime error: cannot apply Div to %s and %s\n", bv_type_name(a), bv_type_name(b));
    exit(1);
}
i64 nv_rem(i64 ah, i64 bh) {
    BV *a = bv_h(ah), *b = bv_h(bh);
    if (a->tag == BV_INT && b->tag == BV_INT) {
        if (b->i == 0) bv_die("modulo by zero");
        return nv_int(a->i % b->i);
    }
    if (bv_is_num(a) && bv_is_num(b)) {
        bv_die("float % is not AOT-supported"); /* interp uses Rust %, gate anyway */
    }
    fprintf(stderr, "runtime error: cannot apply Rem to %s and %s\n", bv_type_name(a), bv_type_name(b));
    exit(1);
}
/* string ordering is byte-lexicographic, matching Rust's str ordering */
static int bv_strcmp(BV *a, BV *b) {
    i64 n = a->s->nbytes < b->s->nbytes ? a->s->nbytes : b->s->nbytes;
    int c = memcmp(a->s->utf8, b->s->utf8, n);
    if (c) return c;
    return (a->s->nbytes > b->s->nbytes) - (a->s->nbytes < b->s->nbytes);
}
i64 nv_cmp_lt(i64 ah, i64 bh) {
    BV *a = bv_h(ah), *b = bv_h(bh);
    if (a->tag == BV_STR && b->tag == BV_STR) return nv_bool(bv_strcmp(a, b) < 0);
    return nv_bool(bv_as_f(a) < bv_as_f(b));
}
i64 nv_cmp_le(i64 ah, i64 bh) {
    BV *a = bv_h(ah), *b = bv_h(bh);
    if (a->tag == BV_STR && b->tag == BV_STR) return nv_bool(bv_strcmp(a, b) <= 0);
    return nv_bool(bv_as_f(a) <= bv_as_f(b));
}
i64 nv_cmp_gt(i64 ah, i64 bh) {
    BV *a = bv_h(ah), *b = bv_h(bh);
    if (a->tag == BV_STR && b->tag == BV_STR) return nv_bool(bv_strcmp(a, b) > 0);
    return nv_bool(bv_as_f(a) > bv_as_f(b));
}
i64 nv_cmp_ge(i64 ah, i64 bh) {
    BV *a = bv_h(ah), *b = bv_h(bh);
    if (a->tag == BV_STR && b->tag == BV_STR) return nv_bool(bv_strcmp(a, b) >= 0);
    return nv_bool(bv_as_f(a) >= bv_as_f(b));
}

i64 nv_bit(i64 ah, i64 bh, i64 op) {
    BV *a = bv_h(ah), *b = bv_h(bh);
    if (a->tag != BV_INT || b->tag != BV_INT) bv_die("bitwise operators require integer operands");
    switch ((char)op) {
        case '&': return nv_int(a->i & b->i);
        case '|': return nv_int(a->i | b->i);
        case '^': return nv_int(a->i ^ b->i);
        case '<': return nv_int((i64)((uint64_t)a->i << (b->i & 63)));
        default:  return nv_int(a->i >> (b->i & 63));
    }
}

i64 nv_neg(i64 h) {
    BV *v = bv_h(h);
    if (v->tag == BV_INT) return nv_int(-v->i);
    if (v->tag == BV_FLOAT) return nv_float(-v->f);
    bv_die("cannot negate non-number");
    return nv_null();
}
i64 nv_not(i64 h) { return nv_bool(!nv_truthy(h)); }
i64 nv_bitnot(i64 h) {
    BV *v = bv_h(h);
    if (v->tag != BV_INT) bv_die("cannot apply BitNot to non-int");
    return nv_int(~v->i);
}

i64 nv_as_int(i64 h) {
    BV *v = bv_h(h);
    if (v->tag != BV_INT) { fprintf(stderr, "runtime error: expected integer, got %s\n", bv_type_name(v)); exit(1); }
    return v->i;
}
