/* Nova AOT runtime — tagged refcounted values (int/float/bool/null/str/array).
 * Compiled together with the generated program under -O3 -flto so LLVM inlines
 * these ops into program code. Semantics mirror src/interp.rs exactly; the
 * build-time byte-diff gate enforces it. Written from scratch for Nova. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef int64_t i64;

enum { NV_INT, NV_FLOAT, NV_BOOL, NV_NULL, NV_STR, NV_ARR };

typedef struct NVStr {
    i64 rc;
    i64 nchars;      /* Nova indexes strings by Unicode char */
    i64 nbytes;
    i64 *coff;       /* byte offset of each char (nchars+1 entries) */
    char *utf8;
} NVStr;

struct NV;
typedef struct NVArr {
    i64 rc;
    i64 len, cap;
    struct NV *items;
} NVArr;

typedef struct NV {
    uint8_t tag;
    union { i64 i; double f; NVStr *s; NVArr *a; };
} NV;

static void nv_die(const char *msg) {
    fprintf(stderr, "runtime error: %s\n", msg);
    exit(1);
}
static void nv_dief(const char *fmt, i64 a, i64 b) {
    fprintf(stderr, "runtime error: ");
    fprintf(stderr, fmt, (long long)a, (long long)b);
    fprintf(stderr, "\n");
    exit(1);
}

static NV nv_int(i64 v)   { NV x; x.tag = NV_INT;   x.i = v; return x; }
static NV nv_float(double v){ NV x; x.tag = NV_FLOAT; x.f = v; return x; }
static NV nv_bool(i64 v)  { NV x; x.tag = NV_BOOL;  x.i = v != 0; return x; }
static NV nv_null(void)   { NV x; x.tag = NV_NULL;  x.i = 0; return x; }

static void nv_release(NV v);

static void nvstr_free(NVStr *s) { free(s->coff); free(s->utf8); free(s); }
static void nvarr_free(NVArr *a) {
    for (i64 i = 0; i < a->len; i++) nv_release(a->items[i]);
    free(a->items);
    free(a);
}
static NV nv_retain(NV v) {
    if (v.tag == NV_STR) v.s->rc++;
    else if (v.tag == NV_ARR) v.a->rc++;
    return v;
}
static void nv_release(NV v) {
    if (v.tag == NV_STR) { if (--v.s->rc == 0) nvstr_free(v.s); }
    else if (v.tag == NV_ARR) { if (--v.a->rc == 0) nvarr_free(v.a); }
}

/* build a string value from UTF-8 bytes, computing per-char byte offsets */
static NV nv_str_n(const char *bytes, i64 nbytes) {
    NVStr *s = malloc(sizeof(NVStr));
    s->rc = 1;
    s->nbytes = nbytes;
    s->utf8 = malloc(nbytes + 1);
    memcpy(s->utf8, bytes, nbytes);
    s->utf8[nbytes] = 0;
    i64 nchars = 0;
    for (i64 i = 0; i < nbytes; i++)
        if (((unsigned char)bytes[i] & 0xC0) != 0x80) nchars++;
    s->nchars = nchars;
    s->coff = malloc(sizeof(i64) * (nchars + 1));
    i64 c = 0;
    for (i64 i = 0; i < nbytes; i++)
        if (((unsigned char)bytes[i] & 0xC0) != 0x80) s->coff[c++] = i;
    s->coff[nchars] = nbytes;
    NV v; v.tag = NV_STR; v.s = s;
    return v;
}
static NV nv_str(const char *cstr) { return nv_str_n(cstr, (i64)strlen(cstr)); }

static NV nv_arr(i64 cap) {
    NVArr *a = malloc(sizeof(NVArr));
    a->rc = 1;
    a->len = 0;
    a->cap = cap > 4 ? cap : 4;
    a->items = malloc(sizeof(NV) * a->cap);
    NV v; v.tag = NV_ARR; v.a = a;
    return v;
}
/* takes ownership of item */
static void nv_arr_push(NV arr, NV item) {
    if (arr.tag != NV_ARR) nv_die("push expects array");
    NVArr *a = arr.a;
    if (a->len == a->cap) {
        a->cap *= 2;
        a->items = realloc(a->items, sizeof(NV) * a->cap);
    }
    a->items[a->len++] = item;
}
static NV nv_pop(NV arr) {
    if (arr.tag != NV_ARR) nv_die("pop expects array");
    NVArr *a = arr.a;
    if (a->len == 0) return nv_null();
    return a->items[--a->len]; /* ownership moves to caller */
}

static const char *nv_type_name(NV v) {
    switch (v.tag) {
        case NV_INT: return "int";
        case NV_FLOAT: return "float";
        case NV_BOOL: return "bool";
        case NV_NULL: return "null";
        case NV_STR: return "string";
        default: return "array";
    }
}

static i64 nv_len(NV v) {
    if (v.tag == NV_ARR) return v.a->len;
    if (v.tag == NV_STR) return v.s->nchars;
    fprintf(stderr, "runtime error: len expects array or string, got %s\n", nv_type_name(v));
    exit(1);
}

static i64 nv_truthy(NV v) {
    switch (v.tag) {
        case NV_BOOL: case NV_INT: return v.i != 0;
        case NV_NULL: return 0;
        default: return 1;
    }
}

static i64 nv_eq(NV a, NV b) {
    if (a.tag == NV_INT && b.tag == NV_INT) return a.i == b.i;
    if (a.tag == NV_FLOAT && b.tag == NV_FLOAT) return a.f == b.f;
    if (a.tag == NV_INT && b.tag == NV_FLOAT) return (double)a.i == b.f;
    if (a.tag == NV_FLOAT && b.tag == NV_INT) return a.f == (double)b.i;
    if (a.tag == NV_BOOL && b.tag == NV_BOOL) return a.i == b.i;
    if (a.tag == NV_NULL && b.tag == NV_NULL) return 1;
    if (a.tag == NV_STR && b.tag == NV_STR)
        return a.s->nbytes == b.s->nbytes && memcmp(a.s->utf8, b.s->utf8, a.s->nbytes) == 0;
    if (a.tag == NV_ARR && b.tag == NV_ARR) {
        if (a.a->len != b.a->len) return 0;
        for (i64 i = 0; i < a.a->len; i++)
            if (!nv_eq(a.a->items[i], b.a->items[i])) return 0;
        return 1;
    }
    return 0;
}

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

/* Rust {} on f64 = shortest string that round-trips. Try increasing precision
 * with %g (which Rust's output is a subset of, modulo e-notation styling). */
static void fmt_f64(SB *sb, double x) {
    char tmp[64];
    if (x != x) { sb_puts(sb, "NaN"); return; }
    if (x > 1.7e308 && x / 2 == x) { sb_puts(sb, "inf"); return; }
    if (x < -1.7e308 && x / 2 == x) { sb_puts(sb, "-inf"); return; }
    double frac = x - (double)(i64)x;
    if (frac == 0.0 && x < 9.2e18 && x > -9.2e18) {
        /* integral & finite: Rust prints {:.1} => "3.0" (interp special-case) */
        snprintf(tmp, sizeof tmp, "%.1f", x);
        sb_puts(sb, tmp);
        return;
    }
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(tmp, sizeof tmp, "%.*g", prec, x);
        if (strtod(tmp, NULL) == x) break;
    }
    /* normalize C e-notation (1e+05 / 1.5e-07) to Rust style (1e5 / 1.5e-7) */
    char out[64]; i64 o = 0;
    for (char *p = tmp; *p; p++) {
        if (*p == 'e') {
            out[o++] = 'e';
            p++;
            if (*p == '+') p++;
            else if (*p == '-') { out[o++] = '-'; p++; }
            while (*p == '0' && *(p + 1)) p++;
            while (*p) out[o++] = *p++;
            break;
        }
        out[o++] = *p;
    }
    out[o] = 0;
    sb_puts(sb, out);
}

static void nv_fmt(SB *sb, NV v, int quote_strings) {
    char tmp[32];
    switch (v.tag) {
        case NV_INT:
            snprintf(tmp, sizeof tmp, "%lld", (long long)v.i);
            sb_puts(sb, tmp);
            break;
        case NV_FLOAT: fmt_f64(sb, v.f); break;
        case NV_BOOL: sb_puts(sb, v.i ? "true" : "false"); break;
        case NV_NULL: sb_puts(sb, "null"); break;
        case NV_STR:
            if (quote_strings) { sb_puts(sb, "\""); sb_put(sb, v.s->utf8, v.s->nbytes); sb_puts(sb, "\""); }
            else sb_put(sb, v.s->utf8, v.s->nbytes);
            break;
        case NV_ARR:
            sb_puts(sb, "[");
            for (i64 i = 0; i < v.a->len; i++) {
                if (i) sb_puts(sb, ", ");
                nv_fmt(sb, v.a->items[i], 1); /* strings inside arrays get quotes */
            }
            sb_puts(sb, "]");
            break;
    }
}

/* borrows v */
static void nv_print(NV v) {
    SB sb; sb_init(&sb);
    nv_fmt(&sb, v, 0);
    fwrite(sb.buf, 1, sb.len, stdout);
    fputc('\n', stdout);
    free(sb.buf);
}

/* owned string of v's display form (for f-strings / string +) */
static NV nv_tostr(NV v) {
    SB sb; sb_init(&sb);
    nv_fmt(&sb, v, 0);
    NV r = nv_str_n(sb.buf, sb.len);
    free(sb.buf);
    return r;
}

/* ---- operators (mirror eval_binop; args borrowed, result owned) ---- */

static double nv_as_f(NV v) {
    if (v.tag == NV_INT) return (double)v.i;
    if (v.tag == NV_FLOAT) return v.f;
    nv_die("expected a number");
    return 0;
}
static int nv_is_num(NV v) { return v.tag == NV_INT || v.tag == NV_FLOAT; }

static NV nv_concat2(NV a, NV b) { /* string + value / value + string */
    NV sa = a.tag == NV_STR ? nv_retain(a) : nv_tostr(a);
    NV sb2 = b.tag == NV_STR ? nv_retain(b) : nv_tostr(b);
    i64 nbytes = sa.s->nbytes + sb2.s->nbytes;
    char *buf = malloc(nbytes);
    memcpy(buf, sa.s->utf8, sa.s->nbytes);
    memcpy(buf + sa.s->nbytes, sb2.s->utf8, sb2.s->nbytes);
    NV r = nv_str_n(buf, nbytes);
    free(buf);
    nv_release(sa); nv_release(sb2);
    return r;
}

static NV nv_add(NV a, NV b) {
    if (a.tag == NV_INT && b.tag == NV_INT) return nv_int(a.i + b.i); /* overflow -> gate */
    if (a.tag == NV_STR || b.tag == NV_STR) return nv_concat2(a, b);
    if (nv_is_num(a) && nv_is_num(b)) return nv_float(nv_as_f(a) + nv_as_f(b));
    fprintf(stderr, "runtime error: cannot apply Add to %s and %s\n", nv_type_name(a), nv_type_name(b));
    exit(1);
}
static NV nv_sub(NV a, NV b) {
    if (a.tag == NV_INT && b.tag == NV_INT) return nv_int(a.i - b.i);
    if (nv_is_num(a) && nv_is_num(b)) return nv_float(nv_as_f(a) - nv_as_f(b));
    fprintf(stderr, "runtime error: cannot apply Sub to %s and %s\n", nv_type_name(a), nv_type_name(b));
    exit(1);
}
static NV nv_mul(NV a, NV b) {
    if (a.tag == NV_INT && b.tag == NV_INT) return nv_int(a.i * b.i);
    if (nv_is_num(a) && nv_is_num(b)) return nv_float(nv_as_f(a) * nv_as_f(b));
    fprintf(stderr, "runtime error: cannot apply Mul to %s and %s\n", nv_type_name(a), nv_type_name(b));
    exit(1);
}
static NV nv_div(NV a, NV b) {
    if (a.tag == NV_INT && b.tag == NV_INT) {
        if (b.i == 0) nv_die("division by zero");
        return nv_int(a.i / b.i);
    }
    if (nv_is_num(a) && nv_is_num(b)) return nv_float(nv_as_f(a) / nv_as_f(b));
    fprintf(stderr, "runtime error: cannot apply Div to %s and %s\n", nv_type_name(a), nv_type_name(b));
    exit(1);
}
static NV nv_rem(NV a, NV b) {
    if (a.tag == NV_INT && b.tag == NV_INT) {
        if (b.i == 0) nv_die("modulo by zero");
        return nv_int(a.i % b.i);
    }
    if (nv_is_num(a) && nv_is_num(b)) {
        double x = nv_as_f(a), y = nv_as_f(b);
        double r = x - (double)((i64)(x / y)) * y; /* fmod without libm */
        (void)r;
        nv_die("float % is not AOT-supported"); /* interp uses Rust %, gate anyway */
    }
    fprintf(stderr, "runtime error: cannot apply Rem to %s and %s\n", nv_type_name(a), nv_type_name(b));
    exit(1);
}
/* string ordering is byte-lexicographic, matching Rust's str ordering */
static int nv_strcmp(NV a, NV b) {
    i64 n = a.s->nbytes < b.s->nbytes ? a.s->nbytes : b.s->nbytes;
    int c = memcmp(a.s->utf8, b.s->utf8, n);
    if (c) return c;
    return (a.s->nbytes > b.s->nbytes) - (a.s->nbytes < b.s->nbytes);
}
static NV nv_cmp_lt(NV a, NV b) {
    if (a.tag == NV_STR && b.tag == NV_STR) return nv_bool(nv_strcmp(a, b) < 0);
    return nv_bool(nv_as_f(a) < nv_as_f(b));
}
static NV nv_cmp_le(NV a, NV b) {
    if (a.tag == NV_STR && b.tag == NV_STR) return nv_bool(nv_strcmp(a, b) <= 0);
    return nv_bool(nv_as_f(a) <= nv_as_f(b));
}
static NV nv_cmp_gt(NV a, NV b) {
    if (a.tag == NV_STR && b.tag == NV_STR) return nv_bool(nv_strcmp(a, b) > 0);
    return nv_bool(nv_as_f(a) > nv_as_f(b));
}
static NV nv_cmp_ge(NV a, NV b) {
    if (a.tag == NV_STR && b.tag == NV_STR) return nv_bool(nv_strcmp(a, b) >= 0);
    return nv_bool(nv_as_f(a) >= nv_as_f(b));
}

static NV nv_bit(NV a, NV b, char op) {
    if (a.tag != NV_INT || b.tag != NV_INT) nv_die("bitwise operators require integer operands");
    switch (op) {
        case '&': return nv_int(a.i & b.i);
        case '|': return nv_int(a.i | b.i);
        case '^': return nv_int(a.i ^ b.i);
        case '<': return nv_int((i64)((uint64_t)a.i << (b.i & 63)));
        default:  return nv_int(a.i >> (b.i & 63));
    }
}

static NV nv_neg(NV v) {
    if (v.tag == NV_INT) return nv_int(-v.i);
    if (v.tag == NV_FLOAT) return nv_float(-v.f);
    nv_die("cannot negate non-number");
    return nv_null();
}
static NV nv_not(NV v) { return nv_bool(!nv_truthy(v)); }
static NV nv_bitnot(NV v) {
    if (v.tag != NV_INT) nv_die("cannot apply BitNot to non-int");
    return nv_int(~v.i);
}

/* index read: mirror index_get (array int / str char); result owned */
static NV nv_index(NV base, NV idx) {
    if (base.tag == NV_ARR) {
        if (idx.tag != NV_INT) nv_die("expected integer, got non-int index");
        i64 i = idx.i;
        if (i < 0 || i >= base.a->len) nv_dief("index %lld out of bounds (len %lld)", i, base.a->len);
        return nv_retain(base.a->items[i]);
    }
    if (base.tag == NV_STR) {
        if (idx.tag != NV_INT) nv_die("expected integer, got non-int index");
        i64 i = idx.i;
        if (i < 0 || i >= base.s->nchars) nv_dief("string index %lld out of bounds (len %lld)", i, base.s->nchars);
        i64 lo = base.s->coff[i], hi = base.s->coff[i + 1];
        return nv_str_n(base.s->utf8 + lo, hi - lo);
    }
    fprintf(stderr, "runtime error: cannot index into %s\n", nv_type_name(base));
    exit(1);
}

/* base[idx] = v (array only in the AOT tier); consumes v */
static void nv_index_set(NV base, NV idx, NV v) {
    if (base.tag != NV_ARR) { fprintf(stderr, "runtime error: cannot index-assign into %s\n", nv_type_name(base)); exit(1); }
    if (idx.tag != NV_INT) nv_die("expected integer, got non-int index");
    i64 i = idx.i;
    if (i < 0 || i >= base.a->len) nv_dief("index %lld out of bounds (len %lld)", i, base.a->len);
    nv_release(base.a->items[i]);
    base.a->items[i] = v;
}

/* slice with Nova clamping (mirror do_slice); has_lo/has_hi flag open ends */
static NV nv_slice(NV base, i64 lo, int has_lo, i64 hi, int has_hi, int inclusive) {
    i64 len;
    if (base.tag == NV_ARR) len = base.a->len;
    else if (base.tag == NV_STR) len = base.s->nchars;
    else { fprintf(stderr, "runtime error: cannot slice %s\n", nv_type_name(base)); exit(1); }
    i64 start = 0, end = len;
    if (has_lo) { start = lo < 0 ? (len + lo > 0 ? len + lo : 0) : (lo < len ? lo : len); }
    if (has_hi) {
        i64 x = hi < 0 ? len + hi : hi;
        if (inclusive) x += 1;
        end = x < 0 ? 0 : (x > len ? len : x);
    }
    if (end < start) end = start;
    if (base.tag == NV_ARR) {
        NV out = nv_arr(end - start + 1);
        for (i64 i = start; i < end; i++) nv_arr_push(out, nv_retain(base.a->items[i]));
        return out;
    }
    i64 b0 = base.s->coff[start], b1 = base.s->coff[end];
    return nv_str_n(base.s->utf8 + b0, b1 - b0);
}

/* integer range lo..hi / lo..=hi as an array (mirror build_range) */
static NV nv_range(i64 lo, i64 hi, int inclusive) {
    i64 last = inclusive ? hi : hi - 1;
    NV out = nv_arr(last >= lo ? last - lo + 1 : 1);
    for (i64 i = lo; i <= last; i++) nv_arr_push(out, nv_int(i));
    return out;
}

static i64 nv_as_int(NV v) {
    if (v.tag != NV_INT) { fprintf(stderr, "runtime error: expected integer, got %s\n", nv_type_name(v)); exit(1); }
    return v.i;
}
static double nv_as_float(NV v) {
    if (v.tag != NV_FLOAT) { fprintf(stderr, "runtime error: expected float, got %s\n", nv_type_name(v)); exit(1); }
    return v.f;
}
