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

enum { BV_INT, BV_FLOAT, BV_BOOL, BV_NULL, BV_STR, BV_ARR, BV_STRUCT, BV_MAP, BV_ENUM };

typedef struct BStr {
    i64 nchars;      /* Nova indexes strings by Unicode char */
    i64 nbytes;
    i64 *coff;       /* byte offset of each char (nchars+1 entries) */
    char *utf8;
} BStr;

/* dynamic array of value handles (arena-allocated; grows, never freed) */
typedef struct BArr {
    i64 len, cap;
    i64 *items;      /* each element is a value handle (i64) */
} BArr;

/* struct instance: type name + field-name/value handles, in SORTED field order
 * (matching the interpreter's Display, which sorts keys). Field access is by
 * name (linear scan over the few fields) so no compile-time slot map is needed. */
typedef struct BStruct {
    i64 type_name;   /* handle to a BV_STR */
    i64 nfields;
    i64 *names;      /* handles to BV_STR field names (sorted) */
    i64 *values;     /* value handles, parallel to names */
} BStruct;

/* insertion-ordered key/value pairs with linear lookup — mirrors the
 * interpreter's Value::Map (a Vec<(Value, Value)>), NOT a hash map. */
typedef struct BMap {
    i64 len, cap;
    i64 *keys;       /* handles */
    i64 *vals;       /* handles */
} BMap;

/* enum instance: variant name + tuple payload handles (mirrors EnumVal) */
typedef struct BEnum {
    i64 variant;     /* handle to BV_STR */
    i64 ndata;
    i64 *data;       /* payload value handles */
} BEnum;

typedef struct BV {
    uint8_t tag;
    union { i64 i; double f; BStr *s; BArr *a; BStruct *st; BMap *m; BEnum *en; };
} BV;

/* handle <-> pointer. Allocate-and-leak: a boxed value lives for the whole run. */
static BV *bv_h(i64 h) { return (BV *)(intptr_t)h; }
static i64 bv_mk(BV v) { BV *p = malloc(sizeof(BV)); *p = v; return (i64)(intptr_t)p; }

static void bv_die(const char *msg) {
    fprintf(stderr, "runtime error: %s\n", msg);
    exit(1);
}
static i64 bv_eq(BV *a, BV *b); /* forward: map key comparison (values_eq) */
static void bv_dief(const char *fmt, i64 a, i64 b) {
    fprintf(stderr, "runtime error: ");
    fprintf(stderr, fmt, (long long)a, (long long)b);
    fprintf(stderr, "\n");
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
    s->coff = malloc(sizeof(i64) * (nchars + 1));
    i64 c = 0;
    for (i64 i = 0; i < nbytes; i++)
        if (((unsigned char)bytes[i] & 0xC0) != 0x80) s->coff[c++] = i;
    s->coff[nchars] = nbytes;
    return s;
}
i64 nv_str(i64 ptr, i64 len) {
    BV x; x.tag = BV_STR; x.s = bstr_new((const char *)(intptr_t)ptr, len);
    return bv_mk(x);
}

/* ---- arrays (arena; handle elements) ---- */
i64 nv_arr(i64 cap) {
    BArr *a = malloc(sizeof(BArr));
    a->len = 0;
    a->cap = cap > 4 ? cap : 4;
    a->items = malloc(sizeof(i64) * a->cap);
    BV x; x.tag = BV_ARR; x.a = a;
    return bv_mk(x);
}
void nv_arr_push(i64 arrh, i64 itemh) {
    BV *v = bv_h(arrh);
    if (v->tag != BV_ARR) bv_die("push expects array");
    BArr *a = v->a;
    if (a->len == a->cap) { a->cap *= 2; a->items = realloc(a->items, sizeof(i64) * a->cap); }
    a->items[a->len++] = itemh;
}
i64 nv_pop(i64 arrh) {
    BV *v = bv_h(arrh);
    if (v->tag != BV_ARR) bv_die("pop expects array");
    BArr *a = v->a;
    if (a->len == 0) return nv_null();
    return a->items[--a->len];
}

/* ---- maps (insertion-ordered pairs; linear lookup by key, values_eq) ---- */
i64 nv_map(i64 cap) {
    BMap *m = malloc(sizeof(BMap));
    m->len = 0;
    m->cap = cap > 4 ? cap : 4;
    m->keys = malloc(sizeof(i64) * m->cap);
    m->vals = malloc(sizeof(i64) * m->cap);
    BV x; x.tag = BV_MAP; x.m = m;
    return bv_mk(x);
}
/* insert or overwrite key -> value (mirrors interp map insert / index_set) */
void nv_map_put(i64 mh, i64 kh, i64 vh) {
    BV *v = bv_h(mh);
    if (v->tag != BV_MAP) bv_die("expected map");
    BMap *m = v->m;
    for (i64 i = 0; i < m->len; i++)
        if (bv_eq(bv_h(m->keys[i]), bv_h(kh))) { m->vals[i] = vh; return; }
    if (m->len == m->cap) {
        m->cap *= 2;
        m->keys = realloc(m->keys, sizeof(i64) * m->cap);
        m->vals = realloc(m->vals, sizeof(i64) * m->cap);
    }
    m->keys[m->len] = kh;
    m->vals[m->len] = vh;
    m->len++;
}
/* key lookup -> value, or null if absent (mirrors index_get for maps) */
i64 nv_map_get(i64 mh, i64 kh) {
    BMap *m = bv_h(mh)->m;
    for (i64 i = 0; i < m->len; i++)
        if (bv_eq(bv_h(m->keys[i]), bv_h(kh))) return m->vals[i];
    return nv_null();
}
/* the map's keys as an array, in insertion order (for foreach / keys()) */
i64 nv_keys(i64 mh) {
    BV *v = bv_h(mh);
    if (v->tag != BV_MAP) bv_die("keys expects map");
    BMap *m = v->m;
    i64 out = nv_arr(m->len);
    for (i64 i = 0; i < m->len; i++) nv_arr_push(out, m->keys[i]);
    return out;
}

/* ---- structs (handle name/value slots; field access by name) ---- */
i64 nv_struct(i64 type_name_h, i64 nfields) {
    BStruct *s = malloc(sizeof(BStruct));
    s->type_name = type_name_h;
    s->nfields = nfields;
    s->names = malloc(sizeof(i64) * (nfields > 0 ? nfields : 1));
    s->values = malloc(sizeof(i64) * (nfields > 0 ? nfields : 1));
    BV x; x.tag = BV_STRUCT; x.st = s;
    return bv_mk(x);
}
/* set slot `i` during construction: field name + value */
void nv_struct_set(i64 sh, i64 i, i64 name_h, i64 val_h) {
    BStruct *s = bv_h(sh)->st;
    s->names[i] = name_h;
    s->values[i] = val_h;
}
/* find the slot whose name equals `name_h` (both BV_STR); -1 if absent */
static i64 bstruct_slot(BStruct *s, i64 name_h) {
    BStr *want = bv_h(name_h)->s;
    for (i64 i = 0; i < s->nfields; i++) {
        BStr *have = bv_h(s->names[i])->s;
        if (have->nbytes == want->nbytes && memcmp(have->utf8, want->utf8, want->nbytes) == 0)
            return i;
    }
    return -1;
}
i64 nv_field(i64 sh, i64 name_h) {
    BV *v = bv_h(sh);
    if (v->tag != BV_STRUCT) { fprintf(stderr, "runtime error: cannot access field of non-struct\n"); exit(1); }
    i64 i = bstruct_slot(v->st, name_h);
    if (i < 0) { fprintf(stderr, "runtime error: no such field: %s\n", bv_h(name_h)->s->utf8); exit(1); }
    return v->st->values[i];
}
void nv_field_set(i64 sh, i64 name_h, i64 val_h) {
    BV *v = bv_h(sh);
    if (v->tag != BV_STRUCT) { fprintf(stderr, "runtime error: cannot assign field of non-struct\n"); exit(1); }
    i64 i = bstruct_slot(v->st, name_h);
    if (i < 0) { fprintf(stderr, "runtime error: no such field: %s\n", bv_h(name_h)->s->utf8); exit(1); }
    v->st->values[i] = val_h;
}

/* ---- enums (tagged value + tuple payload) ---- */
i64 nv_enum(i64 variant_h, i64 ndata) {
    BEnum *e = malloc(sizeof(BEnum));
    e->variant = variant_h;
    e->ndata = ndata;
    e->data = malloc(sizeof(i64) * (ndata > 0 ? ndata : 1));
    BV x; x.tag = BV_ENUM; x.en = e;
    return bv_mk(x);
}
void nv_enum_set(i64 eh, i64 i, i64 val_h) { bv_h(eh)->en->data[i] = val_h; }
i64 nv_enum_data(i64 eh, i64 i) { return bv_h(eh)->en->data[i]; }
i64 nv_enum_arity(i64 eh) { return bv_h(eh)->en->ndata; }
/* is `eh` an enum whose variant name == the string `name_h`? */
i64 nv_enum_is(i64 eh, i64 name_h) {
    BV *v = bv_h(eh);
    if (v->tag != BV_ENUM) return 0;
    return bv_eq(bv_h(v->en->variant), bv_h(name_h));
}

/* ---- pattern-match introspection (used by BoxGen's compiled match) ---- */
i64 nv_tag(i64 h) { return bv_h(h)->tag; }
i64 nv_is_arr(i64 h) { return bv_h(h)->tag == BV_ARR; } /* tuple/slice patterns match arrays only */
/* int value in lo..hi (or lo..=hi); 0 if not an int or out of range */
i64 nv_in_range(i64 h, i64 lo, i64 hi, i64 inclusive) {
    BV *v = bv_h(h);
    if (v->tag != BV_INT) return 0;
    if (inclusive) return v->i >= lo && v->i <= hi;
    return v->i >= lo && v->i < hi;
}
/* is `h` a struct of type `name_h` (empty name matches any struct)? */
i64 nv_struct_is(i64 h, i64 name_h) {
    BV *v = bv_h(h);
    if (v->tag != BV_STRUCT) return 0;
    if (bv_h(name_h)->s->nbytes == 0) return 1;
    return bv_eq(bv_h(v->st->type_name), bv_h(name_h));
}
/* does struct `h` have a field named `name_h`? (for struct patterns) */
i64 nv_has_field(i64 h, i64 name_h) {
    BV *v = bv_h(h);
    if (v->tag != BV_STRUCT) return 0;
    return bstruct_slot(v->st, name_h) >= 0;
}
void nv_no_match(void) {
    fprintf(stderr, "runtime error: no match arm matched (non-exhaustive match)\n");
    exit(1);
}

static const char *bv_type_name(BV *v) {
    switch (v->tag) {
        case BV_INT: return "int";
        case BV_FLOAT: return "float";
        case BV_BOOL: return "bool";
        case BV_NULL: return "null";
        case BV_STR: return "string";
        case BV_ARR: return "array";
        case BV_STRUCT: return "struct";
        case BV_MAP: return "map";
        default: return "enum";
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
    if (v->tag == BV_ARR) return nv_int(v->a->len);
    if (v->tag == BV_STR) return nv_int(v->s->nchars);
    /* like interp.rs, len() does NOT accept a map — it errors */
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
    if (a->tag == BV_ARR && b->tag == BV_ARR) {
        if (a->a->len != b->a->len) return 0;
        for (i64 i = 0; i < a->a->len; i++)
            if (!bv_eq(bv_h(a->a->items[i]), bv_h(b->a->items[i]))) return 0;
        return 1;
    }
    if (a->tag == BV_STRUCT && b->tag == BV_STRUCT) {
        BStruct *x = a->st, *y = b->st;
        if (x->nfields != y->nfields) return 0;
        if (!bv_eq(bv_h(x->type_name), bv_h(y->type_name))) return 0;
        for (i64 i = 0; i < x->nfields; i++) {
            if (!bv_eq(bv_h(x->names[i]), bv_h(y->names[i]))) return 0;
            if (!bv_eq(bv_h(x->values[i]), bv_h(y->values[i]))) return 0;
        }
        return 1;
    }
    if (a->tag == BV_MAP && b->tag == BV_MAP) {
        BMap *x = a->m, *y = b->m;
        if (x->len != y->len) return 0;
        for (i64 i = 0; i < x->len; i++) {   /* order-insensitive: each a-key in b */
            i64 j = 0;
            for (; j < y->len; j++)
                if (bv_eq(bv_h(x->keys[i]), bv_h(y->keys[j]))
                    && bv_eq(bv_h(x->vals[i]), bv_h(y->vals[j]))) break;
            if (j == y->len) return 0;
        }
        return 1;
    }
    if (a->tag == BV_ENUM && b->tag == BV_ENUM) {
        BEnum *x = a->en, *y = b->en;
        if (!bv_eq(bv_h(x->variant), bv_h(y->variant))) return 0;
        if (x->ndata != y->ndata) return 0;
        for (i64 i = 0; i < x->ndata; i++)
            if (!bv_eq(bv_h(x->data[i]), bv_h(y->data[i]))) return 0;
        return 1;
    }
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

static void bv_fmt(SB *sb, BV *v, int quote_strings) {
    char tmp[32];
    switch (v->tag) {
        case BV_INT:
            snprintf(tmp, sizeof tmp, "%lld", (long long)v->i);
            sb_puts(sb, tmp);
            break;
        case BV_FLOAT: fmt_f64(sb, v->f); break;
        case BV_BOOL: sb_puts(sb, v->i ? "true" : "false"); break;
        case BV_NULL: sb_puts(sb, "null"); break;
        case BV_STR:
            if (quote_strings) { sb_puts(sb, "\""); sb_put(sb, v->s->utf8, v->s->nbytes); sb_puts(sb, "\""); }
            else sb_put(sb, v->s->utf8, v->s->nbytes);
            break;
        case BV_ARR:
            sb_puts(sb, "[");
            for (i64 i = 0; i < v->a->len; i++) {
                if (i) sb_puts(sb, ", ");
                bv_fmt(sb, bv_h(v->a->items[i]), 1); /* strings inside arrays get quotes */
            }
            sb_puts(sb, "]");
            break;
        case BV_STRUCT: {
            /* `TypeName { name: value, ... }` — fields already in sorted order,
             * inner strings quoted, byte-identical to interp.rs Display. */
            BStruct *s = v->st;
            sb_put(sb, bv_h(s->type_name)->s->utf8, bv_h(s->type_name)->s->nbytes);
            sb_puts(sb, " { ");
            for (i64 i = 0; i < s->nfields; i++) {
                if (i) sb_puts(sb, ", ");
                sb_put(sb, bv_h(s->names[i])->s->utf8, bv_h(s->names[i])->s->nbytes);
                sb_puts(sb, ": ");
                bv_fmt(sb, bv_h(s->values[i]), 1);
            }
            sb_puts(sb, " }");
            break;
        }
        case BV_MAP: {
            /* `{k: v, ...}` in insertion order, keys and string values quoted,
             * byte-identical to interp.rs Display. */
            BMap *m = v->m;
            sb_puts(sb, "{");
            for (i64 i = 0; i < m->len; i++) {
                if (i) sb_puts(sb, ", ");
                bv_fmt(sb, bv_h(m->keys[i]), 1);
                sb_puts(sb, ": ");
                bv_fmt(sb, bv_h(m->vals[i]), 1);
            }
            sb_puts(sb, "}");
            break;
        }
        case BV_ENUM: {
            /* unit variant -> `Name`; with payload -> `Name(d0, d1)` (data quoted
             * like array elements), byte-identical to interp.rs Display. */
            BEnum *e = v->en;
            sb_put(sb, bv_h(e->variant)->s->utf8, bv_h(e->variant)->s->nbytes);
            if (e->ndata > 0) {
                sb_puts(sb, "(");
                for (i64 i = 0; i < e->ndata; i++) {
                    if (i) sb_puts(sb, ", ");
                    bv_fmt(sb, bv_h(e->data[i]), 1);
                }
                sb_puts(sb, ")");
            }
            break;
        }
    }
}

void nv_print(i64 h) {
    SB sb; sb_init(&sb);
    bv_fmt(&sb, bv_h(h), 0);
    fwrite(sb.buf, 1, sb.len, stdout);
    fputc('\n', stdout);
    free(sb.buf);
}

/* owned string of h's display form (for f-strings / string +) */
i64 nv_tostr(i64 h) {
    BV *v = bv_h(h);
    if (v->tag == BV_STR) return h; /* already a string: same handle */
    SB sb; sb_init(&sb);
    bv_fmt(&sb, v, 0);
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

/* index read: mirror index_get (map key / array int / str char); handle result */
i64 nv_index(i64 baseh, i64 idxh) {
    BV *base = bv_h(baseh), *idx = bv_h(idxh);
    if (base->tag == BV_MAP) return nv_map_get(baseh, idxh); /* missing key -> null */
    if (base->tag == BV_ARR) {
        if (idx->tag != BV_INT) bv_die("expected integer, got non-int index");
        i64 i = idx->i;
        if (i < 0 || i >= base->a->len) bv_dief("index %lld out of bounds (len %lld)", i, base->a->len);
        return base->a->items[i]; /* shared handle, like the interp's Rc element */
    }
    if (base->tag == BV_STR) {
        if (idx->tag != BV_INT) bv_die("expected integer, got non-int index");
        i64 i = idx->i;
        if (i < 0 || i >= base->s->nchars) bv_dief("string index %lld out of bounds (len %lld)", i, base->s->nchars);
        i64 lo = base->s->coff[i], hi = base->s->coff[i + 1];
        return nv_str((i64)(intptr_t)(base->s->utf8 + lo), hi - lo);
    }
    fprintf(stderr, "runtime error: cannot index into %s\n", bv_type_name(base));
    exit(1);
}

/* base[idx] = v (array positional / map key insert-or-update) */
void nv_index_set(i64 baseh, i64 idxh, i64 vh) {
    BV *base = bv_h(baseh), *idx = bv_h(idxh);
    if (base->tag == BV_MAP) { nv_map_put(baseh, idxh, vh); return; }
    if (base->tag != BV_ARR) { fprintf(stderr, "runtime error: cannot index-assign into %s\n", bv_type_name(base)); exit(1); }
    if (idx->tag != BV_INT) bv_die("expected integer, got non-int index");
    i64 i = idx->i;
    if (i < 0 || i >= base->a->len) bv_dief("index %lld out of bounds (len %lld)", i, base->a->len);
    base->a->items[i] = vh;
}

/* slice with Nova clamping (mirror nova_rt.c do_slice); has_lo/has_hi flag open ends */
i64 nv_slice(i64 baseh, i64 lo, i64 has_lo, i64 hi, i64 has_hi, i64 inclusive) {
    BV *base = bv_h(baseh);
    i64 len;
    if (base->tag == BV_ARR) len = base->a->len;
    else if (base->tag == BV_STR) len = base->s->nchars;
    else { fprintf(stderr, "runtime error: cannot slice %s\n", bv_type_name(base)); exit(1); }
    i64 start = 0, end = len;
    if (has_lo) { start = lo < 0 ? (len + lo > 0 ? len + lo : 0) : (lo < len ? lo : len); }
    if (has_hi) {
        i64 x = hi < 0 ? len + hi : hi;
        if (inclusive) x += 1;
        end = x < 0 ? 0 : (x > len ? len : x);
    }
    if (end < start) end = start;
    if (base->tag == BV_ARR) {
        i64 out = nv_arr(end - start + 1);
        for (i64 i = start; i < end; i++) nv_arr_push(out, base->a->items[i]);
        return out;
    }
    i64 b0 = base->s->coff[start], b1 = base->s->coff[end];
    return nv_str((i64)(intptr_t)(base->s->utf8 + b0), b1 - b0);
}

/* integer range lo..hi / lo..=hi as an array (mirror build_range) */
i64 nv_range(i64 lo, i64 hi, i64 inclusive) {
    i64 last = inclusive ? hi : hi - 1;
    i64 out = nv_arr(last >= lo ? last - lo + 1 : 1);
    for (i64 i = lo; i <= last; i++) nv_arr_push(out, nv_int(i));
    return out;
}

/* iteration snapshot for foreach: a shallow copy of an array (so body mutation
 * doesn't affect the loop), or the value itself for strings/ranges. */
i64 nv_iter(i64 h) {
    BV *v = bv_h(h);
    if (v->tag == BV_ARR) return nv_slice(h, 0, 0, 0, 0, 0);
    if (v->tag == BV_MAP) return nv_keys(h); /* foreach over a map iterates its keys */
    return h;
}

i64 nv_as_int(i64 h) {
    BV *v = bv_h(h);
    if (v->tag != BV_INT) { fprintf(stderr, "runtime error: expected integer, got %s\n", bv_type_name(v)); exit(1); }
    return v->i;
}
