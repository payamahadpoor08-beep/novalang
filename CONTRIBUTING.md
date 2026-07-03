# Contributing to Nova

## The one rule

**The interpreter is the oracle.** Every execution tier (VM, JIT, embed, AOT)
must produce byte-identical output to `nova run` on every program. A change
that makes a tier faster but different is a bug, full stop. The corpus and the
AOT byte-diff gate enforce this automatically — your job is to keep them green
and to extend them when you add behavior.

## Building

```bash
cd nova
cargo build --release        # must produce ZERO warnings
```

## The verification matrix (run before every PR)

```bash
cargo test --release                     # unit + differential tests
bash tests/run_corpus.sh                 # 4 VM modes + AOT tier census
for m in list sort mathx strx ds func json setx fmtx datex; do
  ./target/release/nova test std/$m.nova # stdlib test blocks
done
```

## Adding a feature

1. Implement it in the **interpreter first** (`src/interp.rs`) — that defines
   the semantics.
2. Extend the VM (`src/bytecode.rs`) natively if possible, or let the
   delegation machinery (`EvalAst`) carry it — coverage must never regress.
3. If it is JIT-eligible (pure scalar/array code), extend the eligibility
   analysis and codegen in `src/jit.rs`; anything that can leave the eligible
   world must **deopt**, never diverge.
4. **Add a corpus program** under `tests/corpus/` that exercises the feature
   with printed output — `run_corpus.sh` then locks all tiers together
   forever. Add Rust `#[test]`s using the `same(...)`/`same_jit(...)` helpers.
5. Run the matrix above. All green, zero warnings, or it does not merge.

## Code style

- Match the file's existing style. Comments state invariants and constraints,
  not narration.
- No stub/toy code; if a tier cannot support something, it must fall back
  loudly (documented message) rather than approximate.

## Standard library

`std/*.nova` modules are written in Nova with `test "..." { }` blocks; run
them with `nova test std/<module>.nova`. New modules need tests for every
exported function, including edge cases.
