# State Migration

When a program's data shape changes across a code/schema update, you often need
to preserve existing state — transform values that were built against the *old*
struct definition into the *new* one. Nova makes this a first-class language
construct.

## Syntax

```nova
migrate from Old to New {
    // body: an expression block that produces a `New`-shaped value
}
```

- `Old` and `New` are struct names.
- The body runs with:
  - `old` bound to the incoming value, and
  - each of the **old struct's fields** also bound directly by name (for
    convenience, so you can write `age` instead of `old.age`).
- The block's trailing expression is the produced `New` value.

Apply a migration with the `migrate(value)` builtin. It looks up the migration
whose `from` type matches the value's struct name, runs its body, and returns the
new value:

```nova
struct UserV1 { name: Str, age: Int }
struct UserV2 { name: Str, age: Int, active: Bool }

migrate from UserV1 to UserV2 {
    UserV2 { name: name, age: age + 1, active: true }
}

fn main() {
    let old = UserV1 { name: "Payam", age: 29 }
    let new = migrate(old)     // -> UserV2 { name: "Payam", age: 30, active: true }
    print(new.active)          // true
}
```

## Semantics & guarantees

- **One migration per source type.** Migrations are keyed by their `from` type;
  the matching one is selected from the runtime struct name of the argument.
- `migrate(x)` on a value with no registered migration is a runtime error
  (`no migration defined from `T``), and on a non-struct value it errors too.
- **Byte-identical across tiers.** Migrations run through the same interpreter
  path on `nova run` and `nova vm` (and the JIT/AOT tiers), so the produced value
  and any printed output are identical everywhere. See
  `tests/corpus/state_migration.nova`.

## Limits (honest)

- Migrations are applied explicitly via `migrate(value)`; there is no automatic
  persistence layer or on-disk versioning yet — you call it where you load old
  state. Chained migrations (V1→V2→V3) are done by calling `migrate` twice.
- The body is ordinary Nova code, so it can compute defaults, drop fields, rename,
  or split values however you need.
