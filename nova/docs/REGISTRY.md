# Nova package registry

Nova ships a **real, self-hostable package registry** built on `nova.hgx` (Nova's
`Cargo.toml`). It is std-only — no external crates, no vendored TLS — and the
whole loop is exercised end-to-end by `tests/registry_smoke.sh`:

> publish → `nova registry` HTTP serve → resolve (SemVer) → fetch → SHA-256
> verify → unpack → vendor → run **byte-identical** across tiers.

## Declaring dependencies — `nova.hgx`

```toml
[package]
name    = "app"
version = "0.1.0"
entry   = "main.nova"

[registry]
default = "http://127.0.0.1:7879"     # an index base (http:// or a local dir)
corp    = "/srv/nova-index"           # a named registry

[dependencies]
json     = "^1.2"                                   # SemVer requirement (caret)
http     = { version = "2.0", features = ["tls"] }  # inline table
liblocal = { path = "./liblocal" }                  # path dependency
plugin   = { git = "https://ex/plugin.git", rev = "abc123" }

[abilities]                                         # two-mode attributes (below)
audited = ["trace", "profile"]
heal    = { attr = "self_healing", args = "attempts: 3", targets = ["fetch"] }
```

### SemVer requirements

`VersionReq` supports the full cargo-style grammar: `*`, exact `=1.2.3`, caret
`^1.2.3`, tilde `~1.2.3`, ranges `>=1.0, <2.0`, wildcards `1.*` / `1.2.*`, and a
bare `1.2.3` (which means `^1.2.3`). Pre-releases order below their release
(`1.0.0-alpha < 1.0.0`).

## Commands

| command | effect |
|---|---|
| `nova add <name>@<req>` | add a `[dependencies]` entry to `nova.hgx`, then install |
| `nova add <name> --git <url> [--rev <sha>]` / `--path <p>` | add a git / path dep |
| `nova add <file.nova> [name]` | offline: vendor a local file into `nova_modules/` |
| `nova remove <name>` | drop the dep from `nova.hgx` + delete its vendored files |
| `nova install` | resolve `[dependencies]`, write `nova.lock`, fetch+verify+vendor |
| `nova update` | re-resolve ignoring the lock (pick newest satisfying versions) |
| `nova tree` | print the resolved dependency set |
| `nova publish <index-dir>` | pack this project + append it to a registry index |
| `nova registry <index-dir> [--port N]` | serve an index over HTTP |
| `nova deps` | list declared dependencies |

## The resolver + lockfile

`resolve` reads the root deps and the index, and for each **registry** dep picks
the **highest** published version that satisfies *every* accumulated requirement,
unifying shared transitive deps to a single version and reporting a hard error on
a genuine conflict. The result is written to `nova.lock` (a reproducible
`[[package]]` list with `name`/`version`/`source`/`checksum`); a subsequent
`nova install` replays the lock exactly, while `nova update` re-resolves.

## Registry index format

An index is a directory (a local path, or an `http://` base served by
`nova registry`) with one file per package:

```
<name>/index.txt          # one published version per line:
   <version> <sha256> <archive-file> [dep=req dep2=req ...]
<name>/<name>-<version>.nvpkg   # the package archive
```

`parse_index_file` reads `index.txt`; `build_index` fetches the referenced
packages transitively.

## The `.nvpkg` archive

A package is a **deterministic** flat archive so its SHA-256 is reproducible:

```
NVPKG1\n
<path>\t<len>\n<...len bytes...>      # repeated, files in sorted path order
```

`pack_dir` packs a project (skipping `.*`, `nova_modules/`, `build/`, `target/`);
`unpack_into` restores it with path-traversal protection (no absolute or `..`
paths). `nova install` verifies the downloaded archive's SHA-256 against the lock
before vendoring — a tampered download is rejected.

## Sources

* **registry** — fetched from the index base, checksum-verified, unpacked into
  `nova_modules/<name>/`.
* **path** — copied from a local directory/file into `nova_modules/`.
* **git** — shallow-cloned (optionally checked out at `rev`), `.git` stripped.

All three land in `nova_modules/`, where the import resolver finds them, so a
program just writes `use "<name>/..."`.

## SHA-256

A complete, std-only SHA-256 (`sha256_hex`) — verified against the standard
vectors (`""` → `e3b0…b855`, `"abc"` → `ba78…15ad`) — is the integrity backbone:
every downloaded archive is hashed and compared to the locked checksum.

## Two-mode abilities

The user's design: **attributes/abilities are two-mode** — they may be declared
on the code (`#[trace] fn f() {…}`) *or* project-wide in `nova.hgx`
`[abilities]`. Manifest abilities are merged onto matching functions at load
(`apply_abilities`): an empty `targets` applies to every function; an on-code
declaration always wins (never duplicated). This lets a project turn on tracing,
profiling, retries, contracts, etc. from the manifest without touching sources.

## Honest limits

* **https:// fetching** needs a TLS stack Nova deliberately does not vendor (the
  whole toolchain is external-crate-free). Use a `git`/`path` dependency, or an
  `http://` / local-directory index (which `nova registry` serves). Reported
  clearly at fetch time — never faked.
* **git/path transitive deps** are vendored directly; transitive *registry*
  resolution is fully modelled by the resolver + index.
