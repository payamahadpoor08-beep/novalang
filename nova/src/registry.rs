// Nova's package registry client + resolver — the engine behind `nova add`,
// `nova install`, `nova update`, `nova tree`, `nova publish`, and `nova registry`.
//
// Design (real, self-hostable — no external crates):
//   * `nova.hgx` `[dependencies]` declare deps by semver requirement, path, or git.
//   * A **registry index** is a directory (a git repo, a served folder, or a
//     local dir): `<name>/index.txt` lists published versions, one per line:
//         <version> <sha256-of-tarball> <url> [dep=req, dep2=req, ...]
//     and the package source lives at `<url>` (a `.nova` file or a tar of files).
//   * The **resolver** reads the root deps + the index, picks the highest version
//     satisfying every requirement, unifies shared deps, reports conflicts, and
//     writes a reproducible `nova.lock`.
//   * `install` vendors each resolved package into `nova_modules/<name>/` (or
//     `nova_modules/<name>.nova`) and verifies its sha256 against the lock.
//
// This module is pure/std-only; `main.rs` wires the CLI + HTTP fetch/serve.

use std::collections::BTreeMap;
use std::fmt;

// ---- semantic versions ------------------------------------------------------

#[derive(Clone, PartialEq, Eq)]
pub struct SemVer { pub major: u64, pub minor: u64, pub patch: u64, pub pre: String }

impl SemVer {
    pub fn parse(s: &str) -> Result<SemVer, String> {
        let s = s.trim();
        let (core, pre) = match s.split_once('-') { Some((c, p)) => (c, p.to_string()), None => (s, String::new()) };
        let mut it = core.split('.');
        let g = |o: Option<&str>| -> Result<u64, String> {
            o.unwrap_or("0").parse().map_err(|_| format!("bad version `{}`", s))
        };
        let major = g(it.next())?; let minor = g(it.next())?; let patch = g(it.next())?;
        Ok(SemVer { major, minor, patch, pre })
    }
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if !self.pre.is_empty() { write!(f, "-{}", self.pre)?; }
        Ok(())
    }
}

impl PartialOrd for SemVer { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl Ord for SemVer {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        (self.major, self.minor, self.patch).cmp(&(o.major, o.minor, o.patch)).then_with(|| {
            // a pre-release is lower than the same core release
            match (self.pre.is_empty(), o.pre.is_empty()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => self.pre.cmp(&o.pre),
            }
        })
    }
}

// ---- version requirements ---------------------------------------------------
// Supports: `*`, `1.2.3` / `=1.2.3` (exact), `^1.2.3` (caret), `~1.2.3` (tilde),
// `>=`, `>`, `<=`, `<`, comma-separated conjunctions, and `1.2.*` wildcards.

#[derive(Clone)]
pub struct VersionReq { comparators: Vec<Comp> }

#[derive(Clone)]
enum Comp { Any, Exact(SemVer), Ge(SemVer), Gt(SemVer), Le(SemVer), Lt(SemVer), Caret(SemVer), Tilde(SemVer) }

impl VersionReq {
    pub fn parse(s: &str) -> Result<VersionReq, String> {
        let s = s.trim();
        if s.is_empty() || s == "*" { return Ok(VersionReq { comparators: vec![Comp::Any] }); }
        let mut comparators = Vec::new();
        for part in s.split(',') {
            let p = part.trim();
            let c = if let Some(v) = p.strip_prefix("^") { Comp::Caret(SemVer::parse(v)?) }
                else if let Some(v) = p.strip_prefix("~") { Comp::Tilde(SemVer::parse(v)?) }
                else if let Some(v) = p.strip_prefix(">=") { Comp::Ge(SemVer::parse(v)?) }
                else if let Some(v) = p.strip_prefix("<=") { Comp::Le(SemVer::parse(v)?) }
                else if let Some(v) = p.strip_prefix(">") { Comp::Gt(SemVer::parse(v)?) }
                else if let Some(v) = p.strip_prefix("<") { Comp::Lt(SemVer::parse(v)?) }
                else if let Some(v) = p.strip_prefix("=") { Comp::Exact(SemVer::parse(v)?) }
                else if p.contains('*') { return Ok(Self::from_wildcard(p)?); }
                else { Comp::Caret(SemVer::parse(p)?) }; // bare "1.2.3" means ^1.2.3 (cargo-style)
            comparators.push(c);
        }
        Ok(VersionReq { comparators })
    }

    fn from_wildcard(p: &str) -> Result<VersionReq, String> {
        // 1.* -> >=1.0.0, <2.0.0 ; 1.2.* -> >=1.2.0, <1.3.0
        let segs: Vec<&str> = p.split('.').collect();
        let n = |i: usize| segs.get(i).and_then(|s| s.parse::<u64>().ok());
        let comparators = match (n(0), n(1)) {
            (Some(maj), None) => vec![Comp::Ge(SemVer { major: maj, minor: 0, patch: 0, pre: String::new() }),
                                       Comp::Lt(SemVer { major: maj + 1, minor: 0, patch: 0, pre: String::new() })],
            (Some(maj), Some(min)) => vec![Comp::Ge(SemVer { major: maj, minor: min, patch: 0, pre: String::new() }),
                                            Comp::Lt(SemVer { major: maj, minor: min + 1, patch: 0, pre: String::new() })],
            _ => vec![Comp::Any],
        };
        Ok(VersionReq { comparators })
    }

    pub fn matches(&self, v: &SemVer) -> bool { self.comparators.iter().all(|c| c.matches(v)) }
}

impl Comp {
    fn matches(&self, v: &SemVer) -> bool {
        match self {
            Comp::Any => true,
            Comp::Exact(r) => v == r,
            Comp::Ge(r) => v >= r, Comp::Gt(r) => v > r, Comp::Le(r) => v <= r, Comp::Lt(r) => v < r,
            Comp::Caret(r) => {
                // ^1.2.3 -> >=1.2.3, <2.0.0 ; ^0.2.3 -> >=0.2.3, <0.3.0 ; ^0.0.3 -> =0.0.3
                if v < r { return false; }
                if r.major > 0 { v.major == r.major }
                else if r.minor > 0 { v.major == 0 && v.minor == r.minor }
                else { v.major == 0 && v.minor == 0 && v.patch == r.patch }
            }
            Comp::Tilde(r) => v >= r && v.major == r.major && v.minor == r.minor,
        }
    }
}

// ---- SHA-256 (real, std-only) ----------------------------------------------

pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    let mut h: [u32; 8] = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes([chunk[i*4], chunk[i*4+1], chunk[i*4+2], chunk[i*4+3]]); }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) = (h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
        }
        for (i, v) in [a,b,c,d,e,f,g,hh].iter().enumerate() { h[i] = h[i].wrapping_add(*v); }
    }
    h.iter().map(|x| format!("{:08x}", x)).collect()
}

// ---- dependency model + resolver -------------------------------------------

#[derive(Clone)]
pub enum Source { Registry(VersionReq), Path(String), Git { url: String, rev: Option<String> } }

#[derive(Clone)]
pub struct Dep { pub name: String, pub source: Source }

// one line of `<name>/index.txt`: version + checksum + url + transitive deps
#[derive(Clone)]
pub struct IndexEntry { pub version: SemVer, pub sha256: String, pub url: String, pub deps: Vec<(String, String)> }

// the resolved, locked package set
#[derive(Clone)]
pub struct Locked { pub name: String, pub version: String, pub source: String, pub sha256: String }

// resolve root deps against an index (`name -> published entries`). Registry deps
// pick the highest matching version; path/git resolve directly. Shared deps are
// unified to a single version that satisfies every requirement, else a conflict.
pub fn resolve(root: &[Dep], index: &BTreeMap<String, Vec<IndexEntry>>) -> Result<Vec<Locked>, String> {
    let mut chosen: BTreeMap<String, Locked> = BTreeMap::new();
    let mut reqs: BTreeMap<String, Vec<VersionReq>> = BTreeMap::new();
    // queue of (name, source); expand transitively
    let mut queue: Vec<Dep> = root.to_vec();
    while let Some(dep) = queue.pop() {
        match &dep.source {
            Source::Path(p) => { chosen.insert(dep.name.clone(), Locked { name: dep.name.clone(), version: "0.0.0".into(), source: format!("path:{}", p), sha256: String::new() }); }
            Source::Git { url, rev } => { chosen.insert(dep.name.clone(), Locked { name: dep.name.clone(), version: "0.0.0".into(), source: format!("git:{}{}", url, rev.as_ref().map(|r| format!("#{}", r)).unwrap_or_default()), sha256: String::new() }); }
            Source::Registry(req) => {
                reqs.entry(dep.name.clone()).or_default().push(req.clone());
                let entries = index.get(&dep.name).ok_or_else(|| format!("no such package `{}` in the registry", dep.name))?;
                // highest version satisfying ALL accumulated requirements for this name
                let all = &reqs[&dep.name];
                let best = entries.iter().filter(|e| all.iter().all(|r| r.matches(&e.version))).max_by(|a, b| a.version.cmp(&b.version));
                let best = best.ok_or_else(|| format!("version conflict for `{}`: no version satisfies all of {} requirement(s)", dep.name, all.len()))?;
                // if a different version was already chosen and it no longer satisfies, error
                if let Some(prev) = chosen.get(&dep.name) {
                    if prev.version != best.version.to_string() {
                        return Err(format!("version conflict for `{}`: {} vs {}", dep.name, prev.version, best.version));
                    }
                }
                chosen.insert(dep.name.clone(), Locked { name: dep.name.clone(), version: best.version.to_string(), source: format!("registry:{}", best.url), sha256: best.sha256.clone() });
                for (dn, dr) in &best.deps { queue.push(Dep { name: dn.clone(), source: Source::Registry(VersionReq::parse(dr)?) }); }
            }
        }
    }
    let mut out: Vec<Locked> = chosen.into_values().collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

// ---- lockfile (nova.lock) ---------------------------------------------------

pub fn write_lock(locked: &[Locked]) -> String {
    let mut s = String::from("# nova.lock — generated; do not edit by hand\n");
    for l in locked {
        s.push_str(&format!("[[package]]\nname = \"{}\"\nversion = \"{}\"\nsource = \"{}\"\n", l.name, l.version, l.source));
        if !l.sha256.is_empty() { s.push_str(&format!("checksum = \"{}\"\n", l.sha256)); }
        s.push('\n');
    }
    s
}

pub fn read_lock(text: &str) -> Vec<Locked> {
    let mut out = Vec::new();
    let mut cur: Option<Locked> = None;
    for line in text.lines() {
        let l = line.trim();
        if l == "[[package]]" { if let Some(p) = cur.take() { out.push(p); } cur = Some(Locked { name: String::new(), version: String::new(), source: String::new(), sha256: String::new() }); }
        else if let Some((k, v)) = l.split_once('=') {
            let v = v.trim().trim_matches('"');
            if let Some(c) = cur.as_mut() {
                match k.trim() { "name" => c.name = v.into(), "version" => c.version = v.into(), "source" => c.source = v.into(), "checksum" => c.sha256 = v.into(), _ => {} }
            }
        }
    }
    if let Some(p) = cur { out.push(p); }
    out
}

// parse one `<name>/index.txt` file into published entries
pub fn parse_index_file(text: &str) -> Vec<IndexEntry> {
    let mut out = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') { continue; }
        let mut it = l.split_whitespace();
        let (Some(ver), Some(sha), Some(url)) = (it.next(), it.next(), it.next()) else { continue };
        let Ok(version) = SemVer::parse(ver) else { continue };
        let deps: Vec<(String, String)> = it.filter_map(|kv| kv.split_once('=').map(|(a, b)| (a.to_string(), b.to_string()))).collect();
        out.push(IndexEntry { version, sha256: sha.to_string(), url: url.to_string(), deps });
    }
    out
}

// ---- manifest → resolver deps ----------------------------------------------

use crate::config::{HgxConfig, ManifestDep};

// Turn `[dependencies]` manifest entries into resolver `Dep`s. path/git win over
// version; a bare version becomes a Registry requirement.
pub fn deps_from_manifest(deps: &[ManifestDep]) -> Result<Vec<Dep>, String> {
    let mut out = Vec::new();
    for d in deps {
        let source = if let Some(p) = &d.path { Source::Path(p.clone()) }
            else if let Some(g) = &d.git { Source::Git { url: g.clone(), rev: d.rev.clone() } }
            else { Source::Registry(VersionReq::parse(d.version.as_deref().unwrap_or("*"))?) };
        out.push(Dep { name: d.name.clone(), source });
    }
    Ok(out)
}

// ---- the `.nvpkg` package archive (deterministic, std-only) ------------------
// A published package is a flat archive: a `NVPKG1` header, then per file a
// `<path>\t<len>\n` line followed by exactly <len> raw bytes. Files are packed in
// sorted path order so the archive (and thus its sha256) is reproducible. Single
// `.nova` packages skip the archive and are served/vendored as the bare file.

pub fn pack_dir(dir: &std::path::Path) -> Result<Vec<u8>, String> {
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    collect_files(dir, dir, &mut files)?;
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = b"NVPKG1\n".to_vec();
    for (rel, bytes) in &files {
        out.extend_from_slice(format!("{}\t{}\n", rel, bytes.len()).as_bytes());
        out.extend_from_slice(bytes);
    }
    Ok(out)
}

fn collect_files(root: &std::path::Path, dir: &std::path::Path, out: &mut Vec<(String, Vec<u8>)>) -> Result<(), String> {
    let mut entries: Vec<_> = std::fs::read_dir(dir).map_err(|e| format!("cannot read {}: {}", dir.display(), e))?
        .filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with('.') || name == "nova_modules" || name == "build" || name == "target" { continue; }
        if p.is_dir() { collect_files(root, &p, out)?; }
        else {
            let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().replace('\\', "/");
            let bytes = std::fs::read(&p).map_err(|e| format!("cannot read {}: {}", p.display(), e))?;
            out.push((rel, bytes));
        }
    }
    Ok(())
}

pub fn unpack_into(archive: &[u8], dest: &std::path::Path) -> Result<(), String> {
    let body = archive.strip_prefix(b"NVPKG1\n").ok_or("not a NVPKG1 archive")?;
    let mut i = 0usize;
    while i < body.len() {
        let nl = body[i..].iter().position(|&b| b == b'\n').ok_or("truncated archive header")? + i;
        let header = std::str::from_utf8(&body[i..nl]).map_err(|_| "bad archive header")?;
        let (rel, len) = header.split_once('\t').ok_or("bad archive header line")?;
        let len: usize = len.parse().map_err(|_| "bad archive length")?;
        i = nl + 1;
        if i + len > body.len() { return Err("truncated archive body".into()); }
        let bytes = &body[i..i + len];
        i += len;
        // path safety: reject absolute / parent-escaping paths
        if rel.starts_with('/') || rel.split('/').any(|c| c == "..") { return Err(format!("unsafe path in archive: {}", rel)); }
        let path = dest.join(rel);
        if let Some(parent) = path.parent() { std::fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
        std::fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {}", path.display(), e))?;
    }
    Ok(())
}

// ---- minimal HTTP client (plain http://, std-only, no TLS) -------------------
// Closes the loop with `nova registry serve`. https:// needs a TLS stack we don't
// vendor; use a git dep or a local/`http://` index there (reported honestly).

pub fn fetch(url: &str) -> Result<Vec<u8>, String> {
    if let Some(rest) = url.strip_prefix("file://") { return std::fs::read(rest).map_err(|e| e.to_string()); }
    if url.starts_with("http://") { return http_get(url); }
    if url.starts_with("https://") {
        return Err(format!("https fetch needs a TLS stack Nova does not vendor: {}\n  use a `git`/`path` dep or an http:// / local index", url));
    }
    // a bare filesystem path (local index)
    std::fs::read(url).map_err(|e| format!("cannot read {}: {}", url, e))
}

fn http_get(url: &str) -> Result<Vec<u8>, String> {
    use std::io::{Read, Write};
    let rest = url.strip_prefix("http://").unwrap();
    let (authority, path) = rest.split_once('/').map(|(a, p)| (a, format!("/{}", p))).unwrap_or((rest, "/".into()));
    let (host, port) = authority.split_once(':').map(|(h, p)| (h.to_string(), p.parse::<u16>().unwrap_or(80))).unwrap_or((authority.to_string(), 80));
    let mut stream = std::net::TcpStream::connect((host.as_str(), port)).map_err(|e| format!("connect {}: {}", authority, e))?;
    let req = format!("GET {} HTTP/1.0\r\nHost: {}\r\nUser-Agent: nova-pkg\r\nConnection: close\r\n\r\n", path, host);
    stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    let split = buf.windows(4).position(|w| w == b"\r\n\r\n").ok_or("no HTTP header terminator")?;
    let head = String::from_utf8_lossy(&buf[..split]);
    let status = head.lines().next().unwrap_or("");
    if !status.contains(" 200") { return Err(format!("http {}: {}", url, status.trim())); }
    Ok(buf[split + 4..].to_vec())
}

// ---- registry index loading -------------------------------------------------
// An index is a directory (local path or an http:// base) with one file per
// package: `<name>/index.txt`. `_names.txt` (optional) lists packages so the
// resolver can pull only what it needs; we fetch lazily per name here.

pub fn load_index_entries(base: &str, name: &str) -> Result<Vec<IndexEntry>, String> {
    let sep = if base.ends_with('/') { "" } else { "/" };
    let url = format!("{}{}{}/index.txt", base, sep, name);
    let bytes = fetch(&url)?;
    let text = String::from_utf8_lossy(&bytes);
    Ok(parse_index_file(&text))
}

// Build the index map the resolver needs, fetching each referenced package's
// `index.txt` transitively (BFS over dep names).
pub fn build_index(base: &str, roots: &[Dep]) -> Result<BTreeMap<String, Vec<IndexEntry>>, String> {
    let mut idx: BTreeMap<String, Vec<IndexEntry>> = BTreeMap::new();
    let mut queue: Vec<String> = roots.iter().filter_map(|d| match d.source { Source::Registry(_) => Some(d.name.clone()), _ => None }).collect();
    while let Some(name) = queue.pop() {
        if idx.contains_key(&name) { continue; }
        let entries = load_index_entries(base, &name)?;
        for e in &entries { for (dn, _) in &e.deps { if !idx.contains_key(dn) { queue.push(dn.clone()); } } }
        idx.insert(name, entries);
    }
    Ok(idx)
}

// ---- two-mode abilities: apply manifest attributes onto the AST --------------
// The user's design — attributes may live in `nova.hgx` instead of on the code.
// After a program is loaded, merge each `[abilities]` entry onto its target
// functions (empty targets = every user fn), skipping any fn that already
// carries that attribute so on-code declarations win / never duplicate.
pub fn apply_abilities(program: &mut crate::ast::Program, abilities: &[crate::config::ManifestAbility]) {
    use crate::ast::{Item, Attr};
    for ab in abilities {
        let raw = if ab.args.is_empty() { ab.attr.clone() } else { format!("{}({})", ab.attr, ab.args) };
        let args: Vec<(String, String)> = ab.args.split(',').filter_map(|kv| {
            let kv = kv.trim();
            if kv.is_empty() { return None; }
            match kv.split_once(':') { Some((k, v)) => Some((k.trim().to_string(), v.trim().to_string())), None => Some((String::new(), kv.to_string())) }
        }).collect();
        for item in program.items.iter_mut() {
            if let Item::Func(f) = item {
                let hit = ab.targets.is_empty() || ab.targets.iter().any(|t| t == &f.name);
                if hit && !f.attrs.iter().any(|a| a.name == ab.attr) {
                    f.attrs.push(Attr { name: ab.attr.clone(), args: args.clone(), exprs: Vec::new(), raw: raw.clone() });
                }
            }
        }
    }
}

// ---- HTTP registry server ---------------------------------------------------
// `nova registry serve <dir> [--port N]` — serves the index dir over http:// so
// `fetch` above can pull `index.txt` files and package archives. Real, threaded,
// std-only; path-safe (no `..` escape out of the served root).
pub fn serve(dir: &std::path::Path, port: u16) -> Result<(), String> {
    use std::io::{Read, Write};
    let root = std::fs::canonicalize(dir).map_err(|e| format!("cannot open {}: {}", dir.display(), e))?;
    let listener = std::net::TcpListener::bind(("0.0.0.0", port)).map_err(|e| format!("bind :{}: {}", port, e))?;
    eprintln!("nova registry serving {} on http://0.0.0.0:{}", root.display(), port);
    for conn in listener.incoming() {
        let root = root.clone();
        let mut stream = match conn { Ok(s) => s, Err(_) => continue };
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);
            let path = req.lines().next().and_then(|l| l.split_whitespace().nth(1)).unwrap_or("/");
            let rel = path.trim_start_matches('/').split('?').next().unwrap_or("");
            let (code, body): (&str, Vec<u8>) = if rel.is_empty() || rel.split('/').any(|c| c == "..") {
                ("400 Bad Request", b"bad path".to_vec())
            } else {
                match std::fs::read(root.join(rel)) {
                    Ok(b) => ("200 OK", b),
                    Err(_) => ("404 Not Found", b"not found".to_vec()),
                }
            };
            let header = format!("HTTP/1.0 {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", code, body.len());
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&body);
        });
    }
    Ok(())
}

// ---- publish: append a package version to a local index dir ------------------
// Packs the project, computes its sha256, copies the archive into the index dir
// as `<name>-<version>.nvpkg`, and appends the index.txt line (idempotent per
// version). This makes any directory a real, self-hostable registry.
pub fn publish(project: &std::path::Path, index_dir: &std::path::Path, cfg: &HgxConfig) -> Result<String, String> {
    let ver = SemVer::parse(&cfg.version)?;
    let archive = pack_dir(project)?;
    let sha = sha256_hex(&archive);
    let pkg_dir = index_dir.join(&cfg.name);
    std::fs::create_dir_all(&pkg_dir).map_err(|e| e.to_string())?;
    let file = format!("{}-{}.nvpkg", cfg.name, ver);
    std::fs::write(pkg_dir.join(&file), &archive).map_err(|e| e.to_string())?;
    let deps: Vec<String> = cfg.dependencies.iter().filter(|d| d.version.is_some())
        .map(|d| format!("{}={}", d.name, d.version.clone().unwrap())).collect();
    let line = format!("{} {} {} {}\n", ver, sha, file, deps.join(" ")).trim_end().to_string() + "\n";
    let index_txt = pkg_dir.join("index.txt");
    let existing = std::fs::read_to_string(&index_txt).unwrap_or_default();
    let mut lines: Vec<String> = existing.lines()
        .filter(|l| !l.trim().is_empty() && !l.split_whitespace().next().map(|v| v == ver.to_string()).unwrap_or(false))
        .map(|s| s.to_string()).collect();
    lines.push(line.trim_end().to_string());
    lines.sort();
    std::fs::write(&index_txt, lines.join("\n") + "\n").map_err(|e| e.to_string())?;
    Ok(sha)
}

// ---- install driver: resolve + lock + fetch + verify + vendor ---------------

// The end-to-end `nova install` / `nova update`. Reads `[dependencies]` +
// `[registry]` from the manifest, resolves against the index (respecting a
// reproducible `nova.lock` unless `update`), then fetches every package, verifies
// its sha256, and vendors it into `nova_modules/` where `resolve_import` finds it.
pub fn install(cfg: &HgxConfig, update: bool) -> Result<Vec<Locked>, String> {
    let deps = deps_from_manifest(&cfg.dependencies)?;
    if deps.is_empty() { return Ok(Vec::new()); }
    let has_registry = deps.iter().any(|d| matches!(d.source, Source::Registry(_)));
    let base = cfg.registries.iter().find(|(n, _)| n == "default")
        .or_else(|| cfg.registries.first()).map(|(_, u)| u.clone());
    if has_registry && base.is_none() {
        return Err("registry dependencies declared but no `[registry]` endpoint configured in nova.hgx".into());
    }
    let lock_exists = std::path::Path::new("nova.lock").exists();
    let locked: Vec<Locked> = if !update && lock_exists {
        read_lock(&std::fs::read_to_string("nova.lock").map_err(|e| e.to_string())?)
    } else {
        let index = if let Some(b) = &base { build_index(b, &deps)? } else { BTreeMap::new() };
        let l = resolve(&deps, &index)?;
        std::fs::write("nova.lock", write_lock(&l)).map_err(|e| e.to_string())?;
        l
    };
    std::fs::create_dir_all("nova_modules").map_err(|e| e.to_string())?;
    for l in &locked { vendor_one(l, base.as_deref())?; }
    Ok(locked)
}

fn vendor_one(l: &Locked, base: Option<&str>) -> Result<(), String> {
    let modules = std::path::Path::new("nova_modules");
    let dest_dir = modules.join(&l.name);
    if let Some(url) = l.source.strip_prefix("registry:") {
        let base = base.ok_or("registry dependency without a configured registry base")?;
        let sep = if base.ends_with('/') { "" } else { "/" };
        let full = format!("{}{}{}/{}", base, sep, l.name, url);
        let bytes = fetch(&full)?;
        let got = sha256_hex(&bytes);
        if !l.sha256.is_empty() && got != l.sha256 {
            return Err(format!("checksum mismatch for `{}` {}: manifest/lock says {}, download is {}", l.name, l.version, l.sha256, got));
        }
        if url.ends_with(".nvpkg") {
            let _ = std::fs::remove_dir_all(&dest_dir);
            unpack_into(&bytes, &dest_dir)?;
        } else {
            std::fs::write(modules.join(format!("{}.nova", l.name)), &bytes).map_err(|e| e.to_string())?;
        }
    } else if let Some(p) = l.source.strip_prefix("path:") {
        copy_source(std::path::Path::new(p), &l.name)?;
    } else if let Some(g) = l.source.strip_prefix("git:") {
        let (url, rev) = match g.split_once('#') { Some((u, r)) => (u, Some(r)), None => (g, None) };
        git_vendor(url, rev, &dest_dir)?;
    }
    Ok(())
}

// Copy a path dependency into nova_modules: a file → `<name>.nova`, a dir → a
// recursive copy under `nova_modules/<name>/`.
fn copy_source(src: &std::path::Path, name: &str) -> Result<(), String> {
    let modules = std::path::Path::new("nova_modules");
    if src.is_file() {
        let bytes = std::fs::read(src).map_err(|e| format!("cannot read {}: {}", src.display(), e))?;
        std::fs::write(modules.join(format!("{}.nova", name)), bytes).map_err(|e| e.to_string())?;
    } else if src.is_dir() {
        let dest = modules.join(name);
        let _ = std::fs::remove_dir_all(&dest);
        copy_tree(src, &dest)?;
    } else {
        return Err(format!("path dependency `{}` not found: {}", name, src.display()));
    }
    Ok(())
}

fn copy_tree(src: &std::path::Path, dst: &std::path::Path) -> Result<(), String> {
    std::fs::create_dir_all(dst).map_err(|e| e.to_string())?;
    for entry in std::fs::read_dir(src).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') || name == *"nova_modules" || name == *"build" || name == *"target" { continue; }
        let (from, to) = (entry.path(), dst.join(&name));
        if from.is_dir() { copy_tree(&from, &to)?; }
        else { std::fs::copy(&from, &to).map_err(|e| e.to_string())?; }
    }
    Ok(())
}

// Vendor a git dependency: shallow-clone, optionally checkout a rev, strip .git.
fn git_vendor(url: &str, rev: Option<&str>, dest: &std::path::Path) -> Result<(), String> {
    let _ = std::fs::remove_dir_all(dest);
    let run = |args: &[&str]| -> Result<(), String> {
        let st = std::process::Command::new("git").args(args)
            .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
            .status().map_err(|e| format!("git not available: {}", e))?;
        if st.success() { Ok(()) } else { Err(format!("git {:?} failed", args)) }
    };
    run(&["clone", "--depth", "1", url, &dest.to_string_lossy()])?;
    if let Some(r) = rev {
        run(&["-C", &dest.to_string_lossy(), "fetch", "--depth", "1", "origin", r])?;
        run(&["-C", &dest.to_string_lossy(), "checkout", r])?;
    }
    let _ = std::fs::remove_dir_all(dest.join(".git"));
    Ok(())
}

// Pretty-print the resolved dependency set as a tree (from the lock / a resolve).
pub fn tree(locked: &[Locked]) -> String {
    let mut out = String::new();
    for l in locked {
        let kind = l.source.split(':').next().unwrap_or("registry");
        out.push_str(&format!("├─ {} {} ({})\n", l.name, l.version, kind));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn semver_order_and_pre() {
        assert!(SemVer::parse("1.2.3").unwrap() > SemVer::parse("1.2.2").unwrap());
        assert!(SemVer::parse("2.0.0").unwrap() > SemVer::parse("1.9.9").unwrap());
        assert!(SemVer::parse("1.0.0-alpha").unwrap() < SemVer::parse("1.0.0").unwrap());
    }
    #[test] fn caret_tilde_wildcard() {
        let v = |s: &str| SemVer::parse(s).unwrap();
        assert!(VersionReq::parse("^1.2.3").unwrap().matches(&v("1.9.0")));
        assert!(!VersionReq::parse("^1.2.3").unwrap().matches(&v("2.0.0")));
        assert!(VersionReq::parse("~1.2.3").unwrap().matches(&v("1.2.9")));
        assert!(!VersionReq::parse("~1.2.3").unwrap().matches(&v("1.3.0")));
        assert!(VersionReq::parse("1.*").unwrap().matches(&v("1.5.0")));
        assert!(VersionReq::parse(">=1.0, <2.0").unwrap().matches(&v("1.4.0")));
        assert!(!VersionReq::parse(">=1.0, <2.0").unwrap().matches(&v("2.1.0")));
    }
    #[test] fn sha256_known_vector() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
    #[test] fn resolver_picks_highest_and_unifies() {
        let mut idx = BTreeMap::new();
        idx.insert("json".to_string(), vec![
            IndexEntry { version: SemVer::parse("1.0.0").unwrap(), sha256: "a".into(), url: "json-1.0.0.nova".into(), deps: vec![] },
            IndexEntry { version: SemVer::parse("1.4.0").unwrap(), sha256: "b".into(), url: "json-1.4.0.nova".into(), deps: vec![] },
            IndexEntry { version: SemVer::parse("2.0.0").unwrap(), sha256: "c".into(), url: "json-2.0.0.nova".into(), deps: vec![] },
        ]);
        let root = vec![Dep { name: "json".into(), source: Source::Registry(VersionReq::parse("^1.0").unwrap()) }];
        let locked = resolve(&root, &idx).unwrap();
        assert_eq!(locked.len(), 1);
        assert_eq!(locked[0].version, "1.4.0"); // highest ^1.0, not 2.0.0
    }
    #[test] fn resolver_reports_conflict() {
        let mut idx = BTreeMap::new();
        idx.insert("x".to_string(), vec![IndexEntry { version: SemVer::parse("1.0.0").unwrap(), sha256: "a".into(), url: "x.nova".into(), deps: vec![] }]);
        let root = vec![Dep { name: "x".into(), source: Source::Registry(VersionReq::parse("^2.0").unwrap()) }];
        assert!(resolve(&root, &idx).is_err());
    }
    #[test] fn pack_unpack_roundtrip_is_deterministic() {
        let tmp = std::env::temp_dir().join(format!("nvpkg_test_{}", std::process::id()));
        let src = tmp.join("src"); let dst = tmp.join("out");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(src.join("sub")).unwrap();
        std::fs::write(src.join("a.nova"), b"fn a() { 1 }\n").unwrap();
        std::fs::write(src.join("sub").join("b.nova"), b"fn b() { 2 }\n").unwrap();
        let arc1 = pack_dir(&src).unwrap();
        let arc2 = pack_dir(&src).unwrap();
        assert_eq!(arc1, arc2, "archive must be reproducible");
        assert_eq!(sha256_hex(&arc1), sha256_hex(&arc2));
        unpack_into(&arc1, &dst).unwrap();
        assert_eq!(std::fs::read(dst.join("a.nova")).unwrap(), b"fn a() { 1 }\n");
        assert_eq!(std::fs::read(dst.join("sub").join("b.nova")).unwrap(), b"fn b() { 2 }\n");
        let _ = std::fs::remove_dir_all(&tmp);
    }
    #[test] fn unpack_rejects_unsafe_paths() {
        let mut arc = b"NVPKG1\n".to_vec();
        arc.extend_from_slice(b"../evil\t3\nbad");
        assert!(unpack_into(&arc, std::path::Path::new("/tmp")).is_err());
    }
    #[test] fn abilities_apply_project_wide_and_targeted() {
        use crate::config::ManifestAbility;
        let mut prog = crate::parser::parse_program("fn fetch() { 1 }\nfn other() { 2 }\n").unwrap();
        let abs = vec![
            ManifestAbility { attr: "trace".into(), args: String::new(), targets: vec![] },
            ManifestAbility { attr: "self_healing".into(), args: "attempts: 3".into(), targets: vec!["fetch".into()] },
        ];
        apply_abilities(&mut prog, &abs);
        let get = |n: &str| -> Vec<String> { prog.items.iter().find_map(|it| match it {
            crate::ast::Item::Func(f) if f.name == n => Some(f.attrs.iter().map(|a| a.name.clone()).collect()), _ => None }).unwrap() };
        assert!(get("fetch").contains(&"trace".to_string()));
        assert!(get("fetch").contains(&"self_healing".to_string()));
        assert!(get("other").contains(&"trace".to_string()));
        assert!(!get("other").contains(&"self_healing".to_string())); // targeted only
    }
    #[test] fn deps_from_manifest_maps_sources() {
        use crate::config::ManifestDep;
        let ds = deps_from_manifest(&[
            ManifestDep { name: "json".into(), version: Some("^1".into()), ..Default::default() },
            ManifestDep { name: "loc".into(), path: Some("../loc".into()), ..Default::default() },
            ManifestDep { name: "g".into(), git: Some("u".into()), rev: Some("r".into()), ..Default::default() },
        ]).unwrap();
        assert!(matches!(ds[0].source, Source::Registry(_)));
        assert!(matches!(ds[1].source, Source::Path(_)));
        assert!(matches!(ds[2].source, Source::Git { .. }));
    }
}
