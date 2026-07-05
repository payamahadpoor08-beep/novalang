// nova.hgx — the per-project configuration file (Nova's Cargo.toml).
//
// Syntax is a strict TOML subset — sections and `key = value` pairs where a
// value is a double-quoted string or an integer, with `#` comments:
//
//     [package]
//     name = "myapp"
//     version = "0.1.0"
//     entry = "src/main.nova"
//
//     [build]
//     opt-level = "release"      # or "debug"
//     jit-threshold = 100        # tiering threshold override
//
//     [target]
//     default = "pc"
//
// Parsed by hand (~a page of code) instead of pulling the `toml` crate: that
// crate brings the whole serde dependency chain for six keys. Unknown keys are
// ignored (forward compatibility); malformed lines are hard errors.

use std::path::Path;

#[derive(Debug, Clone)]
pub struct HgxConfig {
    pub name: String,
    pub version: String,
    pub entry: String,
    pub opt_level: String,
    pub jit_threshold: Option<u64>,
    pub target_default: String,
    // `[dependencies]` — each `name = "^1.2"` or `name = { version, git, rev,
    // path, registry, features }`. The package manager (registry.rs) resolves
    // these against the index / path / git into a reproducible `nova.lock`.
    pub dependencies: Vec<ManifestDep>,
    // `[registry]` / `[registries]` — named registry index endpoints
    // (`crates = "https://…"` or a local dir). "default" is the fallback.
    pub registries: Vec<(String, String)>,
    // `[abilities]` — the user's *two-mode* attribute design: attributes/abilities
    // may be declared here (project-wide) instead of on the code, and are merged
    // onto the matching functions at load. `name = "trace"`, `name = ["trace",
    // "profile"]`, or `name = { attr = "self_healing", args = "attempts: 3",
    // targets = ["fetch", "save"] }`. An empty `targets` applies to every fn.
    pub abilities: Vec<ManifestAbility>,
}

// A dependency line from `[dependencies]`.
#[derive(Debug, Clone, Default)]
pub struct ManifestDep {
    pub name: String,
    pub version: Option<String>,  // semver requirement, e.g. "^1.2.3"
    pub git: Option<String>,
    pub rev: Option<String>,
    pub path: Option<String>,
    pub registry: Option<String>, // named registry from `[registry]`
    pub features: Vec<String>,
}

// A manifest-declared attribute/ability (two-mode: manifest OR on code).
#[derive(Debug, Clone, Default)]
pub struct ManifestAbility {
    pub attr: String,             // attribute name, e.g. "trace"
    pub args: String,             // raw arg text, e.g. "attempts: 3" (may be empty)
    pub targets: Vec<String>,     // function names; empty = apply to every fn
}

impl Default for HgxConfig {
    fn default() -> Self {
        HgxConfig {
            name: "app".into(),
            version: "0.1.0".into(),
            entry: "src/main.nova".into(),
            opt_level: "release".into(),
            jit_threshold: None,
            target_default: "pc".into(),
            dependencies: Vec::new(),
            registries: Vec::new(),
            abilities: Vec::new(),
        }
    }
}

// Load `nova.hgx` from `dir`. Returns None when the file doesn't exist (callers
// keep their current behavior), Some(Err) when it exists but is malformed.
pub fn load_hgx(dir: &Path) -> Option<Result<HgxConfig, String>> {
    let path = dir.join("nova.hgx");
    let text = std::fs::read_to_string(&path).ok()?;
    Some(parse_hgx(&text))
}

pub fn parse_hgx(text: &str) -> Result<HgxConfig, String> {
    let mut cfg = HgxConfig::default();
    let mut section = String::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() { continue; }
        if let Some(name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            let name = name.trim();
            if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
                return Err(format!("nova.hgx line {}: bad section name `{}`", lineno + 1, name));
            }
            section = name.to_string();
            continue;
        }
        let (key, rhs) = line.split_once('=')
            .ok_or_else(|| format!("nova.hgx line {}: expected `key = value`, got `{}`", lineno + 1, line))?;
        let key = key.trim();
        let rhs = rhs.trim();
        // Sections whose values are structured (inline tables / arrays) are
        // handled specially before the scalar `parse_value` path.
        match section.as_str() {
            "dependencies" | "deps" => {
                cfg.dependencies.push(parse_dep(key, rhs)
                    .map_err(|e| format!("nova.hgx line {}: {}", lineno + 1, e))?);
                continue;
            }
            "registry" | "registries" => {
                let url = parse_value(rhs).and_then(|v| v.as_str())
                    .map_err(|e| format!("nova.hgx line {}: {}", lineno + 1, e))?;
                cfg.registries.push((key.to_string(), url));
                continue;
            }
            "abilities" | "attributes" => {
                cfg.abilities.extend(parse_abilities(key, rhs)
                    .map_err(|e| format!("nova.hgx line {}: {}", lineno + 1, e))?);
                continue;
            }
            _ => {}
        }
        let value = parse_value(rhs)
            .map_err(|e| format!("nova.hgx line {}: {}", lineno + 1, e))?;
        match (section.as_str(), key) {
            ("package", "name") => cfg.name = value.as_str()?,
            ("package", "version") => cfg.version = value.as_str()?,
            ("package", "entry") => cfg.entry = value.as_str()?,
            ("build", "opt-level") => {
                let v = value.as_str()?;
                if v != "release" && v != "debug" {
                    return Err(format!("nova.hgx line {}: opt-level must be \"release\" or \"debug\"", lineno + 1));
                }
                cfg.opt_level = v;
            }
            ("build", "jit-threshold") => cfg.jit_threshold = Some(value.as_int()?),
            ("target", "default") => cfg.target_default = value.as_str()?,
            _ => {} // unknown section/key: ignored for forward compatibility
        }
    }
    Ok(cfg)
}

// strip a `#` comment, respecting `#` inside double-quoted strings
fn strip_comment(line: &str) -> &str {
    let mut in_str = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_str = !in_str,
            '#' if !in_str => return &line[..i],
            _ => {}
        }
    }
    line
}

enum HgxValue { Str(String), Int(u64), Arr(Vec<String>) }

impl HgxValue {
    fn as_str(self) -> Result<String, String> {
        match self {
            HgxValue::Str(s) => Ok(s),
            HgxValue::Int(n) => Err(format!("expected a quoted string, got integer {}", n)),
            HgxValue::Arr(_) => Err("expected a quoted string, got an array".to_string()),
        }
    }
    fn as_int(self) -> Result<u64, String> {
        match self {
            HgxValue::Int(n) => Ok(n),
            HgxValue::Str(s) => Err(format!("expected an integer, got \"{}\"", s)),
            HgxValue::Arr(_) => Err("expected an integer, got an array".to_string()),
        }
    }
    fn as_arr(self) -> Result<Vec<String>, String> {
        match self {
            HgxValue::Arr(a) => Ok(a),
            HgxValue::Str(s) => Err(format!("expected an array, got \"{}\"", s)),
            HgxValue::Int(n) => Err(format!("expected an array, got integer {}", n)),
        }
    }
}

// Parse one `[dependencies]` entry: `name = "^1.2"` (version shorthand) or
// `name = { version = "1", git = "url", rev = "sha", path = "..", registry =
// "crates", features = ["a", "b"] }`.
fn parse_dep(name: &str, rhs: &str) -> Result<ManifestDep, String> {
    let mut d = ManifestDep { name: name.to_string(), ..Default::default() };
    if rhs.starts_with('{') {
        for (k, v) in parse_inline_table(rhs)? {
            match k.as_str() {
                "version" => d.version = Some(v.as_str()?),
                "git" => d.git = Some(v.as_str()?),
                "rev" => d.rev = Some(v.as_str()?),
                "path" => d.path = Some(v.as_str()?),
                "registry" => d.registry = Some(v.as_str()?),
                "features" => d.features = v.as_arr()?,
                other => return Err(format!("unknown dependency key `{}`", other)),
            }
        }
        if d.version.is_none() && d.git.is_none() && d.path.is_none() {
            return Err(format!("dependency `{}` needs a version, git, or path", name));
        }
    } else {
        d.version = Some(parse_value(rhs)?.as_str()?);
    }
    Ok(d)
}

// Parse one `[abilities]` entry into one-or-more ManifestAbility. Forms:
//   fast    = "trace"                       # attr, all fns
//   audited = ["trace", "profile"]          # several attrs, all fns
//   heal    = { attr = "self_healing", args = "attempts: 3", targets = ["a"] }
fn parse_abilities(key: &str, rhs: &str) -> Result<Vec<ManifestAbility>, String> {
    if rhs.starts_with('{') {
        let mut a = ManifestAbility::default();
        for (k, v) in parse_inline_table(rhs)? {
            match k.as_str() {
                "attr" | "attribute" => a.attr = v.as_str()?,
                "args" => a.args = v.as_str()?,
                "targets" => a.targets = v.as_arr()?,
                other => return Err(format!("unknown ability key `{}`", other)),
            }
        }
        if a.attr.is_empty() { return Err(format!("ability `{}` needs an `attr`", key)); }
        Ok(vec![a])
    } else if rhs.starts_with('[') {
        Ok(parse_string_array(rhs)?.into_iter()
            .map(|attr| ManifestAbility { attr, ..Default::default() }).collect())
    } else {
        Ok(vec![ManifestAbility { attr: parse_value(rhs)?.as_str()?, ..Default::default() }])
    }
}

// A minimal inline-table parser: `{ k = v, k2 = [..], .. }`. Values are strings,
// ints, or string arrays. Commas separate; splitting respects brackets+quotes.
fn parse_inline_table(s: &str) -> Result<Vec<(String, HgxValue)>, String> {
    let inner = s.strip_prefix('{').and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| format!("malformed inline table: {}", s))?.trim();
    let mut out = Vec::new();
    for field in split_top(inner, ',') {
        let field = field.trim();
        if field.is_empty() { continue; }
        let (k, v) = field.split_once('=')
            .ok_or_else(|| format!("inline table field needs `key = value`: {}", field))?;
        let v = v.trim();
        let val = if v.starts_with('[') { HgxValue::Arr(parse_string_array(v)?) }
                  else { parse_value(v)? };
        out.push((k.trim().to_string(), val));
    }
    Ok(out)
}

// `["a", "b", "c"]` -> vec of strings.
fn parse_string_array(s: &str) -> Result<Vec<String>, String> {
    let inner = s.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| format!("malformed array: {}", s))?.trim();
    if inner.is_empty() { return Ok(Vec::new()); }
    split_top(inner, ',').into_iter()
        .map(|e| parse_value(e.trim()).and_then(|v| v.as_str()))
        .collect()
}

// Split `s` on `sep`, but only at bracket depth 0 and outside quotes.
fn split_top(s: &str, sep: char) -> Vec<String> {
    let (mut out, mut buf, mut depth, mut in_str) = (Vec::new(), String::new(), 0i32, false);
    for c in s.chars() {
        match c {
            '"' => { in_str = !in_str; buf.push(c); }
            '[' | '{' if !in_str => { depth += 1; buf.push(c); }
            ']' | '}' if !in_str => { depth -= 1; buf.push(c); }
            _ if c == sep && depth == 0 && !in_str => { out.push(std::mem::take(&mut buf)); }
            _ => buf.push(c),
        }
    }
    if !buf.trim().is_empty() { out.push(buf); }
    out
}

fn parse_value(v: &str) -> Result<HgxValue, String> {
    if let Some(inner) = v.strip_prefix('"') {
        let inner = inner.strip_suffix('"')
            .ok_or_else(|| format!("unterminated string: {}", v))?;
        if inner.contains('"') {
            return Err(format!("nested quote in string: {}", v));
        }
        return Ok(HgxValue::Str(inner.to_string()));
    }
    v.parse::<u64>()
        .map(HgxValue::Int)
        .map_err(|_| format!("expected a quoted string or integer, got `{}`", v))
}

#[cfg(test)]
mod hgx_tests {
    use super::*;

    #[test] fn full_config_parses() {
        let c = parse_hgx(concat!(
            "# my app\n[package]\nname = \"myapp\"\nversion = \"1.2.3\"\n",
            "entry = \"src/app.nova\"\n\n[build]\nopt-level = \"debug\"\n",
            "jit-threshold = 250   # hot after 250 calls\n[target]\ndefault = \"pc\"\n",
        )).unwrap();
        assert_eq!(c.name, "myapp");
        assert_eq!(c.version, "1.2.3");
        assert_eq!(c.entry, "src/app.nova");
        assert_eq!(c.opt_level, "debug");
        assert_eq!(c.jit_threshold, Some(250));
        assert_eq!(c.target_default, "pc");
    }
    #[test] fn defaults_apply() {
        let c = parse_hgx("[package]\nname = \"x\"\n").unwrap();
        assert_eq!(c.entry, "src/main.nova");
        assert_eq!(c.opt_level, "release");
        assert_eq!(c.jit_threshold, None);
    }
    #[test] fn unknown_keys_ignored() {
        assert!(parse_hgx("[package]\nfuture-key = \"ok\"\n[future-section]\nx = 1\n").is_ok());
    }
    #[test] fn malformed_is_error() {
        assert!(parse_hgx("[package]\nname myapp\n").is_err());   // missing =
        assert!(parse_hgx("[package]\nname = myapp\n").is_err()); // unquoted string
        assert!(parse_hgx("[build]\njit-threshold = \"x\"\n").is_err()); // wrong type
        assert!(parse_hgx("[build]\nopt-level = \"fast\"\n").is_err()); // bad enum
        assert!(parse_hgx("[bad section]\n").is_err());
    }
    #[test] fn comments_and_hash_in_string() {
        let c = parse_hgx("[package]\nname = \"a#b\" # trailing\n").unwrap();
        assert_eq!(c.name, "a#b");
    }
    #[test] fn dependencies_shorthand_and_table() {
        let c = parse_hgx(concat!(
            "[package]\nname = \"x\"\n[dependencies]\n",
            "json = \"^1.2\"\n",
            "http = { version = \"2.0\", features = [\"tls\", \"gzip\"] }\n",
            "local = { path = \"../local\" }\n",
            "gitdep = { git = \"https://ex/g.git\", rev = \"abc\" }\n",
        )).unwrap();
        assert_eq!(c.dependencies.len(), 4);
        let json = c.dependencies.iter().find(|d| d.name == "json").unwrap();
        assert_eq!(json.version.as_deref(), Some("^1.2"));
        let http = c.dependencies.iter().find(|d| d.name == "http").unwrap();
        assert_eq!(http.version.as_deref(), Some("2.0"));
        assert_eq!(http.features, vec!["tls", "gzip"]);
        assert_eq!(c.dependencies.iter().find(|d| d.name == "local").unwrap().path.as_deref(), Some("../local"));
        let g = c.dependencies.iter().find(|d| d.name == "gitdep").unwrap();
        assert_eq!(g.git.as_deref(), Some("https://ex/g.git"));
        assert_eq!(g.rev.as_deref(), Some("abc"));
    }
    #[test] fn registries_and_abilities() {
        let c = parse_hgx(concat!(
            "[package]\nname = \"x\"\n",
            "[registry]\ndefault = \"https://reg.nova/index\"\ncorp = \"/srv/index\"\n",
            "[abilities]\nfast = \"trace\"\naudited = [\"log\", \"profile\"]\n",
            "heal = { attr = \"self_healing\", args = \"attempts: 3\", targets = [\"fetch\"] }\n",
        )).unwrap();
        assert_eq!(c.registries.len(), 2);
        assert_eq!(c.registries[0], ("default".into(), "https://reg.nova/index".into()));
        // fast(1) + audited(2) + heal(1) = 4 abilities
        assert_eq!(c.abilities.len(), 4);
        let heal = c.abilities.iter().find(|a| a.attr == "self_healing").unwrap();
        assert_eq!(heal.args, "attempts: 3");
        assert_eq!(heal.targets, vec!["fetch"]);
        assert!(c.abilities.iter().any(|a| a.attr == "trace" && a.targets.is_empty()));
    }
    #[test] fn bad_dependency_is_error() {
        assert!(parse_hgx("[dependencies]\nfoo = { features = [\"x\"] }\n").is_err()); // no ver/git/path
        assert!(parse_hgx("[dependencies]\nfoo = { bogus = \"1\" }\n").is_err());
    }
}
