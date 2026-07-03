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
        let (key, value) = line.split_once('=')
            .ok_or_else(|| format!("nova.hgx line {}: expected `key = value`, got `{}`", lineno + 1, line))?;
        let key = key.trim();
        let value = parse_value(value.trim())
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

enum HgxValue { Str(String), Int(u64) }

impl HgxValue {
    fn as_str(self) -> Result<String, String> {
        match self {
            HgxValue::Str(s) => Ok(s),
            HgxValue::Int(n) => Err(format!("expected a quoted string, got integer {}", n)),
        }
    }
    fn as_int(self) -> Result<u64, String> {
        match self {
            HgxValue::Int(n) => Ok(n),
            HgxValue::Str(s) => Err(format!("expected an integer, got \"{}\"", s)),
        }
    }
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
}
