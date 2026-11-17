//! Step 5: the definition of done for one port.
//!
//! fmt, clippy (warnings are errors), cargo tests, no unexplained `unsafe`, every function
//! ported, the built module exposes the original API, and the recorded behavior replays cleanly.

use std::path::Path;
use std::process::Command;

use anyhow::Result;
use serde::Serialize;

use crate::bridge::Manifest;
use crate::harness::{self, CASES, DiffOptions};

#[derive(Debug, Serialize)]
pub struct Check {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

/// Bindings `bridge` writes into `src/`; everything else there is the port.
const GENERATED: [&str; 7] = [
    "lib.rs",
    "js.rs",
    "pyconv.rs",
    "pynum.rs",
    "rbconv.rs",
    "utf16.rs",
    "wire.rs",
];

/// How many times the source's lines the port may take before `gate` calls it bloated, plus a
/// fixed allowance for the imports and signatures every port has.
pub const MAX_GROWTH: f64 = 3.0;
const ALLOWANCE: f64 = 50.0;

fn code_lines(text: &str, comment: &[&str]) -> usize {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !comment.iter().any(|c| l.starts_with(c)))
        .count()
}

fn files_under(path: &Path, extensions: &[&str]) -> Vec<std::path::PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }
    crate::walk::builder(path)
        .build()
        .flatten()
        .map(ignore::DirEntry::into_path)
        .filter(|p| {
            p.extension()
                .is_some_and(|e| extensions.contains(&e.to_string_lossy().as_ref()))
        })
        .collect()
}

/// Lines of Rust in the port against lines of the source unit: a port several times larger
/// usually re-implements what a crate already does.
fn size_check(crate_dir: &Path, manifest: &Manifest) -> Option<Check> {
    let source_comments: &[&str] = match manifest.language {
        crate::lang::Lang::Python | crate::lang::Lang::Ruby => &["#"],
        crate::lang::Lang::Php => &["//", "#", "/*", "*"],
        _ => &["//", "/*", "*"],
    };
    let source: usize = files_under(
        &manifest.root.join(&manifest.unit),
        manifest.language.extensions(),
    )
    .iter()
    .filter_map(|p| std::fs::read_to_string(p).ok())
    .map(|t| code_lines(&t, source_comments))
    .sum();
    if source == 0 {
        return None;
    }
    let rust: usize = files_under(&crate_dir.join("src"), &["rs"])
        .iter()
        .filter(|p| {
            p.file_name()
                .is_none_or(|n| !GENERATED.contains(&n.to_string_lossy().as_ref()))
        })
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .map(|t| code_lines(&t, &["//"]))
        .sum();
    let growth = rust as f64 / source as f64;
    let passed = rust as f64 <= MAX_GROWTH * source as f64 + ALLOWANCE;
    Some(Check {
        name: "size",
        passed,
        detail: format!(
            "{rust} lines of Rust for {source} lines of {} ({growth:.1}x){}",
            manifest.language.name(),
            if passed {
                String::new()
            } else {
                format!(
                    "; over {MAX_GROWTH}x usually means code a crate already provides, or a case \
                     handled one by one: use the crates `analyze` lists and fold repeated code"
                )
            }
        ),
    })
}

#[derive(Debug, Serialize)]
pub struct GateOutcome {
    pub passed: bool,
    pub checks: Vec<Check>,
}

pub fn gate(crate_dir: &Path, node_args: &[String]) -> Result<GateOutcome> {
    let manifest = Manifest::load(crate_dir)?;
    let mut checks = rust_checks(crate_dir, &crate_dir.join("target"));
    checks.push(ported_check(crate_dir));

    checks.extend(size_check(crate_dir, &manifest));
    checks.extend(crate::error_paths::checks(crate_dir, &manifest));
    let built = harness::build(crate_dir);
    let built_ok = built.is_ok();
    checks.push(Check {
        name: "build",
        passed: built_ok,
        detail: match built {
            Ok(path) => format!("built {}", path.display()),
            Err(e) => last_lines(&format!("{e:#}"), 15),
        },
    });

    if !crate_dir.join(CASES).exists() {
        checks.push(Check {
            name: "behavior",
            passed: false,
            detail: "no recorded cases: run `rust-anything record` first".into(),
        });
    } else if built_ok {
        match harness::diff(&DiffOptions {
            crate_dir,
            module: None,
            ignore_key_order: false,
            typed_only: false,
            float_tolerance: None,
            node_args,
        }) {
            Ok(d) => {
                checks.push(Check {
                    name: "api",
                    passed: d.missing.is_empty(),
                    detail: if d.missing.is_empty() {
                        format!(
                            "exports {}",
                            manifest
                                .exports
                                .iter()
                                .map(|f| f.name.clone())
                                .chain(manifest.classes.iter().map(|c| format!("class {}", c.name)))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    } else {
                        format!("missing exports: {}", d.missing.join(", "))
                    },
                });
                checks.push(Check {
                    name: "behavior",
                    passed: d.failed == 0,
                    detail: format!("{}/{} recorded cases match", d.passed, d.total),
                });
            }
            Err(e) => checks.push(Check {
                name: "behavior",
                passed: false,
                detail: format!("{e:#}"),
            }),
        }
    }

    Ok(GateOutcome {
        passed: checks.iter().all(|c| c.passed),
        checks,
    })
}

/// The checks that hold for any Rust crate, bindings or not: fmt, clippy (warnings are errors),
/// cargo test, and `unsafe` only with a `// SAFETY:` comment.
pub fn rust_checks(crate_dir: &Path, target_dir: &Path) -> Vec<Check> {
    let cargo = |args: &[&str]| -> (bool, String) {
        let mut cmd = Command::new("cargo");
        cmd.args(args)
            .arg("--manifest-path")
            .arg(crate_dir.join("Cargo.toml"))
            .env("CARGO_TARGET_DIR", target_dir);
        // Lint arguments go after `--`, so the manifest path must come before them.
        match cmd.output() {
            Ok(o) => (
                o.status.success(),
                last_lines(&String::from_utf8_lossy(&o.stderr), 15),
            ),
            Err(e) => (false, e.to_string()),
        }
    };
    let mut checks = Vec::new();

    let (ok, out) = cargo(&["fmt", "--check"]);
    checks.push(Check {
        name: "fmt",
        passed: ok,
        detail: if ok {
            "formatted".into()
        } else {
            format!("run `cargo fmt`\n{out}")
        },
    });

    let mut clippy = Command::new("cargo");
    clippy
        .args(["clippy", "--all-targets", "--manifest-path"])
        .arg(crate_dir.join("Cargo.toml"))
        .args(["--", "-D", "warnings"])
        .env("CARGO_TARGET_DIR", target_dir);
    let (ok, out) = match clippy.output() {
        Ok(o) => (
            o.status.success(),
            last_lines(&String::from_utf8_lossy(&o.stderr), 15),
        ),
        Err(e) => (false, e.to_string()),
    };
    checks.push(Check {
        name: "clippy",
        passed: ok,
        detail: if ok { "no warnings".into() } else { out },
    });

    let (ok, out) = cargo(&["test", "--quiet"]);
    checks.push(Check {
        name: "cargo test",
        passed: ok,
        detail: if ok { "passed".into() } else { out },
    });

    let (unexplained, explained) = unsafe_blocks(&crate_dir.join("src"));
    checks.push(Check {
        name: "unsafe",
        passed: unexplained.is_empty(),
        detail: if unexplained.is_empty() {
            format!("{explained} unsafe uses, each with a `// SAFETY:` comment")
        } else {
            format!(
                "unsafe without a `// SAFETY:` comment: {}",
                unexplained.join(", ")
            )
        },
    });

    checks
}

/// No function in `src/port.rs` is still `unimplemented!`.
pub fn ported_check(crate_dir: &Path) -> Check {
    let path = crate_dir.join("src/port.rs");
    let port = match std::fs::read_to_string(&path) {
        Ok(port) => port,
        Err(error) => {
            return Check {
                name: "ported",
                passed: false,
                detail: format!("cannot read {}: {error}", path.display()),
            };
        }
    };
    let todo = port.matches("unimplemented!(").count();
    Check {
        name: "ported",
        passed: todo == 0,
        detail: if todo == 0 {
            "nothing `unimplemented!` in port.rs".into()
        } else {
            format!("{todo} functions in port.rs still `unimplemented!`")
        },
    }
}

/// `unsafe` uses in Rust sources, split into those without and with a `// SAFETY:` comment just above.
fn unsafe_blocks(src: &Path) -> (Vec<String>, usize) {
    let mut unexplained = Vec::new();
    let mut explained = 0;
    let mut stack = vec![src.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            let lines: Vec<&str> = text.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                let uses_unsafe = code
                    .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .any(|word| word == "unsafe");
                if !uses_unsafe {
                    continue;
                }
                let has_safety = lines[..i]
                    .iter()
                    .rev()
                    .take_while(|l| l.trim_start().starts_with("//"))
                    .any(|l| l.contains("SAFETY:"));
                if has_safety {
                    explained += 1;
                } else {
                    unexplained.push(format!("{}:{}", path.display(), i + 1));
                }
            }
        }
    }
    (unexplained, explained)
}

fn last_lines(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::unsafe_blocks;

    #[test]
    fn requires_safety_comments() {
        let dir = std::env::temp_dir().join(format!("rust-anything-gate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("a.rs"),
            "#![forbid(unsafe_code)]\n// SAFETY: checked above\nunsafe { x() }\nlet y = unsafe { z() };\n// unsafe in a comment\n",
        )
        .unwrap();
        let (unexplained, explained) = unsafe_blocks(&dir);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(explained, 1);
        assert_eq!(unexplained.len(), 1);
        assert!(unexplained[0].ends_with("a.rs:4"));
    }

    #[test]
    fn missing_port_source_fails_the_ported_check() {
        let dir = crate::app::temp_dir("gate-port").unwrap();
        let check = super::ported_check(&dir);
        std::fs::remove_dir(dir).unwrap();
        assert!(!check.passed);
        assert!(check.detail.contains("cannot read"));
        assert!(check.detail.contains("port.rs"));
    }
}
