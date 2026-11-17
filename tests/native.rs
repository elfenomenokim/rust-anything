//! Bridges built for real: compile the generated crate, load it from the original language and
//! replay the recording. Run on every CI platform, since linking differs between them.

use std::path::{Path, PathBuf};
use std::process::Command;

use rust_anything::bridge::{self, Kind};
use rust_anything::harness::{self, DiffOptions, RecordOptions};

fn available(program: &str) -> bool {
    Command::new(program)
        .arg(if program == "go" {
            "version"
        } else {
            "--version"
        })
        .output()
        .is_ok_and(|o| o.status.success())
}

fn project(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "rust-anything-native-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    for (path, text) in files {
        let path = root.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    root
}

/// Bridge `unit`, record the original, port `name` with `body`, build, and replay.
fn round_trip(root: &Path, unit: &str, kind: Kind, name: &str, body: &str) {
    let dir = root.join("port");
    bridge::bridge(&bridge::Options {
        root,
        unit,
        kind: Some(kind),
        out: Some(dir.clone()),
        name: None,
        force: false,
        keep: Vec::new(),
        lang: None,
        batch: Vec::new(),
    })
    .unwrap();
    let recorded = harness::record(&RecordOptions {
        crate_dir: &dir,
        count: 30,
        typed_only: false,
        inputs: None,
        module: None,
        resolve_from: None,
        node_args: &[],
    })
    .unwrap();
    assert!(recorded.cases > 0, "{recorded:?}");

    let port_path = dir.join("src/port.rs");
    let port = std::fs::read_to_string(&port_path).unwrap();
    let start = port
        .find(&format!("pub fn {name}("))
        .unwrap_or_else(|| panic!("no {name} in\n{port}"));
    let open = start + port[start..].find('{').unwrap();
    let close = open + port[open..].find("\n}").unwrap();
    let ported = format!("{}{{\n    {body}{}", &port[..open], &port[close..]);
    std::fs::write(&port_path, ported).unwrap();

    harness::build(&dir).unwrap();
    let outcome = harness::diff(&DiffOptions {
        crate_dir: &dir,
        module: None,
        ignore_key_order: false,
        typed_only: false,
        node_args: &[],
        float_tolerance: None,
    })
    .unwrap();
    assert_eq!(
        (outcome.failed, outcome.passed),
        (0, outcome.total),
        "{:#?}",
        outcome.mismatches
    );

    std::fs::write(&port_path, std::fs::read(&port_path).unwrap()).unwrap();
    let stale = harness::diff(&DiffOptions {
        crate_dir: &dir,
        module: None,
        ignore_key_order: false,
        typed_only: false,
        node_args: &[],
        float_tolerance: None,
    })
    .unwrap_err();
    assert!(
        format!("{stale:#}").contains("rust-anything build"),
        "{stale:#}"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn napi_addon_builds_loads_and_matches() {
    if !available("node") {
        eprintln!("skipped: node not found");
        return;
    }
    let root = project(
        "napi",
        &[
            ("package.json", r#"{"name":"shout","version":"0.1.0"}"#),
            (
                "src/shout.ts",
                "export function shout(text: string): string {\n  return text + \"!\";\n}\n",
            ),
        ],
    );
    round_trip(
        &root,
        "src/shout.ts",
        Kind::Napi,
        "shout",
        "Ok(format!(\"{text}!\"))",
    );
}

#[test]
fn pyo3_extension_builds_imports_and_matches() {
    if !available("python3") {
        eprintln!("skipped: python3 not found");
        return;
    }
    let root = project(
        "pyo3",
        &[
            (
                "pyproject.toml",
                "[project]\nname = \"shout\"\nversion = \"0.1.0\"\n",
            ),
            (
                "shout/__init__.py",
                "def shout(text: str) -> str:\n    return text + \"!\"\n",
            ),
        ],
    );
    round_trip(
        &root,
        "shout/",
        Kind::Pyo3,
        "shout",
        "Ok(format!(\"{text}!\"))",
    );
}

/// A freshly bridged crate is all `unimplemented!` stubs: replaying it must pass no case, even
/// where the original throws the same kind of error a panic turns into.
fn stubs_fail(root: &Path, unit: &str, kind: Option<Kind>, program: &str) {
    if !available(program) {
        eprintln!("skipped: {program} not found");
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "rust-anything-stubs-{}-{}",
        root.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    bridge::bridge(&bridge::Options {
        root,
        unit,
        kind,
        out: Some(dir.clone()),
        name: None,
        force: false,
        keep: Vec::new(),
        lang: None,
        batch: Vec::new(),
    })
    .unwrap();
    let recorded = harness::record(&RecordOptions {
        crate_dir: &dir,
        count: 20,
        typed_only: false,
        inputs: None,
        module: None,
        resolve_from: None,
        node_args: &[],
    })
    .unwrap();
    assert!(
        recorded.errors > 0,
        "{}: no recorded case throws, so this proves nothing",
        root.display()
    );
    harness::build(&dir).unwrap();
    let outcome = harness::diff(&DiffOptions {
        crate_dir: &dir,
        module: None,
        ignore_key_order: false,
        typed_only: false,
        node_args: &[],
        float_tolerance: None,
    })
    .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(outcome.passed, 0, "a stub matched {outcome:#?}");
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn unported_stubs_never_match_in_typescript() {
    stubs_fail(
        &fixture("classes"),
        "src/stats.ts",
        Some(Kind::Napi),
        "node",
    );
}

#[test]
fn unported_stubs_never_match_in_python() {
    stubs_fail(&fixture("python"), "src/textkit/", None, "python3");
}

#[test]
fn unported_stubs_never_match_a_ruby_runtime_error() {
    let root = project(
        "stubs-ruby",
        &[
            ("shout.gemspec", ""),
            (
                "lib/shout.rb",
                "module Shout\n  def self.shout(s)\n    raise \"empty\" if s.empty?\n    s + \"!\"\n  end\nend\n",
            ),
        ],
    );
    stubs_fail(&root, "lib/shout.rb", None, "ruby");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn unported_stubs_never_match_a_java_runtime_exception() {
    let root = project(
        "stubs-java",
        &[
            (
                "pom.xml",
                "<project xmlns=\"http://maven.apache.org/POM/4.0.0\">\n  <modelVersion>4.0.0</modelVersion>\n  <groupId>com.example</groupId>\n  <artifactId>shout</artifactId>\n  <version>0.1.0</version>\n</project>\n",
            ),
            (
                "src/main/java/com/example/shout/Shout.java",
                "package com.example.shout;\n\npublic final class Shout {\n    public static String shout(String s) {\n        if (s.isEmpty()) {\n            throw new RuntimeException(\"empty\");\n        }\n        return s + \"!\";\n    }\n}\n",
            ),
        ],
    );
    stubs_fail(&root, "src/main/java/com/example/shout/", None, "java");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn unported_stubs_never_match_a_go_panic() {
    let root = project(
        "stubs-go",
        &[
            ("go.mod", "module example.com/shout\n\ngo 1.22\n"),
            (
                "shout/shout.go",
                "package shout\n\nfunc Shout(s string) string {\n\tif s == \"\" {\n\t\tpanic(\"empty\")\n\t}\n\treturn s + \"!\"\n}\n",
            ),
        ],
    );
    stubs_fail(&root, "shout/", None, "go");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn napi_errors_keep_their_class_and_async_exports_reject() {
    if !available("node") {
        eprintln!("skipped: node not found");
        return;
    }
    let root = project(
        "napi-errors",
        &[
            ("package.json", r#"{"name":"guard","version":"0.1.0"}"#),
            (
                "src/guard.ts",
                "export function check(text: string): string {\n  if (text === '') throw new TypeError('empty');\n  return text;\n}\n\
                 export async function later(text: string): Promise<string> {\n  return text;\n}\n",
            ),
        ],
    );
    let dir = root.join("port");
    bridge::bridge(&bridge::Options {
        root: &root,
        unit: "src/guard.ts",
        kind: Some(Kind::Napi),
        out: Some(dir.clone()),
        name: None,
        force: false,
        keep: Vec::new(),
        lang: None,
        batch: Vec::new(),
    })
    .unwrap();
    let port = std::fs::read_to_string(dir.join("src/port.rs")).unwrap();
    let port = port
        .replacen(
            "unimplemented!(\"port `check` from src/guard.ts\")",
            "if text.is_empty() { return Err(\"TypeError: empty\".into()); }\n    Ok(text)",
            1,
        )
        .replacen(
            "unimplemented!(\"port `later` from src/guard.ts\")",
            "panic!(\"boom\")",
            1,
        );
    std::fs::write(dir.join("src/port.rs"), &port).unwrap();
    harness::build(&dir).unwrap();
    let script = r#"
const m = require(process.argv[1])
const out = []
try { m.check('') } catch (e) { out.push(e instanceof TypeError, e.message) }
try { m.check('a\uD800') } catch (e) { out.push(e instanceof TypeError && /lone surrogate/.test(e.message)) }
let threw = false
const p = (() => { try { return m.later(5) } catch { threw = true } })()
out.push(!threw && p instanceof Promise)
;(p ? p.catch(e => e.name) : Promise.resolve(null)).then(name => {
  out.push(name)
  return m.later('x').catch(e => e.code)
}).then(code => { out.push(code); console.log(JSON.stringify(out)) })
"#;
    let output = Command::new("node")
        .args(["-e", script])
        .arg(dir.join("index.cjs"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    std::fs::remove_dir_all(&root).unwrap();
    assert_eq!(
        stdout.trim(),
        r#"[true,"empty",true,true,"TypeError","RUST_PANIC"]"#,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn napi_carries_bytes_maps_sets_dates_and_bigints() {
    if !available("node") {
        eprintln!("skipped: node not found");
        return;
    }
    let root = project(
        "napi-types",
        &[
            ("package.json", r#"{"name":"types","version":"0.1.0"}"#),
            (
                "src/types.ts",
                "export function size(m: Map<string, number>): number { return m.size }\n\
                 export function echo(b: Uint8Array): Uint8Array { return b }\n\
                 export function big(n: bigint): bigint { return n * 2n }\n\
                 export function day(d: Date): Date { return d }\n\
                 export function keys(s: Set<string>): Set<string> { return s }\n",
            ),
        ],
    );
    let dir = root.join("port");
    bridge::bridge(&bridge::Options {
        root: &root,
        unit: "src/types.ts",
        kind: Some(Kind::Napi),
        out: Some(dir.clone()),
        name: None,
        force: false,
        keep: Vec::new(),
        lang: None,
        batch: Vec::new(),
    })
    .unwrap();
    let mut port = std::fs::read_to_string(dir.join("src/port.rs")).unwrap();
    for (name, body) in [
        (
            "size",
            "Ok(m[\"$map\"].as_array().map_or(0, Vec::len) as f64)",
        ),
        ("echo", "Ok(b)"),
        (
            "big",
            "let n: i128 = n[\"$bigint\"].as_str().ok_or(\"not a bigint\")?.parse().map_err(|e| format!(\"{e}\"))?;\n    Ok(serde_json::json!({ \"$bigint\": (n * 2).to_string() }))",
        ),
        ("day", "Ok(d)"),
        ("keys", "Ok(s)"),
    ] {
        let stub = format!("unimplemented!(\"port `{name}` from src/types.ts\")");
        assert!(port.contains(&stub), "{port}");
        port = port.replacen(&stub, body, 1);
    }
    std::fs::write(dir.join("src/port.rs"), &port).unwrap();
    harness::build(&dir).unwrap();
    let script = r#"
const m = require(process.argv[1])
const bytes = m.echo(new Uint8Array([0, 7, 255]))
const buffer = m.echo(Buffer.from('hi'))
const when = new Date('2024-02-29T12:00:00.000Z')
console.log(JSON.stringify([
  m.size(new Map([['a', 1], ['b', 2]])),
  bytes instanceof Uint8Array && [...bytes].join(','),
  Buffer.isBuffer(buffer) && buffer.toString(),
  String(m.big(9007199254740993n)),
  m.day(when) instanceof Date && m.day(when).getTime() === when.getTime(),
  [...m.keys(new Set(['x', 'y']))].join(','),
]))
"#;
    let output = Command::new("node")
        .args(["-e", script])
        .arg(dir.join("index.cjs"))
        .output()
        .unwrap();
    std::fs::remove_dir_all(&root).unwrap();
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        r#"[2,"0,7,255","hi","18014398509481986",true,"x,y"]"#,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn java_records_with_the_bridge_crate_inside_the_project() {
    if !available("javac") {
        eprintln!("skipped: javac not found");
        return;
    }
    let root = project(
        "java-default-out",
        &[
            (
                "pom.xml",
                "<project xmlns=\"http://maven.apache.org/POM/4.0.0\">\n  <modelVersion>4.0.0</modelVersion>\n  <groupId>com.example</groupId>\n  <artifactId>smoke</artifactId>\n  <version>0.1.0</version>\n</project>\n",
            ),
            (
                "src/main/java/com/example/smoke/TextKit.java",
                "package com.example.smoke;\n\npublic final class TextKit {\n    public static int wordCount(String s) {\n        return s.isBlank() ? 0 : s.trim().split(\"\\\\s+\").length;\n    }\n}\n",
            ),
        ],
    );
    let outcome = bridge::bridge(&bridge::Options {
        root: &root,
        unit: "src/main/java/com/example/smoke/",
        kind: None,
        out: None,
        name: None,
        force: false,
        keep: Vec::new(),
        lang: None,
        batch: Vec::new(),
    })
    .unwrap();
    assert!(outcome.crate_dir.ends_with("rust/smoke-rs"), "{outcome:?}");
    let recorded = harness::record(&RecordOptions {
        crate_dir: &outcome.crate_dir,
        count: 10,
        typed_only: false,
        inputs: None,
        module: None,
        resolve_from: None,
        node_args: &[],
    })
    .unwrap();
    std::fs::remove_dir_all(&root).unwrap();
    assert!(recorded.cases > 0, "{recorded:?}");
}
