//! Steps 3 and 5's runtime half: record what the original code does, build the Rust module, and
//! replay the recorded cases against it.
//!
//! Both sides run in Node through small generated scripts, so the comparison sees exactly
//! what a JS caller sees. Values are encoded so `undefined`, `NaN`, `-0`, `Map`, `Set`,
//! `BigInt` and key order survive JSON.

use crate::walk::Canonical;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::bridge::{Kind, Manifest, RType, param_type};
use crate::ts::exports::LocalType;
use crate::ts::shape::{self, Shape};

pub const CASES: &str = "cases.jsonl";
const WORK_DIR: &str = ".rust-anything";

/// Strings that break naive ports: empty, case, whitespace, non-ASCII, astral, combining.
const STRINGS: &[&str] = &[
    "",
    "a",
    "ab",
    "abc",
    "hello",
    "hello world",
    "Hello World",
    "HELLO",
    "  padded  ",
    "a.b.c",
    "ns:key.path",
    "a:b:c",
    ":",
    ".leading",
    "trailing.",
    "a..b",
    "src/a/b.ts",
    "./x/**/*.json",
    "key_name-1",
    "{name} has {count} items",
    "line1\nline2",
    "tab\tseparated",
    "ünïcödé",
    "straße",
    "İstanbul",
    "ΣΊΣΥΦΟΣ",
    "日本語のテキスト",
    "emoji 👍🏽 mix",
    "😀",
    "a😀b😀c",
    "e\u{301}",
    "\u{FEFF}bom",
];
const JSON_VALUES: &str =
    r#"[null, true, 0, "x", [], {}, {"a": "b"}, {"a": {"b": "c"}}, ["a", 1]]"#;

#[derive(Debug, Serialize)]
pub struct RecordOutcome {
    pub path: PathBuf,
    pub cases: usize,
    pub errors: usize,
    /// Cases from the semantic traps (added to the generated ones for every function).
    pub traps: usize,
    pub per_function: BTreeMap<String, usize>,
    /// Functions in files that use regexes: their string parameters also got the regex traps.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub regex_functions: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DiffOutcome {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    /// Exports the Rust module does not provide.
    pub missing: Vec<String>,
    /// The first mismatches: `{ fn, args, expected, actual }`.
    pub mismatches: Vec<Value>,
    /// Mismatches excluded by `--typed-only`: the recorded arguments cannot go through the
    /// declared typed API. They are not counted in `failed`.
    #[serde(default)]
    pub outside_typed_api: usize,
}

pub struct RecordOptions<'a> {
    pub crate_dir: &'a Path,
    /// Generated cases per function.
    pub count: usize,
    /// Only generate cases that fit the declared parameter types: skip the
    /// deliberately-unshaped probes for class-like parameters. Curated `--inputs` are kept.
    pub typed_only: bool,
    /// Extra cases, one `{"fn": name, "args": [...]}` JSON object per line.
    pub inputs: Option<&'a Path>,
    /// Import the original from this JS module instead of the TS source files.
    pub module: Option<&'a Path>,
    /// Resolve npm packages the original imports from this folder's `node_modules`.
    pub resolve_from: Option<&'a Path>,
    pub node_args: &'a [String],
}

pub fn record(opts: &RecordOptions) -> Result<RecordOutcome> {
    let manifest = Manifest::load(opts.crate_dir)?;
    let mut cases: Vec<Value> = Vec::new();
    let module_for = |file: &str| -> Result<String> {
        let path = match opts.module {
            Some(m) => m
                .canonical()
                .with_context(|| format!("resolving {}", m.display()))?,
            None => manifest.root.join(file),
        };
        Ok(path.to_string_lossy().into_owned())
    };
    let mut scopes: BTreeMap<String, Vec<LocalType>> = BTreeMap::new();
    let mut regex_files: BTreeMap<String, bool> = BTreeMap::new();
    let mut regex_functions = Vec::new();
    let mut traps = 0;
    for f in &manifest.exports {
        let module = module_for(&f.file)?;
        if !scopes.contains_key(&f.file) {
            let text = std::fs::read_to_string(manifest.root.join(&f.file)).unwrap_or_default();
            regex_files.insert(
                f.file.clone(),
                crate::risks::uses_regex(manifest.language, &text),
            );
            scopes.insert(
                f.file.clone(),
                crate::bridge::types_in_scope(&manifest.root, &f.file, &text),
            );
        }
        let regex = regex_files[&f.file];
        if regex && f.params.iter().any(|p| param_type(p) == RType::Str) {
            regex_functions.push(f.name.clone());
        }
        let types = &scopes[&f.file];
        let mut all = generate(f, opts.count, types, opts.typed_only);
        let mut seen: std::collections::HashSet<String> = all
            .iter()
            .map(|a| serde_json::to_string(a).unwrap_or_default())
            .collect();
        let base: Vec<Value> = all
            .first()
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for args in trap_args(&f.params, &base, regex) {
            let args = Value::Array(args);
            if seen.insert(serde_json::to_string(&args).unwrap_or_default()) {
                all.push(args);
                traps += 1;
            }
        }
        for (i, mut args) in all.into_iter().enumerate() {
            // Python callers pass ints where floats are expected, and int-ness can leak into
            // results: every fourth case passes integral `number` arguments as ints.
            if manifest.language == crate::lang::Lang::Python && i % 4 == 3 {
                intify(&f.params, &mut args);
            }
            let mut case = json!({ "fn": f.name, "module": module, "args": args });
            split_keyword_only(&f.params, &mut case);
            cases.push(case);
        }
    }
    for class in &manifest.classes {
        let module = module_for(&class.file)?;
        let text = std::fs::read_to_string(manifest.root.join(&class.file)).unwrap_or_default();
        let types = crate::bridge::types_in_scope(&manifest.root, &class.file, &text);
        for mut scenario in scenarios(class, opts.count.max(20), &types, opts.typed_only) {
            scenario["module"] = Value::String(module.clone());
            cases.push(scenario);
        }
    }
    if let Some(path) = opts.inputs {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        for (i, line) in text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
        {
            let mut case: Value = serde_json::from_str(line).with_context(|| {
                format!(
                    "{}:{} (values use the record codec: {{\"$undefined\": true}}, {{\"$num\": \"NaN\"}}, \
                     and {{\"$utf16\": [code units]}} for strings with lone surrogates)",
                    path.display(),
                    i + 1
                )
            })?;
            if let Some(class) = case["class"].as_str() {
                let Some(c) = manifest.classes.iter().find(|c| c.name == class) else {
                    bail!(
                        "{}:{}: `{class}` is not a ported class of this unit",
                        path.display(),
                        i + 1
                    );
                };
                case["module"] = Value::String(module_for(&c.file)?);
                cases.push(case);
                continue;
            }
            let name = case["fn"].as_str().unwrap_or_default().to_string();
            let Some(f) = manifest.exports.iter().find(|f| f.name == name) else {
                bail!(
                    "{}:{}: `{name}` is not an export of this unit",
                    path.display(),
                    i + 1
                );
            };
            case["module"] = Value::String(module_for(&f.file)?);
            cases.push(case);
        }
    }

    let out = if manifest.language == crate::lang::Lang::Python {
        crate::harness_py::record(opts.crate_dir, &cases)?
    } else if manifest.kind == Kind::CAbi {
        crate::harness_c::record(opts.crate_dir, &manifest, &cases)?
    } else if manifest.language == crate::lang::Lang::Go {
        crate::harness_go::record(opts.crate_dir, &manifest, &cases)?
    } else if manifest.language == crate::lang::Lang::Ruby {
        crate::harness_ruby::record(opts.crate_dir, &manifest, &cases)?
    } else if manifest.language == crate::lang::Lang::Java {
        crate::harness_java::record(opts.crate_dir, &manifest, &cases)?
    } else {
        record_node(opts, &manifest, &cases)?
    };
    let text = std::fs::read_to_string(&out)?;
    let mut per_function = BTreeMap::new();
    let mut errors = 0;
    let mut count = 0;
    for line in text.lines().filter(|l| !l.is_empty()) {
        let case: Value = serde_json::from_str(line)?;
        count += 1;
        let class_error = case["results"]
            .as_array()
            .is_some_and(|r| r.iter().any(|x| x.get("error").is_some()));
        errors +=
            usize::from(case.get("error").is_some() || case.get("panic").is_some() || class_error);
        let name = case["fn"]
            .as_str()
            .or_else(|| case["class"].as_str())
            .unwrap_or_default();
        *per_function.entry(name.to_string()).or_default() += 1;
    }
    Ok(RecordOutcome {
        path: out,
        cases: count,
        errors,
        traps,
        per_function,
        regex_functions,
    })
}

/// Runs the recorded inputs through Node: the original TS/JS, with the resolve hook.
fn record_node(opts: &RecordOptions, manifest: &Manifest, cases: &[Value]) -> Result<PathBuf> {
    let work = opts.crate_dir.join(WORK_DIR);
    std::fs::create_dir_all(&work)?;
    let inputs_path = work.join("inputs.json");
    std::fs::write(&inputs_path, serde_json::to_string(cases)?)?;
    let script = work.join("record.mjs");
    std::fs::write(&script, format!("{CODEC_JS}\n{RECORD_JS}"))?;
    let out = opts.crate_dir.join(CASES);
    // TS sources often import the way bundlers resolve (`./errors`, `./types`): a resolve hook
    // lets Node follow them, so the original runs as it is.
    std::fs::write(work.join("resolve.mjs"), RESOLVE_JS)?;
    std::fs::write(work.join("register.mjs"), REGISTER_JS)?;
    let register = work.canonical()?.join("register.mjs");
    let mut node_args = vec![format!("--import={}", file_url(&register))];
    node_args.extend(opts.node_args.iter().cloned());
    let aliases = alias_env(manifest);
    let mut cmd = Command::new("node");
    cmd.args(&node_args)
        .arg(&script)
        .arg(&inputs_path)
        .arg(&out)
        .env("RUST_ANYTHING_PATHS", &aliases)
        .env("RUST_ANYTHING_WORKSPACE", workspace_env(manifest));
    if let Some(dir) = opts.resolve_from {
        let dir = dir
            .canonical()
            .with_context(|| format!("resolving {}", dir.display()))?;
        cmd.env("RUST_ANYTHING_RESOLVE_FROM", dir);
    }
    if let Err(error) = run(&mut cmd, "node") {
        // Enums and parameter properties need Node to transform TS, not only strip types.
        if !format!("{error:#}").contains("ERR_UNSUPPORTED_TYPESCRIPT_SYNTAX") {
            return Err(error);
        }
        let mut retry = Command::new("node");
        retry
            .arg("--experimental-transform-types")
            .arg("--no-warnings=ExperimentalWarning")
            .args(&node_args)
            .arg(&script)
            .arg(&inputs_path)
            .arg(&out)
            .env("RUST_ANYTHING_PATHS", &aliases)
            .env("RUST_ANYTHING_WORKSPACE", workspace_env(manifest));
        if let Some(dir) = opts.resolve_from {
            retry.env("RUST_ANYTHING_RESOLVE_FROM", dir.canonical()?);
        }
        run(&mut retry, "node --experimental-transform-types")?;
    }

    Ok(out)
}

/// Copies a freshly built library into place by renaming, so a process that has the old one
/// loaded keeps a valid file.
pub(crate) fn install(built: &Path, dest: &Path) -> Result<()> {
    let tmp = dest.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::copy(built, &tmp).with_context(|| format!("copying {}", built.display()))?;
    std::fs::rename(&tmp, dest).with_context(|| format!("installing {}", dest.display()))
}

/// The library `diff` loads, when it is older than the crate's sources: a forgotten `build`
/// would otherwise test the previous port.
fn stale_build(crate_dir: &Path, manifest: &Manifest) -> Option<PathBuf> {
    let built = match manifest.kind {
        Kind::Napi => crate_dir.join(format!("{}.node", manifest.lib_name())),
        Kind::Pyo3 => crate::harness_py::native_file(crate_dir, manifest),
        _ => return None,
    };
    let built_at = std::fs::metadata(&built).and_then(|m| m.modified()).ok()?;
    let mut sources = vec![crate_dir.join("Cargo.toml")];
    sources.extend(
        walkdir_rs(&crate_dir.join("src"))
            .into_iter()
            .filter(|p| p.extension().is_some_and(|e| e == "rs")),
    );
    sources
        .iter()
        .filter_map(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
        .any(|changed| changed > built_at)
        .then_some(built)
}

fn walkdir_rs(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walkdir_rs(&path));
        } else {
            out.push(path);
        }
    }
    out
}

pub fn build(crate_dir: &Path) -> Result<PathBuf> {
    let manifest = Manifest::load(crate_dir)?;
    match manifest.kind {
        Kind::Pyo3 => crate::harness_py::build(crate_dir, &manifest),
        Kind::Ffi if manifest.language == crate::lang::Lang::Ruby => {
            crate::harness_ruby::build(crate_dir, &manifest)
        }
        Kind::CAbi | Kind::Ffi => crate::harness_c::build(crate_dir, &manifest),
        Kind::Napi => {
            let target = crate_dir.join("target");
            run(
                Command::new("cargo")
                    .args(["build", "--release", "--manifest-path"])
                    .arg(crate_dir.join("Cargo.toml"))
                    .env("CARGO_TARGET_DIR", &target),
                "cargo build",
            )?;
            let lib = manifest.lib_name();
            let release = target.join("release");
            let built = [
                format!("lib{lib}.so"),
                format!("lib{lib}.dylib"),
                format!("{lib}.dll"),
            ]
            .iter()
            .map(|name| release.join(name))
            .find(|p| p.exists())
            .with_context(|| format!("no native library for `{lib}` in {}", release.display()))?;
            let artifact = crate_dir.join(format!("{lib}.node"));
            install(&built, &artifact)?;
            Ok(artifact)
        }
        Kind::Wasm => {
            let target = crate_dir.join("target");
            run(
                Command::new("cargo")
                    .args([
                        "build",
                        "--release",
                        "--target",
                        "wasm32-unknown-unknown",
                        "--manifest-path",
                    ])
                    .arg(crate_dir.join("Cargo.toml"))
                    .env("CARGO_TARGET_DIR", &target),
                "cargo build --target wasm32-unknown-unknown (needs `rustup target add wasm32-unknown-unknown`)",
            )?;
            // wasm-bindgen's CLI must match the crate's wasm-bindgen version exactly.
            let lock = std::fs::read_to_string(crate_dir.join("Cargo.lock")).unwrap_or_default();
            let wanted = locked_version(&lock, "wasm-bindgen")
                .context("wasm-bindgen missing from Cargo.lock")?;
            let installed = Command::new("wasm-bindgen")
                .arg("--version")
                .output()
                .ok()
                .map(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .rsplit(' ')
                        .next()
                        .unwrap_or_default()
                        .to_string()
                });
            if installed.as_deref() != Some(wanted.as_str()) {
                bail!(
                    "the crate uses wasm-bindgen {wanted}; the CLI must match:\n  cargo install wasm-bindgen-cli --version {wanted} --locked{}",
                    installed
                        .map(|v| format!(" (found {v})"))
                        .unwrap_or_default()
                );
            }
            let lib = manifest.lib_name();
            let wasm = target
                .join("wasm32-unknown-unknown/release")
                .join(format!("{lib}.wasm"));
            run(
                Command::new("wasm-bindgen")
                    .args(["--target", "nodejs", "--out-dir"])
                    .arg(crate_dir.join("pkg"))
                    .arg(&wasm),
                "wasm-bindgen",
            )?;
            Ok(crate_dir.join("pkg").join(format!("{lib}.js")))
        }
    }
}

/// Per-function timing of the original against the Rust module.
/// What `bench` concludes: the geometric-mean speedup and what to do.
#[derive(Debug, Serialize)]
pub struct Verdict {
    pub speedup: f64,
    pub faster: Vec<String>,
    pub slower: Vec<String>,
    pub advice: String,
}

/// Faster means at least 1.1x, slower below 0.9x; between is a tie (timing noise).
pub fn verdict(rows: &[BenchRow], lang: crate::lang::Lang) -> Verdict {
    let faster: Vec<String> = rows
        .iter()
        .filter(|r| r.speedup >= 1.1)
        .map(|r| r.name.clone())
        .collect();
    let slower: Vec<String> = rows
        .iter()
        .filter(|r| r.speedup < 0.9)
        .map(|r| r.name.clone())
        .collect();
    let speedup = if rows.is_empty() {
        1.0
    } else {
        (rows.iter().map(|r| r.speedup.max(1e-9).ln()).sum::<f64>() / rows.len() as f64).exp()
    };
    let advice = if rows.is_empty() {
        "Nothing to compare: record cases that do not throw first.".to_string()
    } else if slower.is_empty() && faster.len() == rows.len() {
        "Ship it: every function is faster in Rust, call included.".to_string()
    } else if slower.is_empty() {
        "Ship it: no function is slower in Rust; some only tie.".to_string()
    } else if lang == crate::lang::Lang::C {
        // C links the port directly: there is no crossing to blame.
        format!(
            "No call overhead in C: these are slower as Rust code. Profile and optimize them, or keep them in C: {}.",
            slower.join(", ")
        )
    } else if speedup >= 1.1 {
        format!(
            "Ship the faster functions. Keep the slower ones in the original (`bridge --keep`) or move their callers to Rust too: {}.",
            slower.join(", ")
        )
    } else {
        "Do not ship this unit for speed: calling across costs more than the work these functions do. \
         Port a bigger unit (these functions together with their callers), so one call does more work."
            .to_string()
    };
    Verdict {
        speedup,
        faster,
        slower,
        advice,
    }
}

/// Rows from two `{ name: { cases, ns } }` timing maps: the original's and the port's.
pub fn bench_rows(
    m: &Manifest,
    original: &serde_json::Map<String, Value>,
    rust: &serde_json::Map<String, Value>,
) -> Vec<BenchRow> {
    m.exports
        .iter()
        .filter_map(|f| {
            let (old, new) = (original.get(&f.name)?, rust.get(&f.name)?);
            let (o, n) = (old["ns"].as_f64()?, new["ns"].as_f64()?);
            Some(BenchRow {
                name: f.name.clone(),
                cases: usize::try_from(old["cases"].as_u64().unwrap_or(0)).unwrap_or(0),
                original_ns: o,
                rust_ns: n,
                speedup: o / n,
            })
        })
        .collect()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchRow {
    #[serde(rename = "fn")]
    pub name: String,
    pub cases: usize,
    /// Nanoseconds per call.
    pub original_ns: f64,
    pub rust_ns: f64,
    /// `original_ns / rust_ns`: above 1 means Rust is faster, bridge included.
    pub speedup: f64,
}

pub struct BenchOptions<'a> {
    pub crate_dir: &'a Path,
    /// Minimum measured time per function and side, in milliseconds.
    pub min_ms: u64,
    pub resolve_from: Option<&'a Path>,
    pub node_args: &'a [String],
}

/// Times the recorded cases against the original and the built Rust module, bridge included.
pub fn bench(opts: &BenchOptions) -> Result<Vec<BenchRow>> {
    let manifest = Manifest::load(opts.crate_dir)?;
    if manifest.language == crate::lang::Lang::Python {
        return crate::harness_py::bench(opts.crate_dir, &manifest, opts.min_ms);
    }
    if manifest.kind == Kind::CAbi {
        return crate::harness_c::bench(opts.crate_dir, &manifest, opts.min_ms);
    }
    if manifest.language == crate::lang::Lang::Go {
        return crate::harness_go::bench(opts.crate_dir, &manifest, opts.min_ms);
    }
    if manifest.language == crate::lang::Lang::Ruby {
        return crate::harness_ruby::bench(opts.crate_dir, &manifest, opts.min_ms);
    }
    if manifest.language == crate::lang::Lang::Java {
        return crate::harness_java::bench(opts.crate_dir, &manifest, opts.min_ms);
    }
    let cases = opts.crate_dir.join(CASES);
    if !cases.exists() {
        bail!(
            "no {CASES} in {} (run `rust-anything record` first)",
            opts.crate_dir.display()
        );
    }
    let modules: BTreeMap<String, String> = manifest
        .exports
        .iter()
        .map(|f| {
            (
                f.name.clone(),
                manifest.root.join(&f.file).to_string_lossy().into_owned(),
            )
        })
        .collect();
    let work = opts.crate_dir.join(WORK_DIR);
    std::fs::create_dir_all(&work)?;
    std::fs::write(work.join("resolve.mjs"), RESOLVE_JS)?;
    std::fs::write(work.join("register.mjs"), REGISTER_JS)?;
    let class_modules: BTreeMap<String, String> = manifest
        .classes
        .iter()
        .map(|c| {
            (
                c.name.clone(),
                manifest.root.join(&c.file).to_string_lossy().into_owned(),
            )
        })
        .collect();
    let modules_path = work.join("bench-modules.json");
    std::fs::write(
        &modules_path,
        serde_json::to_string(&json!({ "functions": modules, "classes": class_modules }))?,
    )?;
    let script = work.join("bench.mjs");
    std::fs::write(&script, format!("{CODEC_JS}\n{BENCH_JS}"))?;
    let out = work.join("bench.json");
    let native = opts.crate_dir.join("index.cjs").canonical()?;
    let register = work.canonical()?.join("register.mjs");
    let mut cmd = Command::new("node");
    cmd.arg("--experimental-transform-types")
        .arg("--no-warnings")
        .arg(format!("--import={}", file_url(&register)))
        .args(opts.node_args)
        .arg(&script)
        .arg(&modules_path)
        .arg(&native)
        .arg(&cases)
        .arg(&out)
        .arg(opts.min_ms.to_string())
        .env("RUST_ANYTHING_PATHS", alias_env(&manifest))
        .env("RUST_ANYTHING_WORKSPACE", workspace_env(&manifest));
    if let Some(dir) = opts.resolve_from {
        cmd.env("RUST_ANYTHING_RESOLVE_FROM", dir.canonical()?);
    }
    run(&mut cmd, "node")?;
    Ok(serde_json::from_str(&std::fs::read_to_string(&out)?)?)
}

pub struct DiffOptions<'a> {
    pub crate_dir: &'a Path,
    /// The module to test; defaults to the crate's `index.cjs`.
    pub module: Option<&'a Path>,
    /// Skip mismatches whose recorded arguments do not fit the declared parameter types;
    /// they are reported separately as `outside_typed_api`, not as failures.
    pub typed_only: bool,
    /// Compare objects regardless of key order.
    pub ignore_key_order: bool,
    pub node_args: &'a [String],
    /// Floats match within this tolerance; the one saved by `record` when `None`.
    pub float_tolerance: Option<f64>,
}

/// The first mismatches a diff reports.
const SHOWN_MISMATCHES: usize = 20;

pub fn diff(opts: &DiffOptions) -> Result<DiffOutcome> {
    let mut outcome = diff_exact(opts)?;
    let manifest = Manifest::load(opts.crate_dir).ok();
    let saved = manifest.as_ref().and_then(|m| m.float_tolerance);
    if let Some(t) = opts.float_tolerance.or(saved) {
        let preserve_key_order = opts.module.is_some()
            || manifest.as_ref().is_none_or(|m| {
                matches!(
                    m.language,
                    crate::lang::Lang::Ts | crate::lang::Lang::Python
                )
            });
        outcome = crate::tolerance::apply(
            outcome,
            crate::tolerance::check(t)?,
            opts.ignore_key_order || !preserve_key_order,
        );
    }
    if opts.typed_only {
        apply_typed_only(opts.crate_dir, &mut outcome)?;
    }
    outcome.mismatches.truncate(SHOWN_MISMATCHES);
    anyhow::ensure!(
        outcome.total > 0,
        "no recorded cases to compare; run `rust-anything record` first"
    );
    Ok(outcome)
}

fn diff_exact(opts: &DiffOptions) -> Result<DiffOutcome> {
    if opts.module.is_none()
        && let Ok(manifest) = Manifest::load(opts.crate_dir)
        && let Some(built) = stale_build(opts.crate_dir, &manifest)
    {
        bail!(
            "{} is older than the crate's sources, so diff would test the previous port; run \
             `rust-anything build {}` first",
            built.display(),
            opts.crate_dir.display()
        );
    }
    if let Ok(manifest) = Manifest::load(opts.crate_dir)
        && manifest.language == crate::lang::Lang::Python
        && opts.module.is_none()
    {
        return crate::harness_py::diff(opts.crate_dir, &manifest, opts.ignore_key_order);
    }
    if let Ok(manifest) = Manifest::load(opts.crate_dir)
        && manifest.kind == Kind::CAbi
        && opts.module.is_none()
    {
        return crate::harness_c::diff(opts.crate_dir, &manifest);
    }
    if let Ok(manifest) = Manifest::load(opts.crate_dir)
        && manifest.language == crate::lang::Lang::Go
        && opts.module.is_none()
    {
        return crate::harness_go::diff(opts.crate_dir, &manifest);
    }
    if let Ok(manifest) = Manifest::load(opts.crate_dir)
        && manifest.language == crate::lang::Lang::Ruby
        && opts.module.is_none()
    {
        return crate::harness_ruby::diff(opts.crate_dir, &manifest);
    }
    if let Ok(manifest) = Manifest::load(opts.crate_dir)
        && manifest.language == crate::lang::Lang::Java
        && opts.module.is_none()
    {
        return crate::harness_java::diff(opts.crate_dir, &manifest);
    }
    let cases = opts.crate_dir.join(CASES);
    if !cases.exists() {
        bail!(
            "no {CASES} in {} (run `rust-anything record` first)",
            opts.crate_dir.display()
        );
    }
    let module = match opts.module {
        Some(m) => m.to_path_buf(),
        None => opts.crate_dir.join("index.cjs"),
    };
    let module = module
        .canonical()
        .with_context(|| format!("resolving {}", module.display()))?;
    let work = opts.crate_dir.join(WORK_DIR);
    std::fs::create_dir_all(&work)?;
    let script = work.join("diff.mjs");
    std::fs::write(&script, format!("{CODEC_JS}\n{DIFF_JS}"))?;
    let out = work.join("diff.json");
    let flag = if opts.ignore_key_order {
        "sort"
    } else {
        "keep"
    };
    run_node(
        opts.node_args,
        &script,
        &[
            module.as_os_str(),
            cases.as_os_str(),
            out.as_os_str(),
            flag.as_ref(),
        ],
    )?;
    let text = std::fs::read_to_string(&out)?;
    Ok(serde_json::from_str(&text)?)
}

/// The version of `package` in a Cargo.lock.
fn locked_version(lock: &str, package: &str) -> Option<String> {
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line.trim() == format!("name = \"{package}\"") {
            let version = lines
                .next()?
                .trim()
                .strip_prefix("version = \"")?
                .strip_suffix('"')?;
            return Some(version.to_string());
        }
    }
    None
}

/// Deterministic inputs for one function: edge cases first, then seeded random mixes.
pub fn generate(
    f: &crate::ts::exports::ExportedFn,
    count: usize,
    types: &[LocalType],
    typed_only: bool,
) -> Vec<Value> {
    let mut rng = Rng::new(f.name.bytes().fold(0x9E37_79B9_7F4A_7C15, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01B3)
    }));
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let mut attempts = 0;
    while out.len() < count && attempts < count * 20 {
        attempts += 1;
        let args = args_for(&f.params, types, &mut rng, typed_only);
        let key = serde_json::to_string(&args).unwrap_or_default();
        if seen.insert(key) {
            out.push(Value::Array(args));
        }
    }
    out
}

/// The semantic traps for a value of type `t`: what most often behaves differently once ported
/// (string length and case, signed division and overflow, float printing, map and sort order).
fn trap_values(t: &RType) -> Vec<Value> {
    match t {
        RType::Str => crate::app::STRING_TRAPS
            .iter()
            .chain(["ΣΊΣΥΦΟΣ", "a😀b", "\u{FEFF}bom"].iter())
            .map(|s| json!(s))
            .collect(),
        RType::Int => [
            -7,
            -1,
            0,
            2_147_483_648,
            -2_147_483_649,
            9_007_199_254_740_993,
            i64::MAX,
            i64::MIN,
        ]
        .iter()
        .map(|n| json!(n))
        .collect(),
        RType::F64 => {
            let mut out: Vec<Value> = [
                -7.5,
                0.1,
                0.30000000000000004,
                1e21,
                1e-7,
                5e-324,
                1.797_693_134_862_315_7e308,
                9_007_199_254_740_992.0,
            ]
            .iter()
            .map(|n| json!(n))
            .collect();
            out.push(json!({ "$num": "-0" }));
            out
        }
        RType::Vec(inner) => match inner.as_ref() {
            RType::Str => vec![json!([]), json!(["b", "a", "B", "é", "10", "9", "a"])],
            RType::Int => vec![json!([]), json!([-7, 3, 0, i64::MAX, -1])],
            RType::F64 => vec![json!([]), json!([0.1, 0.2, -0.5, 1e21])],
            _ => vec![json!([])],
        },
        RType::Opt(inner) => {
            let mut out = vec![json!({ "$undefined": true })];
            out.extend(trap_values(inner));
            out
        }
        RType::Json => crate::app::JSON_TRAPS
            .iter()
            .filter_map(|t| serde_json::from_str(t).ok())
            .chain([json!([3, 1, 2]), json!({})])
            .collect(),
        RType::Bool | RType::Unit => Vec::new(),
    }
}

/// Every trap value in every parameter, one at a time, the other arguments as in `base`.
/// Structured parameters (objects with a known shape) are left as generated.
pub fn trap_args(
    params: &[crate::ts::exports::Param],
    base: &[Value],
    regex: bool,
) -> Vec<Vec<Value>> {
    let mut out = Vec::new();
    if base.len() < params.iter().filter(|p| !p.rest).count() {
        return out;
    }
    for (i, p) in params.iter().enumerate() {
        if p.rest || i >= base.len() {
            continue;
        }
        let t = param_type(p);
        let structured = t.contains_json() && !matches!(t, RType::Json);
        let any = matches!(p.ts_type.trim(), "any" | "unknown" | "object" | "Object");
        if structured || (matches!(t, RType::Json) && !any) {
            continue;
        }
        let mut values = trap_values(&t);
        if regex && t == RType::Str {
            values.extend(crate::risks::REGEX_TRAPS.iter().map(|s| json!(s)));
        }
        for v in values {
            if base[i] != v {
                let mut args = base.to_vec();
                args[i] = v;
                out.push(args);
            }
        }
    }
    out
}

/// One argument list for `params`: values that fit, edge cases, and sometimes omitted optionals.
fn args_for(
    params: &[crate::ts::exports::Param],
    types: &[LocalType],
    rng: &mut Rng,
    typed_only: bool,
) -> Vec<Value> {
    let mut args: Vec<Value> = Vec::new();
    for p in params {
        let t = param_type(p);
        // Structured parameters: values that fit the TS type, and now and then a value
        // that does not, so error paths get recorded too. `--typed-only` skips those
        // unshaped probes for class-like parameters: no recorded instance can exist for
        // them, so they only produce argument-shape noise against a typed port.
        let structured = |rng: &mut Rng| {
            if rng.below(100) < 15 && !(typed_only && is_class_like(&p.ts_type, types)) {
                value(&RType::Json, rng)
            } else {
                shaped(&shape::parse(&p.ts_type), types, rng, 0)
            }
        };
        if p.rest {
            let inner = match t {
                RType::Vec(inner) => *inner,
                other => other,
            };
            for _ in 0..rng.below(3) {
                args.push(value(&inner, rng));
            }
        } else if t.contains_json() {
            let v = structured(rng);
            args.push(if p.optional && rng.below(5) == 0 {
                json!({ "$undefined": true })
            } else {
                v
            });
        } else {
            args.push(value(&t, rng));
        }
    }
    // Search-like functions rarely hit on random text: often derive one string from another.
    let strings: Vec<usize> = (0..args.len()).filter(|&i| args[i].is_string()).collect();
    if strings.len() >= 2 && rng.below(10) < 5 {
        let from = strings[rng.below(strings.len())];
        let to = strings[rng.below(strings.len())];
        if from != to {
            let source: Vec<char> = args[from].as_str().unwrap_or_default().chars().collect();
            if !source.is_empty() {
                let start = rng.below(source.len());
                let end = start + 1 + rng.below(source.len() - start);
                let mut piece: String = source[start..end].iter().collect();
                if rng.below(4) == 0 {
                    piece = piece.to_uppercase();
                }
                args[to] = Value::String(piece);
            }
        }
    }
    // Trailing optional arguments: sometimes omit them, the way JS callers do.
    while args.last().is_some_and(|a| a.get("$undefined").is_some()) && rng.below(2) == 0 {
        args.pop();
    }
    args
}

/// Class scenarios: construct, then call a random sequence of methods.
pub fn scenarios(
    class: &crate::ts::exports::ExportedClass,
    count: usize,
    types: &[LocalType],
    typed_only: bool,
) -> Vec<Value> {
    let mut rng = Rng::new(class.name.bytes().fold(0x2545_F491_4F6C_DD1D, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01B3)
    }));
    (0..count)
        .map(|_| {
            let ctor = args_for(&class.constructor, types, &mut rng, typed_only);
            let steps = if class.methods.is_empty() {
                0
            } else {
                1 + rng.below(12)
            };
            // Stateful classes need calls that meet again (`set(k)` then `get(k)`): string
            // arguments often reuse a string seen earlier in the scenario.
            let mut seen: Vec<Value> = ctor.iter().filter(|v| v.is_string()).cloned().collect();
            let calls: Vec<Value> = (0..steps)
                .map(|_| {
                    let m = &class.methods[rng.below(class.methods.len())];
                    let mut args = args_for(&m.params, types, &mut rng, typed_only);
                    for arg in &mut args {
                        if arg.is_string() {
                            if !seen.is_empty() && rng.below(2) == 0 {
                                *arg = seen[rng.below(seen.len())].clone();
                            } else {
                                seen.push(arg.clone());
                            }
                        }
                    }
                    json!({ "method": m.name, "args": args })
                })
                .collect();
            json!({ "class": class.name, "ctor": ctor, "calls": calls })
        })
        .collect()
}

// ---- typed-only: cases outside the declared typed API -------------------------------------
//
// For a typed production API most mismatches come from argument shapes the port's typed
// signature can never receive (a list where a dataclass belongs, a string where a mapping
// belongs). `--typed-only` splits that noise off: `record` skips the deliberately-unshaped
// probes for class-like parameters, and `diff` excludes mismatches whose recorded arguments
// do not fit the declared parameter types, reporting them separately as `outside_typed_api`
// instead of failures. Both default off: without the flag every recorded case counts.
//
// The checks are conservative on purpose: doubt keeps the case. Only what provably cannot go
// through the typed API is skipped.

/// Words that never name a class-like: primitives, `any`/`unknown`, and the container
/// spellings the language plugins produce (`list`, `dict`, `Record<string, ...>`).
const NON_CLASS_WORDS: &[&str] = &[
    "string",
    "number",
    "int",
    "boolean",
    "void",
    "undefined",
    "null",
    "unknown",
    "any",
    "object",
    "Object",
    "Record",
    "Array",
    "ReadonlyArray",
    "str",
    "float",
    "bool",
    "bytes",
    "bytearray",
    "list",
    "dict",
    "tuple",
    "set",
    "frozenset",
    "None",
    "NoneType",
];

/// Union members at the top level, without splitting inside brackets.
fn top_union_parts(ts_type: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in ts_type.char_indices() {
        match c {
            '<' | '(' | '[' | '{' => depth += 1,
            '>' | ')' | ']' | '}' => depth -= 1,
            '|' if depth == 0 => {
                out.push(ts_type[start..i].trim());
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    out.push(ts_type[start..].trim());
    out.retain(|p| !p.is_empty());
    out
}

fn is_word(part: &str) -> bool {
    !part.is_empty()
        && part
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '$')
}

/// Whether a declared parameter type names a class-like (an interface, a dataclass, or any
/// other named model): the one case where generated values cannot be instances, so the
/// unshaped probes only produce argument-shape noise against a typed port.
fn is_class_like(ts_type: &str, types: &[LocalType]) -> bool {
    is_class_like_in(ts_type, types, 0)
}

fn named_is_class_like(name: &str, types: &[LocalType], depth: usize) -> bool {
    if depth > 8 || !is_word(name) || NON_CLASS_WORDS.contains(&name) {
        return false;
    }
    match types.iter().find(|t| t.name == name) {
        Some(t) if t.alias => is_class_like_in(&t.text, types, depth + 1),
        Some(_) => true,
        // Named but unresolvable here (a Python dataclass, a re-exported type).
        None => true,
    }
}

fn is_class_like_in(ts_type: &str, types: &[LocalType], depth: usize) -> bool {
    if depth > 8 {
        return false;
    }
    for part in top_union_parts(ts_type) {
        let part = part.trim();
        if matches!(part, "undefined" | "null" | "void") {
            continue;
        }
        // Arrays of a class-like are class-like: the elements are instances.
        if let Some(inner) = part.strip_suffix("[]").or_else(|| {
            part.strip_prefix("Array<")
                .and_then(|t| t.strip_suffix('>'))
                .or_else(|| {
                    part.strip_prefix("ReadonlyArray<")
                        .and_then(|t| t.strip_suffix('>'))
                })
        }) {
            if named_is_class_like(inner.trim(), types, depth + 1) {
                return true;
            }
            continue;
        }
        if named_is_class_like(part, types, depth + 1) {
            return true;
        }
    }
    false
}

/// A record-codec marker (`{"$num": ...}`, `{"$tuple": ...}`), if that is what this is.
fn marker_of(arg: &Value) -> Option<&str> {
    let obj = arg.as_object()?;
    if obj.len() != 1 {
        return None;
    }
    let (key, _) = obj.iter().next()?;
    key.starts_with('$').then_some(key.as_str())
}

fn is_undefined(arg: &Value) -> bool {
    marker_of(arg) == Some("$undefined")
}

/// A plain object: for a structural type, not an instance or scalar marker.
fn is_plain_object(arg: &Value) -> bool {
    arg.is_object() && marker_of(arg).is_none()
}

/// Whether one recorded argument fits one declared type member. Conservative: anything this
/// function does not model fits.
fn part_fits(
    part: &str,
    optional: bool,
    arg: &Value,
    types: &[LocalType],
    lang: crate::lang::Lang,
    depth: usize,
) -> bool {
    if depth > 8 {
        return true;
    }
    match part {
        "string" => arg.is_string(),
        "number" | "int" => {
            arg.is_number() || matches!(marker_of(arg), Some("$num" | "$float" | "$int"))
        }
        "boolean" => arg.is_boolean(),
        "unknown" | "any" | "object" | "Object" => true,
        _ => {
            if part.ends_with("[]")
                || part.starts_with("Array<")
                || part.starts_with("ReadonlyArray<")
            {
                // A tuple crosses the Python bridge as a sequence too.
                return arg.is_array() || marker_of(arg) == Some("$tuple");
            }
            if part.starts_with("Record<") {
                return arg.is_object();
            }
            if !is_word(part) || NON_CLASS_WORDS.contains(&part) {
                // Literals, inline objects, builtins, callables: not modeled, fits.
                return true;
            }
            match types.iter().find(|t| t.name == part) {
                Some(t) if t.alias => arg_fits_in(&t.text, optional, arg, types, lang, depth),
                Some(_) => {
                    // A declared interface: the value is a plain object.
                    is_plain_object(arg)
                }
                None if lang == crate::lang::Lang::Python => {
                    // A Python dataclass or other named model is nominal: only an actual
                    // instance (recorded as `$object` under its class name) fits. A dict
                    // or list may convert through the bridge, but the original rejects it,
                    // so there is no defined behavior to compare.
                    arg.as_object()
                        .is_some_and(|o| o.get("$object").is_some_and(|n| n.as_str() == Some(part)))
                }
                // Other languages are structural here (a TS class instance records as a
                // plain object): without scope information, the case stays.
                None => true,
            }
        }
    }
}

/// Whether one recorded argument fits a declared parameter (record codec).
fn arg_fits(
    ts_type: &str,
    optional: bool,
    arg: &Value,
    types: &[LocalType],
    lang: crate::lang::Lang,
) -> bool {
    arg_fits_in(ts_type, optional, arg, types, lang, 0)
}

fn arg_fits_in(
    ts_type: &str,
    optional: bool,
    arg: &Value,
    types: &[LocalType],
    lang: crate::lang::Lang,
    depth: usize,
) -> bool {
    if depth > 8 {
        return true;
    }
    if is_undefined(arg) {
        return optional;
    }
    top_union_parts(ts_type).iter().any(|p| {
        let p = p.trim();
        if matches!(p, "undefined" | "null" | "void") {
            return false;
        }
        part_fits(p, optional, arg, types, lang, depth + 1)
    })
}

/// Whether a whole recorded case can go through the declared typed API: every argument
/// fits its parameter. Conservative: doubt keeps the case.
fn case_fits(
    f: &crate::ts::exports::ExportedFn,
    types: &[LocalType],
    lang: crate::lang::Lang,
    args: &[Value],
    kwargs: &serde_json::Map<String, Value>,
) -> bool {
    use crate::ts::exports::Param;
    let fits = |p: &Param, v: &Value| arg_fits(&p.ts_type, p.optional, v, types, lang);
    let mut pos = 0;
    let mut saw_rest = false;
    for p in &f.params {
        if p.rest {
            saw_rest = true;
            break;
        }
        if let Some(kw) = kwargs.get(&p.name)
            && !fits(p, kw)
        {
            return false;
        }
        if pos < args.len() {
            // A duplicated positional is invalid on both sides (both raise); it only
            // matters here when the case is otherwise well-formed.
            if kwargs.contains_key(&p.name) {
                pos += 1;
                continue;
            }
            if !fits(p, &args[pos]) {
                return false;
            }
            pos += 1;
        } else if !kwargs.contains_key(&p.name) && !p.optional {
            // A missing required argument: the declared signature rejects the call.
            return false;
        }
    }
    // Extra positionals the declared signature does not take.
    !(!saw_rest && pos < args.len())
}

fn types_for<'a>(
    cache: &'a mut BTreeMap<String, Vec<LocalType>>,
    manifest: &Manifest,
    file: &str,
) -> &'a [LocalType] {
    if !cache.contains_key(file) {
        let text = std::fs::read_to_string(manifest.root.join(file)).unwrap_or_default();
        let types = crate::bridge::types_in_scope(&manifest.root, file, &text);
        cache.insert(file.to_string(), types);
    }
    &cache[file]
}

/// Reclassifies mismatches whose recorded arguments cannot go through the declared typed
/// API (`diff --typed-only`): they leave `failed` and are counted in `outside_typed_api`.
/// Class cases and ambiguous lookups stay failures: doubt keeps the case.
fn apply_typed_only(crate_dir: &Path, outcome: &mut DiffOutcome) -> Result<()> {
    if outcome.mismatches.is_empty() {
        return Ok(());
    }
    let manifest = Manifest::load(crate_dir)?;
    let lang = manifest.language;
    let text = std::fs::read_to_string(crate_dir.join(CASES)).unwrap_or_default();
    // Keyword arguments are not repeated in the mismatch entries: recover the full case,
    // and only when the lookup is unambiguous.
    let mut cases: Vec<Value> = Vec::new();
    let mut by_call: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(case) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if case.get("class").is_some() {
            continue;
        }
        let name = case["fn"].as_str().unwrap_or_default();
        let key = format!(
            "{name}\x00{}",
            case.get("args").map_or_else(String::new, Value::to_string)
        );
        by_call.entry(key).or_default().push(cases.len());
        cases.push(case);
    }
    let mut type_cache: BTreeMap<String, Vec<LocalType>> = BTreeMap::new();
    let mut kept = Vec::with_capacity(outcome.mismatches.len());
    let mut outside = 0;
    for m in outcome.mismatches.drain(..) {
        let name = m["fn"].as_str().unwrap_or_default();
        let is_class = manifest.classes.iter().any(|c| c.name == name);
        let export = manifest.exports.iter().find(|e| e.name == name);
        let (Some(f), false) = (export, is_class) else {
            kept.push(m);
            continue;
        };
        let args = m["args"].as_array().cloned().unwrap_or_default();
        let key = format!("{name}\x00{}", Value::Array(args));
        let full = match by_call.get(&key).map(Vec::as_slice) {
            Some(&[only]) => Some(&cases[only]),
            _ => None,
        };
        let Some(full) = full else {
            kept.push(m);
            continue;
        };
        let full_args = full["args"].as_array().cloned().unwrap_or_default();
        let kwargs = full["kwargs"].as_object().cloned().unwrap_or_default();
        let types = types_for(&mut type_cache, &manifest, &f.file).to_vec();
        if case_fits(f, &types, lang, &full_args, &kwargs) {
            kept.push(m);
        } else {
            outside += 1;
        }
    }
    outcome.mismatches = kept;
    outcome.failed = outcome.failed.saturating_sub(outside);
    outcome.outside_typed_api += outside;
    Ok(())
}

/// Property names for generated objects: plain, nested-looking, integer-like (JS orders those
/// first), non-ASCII.
const KEYS: &[&str] = &[
    "a", "b", "nav", "home", "title", "2", "10", "ünï", "😀", "a.b",
];

/// A value that fits a TS shape. Named types resolve against `types`; recursion stops at a depth.
fn shaped(s: &Shape, types: &[LocalType], rng: &mut Rng, depth: usize) -> Value {
    match s {
        Shape::Str => value(&RType::Str, rng),
        Shape::Num => value(&RType::F64, rng),
        Shape::Int => value(&RType::Int, rng),
        Shape::Bool => value(&RType::Bool, rng),
        Shape::Null => Value::Null,
        Shape::Undefined => json!({ "$undefined": true }),
        Shape::LitStr(text) => Value::String(text.clone()),
        Shape::LitNum(n) => json!(n),
        Shape::LitBool(b) => Value::Bool(*b),
        Shape::Union(items) if !items.is_empty() => {
            shaped(&items[rng.below(items.len())], types, rng, depth)
        }
        Shape::Array(item) if depth < 6 => Value::Array(
            (0..rng.below(4))
                .map(|_| shaped(item, types, rng, depth + 1))
                .collect(),
        ),
        Shape::Tuple(items) => Value::Array(
            items
                .iter()
                .map(|i| shaped(i, types, rng, depth + 1))
                .collect(),
        ),
        Shape::Object(props, index) if depth < 6 => {
            let mut out = serde_json::Map::new();
            for (name, optional, prop) in props {
                if *optional && rng.below(4) == 0 {
                    continue;
                }
                out.insert(name.clone(), shaped(prop, types, rng, depth + 1));
            }
            if let Some(item) = index {
                for _ in 0..rng.below(4) {
                    out.insert(
                        KEYS[rng.below(KEYS.len())].to_string(),
                        shaped(item, types, rng, depth + 1),
                    );
                }
            }
            Value::Object(out)
        }
        Shape::Ref(name, args) if depth < 6 => {
            shaped(&shape::resolve(name, args, types), types, rng, depth + 1)
        }
        Shape::Any => value(&RType::Json, rng),
        // Too deep, or empty: a leaf ends the recursion.
        _ => Value::Null,
    }
}

fn value(t: &RType, rng: &mut Rng) -> Value {
    match t {
        RType::Str => {
            let a = STRINGS[rng.below(STRINGS.len())];
            if rng.below(5) == 0 {
                Value::String(format!("{a}{}", STRINGS[rng.below(STRINGS.len())]))
            } else {
                Value::String(a.into())
            }
        }
        RType::Int => {
            // Small, signed, and the edges of 32 and 64 bits.
            let n: [i64; 12] = [
                0,
                1,
                -1,
                2,
                7,
                42,
                100,
                -100,
                255,
                i64::from(i32::MAX),
                i64::from(i32::MIN),
                i64::MAX,
            ];
            json!(n[rng.below(n.len())])
        }
        RType::F64 => {
            let n = [
                0.0,
                1.0,
                -1.0,
                2.0,
                3.0,
                10.0,
                0.5,
                -2.5,
                42.0,
                100.0,
                1000.0,
                1e6,
                2_147_483_648.0,
                9_007_199_254_740_992.0,
            ];
            match rng.below(n.len() + 4) {
                i if i < n.len() => json!(n[i]),
                i if i == n.len() => json!({ "$num": "-0" }),
                i if i == n.len() + 1 => json!({ "$num": "NaN" }),
                i if i == n.len() + 2 => json!({ "$num": "Infinity" }),
                _ => json!({ "$num": "-Infinity" }),
            }
        }
        RType::Bool => Value::Bool(rng.below(2) == 0),
        RType::Unit => json!({ "$undefined": true }),
        RType::Vec(inner) => Value::Array((0..rng.below(4)).map(|_| value(inner, rng)).collect()),
        RType::Opt(inner) => {
            if rng.below(4) == 0 {
                json!({ "$undefined": true })
            } else {
                value(inner, rng)
            }
        }
        RType::Json => {
            let values: Vec<Value> = serde_json::from_str(JSON_VALUES).unwrap_or_default();
            values
                .get(rng.below(values.len().max(1)))
                .cloned()
                .unwrap_or(Value::Null)
        }
    }
}

/// xorshift64*: deterministic, so a recording is reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        usize::try_from(x >> 33).unwrap_or(0) % n
    }
}

fn run_node(node_args: &[String], script: &Path, args: &[&std::ffi::OsStr]) -> Result<()> {
    run(
        Command::new("node").args(node_args).arg(script).args(args),
        "node",
    )
}

fn run(cmd: &mut Command, what: &str) -> Result<()> {
    let output = cmd.output().with_context(|| format!("running {what}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(25).collect();
        bail!(
            "{what} failed ({}):\n{}",
            output.status,
            tail.into_iter().rev().collect::<Vec<_>>().join("\n")
        );
    }
    Ok(())
}

/// Python keyword-only parameters (after `*`) are passed by name: they move from `args` into
/// `kwargs`, and an omitted (`undefined`) one is left out.
/// Integral float arguments of `number` parameters, as ints.
fn intify(params: &[crate::ts::exports::Param], args: &mut Value) {
    let Some(items) = args.as_array_mut() else {
        return;
    };
    for (p, v) in params.iter().zip(items.iter_mut()) {
        if p.ts_type == "number"
            && let Some(x) = v.as_f64()
            && x.fract() == 0.0
            && x.abs() < 1e15
            && !(x == 0.0 && x.is_sign_negative())
        {
            *v = json!(x as i64);
        }
    }
}

fn split_keyword_only(params: &[crate::ts::exports::Param], case: &mut Value) {
    if !params.iter().any(|p| p.keyword_only) {
        return;
    }
    let Some(args) = case["args"].as_array().cloned() else {
        return;
    };
    let mut positional = Vec::new();
    let mut kwargs = serde_json::Map::new();
    for (i, value) in args.into_iter().enumerate() {
        match params.get(i) {
            Some(p) if p.keyword_only => {
                if value.get("$undefined").is_none() {
                    kwargs.insert(p.name.clone(), value);
                }
            }
            _ => positional.push(value),
        }
    }
    case["args"] = Value::Array(positional);
    case["kwargs"] = Value::Object(kwargs);
}

/// Workspace packages as `{ name: absolute source folder }` for the resolve hook: a monorepo
/// imports its own packages by name, and their build output may not exist.
fn workspace_env(manifest: &Manifest) -> String {
    let packages: BTreeMap<String, String> = crate::ts::discover::discover(&manifest.root)
        .map(|ws| {
            ws.packages
                .into_iter()
                .filter(|p| !p.dir.is_empty())
                .map(|p| {
                    let dir = manifest.root.join(&p.dir);
                    let src = if dir.join("src").is_dir() {
                        dir.join("src")
                    } else {
                        dir
                    };
                    (p.name, src.to_string_lossy().into_owned())
                })
                .collect()
        })
        .unwrap_or_default();
    serde_json::to_string(&packages).unwrap_or_else(|_| "{}".into())
}

/// The tsconfig `paths` of the unit's files, as JSON for the resolve hook: `[[pattern, [absolute targets]]]`.
fn alias_env(manifest: &Manifest) -> String {
    let mut entries: Vec<(String, Vec<String>)> = Vec::new();
    let files = manifest
        .exports
        .iter()
        .map(|f| &f.file)
        .chain(manifest.classes.iter().map(|c| &c.file));
    for file in files {
        let dir = file.rsplit_once('/').map_or("", |(d, _)| d);
        for (pattern, targets) in
            crate::ts::tsconfig::PathAliases::nearest(&manifest.root, dir).entries
        {
            if entries.iter().all(|(p, _)| *p != pattern) {
                let absolute = targets
                    .iter()
                    .map(|t| manifest.root.join(t).to_string_lossy().into_owned())
                    .collect();
                entries.push((pattern, absolute));
            }
        }
    }
    serde_json::to_string(&entries).unwrap_or_else(|_| "[]".into())
}

fn file_url(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    if text.starts_with('/') {
        format!("file://{text}")
    } else {
        format!("file:///{text}")
    }
}

const REGISTER_JS: &str =
    "import { register } from 'node:module'\nregister('./resolve.mjs', import.meta.url)\n";

/// Node resolve hook: bundler-style relative imports, and npm packages from
/// `RUST_ANYTHING_RESOLVE_FROM` when the source tree has none installed.
const RESOLVE_JS: &str = r#"
import { existsSync, statSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'

const extra = process.env.RUST_ANYTHING_RESOLVE_FROM
// tsconfig `paths`: [[pattern, [absolute targets]]], longest prefix first.
const aliases = JSON.parse(process.env.RUST_ANYTHING_PATHS || '[]')
  .sort(([a], [b]) => b.split('*')[0].length - a.split('*')[0].length)
// Workspace packages by name: `@x18n/core` and `@x18n/core/sub` resolve to source.
const workspace = JSON.parse(process.env.RUST_ANYTHING_WORKSPACE || '{}')
const workspaceTargets = (specifier) => {
  for (const [name, src] of Object.entries(workspace)) {
    if (specifier === name)
      return [`${src}/index`]
    if (specifier.startsWith(`${name}/`))
      return [`${src}/${specifier.slice(name.length + 1)}`, `${src}/${specifier.slice(name.length + 1)}/index`]
  }
  return []
}
const aliasTargets = (specifier) => {
  for (const [pattern, targets] of aliases) {
    const [prefix, suffix = ''] = pattern.split('*')
    const exact = !pattern.includes('*')
    if (exact ? specifier === pattern : specifier.startsWith(prefix) && specifier.endsWith(suffix) && specifier.length >= prefix.length + suffix.length) {
      const star = exact ? '' : specifier.slice(prefix.length, specifier.length - suffix.length)
      return targets.map(t => t.replace('*', star))
    }
  }
  return []
}
const suffixes = ['.ts', '.tsx', '.mts', '.js', '.mjs', '/index.ts', '/index.mts', '/index.js']
const isFile = p => existsSync(p) && statSync(p).isFile()

export async function resolve(specifier, context, next) {
  // Aliases first: `@/x` is not even a valid package name for Node.
  for (const target of [...aliasTargets(specifier), ...workspaceTargets(specifier)]) {
    const stem = target.replace(/\.js$/, '')
    for (const suffix of ['', ...suffixes]) {
      if (isFile(stem + suffix))
        return next(pathToFileURL(stem + suffix).href, context)
    }
  }
  try {
    return await next(specifier, context)
  }
  catch (error) {
    const missing = error?.code === 'ERR_MODULE_NOT_FOUND' || error?.code === 'ERR_UNSUPPORTED_DIR_IMPORT'
    if (!missing || !context.parentURL)
      throw error
    if (specifier.startsWith('.') || specifier.startsWith('/')) {
      const stem = fileURLToPath(new URL(specifier, context.parentURL)).replace(/\.js$/, '')
      for (const suffix of suffixes) {
        if (isFile(stem + suffix))
          return next(pathToFileURL(stem + suffix).href, context)
      }
    }
    else if (extra) {
      return next(specifier, { ...context, parentURL: pathToFileURL(`${extra}/`).href })
    }
    throw error
  }
}
"#;

/// Encoding shared by the record, diff and shadow scripts. Lossless for what a JS caller
/// can observe: `undefined`, `NaN`, `-0`, BigInt, Map/Set order, key order, lone surrogates.
pub const CODEC_JS: &str = r#"
const enc = (v, stack = []) => {
  if (v === undefined) return { $undefined: true }
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return { $num: 'NaN' }
    if (!Number.isFinite(v)) return { $num: v > 0 ? 'Infinity' : '-Infinity' }
    if (Object.is(v, -0)) return { $num: '-0' }
    return v
  }
  // Strings JS split inside a surrogate pair are not valid Unicode: keep their code units.
  if (typeof v === 'string' && !v.isWellFormed()) return { $utf16: Array.from({ length: v.length }, (_, i) => v.charCodeAt(i)) }
  if (typeof v === 'bigint') return { $bigint: String(v) }
  if (typeof v === 'function') return { $function: v.name }
  if (typeof v === 'symbol') return { $symbol: String(v.description) }
  if (v === null || typeof v !== 'object') return v
  if (stack.includes(v)) return { $cycle: true }
  const next = [...stack, v]
  if (Array.isArray(v)) return v.map(x => enc(x, next))
  if (typeof Buffer !== 'undefined' && Buffer.isBuffer(v)) return { $buffer: [...v] }
  if (v instanceof Uint8Array) return { $bytes: [...v] }
  if (v instanceof Map) return { $map: [...v].map(([k, x]) => [enc(k, next), enc(x, next)]) }
  if (v instanceof Set) return { $set: [...v].map(x => enc(x, next)) }
  if (v instanceof Date) return { $date: Number.isNaN(v.getTime()) ? 'Invalid' : v.toISOString() }
  if (v instanceof RegExp) return { $regexp: String(v) }
  if (v instanceof Error) return { $error: v.message }
  return Object.fromEntries(Object.keys(v).map(k => [k, enc(v[k], next)]))
}
const dec = (v) => {
  if (Array.isArray(v)) return v.map(dec)
  if (v === null || typeof v !== 'object') return v
  if (v.$undefined) return undefined
  if ('$num' in v) return Number(v.$num)
  if ('$bigint' in v) return BigInt(v.$bigint)
  if ('$utf16' in v) return String.fromCharCode(...v.$utf16)
  if ('$map' in v) return new Map(v.$map.map(([k, x]) => [dec(k), dec(x)]))
  if ('$set' in v) return new Set(v.$set.map(dec))
  if ('$date' in v) return new Date(v.$date)
  if ('$bytes' in v) return Uint8Array.from(v.$bytes)
  if ('$buffer' in v) return Buffer.from(v.$buffer)
  return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, dec(x)]))
}
const canon = (v, sort) => {
  if (Array.isArray(v)) return v.map(x => canon(x, sort))
  if (v === null || typeof v !== 'object') return v
  const keys = Object.keys(v)
  if (sort) keys.sort()
  return Object.fromEntries(keys.map(k => [k, canon(v[k], sort)]))
}
const same = (a, b, sort = false) => JSON.stringify(canon(a, sort)) === JSON.stringify(canon(b, sort))
"#;

const RECORD_JS: &str = r#"
import { readFileSync, writeFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

const [inputsPath, outPath] = process.argv.slice(2)
const inputs = JSON.parse(readFileSync(inputsPath, 'utf8'))
const modules = new Map()
const lines = []
// A class scenario: construct, then call each method in turn; every step's result is recorded.
const runScenario = async (Class, c) => {
  const results = []
  let instance
  try {
    instance = new Class(...c.ctor.map(dec))
    results.push({ ok: true })
  }
  catch (error) {
    results.push({ error: String(error?.message ?? error) })
    return results
  }
  for (const call of c.calls) {
    try {
      results.push({ ok: enc(await instance[call.method](...call.args.map(dec))) })
    }
    catch (error) {
      results.push({ error: String(error?.message ?? error) })
    }
  }
  return results
}
for (const c of inputs) {
  let mod = modules.get(c.module)
  if (!mod) {
    mod = await import(pathToFileURL(c.module).href)
    modules.set(c.module, mod)
  }
  if (c.class) {
    const Class = mod[c.class] ?? mod.default?.[c.class]
    if (typeof Class !== 'function') {
      console.error(`${c.class} is not a class exported by ${c.module}`)
      process.exit(2)
    }
    lines.push(JSON.stringify({ class: c.class, ctor: c.ctor, calls: c.calls, results: await runScenario(Class, c) }))
    continue
  }
  const fn = mod[c.fn] ?? mod.default?.[c.fn]
  if (typeof fn !== 'function') {
    console.error(`${c.fn} is not a function exported by ${c.module}`)
    process.exit(2)
  }
  try {
    const value = await fn(...c.args.map(dec))
    lines.push(JSON.stringify({ fn: c.fn, args: c.args, ok: enc(value) }))
  }
  catch (error) {
    lines.push(JSON.stringify({ fn: c.fn, args: c.args, error: String(error?.message ?? error) }))
  }
}
writeFileSync(outPath, `${lines.join('\n')}\n`)
"#;

const BENCH_JS: &str = r#"
import { readFileSync, writeFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

const [modulesPath, nativePath, casesPath, outPath, minMs] = process.argv.slice(2)
const { functions: modules, classes: classModules } = JSON.parse(readFileSync(modulesPath, 'utf8'))
const loaded = await import(pathToFileURL(nativePath).href)
const native = { ...(loaded.default && typeof loaded.default === 'object' ? loaded.default : {}), ...loaded }
const cases = readFileSync(casesPath, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l))
const byFn = new Map()
const scenarios = new Map()
for (const c of cases.filter(c => c.class)) {
  if (!scenarios.has(c.class)) scenarios.set(c.class, [])
  scenarios.get(c.class).push(c)
}
for (const c of cases.filter(c => !c.class)) {
  if (!byFn.has(c.fn)) byFn.set(c.fn, [])
  byFn.get(c.fn).push(c.args.map(dec))
}
// Time whole passes over the cases until `minMs` has passed; errors count as calls.
const time = (fn, argsList) => {
  const pass = () => {
    for (const args of argsList) {
      try { fn(...args) } catch {}
    }
  }
  pass()
  let passes = 0
  const start = performance.now()
  do { pass(); passes++ } while (performance.now() - start < Number(minMs))
  return (performance.now() - start) * 1e6 / (passes * argsList.length)
}
const rows = []
for (const [name, argsList] of byFn) {
  const original = (await import(pathToFileURL(modules[name]).href))[name]
  const rust = native[name]
  if (typeof original !== 'function' || typeof rust !== 'function') continue
  const originalNs = time(original, argsList)
  const rustNs = time(rust, argsList)
  rows.push({ fn: name, cases: argsList.length, original_ns: originalNs, rust_ns: rustNs, speedup: originalNs / rustNs })
}
// Classes: time whole scenarios (construct, then every call).
const play = (Class, list) => () => {
  for (const c of list) {
    try {
      const instance = new Class(...c.ctor.map(dec))
      for (const call of c.calls) {
        try { instance[call.method](...call.args.map(dec)) } catch {}
      }
    } catch {}
  }
}
for (const [name, list] of scenarios) {
  const original = (await import(pathToFileURL(classModules[name]).href))[name]
  const rust = native[name]
  if (typeof original !== 'function' || typeof rust !== 'function') continue
  const timeAll = (fn) => {
    fn()
    let passes = 0
    const start = performance.now()
    do { fn(); passes++ } while (performance.now() - start < Number(minMs))
    return (performance.now() - start) * 1e6 / (passes * list.length)
  }
  const originalNs = timeAll(play(original, list))
  const rustNs = timeAll(play(rust, list))
  rows.push({ fn: `class ${name}`, cases: list.length, original_ns: originalNs, rust_ns: rustNs, speedup: originalNs / rustNs })
}
writeFileSync(outPath, JSON.stringify(rows))
"#;

const DIFF_JS: &str = r#"
import { readFileSync, writeFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

const [modulePath, casesPath, outPath, keyOrder] = process.argv.slice(2)
const loaded = await import(pathToFileURL(modulePath).href)
const mod = { ...(loaded.default && typeof loaded.default === 'object' ? loaded.default : {}), ...loaded }
const cases = readFileSync(casesPath, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l))
const summary = { total: cases.length, passed: 0, failed: 0, missing: [], mismatches: [] }
const scenario = async (Class, c) => {
  const results = []
  let instance
  try {
    instance = new Class(...c.ctor.map(dec))
    results.push({ ok: true })
  }
  catch (error) {
    results.push({ error: String(error?.message ?? error) })
    return results
  }
  for (const call of c.calls) {
    try {
      results.push({ ok: enc(await instance[call.method](...call.args.map(dec))) })
    }
    catch (error) {
      results.push({ error: String(error?.message ?? error) })
    }
  }
  return results
}
const stepsMatch = (want, got) => want.length === got.length && want.every((w, i) =>
  ('error' in w && 'error' in got[i] && !got[i].error.startsWith('Rust panic:')) || ('ok' in w && 'ok' in got[i] && same(w.ok, got[i].ok, keyOrder === 'sort')))
for (const c of cases) {
  if (c.class) {
    const Class = mod[c.class]
    if (typeof Class !== 'function') {
      if (!summary.missing.includes(c.class)) summary.missing.push(c.class)
      summary.failed++
      continue
    }
    const got = await scenario(Class, c)
    if (stepsMatch(c.results, got)) {
      summary.passed++
    }
    else {
      summary.failed++
      summary.mismatches.push({ fn: c.class, args: [c.ctor, c.calls], expected: c.results, actual: got })
    }
    continue
  }
  const fn = mod[c.fn]
  let actual
  if (typeof fn !== 'function') {
    if (!summary.missing.includes(c.fn)) summary.missing.push(c.fn)
    actual = { missing: true }
  }
  else {
    try {
      actual = { ok: enc(await fn(...c.args.map(dec))) }
    }
    catch (error) {
      actual = { error: String(error?.message ?? error) }
    }
  }
  const expected = 'error' in c ? { error: c.error } : { ok: c.ok }
  // Both sides throwing counts as agreement: error messages are not part of the contract.
  const panicked = 'error' in actual && actual.error.startsWith('Rust panic:')
  const pass = !panicked && ('error' in expected && 'error' in actual) || ('ok' in expected && 'ok' in actual && same(expected.ok, actual.ok, keyOrder === 'sort'))
  if (pass) {
    summary.passed++
  }
  else {
    summary.failed++
    summary.mismatches.push({ fn: c.fn, args: c.args, expected, actual })
  }
}
writeFileSync(outPath, JSON.stringify(summary, null, 2))
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ts::exports::{ExportedFn, Param};

    fn f(params: &[(&str, &str)]) -> ExportedFn {
        ExportedFn {
            name: "f".into(),
            file: "a.ts".into(),
            line: 1,
            params: params
                .iter()
                .map(|(n, t)| Param {
                    name: (*n).into(),
                    ts_type: (*t).into(),
                    optional: false,
                    rest: false,
                    ..Default::default()
                })
                .collect(),
            returns: "number".into(),
            is_async: false,
            is_generator: false,
            type_params: Vec::new(),
            uses_instanceof: false,
            ..Default::default()
        }
    }

    #[test]
    fn generates_deterministic_distinct_inputs() {
        let a = generate(&f(&[("q", "string"), ("t", "string")]), 50, &[], false);
        let b = generate(&f(&[("q", "string"), ("t", "string")]), 50, &[], false);
        assert_eq!(a, b);
        assert_eq!(a.len(), 50);
        let unique: std::collections::HashSet<String> = a.iter().map(|v| v.to_string()).collect();
        assert_eq!(unique.len(), 50);
        assert!(
            a.iter()
                .any(|args| args.to_string().contains('😀') || args.to_string().contains('👍'))
        );
    }

    #[test]
    fn generates_values_that_fit_object_types() {
        let types = crate::ts::exports::local_types(
            "m.ts",
            "export interface Edit { readonly range: { start: number, end: number }, readonly text: string }\n",
        );
        let cases = generate(&f(&[("edits", "readonly Edit[]")]), 60, &types, false);
        let fitting = cases
            .iter()
            .filter_map(|args| args[0].as_array())
            .flatten()
            .filter(|e| e["range"]["start"].is_number() || e["range"]["start"].is_object())
            .count();
        assert!(fitting > 20, "{fitting}");
    }

    #[test]
    fn covers_number_edge_cases() {
        let cases = generate(&f(&[("n", "number")]), 18, &[], false);
        let text = serde_json::to_string(&cases).unwrap();
        assert!(text.contains("NaN") && text.contains("-0"));
    }

    fn req(name: &str, ts: &str) -> Param {
        Param {
            name: name.into(),
            ts_type: ts.into(),
            ..Default::default()
        }
    }

    fn opt(name: &str, ts: &str) -> Param {
        Param {
            name: name.into(),
            ts_type: ts.into(),
            optional: true,
            ..Default::default()
        }
    }

    fn with_params(params: Vec<Param>) -> ExportedFn {
        let mut g = f(&[]);
        g.params = params;
        g
    }

    #[test]
    fn class_like_names_named_models() {
        let types = crate::ts::exports::local_types(
            "m.ts",
            "export interface Edit { text: string }\ntype Name = string;\n",
        );
        // A dataclass, an interface, and an array of either: generated values cannot be
        // instances, so the unshaped probes are noise under `--typed-only`.
        assert!(is_class_like("Thresholds | undefined", &[]));
        assert!(is_class_like("Edit", &types));
        assert!(is_class_like("Edit[]", &types));
        assert!(is_class_like("Array<Edit>", &types));
        // Primitives, aliases of primitives, records and unshaped unknowns stay.
        assert!(!is_class_like("string", &[]));
        assert!(!is_class_like("Name", &types));
        assert!(!is_class_like("Record<string, number>", &[]));
        assert!(!is_class_like("unknown", &[]));
        assert!(!is_class_like("string[]", &types));
    }

    #[test]
    fn typed_only_accepts_what_the_bridge_accepts() {
        let py = crate::lang::Lang::Python;
        // Primitives: the value's JSON kind must match.
        assert!(arg_fits("string", false, &json!("hi"), &[], py));
        assert!(!arg_fits("string", false, &json!(3), &[], py));
        assert!(arg_fits("int", false, &json!(3), &[], py));
        assert!(arg_fits(
            "number",
            false,
            &json!({ "$num": "NaN" }),
            &[],
            py
        ));
        assert!(!arg_fits("boolean", false, &json!(1), &[], py));
        // `None` fits only an optional parameter.
        assert!(arg_fits(
            "Thresholds | undefined",
            true,
            &json!({ "$undefined": true }),
            &[],
            py
        ));
        assert!(!arg_fits(
            "string",
            false,
            &json!({ "$undefined": true }),
            &[],
            py
        ));
        // A named model is nominal in Python: only an instance of that class fits.
        assert!(arg_fits(
            "Thresholds | undefined",
            true,
            &json!({ "$object": "Thresholds", "attrs": {} }),
            &[],
            py
        ));
        assert!(!arg_fits(
            "Thresholds | undefined",
            true,
            &json!({ "a": "b" }),
            &[],
            py
        ));
        assert!(!arg_fits(
            "Thresholds | undefined",
            true,
            &json!([]),
            &[],
            py
        ));
        assert!(!arg_fits(
            "Thresholds | undefined",
            true,
            &json!({ "$object": "Other", "attrs": {} }),
            &[],
            py
        ));
        // Untyped parameters fit everything; arrays fit sequences.
        assert!(arg_fits("unknown", false, &json!([1]), &[], py));
        assert!(arg_fits("string[]", false, &json!(["a"]), &[], py));
        assert!(!arg_fits("string[]", false, &json!(42), &[], py));
    }

    #[test]
    fn typed_only_checks_whole_calls() {
        let py = crate::lang::Lang::Python;
        let g = with_params(vec![req("text", "string"), opt("sep", "string")]);
        let no_kwargs = Default::default();
        assert!(case_fits(&g, &[], py, &[json!("hi")], &no_kwargs));
        assert!(!case_fits(&g, &[], py, &[json!(42)], &no_kwargs));
        assert!(!case_fits(&g, &[], py, &[], &no_kwargs));
        // Keyword arguments are checked against their own parameter.
        let kwargs: serde_json::Map<String, Value> =
            [("sep".to_string(), json!(3))].into_iter().collect();
        assert!(!case_fits(&g, &[], py, &[json!("hi")], &kwargs));
        let kwargs: serde_json::Map<String, Value> =
            [("sep".to_string(), json!("-"))].into_iter().collect();
        assert!(case_fits(&g, &[], py, &[json!("hi")], &kwargs));
    }

    #[test]
    fn typed_only_generation_skips_unshaped_class_probes() {
        let types =
            crate::ts::exports::local_types("m.ts", "export interface Edit { text: string }\n");
        let g = f(&[("edits", "Edit[]")]);
        let typed = generate(&g, 60, &types, true);
        assert_eq!(typed.len(), 60);
        assert!(
            typed.iter().all(|args| args[0].is_array()),
            "{}",
            serde_json::to_string(&typed).unwrap_or_default()
        );
    }
}

#[cfg(test)]
mod verdict_tests {
    use super::*;

    fn row(name: &str, speedup: f64) -> BenchRow {
        BenchRow {
            name: name.into(),
            cases: 1,
            original_ns: speedup,
            rust_ns: 1.0,
            speedup,
        }
    }

    #[test]
    fn advice_follows_the_numbers() {
        let ts = crate::lang::Lang::Ts;
        assert!(
            verdict(&[row("a", 3.0), row("b", 2.0)], ts)
                .advice
                .starts_with("Ship it: every")
        );
        assert!(
            verdict(&[row("a", 3.0), row("b", 1.0)], ts)
                .advice
                .starts_with("Ship it: no function")
        );
        let mixed = verdict(&[row("a", 4.0), row("b", 0.5)], ts);
        assert!(mixed.advice.contains("Keep the slower ones") && mixed.advice.ends_with("b."));
        assert!(
            verdict(&[row("a", 0.2), row("b", 0.3)], ts)
                .advice
                .starts_with("Do not ship")
        );
        assert!(
            verdict(&[row("a", 0.5)], crate::lang::Lang::C)
                .advice
                .starts_with("No call overhead")
        );
    }
}
