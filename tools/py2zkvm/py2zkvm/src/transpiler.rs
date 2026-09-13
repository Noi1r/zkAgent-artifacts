use anyhow::{bail, Context, Result};
use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;
use syn::{Attribute, Item, Meta};

pub fn transpile_python(input: &Path) -> Result<String> {
    transpile_with_depyler(input, OsStr::new("depyler"))
}

fn transpile_with_depyler(input: &Path, depyler: &OsStr) -> Result<String> {
    let input = input
        .canonicalize()
        .with_context(|| format!("Cannot open Python input: {}", input.display()))?;
    let check = Command::new(depyler).arg("--version").output().context(
        "Failed to start depyler; ensure ~/.cargo/bin is in PATH and install depyler 3.25.0",
    )?;
    if !check.status.success() {
        bail!(
            "depyler --version failed ({}): {}",
            check.status,
            String::from_utf8_lossy(&check.stderr).trim()
        );
    }

    // An explicit temporary output cannot consume or overwrite a user's sibling .rs file.
    let directory = tempfile::tempdir().context("Cannot create Depyler output directory")?;
    let output_path = directory.path().join("transpiled.rs");
    let output = Command::new(depyler)
        .arg("transpile")
        .arg(&input)
        .arg("--output")
        .arg(&output_path)
        .output()
        .context("Failed to run depyler transpile")?;
    if !output.status.success() {
        bail!(
            "depyler transpile failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let rust_code = std::fs::read_to_string(&output_path)
        .context("Depyler reported success but did not write the requested Rust output")?;
    clean_for_zkvm(&rust_code)
}

pub(crate) fn clean_for_zkvm(rust_code: &str) -> Result<String> {
    let mut file = syn::parse_file(rust_code)
        .context("Depyler generated invalid Rust; refusing to generate an SP1 project")?;
    clean_items(&mut file.items, true);
    Ok(prettyplease::unparse(&file))
}

fn clean_items(items: &mut Vec<Item>, top_level: bool) {
    items.retain(|item| {
        !item_attributes(item).iter().any(is_test_attribute)
            && !(top_level && is_generated_main(item))
    });
    for item in items {
        if let Item::Mod(module) = item {
            if let Some((_, items)) = &mut module.content {
                clean_items(items, false);
            }
        }
    }
}

fn item_attributes(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(item) => &item.attrs,
        Item::Enum(item) => &item.attrs,
        Item::ExternCrate(item) => &item.attrs,
        Item::Fn(item) => &item.attrs,
        Item::ForeignMod(item) => &item.attrs,
        Item::Impl(item) => &item.attrs,
        Item::Macro(item) => &item.attrs,
        Item::Mod(item) => &item.attrs,
        Item::Static(item) => &item.attrs,
        Item::Struct(item) => &item.attrs,
        Item::Trait(item) => &item.attrs,
        Item::TraitAlias(item) => &item.attrs,
        Item::Type(item) => &item.attrs,
        Item::Union(item) => &item.attrs,
        Item::Use(item) => &item.attrs,
        _ => &[],
    }
}

fn is_test_attribute(attribute: &Attribute) -> bool {
    attribute.path().is_ident("test")
        || attribute.path().is_ident("quickcheck")
        || (attribute.path().is_ident("cfg")
            && attribute
                .parse_args::<syn::Path>()
                .map(|path| path.is_ident("test"))
                .unwrap_or(false))
}

fn is_generated_main(item: &Item) -> bool {
    let Item::Fn(function) = item else {
        return false;
    };
    if function.sig.ident != "main" {
        return false;
    }
    function.attrs.iter().any(|attribute| {
        let Meta::NameValue(value) = &attribute.meta else {
            return false;
        };
        if !value.path.is_ident("doc") {
            return false;
        }
        let syn::Expr::Lit(literal) = &value.value else {
            return false;
        };
        let syn::Lit::Str(doc) = &literal.lit else {
            return false;
        };
        doc.value().contains("DEPYLER-1216")
            || doc
                .value()
                .contains("Auto-generated entry point for standalone compilation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_inline_and_multiline_generated_main() {
        for source in [
            r###"#[doc = r" DEPYLER-1216: Auto-generated entry point for standalone compilation"] pub fn main () -> Result <(), Box<dyn std::error::Error>> { Ok(()) }
pub fn add() -> i32 { 11 }
"###,
            r###"#[doc = "Auto-generated entry point for standalone compilation"]
pub fn main
() -> Result<(), Box<dyn std::error::Error>> {
    let _ = r#"unbalanced { braces } } and fn main()"#;
    if true { let _ = "{"; }
    Ok(())
}
pub fn add() -> i32 { 11 }
"###,
        ] {
            let cleaned = clean_for_zkvm(source).unwrap();
            let file = syn::parse_file(&cleaned).unwrap();
            assert_eq!(file.items.len(), 1);
            assert!(matches!(&file.items[0], Item::Fn(f) if f.sig.ident == "add"));
        }
    }

    #[test]
    fn regression_preserve_business_main_strings_and_nested_main() {
        let source = r###"pub fn main() -> i32 { 7 }
pub fn text() -> &'static str { r#"fn main() { Ok(()) } quickcheck"# }
mod helper { pub fn main() {} }
#[cfg(test)] mod tests { use quickcheck::quickcheck; #[test] fn ignored() {} }
#[test] fn another_test() {}
"###;
        let cleaned = clean_for_zkvm(source).unwrap();
        let file = syn::parse_file(&cleaned).unwrap();
        assert_eq!(file.items.len(), 3);
        assert!(matches!(&file.items[0], Item::Fn(f) if f.sig.ident == "main"));
        assert!(cleaned.contains("fn main() { Ok(()) } quickcheck"));
        assert!(cleaned.contains("mod helper"));
        assert!(!cleaned.contains("mod tests"));
    }

    #[test]
    fn regression_invalid_rust_is_an_error() {
        assert!(clean_for_zkvm("pub fn broken( {").is_err());
    }

    #[cfg(unix)]
    fn fake_depyler(directory: &Path, body: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = directory.join("depyler");
        std::fs::write(&path, format!("#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'depyler 3.25.0'; exit 0; fi\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn regression_failed_transpilation_does_not_use_stale_rust() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.py");
        std::fs::write(&input, "def add(): return 11").unwrap();
        let stale = input.with_extension("rs");
        std::fs::write(&stale, "user-owned Rust file").unwrap();
        let depyler = fake_depyler(directory.path(), "echo 'intentional failure' >&2; exit 17");
        let error = transpile_with_depyler(&input, depyler.as_os_str())
            .unwrap_err()
            .to_string();
        assert!(error.contains("intentional failure"));
        assert_eq!(
            std::fs::read_to_string(stale).unwrap(),
            "user-owned Rust file"
        );
    }

    #[cfg(unix)]
    #[test]
    fn regression_success_uses_explicit_output_without_touching_input_directory() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input with spaces.py");
        std::fs::write(&input, "def add(): return 11").unwrap();
        let stale = input.with_extension("rs");
        std::fs::write(&stale, "user-owned Rust file").unwrap();
        let depyler = fake_depyler(
            directory.path(),
            "printf 'pub fn add() -> i32 { 11 }\\n' > \"$4\"",
        );
        let output = transpile_with_depyler(&input, depyler.as_os_str()).unwrap();
        assert!(output.contains("pub fn add()"));
        assert_eq!(
            std::fs::read_to_string(stale).unwrap(),
            "user-owned Rust file"
        );
    }
}
