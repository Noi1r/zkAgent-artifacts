use crate::generator::{self, Sp1Config};
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

pub fn generate_sp1_project(output_dir: &Path, config: &Sp1Config) -> Result<()> {
    let program_dir = output_dir.join("program");
    let program_src = program_dir.join("src");
    let script_dir = output_dir.join("script");
    let script_src = script_dir.join("src");

    fs::create_dir_all(&program_src)
        .with_context(|| format!("Failed to create {}", program_src.display()))?;
    fs::create_dir_all(&script_src)
        .with_context(|| format!("Failed to create {}", script_src.display()))?;

    let workspace_toml = generate_workspace_toml();
    fs::write(output_dir.join("Cargo.toml"), workspace_toml)?;

    let guest_code = generator::generate_guest_code(config)?;
    fs::write(program_src.join("main.rs"), guest_code)?;

    let guest_toml = generate_guest_cargo_toml(config);
    fs::write(program_dir.join("Cargo.toml"), guest_toml)?;

    let host_code = generator::generate_host_code(config)?;
    fs::write(script_src.join("main.rs"), host_code)?;

    let host_toml = generate_host_cargo_toml(config);
    fs::write(script_dir.join("Cargo.toml"), host_toml)?;

    let build_rs = generate_build_rs();
    fs::write(script_dir.join("build.rs"), build_rs)?;

    let cargo_config_dir = output_dir.join(".cargo");
    fs::create_dir_all(&cargo_config_dir)?;
    fs::write(
        cargo_config_dir.join("config.toml"),
        generate_cargo_config(),
    )?;

    let build_script = generate_build_script(&config.main_function.name);
    fs::write(output_dir.join("build.sh"), build_script)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(output_dir.join("build.sh"))?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(output_dir.join("build.sh"), perms)?;
    }

    let readme = generate_readme(&config.main_function);
    fs::write(output_dir.join("README.md"), readme)?;

    fs::write(
        output_dir.join("rust-toolchain.toml"),
        generate_rust_toolchain(),
    )?;

    Ok(())
}

fn generate_workspace_toml() -> String {
    r#"[workspace]
resolver = "2"
members = ["program", "script"]

[workspace.package]
version = "0.1.0"
edition = "2021"

[workspace.dependencies]
sp1-sdk = "3.4.0"
sp1-zkvm = "3.4.0"
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"
"#
    .to_string()
}

fn generate_guest_cargo_toml(config: &Sp1Config) -> String {
    let mut toml = r#"[package]
name = "program"
version.workspace = true
edition.workspace = true

[dependencies]
sp1-zkvm.workspace = true
serde.workspace = true
"#
    .to_string();

    if config.precompiles {
        toml.push_str(r#"
# Precompile patches for crypto acceleration
[patch.crates-io]
sha2 = { git = "https://github.com/sp1-patches/RustCrypto-hashes", package = "sha2", branch = "patch-sha2-v0.10.8" }
sha3 = { git = "https://github.com/sp1-patches/RustCrypto-hashes", package = "sha3", branch = "patch-sha3-v0.10.8" }
crypto-bigint = { git = "https://github.com/sp1-patches/RustCrypto-bigint", branch = "patch-v0.5.5" }
"#);
    }

    toml
}

fn generate_host_cargo_toml(_config: &Sp1Config) -> String {
    r#"[package]
name = "script"
version.workspace = true
edition.workspace = true

[dependencies]
sp1-sdk.workspace = true
serde.workspace = true
bincode.workspace = true

# Match the arithmetic dependencies in the SP1 3.4.0 release lockfile.
ruint = "=1.12.3"
num-modular = "=0.6.1"

"#
    .to_string()
}

fn generate_cargo_config() -> String {
    r#"[target.riscv32im-succinct-zkvm-elf]
rustflags = ["-C", "target-feature=+crt-static"]
"#
    .to_string()
}

fn generate_build_rs() -> String {
    r#"use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let program = manifest.join("../program");
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(cargo)
        .current_dir(&program)
        .args(["prove", "build", "--output-directory"])
        .arg(&output)
        .args(["--elf-name", "program"])
        .env("CARGO_TARGET_DIR", output.join("guest-target"))
        .status().expect("cannot start cargo prove; install SP1 CLI 5.2.4");
    assert!(status.success(), "SP1 guest compilation failed");
    let elf = output.join("program");
    let bytes = fs::read(&elf).expect("cargo prove did not produce the guest ELF");
    assert!(bytes.len() >= 20 && &bytes[..4] == b"\x7fELF"
            && bytes[4] == 1 && bytes[5] == 1
            && u16::from_le_bytes([bytes[18], bytes[19]]) == 243,
            "SP1 SDK 3.4.0 requires a 32-bit RISC-V ELF; use SP1 CLI 5.2.4, not CLI 6.x");
    println!("cargo:rustc-env=SP1_ELF_program={}", elf.display());
    println!("cargo:rerun-if-changed={}", program.display());
    println!("cargo:rerun-if-changed={}", manifest.join("../Cargo.toml").display());
    println!("cargo:rerun-if-changed={}", manifest.join("../Cargo.lock").display());
    println!("cargo:rerun-if-changed={}", manifest.join("../.cargo/config.toml").display());
    println!("cargo:rerun-if-changed={}", manifest.join("../rust-toolchain.toml").display());
}
"#
    .to_string()
}

fn generate_build_script(function_name: &str) -> String {
    format!(
        r#"#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/script"
echo "Building SP1 guest and host for {}"
cargo build --release
echo "Run the generated host from script/: cargo run --release"
"#,
        function_name
    )
}

fn generate_readme(main_fn: &crate::parser::RustFunction) -> String {
    let params_doc: Vec<String> = main_fn
        .params
        .iter()
        .map(|(name, typ)| format!("- `{}`: `{}`", name, typ))
        .collect();

    format!(
        r#"# SP1 zkVM Project

Generated by py2zkvm - Python to SP1 zkVM transpiler.

## Proving Function

**`{}`**

### Parameters
{}

### Returns
`{}`

## Building

### Prerequisites

1. Install Rust (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup toolchain install 1.94.0 --profile minimal
export PATH="$HOME/.sp1/bin:$HOME/.cargo/bin:$PATH"
```

2. Install the RV32 build tools used by this project:
```bash
curl -fsSL https://sp1up.succinct.xyz | bash
sp1up -v 5.2.4
rustup toolchain list
cargo +1.94.0 prove --version
```

The toolchain list must include `succinct`. This project retains the SP1 3.4.0
SDK and uses the RV32 CLI 5.2.4 with host Cargo 1.94.0.
`cargo prove build` selects the Succinct guest compiler.

### Build

```bash
./build.sh
```

### Run

```bash
cd script
cargo run --release
```

This will:
1. Execute the computation inside the zkVM
2. Generate a zero-knowledge proof
3. Verify the proof
4. Save the proof to `proof.bin`

## Modifying Inputs

Edit `script/src/main.rs` to change the input values passed to the zkVM.

## Project Structure

```
.
├── program/          # Guest program (runs inside zkVM)
│   ├── Cargo.toml
│   └── src/
│       └── main.rs   # Your transpiled Python code
├── script/           # Host program (generates proofs)
│   ├── Cargo.toml
│   └── src/
│       └── main.rs   # Proof generation code
├── Cargo.toml        # Workspace config
└── README.md
```

## License

Generated code - use as you wish.
"#,
        main_fn.name,
        params_doc.join("\n"),
        main_fn.return_type,
    )
}

fn generate_rust_toolchain() -> String {
    r#"[toolchain]
channel = "1.94.0"
profile = "minimal"
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_workspace_toml() {
        let toml = generate_workspace_toml();
        assert!(toml.contains("[workspace]"));
        assert!(toml.contains("sp1-sdk"));
    }
}
