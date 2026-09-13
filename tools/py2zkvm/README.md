# py2zkvm

Python to SP1 zkVM transpiler. This repo turns a Python function into a provable SP1 project and runs proof generation + verification.

## Prerequisites

On Ubuntu/Debian, install the native build dependencies first (omit `sudo` as root):

```bash
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev
```

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup toolchain install 1.94.0 --profile minimal
export PATH="$HOME/.sp1/bin:$HOME/.cargo/bin:$PATH"

# SP1 succinct toolchain
curl -fsSL https://sp1up.succinct.xyz | bash
sp1up -v 5.2.4
source /root/.bashrc
rustup toolchain list
cargo +1.94.0 prove --version

# depyler (Python -> Rust)
# Use the version tested with this artifact.
cargo install depyler --version 3.25.0 --locked
depyler --version
```

## Build

```bash
cd py2zkvm
cargo build --release
```


## Quick Start

```bash
cd ..
./py2zkvm/target/release/py2zkvm transpile -i examples/add.py -o sp1_add --main-fn add

cd sp1_add/script
cargo run --release

stat -c '%s' proof.bin
```

