
# Coral

Coral generates zero-knowledge proofs that a committed document is consistent
with a public context-free grammar. The system is described in our paper:
[Coral: Fast Succinct Non-Interactive Zero-Knowledge CFG Proofs](https://eprint.iacr.org/2025/1420).

## Prerequisites

- Rust toolchain with `cargo`.
- A Perpetual Powers of Tau (PPoT) file available at the repo root as
  `./ppot_0080_23.ptau`. Coral is currently hardcoded to look for this file.

Download (about 9GB):

```bash
curl -L -o ppot_0080_23.ptau \
  https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_23.ptau
```

## Build

```bash
cargo build --release --locked
```


## Prove + Verify a JSON Document


Run the end-to-end flow (commit + prove + verify) using the corresponding tool grammar:

```bash
cargo run --release -- \
  --e2e -d tests/test_docs/json/weather.json -g grammars/weather.pest -b 100
  
cargo run --release -- \
  --e2e -d tests/test_docs/json/code.json -g grammars/coding.pest -b 100
```

On success, Coral emits:
- `prover_data.cmt` (prover commitment)
- `verifier_data.cmt` (verifier commitment)
- `to_verify.proof` (proof)

It also prints:
- `Proving time (ms)`
- `Verification time (ms)`
- `Proof size (bytes)`

