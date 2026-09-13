# zkAgent

Implementation and benchmark programs for zkAgent.

Run the commands below from the repository root. The example uses GPT-2 124M,
Q20 quantization, and 32 threads.

## 1. Install dependencies

On Ubuntu/Debian, install the build tools, GMP development libraries, and Python
virtual-environment support. Omit `sudo` when running as root.

```bash
sudo apt update
sudo apt install -y build-essential cmake libgmp-dev python3-venv pkg-config libssl-dev
```

Initialize the dependencies included as submodules and install the Python packages
in a virtual environment.

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 2. Prepare the model

Download GPT-2 124M, then prepare the quantized model used by the benchmarks.

```bash
python3 scripts/download_gpt2_124m.py
cp backup/vocab.json backup/merges.txt model/gpt2-124m/

python3 quant/prepare_gpt2_quant.py \
  --model-dir ./model/gpt2-124m \
  --Q 20
```

The download script also saves the tokenizer files. If `vocab.json` and
`merges.txt` are missing from the model directory, copy them from `backup/`.

## 3. Build

Compile the benchmark executables.

```bash
./build.sh Release
```

Executables are written to `cmake-build-release/src/`.

## 4. Benchmarks (Tables 1–6)

The benchmarks below follow the paper's table order. Use the common launcher
from the repository root. Results are saved under
`benchmark_logs/<table1..table6|decode>/<timestamp>_<pid>/`.

| Entry | Purpose | Peak RSS (GiB) | Total runtime (s) | Circuit/plan build (s, included) |
| --- | --- | --- | --- | --- |
| `table1` | GPT-2 embedding preprocessing | 6.71 | 945.60 | — |
| `table2` | Online zkEmbed lookup | 1.57 | 11.08 | — |
| `table3` | Cross-model embedding preprocessing | — | — | — |
| `table4` | RoPE computation-core comparison | 0.59 | 111.24 | — |
| `table5` | Transformer component benchmark | See below | See below | See below |
| `table6` | Greedy inference core | See below | See below | See below |
### Table 1: GPT-2 embedding preprocessing

The GPT-2 embedding benchmarks read
`./model/gpt2-124m/quantized/tensors/embedding_token_i32.bin`, generated in step 2.
Tables 1–3 default to 32 threads. Set `THREADS` to choose the thread count;
when unset, an explicit `OMP_NUM_THREADS` is also respected.
To read a GPT-2 embedding file elsewhere, set `GPT2_EMBEDDING_PATH`.

```bash
bash benchmarks/run.sh table1
```

Table 1 measures fresh preprocessing and writes the embedding cache. Tables 2
and 6 reuse this cache for the same prepared model, or build it automatically
if missing. Tables 1 and 3 report FFT planning and table precomputation,
excluding SRS setup, input copying, and cache I/O.

### Table 2: Online zkEmbed lookup

Run the online embedding lookup benchmark:

```bash
THREADS=32 bash benchmarks/run.sh table2
```

### Table 3: Cross-model embedding preprocessing

Select one model, for example GPT-2 Small:

```bash
ZKEMBED_MODEL=gpt2-small bash benchmarks/run.sh table3
```

Run all six additional Table 3 models in sequence:

```bash
THREADS=32 ZKEMBED_MODEL=all bash benchmarks/run.sh table3
```

This runs `tinyllama`, `gpt2-medium`, `gpt2-xl`, `gpt3`, `deepseek-v2`, and
`deepseek-v3`. Each model's log, resource JSON, and Markdown report are saved
under `benchmark_logs/table3/<timestamp>_<pid>/`, with run statuses in
`summary.tsv`. Failed runs are recorded and the remaining models continue.

`ZKEMBED_MODEL` accepts `gpt2-small`, `tinyllama`, `gpt2-medium`, `gpt2-xl`,
`gpt3`, `deepseek-v2`, or `deepseek-v3`.
The six additional models use randomly generated embedding matrices with their
configured vocabulary sizes and embedding dimensions.

### Table 4: RoPE benchmark

Run all Table 4 configurations:

```bash
bash benchmarks/run.sh table4
```

This compares Thaler and Hadamard at `T=32,128,512`, `d=64,128`, `H=12`,
plus the Llama-2-7B RoPE shape (`T=512`, `d=128`, `H=32`), using one thread.
Results are saved to `benchmark_logs/table4/<timestamp>_<pid>/report.md`.

To run one shape, supply `seq_len head_dim heads`:

```bash
bash benchmarks/run.sh table4 32 64 12
```

### Table 5: Transformer component benchmark

After building and preparing the model, run Table 5 directly:

```bash
bash benchmarks/run.sh table5
```

The script uses the prepared model in `./model/gpt2-124m`, 32 threads,
sequence lengths `32, 64, 128, 256, 512`, and both SHOUT and LogUp-GKR.
Inputs are bundled in `demo/table5/gpt2_tokens_T{32,64,128,256,512}.txt`.
Logs and the TSV summary are saved under `benchmark_logs/table5/<timestamp>_<pid>/`.

Observed resources per configuration (GPT-2 124M, 32 threads). Total runtime
is process wall time, including preparation, proving, and verification.
Circuit/plan construction is included in that total.

| T | Backend | Cache | Total runtime (s) | Peak RSS (GiB) | Circuit/plan build (s, included) |
| --- | --- | --- | --- | --- | --- |
| 32 | SHOUT | fresh | 74.35 | 40.46 | 29.66 |
| 64 | SHOUT | fresh | 118.20 | 53.85 | 63.17 |
| 128 | SHOUT | fresh | 245.89 | 84.09 | 161.44 |
| 256 | SHOUT | fresh | 670.59 | 194.96 | 514.70 |
| 512 | SHOUT | fresh | 2851.54 | 463.95 | 2544.30 |
| 32 | LogUp-GKR | hit | 43.92 | 33.89 | 0.00 |
| 64 | LogUp-GKR | hit | 85.86 | 47.45 | 0.00 |
| 128 | LogUp-GKR | hit | 201.02 | 78.69 | 0.00 |
| 256 | LogUp-GKR | hit | 541.76 | 194.68 | 0.00 |
| 512 | LogUp-GKR | hit | 1527.97 | 475.61 | 0.00 |

The SHOUT runs built fresh caches; the LogUp-GKR runs reused them, so their
total runtimes include different preparation work.

### Table 6: Greedy inference core

Run the greedy inference proof on either bundled demo transcript. Cached
preparation is reused on later runs. Regenerate cached preparation after
changing model weights, quantization, or circuit layout.

If Table 1 has not been run and the embedding cache is missing, Table 6 performs
preprocessing automatically. Its time is included in `Preparation: total`;
add `--debug` to display the preprocessing messages.

| Workload | Backend | Cache | Wall (s) | Peak RSS (GiB) | Circuit/plan (s, included) |
|---|---|---|---:|---:|---:|
| Weather | SHOUT | hit | 696.27 | 210.32 | 93.21 |
| Coding | SHOUT | fresh | 727.17 | 211.23 | 519.59 |
| Weather | LogUp-GKR | hit | 249.99 | 208.98 | 0.00 |
| Coding | LogUp-GKR | hit | 247.54 | 208.96 | 0.00

#### SHOUT

```bash
bash benchmarks/run.sh table6 \
  --model-dir ./model/gpt2-124m \
  --token-file ./demo/weather_tokens.txt \
  --seq-len 152 \
  --prompt-len 130 \
  --threads 32 \
  --cache-dir ./cache \
  --use-cache \
  --lookup-backend shout

bash benchmarks/run.sh table6 \
  --model-dir ./model/gpt2-124m \
  --token-file ./demo/coding_tokens.txt \
  --seq-len 143 \
  --prompt-len 129 \
  --threads 32 \
  --cache-dir ./cache \
  --use-cache \
  --lookup-backend shout
```

#### LogUp-GKR

```bash
bash benchmarks/run.sh table6 \
  --model-dir ./model/gpt2-124m \
  --token-file ./demo/weather_tokens.txt \
  --seq-len 152 \
  --prompt-len 130 \
  --threads 32 \
  --cache-dir ./cache \
  --use-cache \
  --lookup-backend logup-gkr

bash benchmarks/run.sh table6 \
  --model-dir ./model/gpt2-124m \
  --token-file ./demo/coding_tokens.txt \
  --seq-len 143 \
  --prompt-len 129 \
  --threads 32 \
  --cache-dir ./cache \
  --use-cache \
  --lookup-backend logup-gkr
```

#### Tool execution

Run the Table 6 tools separately by following their own setup and execution instructions:

- [Coral](tools/coral/README.md): JSON format proofs for tool calls.
- [Weather zkTLS](tools/zktls/weather-zktls-example/README.md): weather API attestations.
- [Code zkVM](tools/py2zkvm/README.md): Python-to-SP1 execution proofs.

## 5. Additional evaluations

### Standalone decoding

Run the standalone decoding benchmark:

```bash
bash benchmarks/run.sh decode
```

This runs Greedy and Top-k (k=40) with SHOUT and LogUp-GKR, using N=50,257,
T=512, 32 threads, and batched proofs. Logs and `summary.csv` are saved under
`benchmark_logs/decode/<timestamp>_<pid>/`.

### Quantization accuracy

The commands below use `--device cuda` and require an NVIDIA GPU with a working
driver (`nvidia-smi` should succeed). On a CPU-only machine, use `--device cpu`;
large-model experiments will run substantially slower.

These experiments load Hugging Face checkpoints. Before running calibration,
obtain Hugging Face access to [Llama-2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
and [Llama-2 13B](https://huggingface.co/meta-llama/Llama-2-13b-hf), then log in
on the machine running the experiment:

```bash
hf auth login
```

#### Calibration-bound exceedance table

Run the calibration-bound exceedance experiment:

```bash
python3 quant/calib_overflow.py \
  --models "openai-community/gpt2,TinyLlama/TinyLlama-1.1B-Chat-v1.0,meta-llama/Llama-2-7b-hf,meta-llama/Llama-2-13b-hf" \
  --calib-dataset wt2-train \
  --test-datasets wt2,wt103,c4 \
  --calib-tokens 50000 \
  --test-tokens 50000 \
  --union-bound-sources calibrated \
  --dtype auto \
  --device cuda \
  --json-out ./benchmark_logs/quant/calibration.json \
  --csv-out ./benchmark_logs/quant/calibration.csv
```

Read `per_token_pct` in `calibration.csv`; run details are in `calibration.json`.

#### Perplexity table

Run the FP32, Naive Dim., and Ours (full) comparison at Q=18, 20, and 22:

```bash
python3 quant/quant_ppl_qsweep.py \
  --model openai-community/gpt2 \
  --dataset all \
  --Q-list 18,20,22 \
  --subset-tokens 200000 \
  --device cuda \
  --csv-out ./benchmark_logs/quant/perplexity.csv \
  --metadata-out ./benchmark_logs/quant/perplexity_metadata.json
```

| Paper row | Output field / preset |
| --- | --- |
| FP32 | `fp_baseline`, repeated for each dataset's quantized rows |
| Naive Dim. | `configuration=naive_dim`, read `ppl` at each `Q` |
| Ours (full) | `configuration=ours_full`, read `ppl` at each `Q` |

Results are saved in `perplexity.csv`; run settings are in `perplexity_metadata.json`.

## Directories

- `src/`: model execution and proof implementations.
- `benchmarks/`: benchmark programs and launchers.
- `scripts/`: model download, tokenization, generation wrappers and cache utilities.
- `quant/`: quantization preparation and accuracy evaluation.
- `tools/`: external tool implementations.
- `backup/`: fallback GPT-2 tokenizer files.
