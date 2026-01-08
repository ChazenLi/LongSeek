# Minimal SMILES Transformer

This folder contains a minimal, trainable seq2seq Transformer for single-step
forward (reactants -> products) and retro (products -> reactants) prediction.

## Requirements

- Python 3.9+
- PyTorch
- RDKit (required for SELFIES and drawing)
- selfies (required for SELFIES tokenization)
- Pillow (optional, for GUI image preview)

## Quick start

```bash
python MoleReact/retro_transformer/train.py --direction retro
```

Forward direction:

```bash
python MoleReact/retro_transformer/train.py --direction forward
```

## Notes

- Tokenization supports atom-aware tokens or BPE merges.
- Input CSV must have a `rs>>ps` column.
- Outputs run artifacts under `MoleReact/retro_transformer/runs/`.

## BPE example

```bash
python MoleReact/retro_transformer/train.py --direction retro --tokenizer bpe --bpe-merges 2000
```

## SELFIES example

```bash
python MoleReact/retro_transformer/train.py --direction retro --tokenizer selfies --selfies-roundtrip-check
```

When `tokenizer=selfies`, the input SMILES are canonicalized with RDKit, encoded
to SELFIES, and (optionally) roundtrip-checked before training/inference.

## Prepare SELFIES CSV (optional)

```bash
python MoleReact/retro_transformer/prepare_selfies.py --input MoleReact/uspto_llm_clean.csv --output MoleReact/uspto_llm_selfies.csv --roundtrip-check
```

Filter QC-passed rows while preparing SELFIES:

```bash
python MoleReact/retro_transformer/prepare_selfies.py --input MoleReact/uspto_llm_mapped_qc.csv --output MoleReact/uspto_llm_selfies_qc.csv --roundtrip-check --qc-pass-only
```

Train directly from a SELFIES CSV:

```bash
python MoleReact/retro_transformer/train.py --csv MoleReact/uspto_llm_selfies.csv --tokenizer selfies --selfies-input --selfies-column selfies_rxn
```

## Conditions

Use `--use-conditions` to append solvents/catalyst/temperature/time to the src with a `[COND]` tag.

```bash
python MoleReact/retro_transformer/train.py --direction retro --use-conditions
```

For inference, pass conditions explicitly:

```bash
python MoleReact/retro_transformer/predict.py --checkpoint PATH\\best.pt --vocab PATH\\vocab.json --smiles "O=C(N)C1=CC=CC=C1" --conditions "SOLV=O1CCOCC1|TEMP=160C|TIME=2h"
```

## Predict

```bash
python MoleReact/retro_transformer/predict.py --checkpoint PATH\\best.pt --vocab PATH\\vocab.json --smiles "O=C(N)C1=CC=CC=C1"
```

SELFIES inference prints canonicalized input and SELFIES strings:

```bash
python MoleReact/retro_transformer/predict.py --checkpoint PATH\\best.pt --vocab PATH\\vocab.json --smiles "O=C(N)C1=CC=CC=C1" --top-k 5
```

Expected output includes:

```
product=...
product_canon=...
product_selfies=...
```

## SMILES repair (optional)

Train a small repair model on clean SMILES:

```bash
python MoleReact/retro_transformer/smiles_repair.py --input MoleReact/uspto_llm_clean.csv --epochs 3
```

Use the GUI repair toggle to load the repair checkpoint and compare raw vs repaired outputs.

## GUI

```bash
python MoleReact/retro_transformer/gui.py
```

Notes:

- If `tokenizer=selfies`, the GUI converts input SMILES to canonical SMILES,
  then to SELFIES with roundtrip check before prediction.
- The preview grid includes the input molecule (legend `input`) when RDKit and
  Pillow are available.
