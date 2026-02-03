# WORK 1 : Minimal SMILES Transformer(File:retro-transformer)      

This folder contains a minimal, trainable seq2seq Transformer for single-step
forward (reactants -> products) and retro (products -> reactants) prediction.

Due to the device 4060,  the most important part is the data cleaning and select
this is also the exercise in this application of LLM in chem and bio, and this small
TF got a good performance in new dataset( personally, in this 9M size maybe good)

But the device can not hold more train and bigger framework, cause the whole training cost 12h
or even more and the small model became overfit.

thus this is a trial, but is connected with another multiretro-agent, which return an excellent result 
when using the CLI mode to doing those multistep prediction synthesis work, and those result were passed through
the chemical synthesis worker's judgemnt.

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



<img width="1120" height="640" alt="top_hits" src="https://github.com/user-attachments/assets/cf4d45bd-fd4b-4343-97bd-dc3e599ffd57" />
<img width="1120" height="640" alt="ppl" src="https://github.com/user-attachments/assets/e43ebbda-d0da-4479-bce7-cb2a2a11e922" />
<img width="1120" height="640" alt="loss" src="https://github.com/user-attachments/assets/6e169718-62e9-4b1f-a3fe-ea1be4d6fc35" />



# work 2: multistep retrosynthesis agent         


- If `tokenizer=selfies`, the GUI converts input SMILES to canonical SMILES,
  then to SELFIES with roundtrip check before prediction.
- The preview grid includes the input molecule (legend `input`) when RDKit and
  Pillow are available.



# work 3: a simple guess game that hold by the llm to see the gaming strategy that they will choose
file: guess
<img width="2778" height="2378" alt="combined_chart" src="https://github.com/user-attachments/assets/b971b9d8-52c0-4df3-bee7-4cd05b26a5fc" />

# work 4:    
file : STRUC2VEC; The-basic-exploration;           
this a kind of work that try to visual and find the ball or min space that contains the children structure;             
may be a help for the further vector system design and frame structure design.          
another reason is that it is a kind of trouble to get the validation of pymol...
results:  <img width="945" height="802" alt="93d90575-20f3-4ec7-8508-7862912d78ed" src="https://github.com/user-attachments/assets/a054fc74-0c27-43a7-9304-ca8e8b776b4a" />

<img width="985" height="923" alt="a2085e66-4a29-46ad-ba5e-4bf9212313f1" src="https://github.com/user-attachments/assets/d0b66e33-282c-4a01-ba13-473a464ed8fb" />

# work 5:   
file: the-cellular-automata; trial;
something interesting that use the CA as the approach
trial: it is a kind of work that using the CA as the basic model to simulate the revoery process of the skin. just funny and for fun. maybe better to make a video to see the detailed process.  And this is a very small params trial to know the whole workflow in this kind of program.
results：<img width="2083" height="1484" alt="optimized_simulation_L12" src="https://github.com/user-attachments/assets/0f246dae-6f31-4ea1-aeeb-ff112a82446a" />
<img width="2685" height="1771" alt="parameter_sensitivity_en" src="https://github.com/user-attachments/assets/ceb034a4-0678-45ac-8c4c-a9dee30f19f3" />
<img width="1786" height="734" alt="quick_gamma_test" src="https://github.com/user-attachments/assets/7c0fa8ec-2c47-464a-bf21-894e98968c0c" />
<img width="1980" height="1293" alt="comparison_report_en" src="https://github.com/user-attachments/assets/b4432ee4-1063-425e-8753-429c7060647b" />
<img width="2556" height="2171" alt="comprehensive_visualization_en" src="https://github.com/user-attachments/assets/d303ae56-d08b-449d-a0b6-b2e112f03c52" />
<img width="2234" height="712" alt="mask_visualization_SN15_0" src="https://github.com/user-attachments/assets/6dd0b058-c8c8-4b0c-8e5a-61144dd83fd2" />
<img width="2234" height="714" alt="mask_visualization_SN15_4" src="https://github.com/user-attachments/assets/6abf8fc6-85a1-473d-af32-71f9fa3494ac" />

