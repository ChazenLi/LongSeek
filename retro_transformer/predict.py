import argparse
import json
import os
import warnings

import torch

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency
    Chem = None

try:
    import selfies as sf
except Exception:  # pragma: no cover - optional dependency
    sf = None

from data import decode, encode
from model import Seq2SeqTransformer, beam_search, create_padding_mask


def _require_selfies_support():
    if sf is None:
        raise RuntimeError("selfies is required for SELFIES input.")
    if Chem is None:
        raise RuntimeError("RDKit is required for SELFIES input.")


def _remove_isolated_h(mol):
    rwmol = Chem.RWMol(mol)
    remove_idx = []
    for atom in rwmol.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0:
            remove_idx.append(atom.GetIdx())
    for idx in sorted(remove_idx, reverse=True):
        rwmol.RemoveAtom(idx)
    return rwmol.GetMol()


def _canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _remove_isolated_h(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _canonicalize_mixture(smiles):
    parts = [p for p in smiles.split(".") if p.strip()]
    if not parts:
        return None
    canonical = []
    for part in parts:
        canon = _canonicalize_smiles(part)
        if canon is None:
            return None
        canonical.append(canon)
    return ".".join(canonical)


def _smiles_to_selfies(smiles, roundtrip_check=True):
    _require_selfies_support()
    parts = [p for p in smiles.split(".") if p.strip()]
    if not parts:
        return None
    encoded = []
    for part in parts:
        canon = _canonicalize_smiles(part)
        if canon is None:
            return None
        try:
            sf_text = sf.encoder(canon)
        except Exception:
            return None
        if roundtrip_check:
            try:
                back = sf.decoder(sf_text)
            except Exception:
                return None
            canon_back = _canonicalize_smiles(back)
            if canon_back is None:
                return None
        encoded.append(sf_text)
    return ".".join(encoded)


def _load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    merges = data.get("bpe_merges")
    tokenizer = data.get("tokenizer", "atom")
    return data["stoi"], {int(k): v for k, v in data["itos"].items()}, merges, tokenizer


def predict_one(
    model,
    product_smiles,
    stoi,
    itos,
    merges,
    tokenizer,
    device,
    beam_size,
    max_len,
    length_penalty,
    selfies_input=False,
):
    ids = encode(
        product_smiles,
        stoi,
        merges=merges,
        tokenizer=tokenizer,
        Chem=Chem,
        selfies_input=selfies_input,
    )
    src = torch.tensor([ids], dtype=torch.long, device=device)
    src_padding_mask = create_padding_mask(src, stoi["<pad>"])
    beams = beam_search(
        model,
        src,
        src_padding_mask,
        stoi["<bos>"],
        stoi["<eos>"],
        max_len,
        beam_size,
        device,
    )
    scored = []
    for seq, score, _ in beams:
        text = decode(seq[0], itos, tokenizer=tokenizer)
        length = max(len(text), 1)
        norm = score / (length ** length_penalty)
        scored.append((text, score, norm))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def main():
    warnings.filterwarnings(
        "ignore",
        message="The PyTorch API of nested tensors is in prototype stage",
    )
    parser = argparse.ArgumentParser(description="Retro transformer prediction.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json")
    parser.add_argument("--smiles", help="Product SMILES string")
    parser.add_argument("--input-file", help="One product SMILES per line")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument(
        "--conditions",
        help="Optional condition text appended as [COND]SOLV=...|CAT=...|TEMP=...|TIME=...",
    )
    parser.add_argument(
        "--no-filter-valid",
        action="store_true",
        help="Disable RDKit validity filtering",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable canonical deduplication",
    )
    parser.add_argument(
        "--max-beam",
        type=int,
        default=60,
        help="Maximum beam size when expanding to fill Top-k",
    )
    parser.add_argument("--emb", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff", type=int, default=512)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.vocab):
        raise FileNotFoundError(f"vocab not found: {args.vocab}")

    stoi, itos, merges, tokenizer = _load_vocab(args.vocab)
    if tokenizer == "selfies":
        _require_selfies_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqTransformer(
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        emb_size=args.emb,
        nhead=args.heads,
        src_vocab_size=len(stoi),
        tgt_vocab_size=len(stoi),
        dim_feedforward=args.ff,
        dropout=args.dropout,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    inputs = []
    if args.smiles:
        inputs.append(args.smiles.strip())
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            inputs.extend([line.strip() for line in f if line.strip()])
    if not inputs:
        raise ValueError("Provide --smiles or --input-file.")

    for smi in inputs:
        src = smi
        if args.conditions:
            if tokenizer == "selfies":
                raise ValueError("SELFIES tokenizer does not support [COND] inputs.")
            cond = args.conditions.strip().replace(" ", "_")
            if not cond.startswith("[COND]"):
                cond = "[COND]" + cond
            src = smi + cond
        canon = None
        sf_text = None
        selfies_input = False
        if tokenizer == "selfies":
            canon = _canonicalize_mixture(smi)
            if canon is None:
                raise ValueError("RDKit canonicalization failed for input SMILES.")
            sf_text = _smiles_to_selfies(canon, roundtrip_check=True)
            if not sf_text:
                raise ValueError("SELFIES conversion failed or roundtrip check failed.")
            src = sf_text
            selfies_input = True
        filtered = []
        seen = set()
        beam_size = max(args.top_k, 1)
        while len(filtered) < args.top_k and beam_size <= args.max_beam:
            preds = predict_one(
                model,
                src,
                stoi,
                itos,
                merges,
                tokenizer,
                device,
                beam_size=beam_size,
                max_len=args.max_len,
                length_penalty=args.length_penalty,
                selfies_input=selfies_input,
            )
            filtered = []
            seen = set()
            for text, score, norm in preds:
                if not args.no_filter_valid and Chem is not None:
                    mol = Chem.MolFromSmiles(text)
                    if mol is None:
                        continue
                    if args.no_dedupe:
                        filtered.append((text, score, norm))
                    else:
                        canon = Chem.MolToSmiles(mol, canonical=True)
                        if canon in seen:
                            continue
                        seen.add(canon)
                        filtered.append((canon, score, norm))
                else:
                    filtered.append((text, score, norm))
                if len(filtered) >= args.top_k:
                    break
            if len(filtered) >= args.top_k:
                break
            beam_size = min(args.max_beam, beam_size * 2)
        print(f"product={smi}")
        if tokenizer == "selfies":
            print(f"product_canon={canon}")
            print(f"product_selfies={sf_text}")
        for i, (text, score, norm) in enumerate(filtered, start=1):
            print(f"{i}\t{norm:.4f}\t{text}")


if __name__ == "__main__":
    main()
