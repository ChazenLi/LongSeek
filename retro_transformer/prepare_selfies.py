import argparse
import csv
import json
import os

try:
    import selfies as sf
except Exception:
    sf = None

try:
    from rdkit import Chem
except Exception:
    Chem = None


def _require():
    if sf is None:
        raise RuntimeError("selfies is required.")
    if Chem is None:
        raise RuntimeError("RDKit is required.")


def _remove_isolated_h(mol):
    rwmol = Chem.RWMol(mol)
    remove_idx = []
    for atom in rwmol.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0:
            remove_idx.append(atom.GetIdx())
    for idx in sorted(remove_idx, reverse=True):
        rwmol.RemoveAtom(idx)
    return rwmol.GetMol()


def _canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _remove_isolated_h(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _encode_side(side, roundtrip_check):
    parts = [p for p in side.split(".") if p.strip()]
    encoded = []
    for part in parts:
        canon = _canonicalize(part)
        if canon is None:
            return None
        try:
            s = sf.encoder(canon)
        except Exception:
            return None
        if roundtrip_check:
            try:
                back = sf.decoder(s)
            except Exception:
                return None
            canon_back = _canonicalize(back)
            if canon_back is None:
                return None
        encoded.append(s)
    return ".".join(encoded)


def main():
    parser = argparse.ArgumentParser(description="Prepare SELFIES CSV.")
    parser.add_argument(
        "--input",
        default=os.path.join("MoleReact", "uspto_llm_clean.csv"),
        help="Input CSV with rs>>ps",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("MoleReact", "uspto_llm_selfies.csv"),
        help="Output CSV with selfies_rxn column",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--roundtrip-check", action="store_true")
    parser.add_argument("--qc-pass-only", action="store_true")
    parser.add_argument("--qc-field", default="qc_pass")
    parser.add_argument("--progress-every", type=int, default=20000)
    args = parser.parse_args()

    _require()
    total = 0
    kept = 0
    skipped = 0
    skipped_qc = 0

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f_in, open(
        args.output, "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        if "rs>>ps" not in reader.fieldnames:
            raise ValueError("Missing 'rs>>ps' column.")
        fieldnames = list(reader.fieldnames) + ["selfies_rxn"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            total += 1
            if args.qc_pass_only:
                val = str(row.get(args.qc_field, "")).strip().lower()
                if val not in ("true", "1", "yes"):
                    skipped_qc += 1
                    continue
            rxn = row.get("rs>>ps", "")
            if ">>" not in rxn:
                skipped += 1
                continue
            r, p = rxn.split(">>", 1)
            r_sf = _encode_side(r, args.roundtrip_check)
            p_sf = _encode_side(p, args.roundtrip_check)
            if r_sf is None or p_sf is None:
                skipped += 1
                continue
            row["selfies_rxn"] = f"{r_sf}>>{p_sf}"
            writer.writerow(row)
            kept += 1
            if args.progress_every and kept % args.progress_every == 0:
                print(f"progress kept={kept} total={total} skipped={skipped} skipped_qc={skipped_qc}")
            if args.max_rows and kept >= args.max_rows:
                break

    report = {
        "total": total,
        "kept": kept,
        "skipped": skipped,
        "skipped_qc": skipped_qc,
        "output": args.output,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f"task=prepare_selfies status=done output={args.output} kept={kept}")


if __name__ == "__main__":
    main()
