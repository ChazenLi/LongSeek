import csv
import random
from collections import Counter

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

_TWO_CHAR = {
    "Cl",
    "Br",
    "Si",
    "Na",
    "Ca",
    "Li",
    "Mg",
    "Al",
    "Sn",
    "Ag",
    "Fe",
    "Zn",
    "Cu",
    "Mn",
    "Hg",
    "Pb",
    "Pt",
    "Pd",
    "Ni",
    "Co",
    "Se",
    "As",
    "Au",
    "Bi",
}

try:
    import selfies as _selfies
except Exception:  # pragma: no cover - optional dependency
    _selfies = None

_CANON_STATS = {"h_removed": 0, "sanitize_fail": 0, "encode_fail": 0}


def reset_canon_stats():
    for k in _CANON_STATS:
        _CANON_STATS[k] = 0


def get_canon_stats():
    return dict(_CANON_STATS)


def _require_selfies():
    if _selfies is None:
        raise RuntimeError("SELFIES not installed. Install selfies to use this tokenizer.")


def _remove_isolated_h(mol, Chem):
    rwmol = Chem.RWMol(mol)
    remove_idx = []
    for atom in rwmol.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0:
            remove_idx.append(atom.GetIdx())
    for idx in sorted(remove_idx, reverse=True):
        rwmol.RemoveAtom(idx)
    if remove_idx:
        _CANON_STATS["h_removed"] += len(remove_idx)
    return rwmol.GetMol()


def _canonicalize(smiles, Chem):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _remove_isolated_h(mol, Chem)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        _CANON_STATS["sanitize_fail"] += 1
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def tokenize_smiles(smiles):
    tokens = []
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        if ch == "[":
            j = smiles.find("]", i + 1)
            if j == -1:
                tokens.append(ch)
                i += 1
            else:
                tokens.append(smiles[i : j + 1])
                i = j + 1
            continue
        if ch == "%" and i + 2 < len(smiles) and smiles[i + 1 : i + 3].isdigit():
            tokens.append(smiles[i : i + 3])
            i += 3
            continue
        if i + 1 < len(smiles) and smiles[i : i + 2] in _TWO_CHAR:
            tokens.append(smiles[i : i + 2])
            i += 2
            continue
        tokens.append(ch)
        i += 1
    return tokens


def tokenize_text(
    text,
    tokenizer="atom",
    merges=None,
    Chem=None,
    roundtrip_check=False,
    selfies_input=False,
):
    if tokenizer == "selfies":
        _require_selfies()
        if Chem is None:
            raise RuntimeError("RDKit is required for SELFIES tokenizer.")
        if selfies_input:
            sf = text
        else:
            canon = _canonicalize(text, Chem)
            if canon is None:
                return None
            try:
                sf = _selfies.encoder(canon)
            except Exception:
                _CANON_STATS["encode_fail"] += 1
                return None
        tokens = list(_selfies.split_selfies(sf))
        if roundtrip_check:
            try:
                decoded = _selfies.decoder(sf)
            except Exception:
                return None
            canon_back = _canonicalize(decoded, Chem)
            if canon_back is None:
                return None
        return tokens

    tokens = tokenize_smiles(text)
    if merges:
        tokens = apply_bpe(tokens, merges)
    return tokens


def _get_pair_stats(token_lists):
    stats = Counter()
    for tokens in token_lists:
        for i in range(len(tokens) - 1):
            stats[(tokens[i], tokens[i + 1])] += 1
    return stats


def _merge_pair(tokens, pair):
    merged = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def train_bpe(token_lists, num_merges=1000, min_pair_freq=2):
    merges = []
    current = [list(toks) for toks in token_lists]
    for _ in range(num_merges):
        stats = _get_pair_stats(current)
        if not stats:
            break
        (pair, freq) = stats.most_common(1)[0]
        if freq < min_pair_freq:
            break
        merges.append(pair)
        current = [_merge_pair(toks, pair) for toks in current]
    return merges, current


def apply_bpe(tokens, merges):
    out = list(tokens)
    for pair in merges:
        out = _merge_pair(out, tuple(pair))
    return out


def _parse_listish(value):
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null", "[]"):
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip().strip("'\"") for p in s.split(",")]
    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        cleaned.append(p.replace(" ", "_"))
    return cleaned


def build_conditions(row):
    parts = []
    solvents = _parse_listish(row.get("solvents"))
    catalyst = _parse_listish(row.get("catalyst"))
    temperature = _parse_listish(row.get("temperature"))
    time = _parse_listish(row.get("time"))
    if solvents:
        parts.append("SOLV=" + "+".join(solvents))
    if catalyst:
        parts.append("CAT=" + "+".join(catalyst))
    if temperature:
        parts.append("TEMP=" + "+".join(temperature))
    if time:
        parts.append("TIME=" + "+".join(time))
    if not parts:
        return ""
    return "[COND]" + "|".join(parts)


def read_reactions(
    csv_path,
    max_rows=None,
    include_conditions=False,
    qc_pass_only=False,
    qc_field="qc_pass",
    selfies_input=False,
    selfies_column="selfies_rxn",
):
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if selfies_input:
            if selfies_column not in reader.fieldnames:
                raise ValueError(f"Missing '{selfies_column}' column in CSV.")
        elif "rs>>ps" not in reader.fieldnames:
            raise ValueError("Missing 'rs>>ps' column in CSV.")
        for i, row in enumerate(reader):
            if qc_pass_only:
                val = str(row.get(qc_field, "")).strip().lower()
                if val not in ("true", "1", "yes"):
                    continue
            rxn = row.get(selfies_column if selfies_input else "rs>>ps", "")
            if ">>" not in rxn:
                continue
            reactants, products = rxn.split(">>", 1)
            cond = build_conditions(row) if include_conditions else ""
            rows.append({"reactants": reactants, "products": products, "conditions": cond})
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def make_pairs(rows, direction="retro", include_conditions=False):
    pairs = []
    for row in rows:
        reactants = row["reactants"]
        products = row["products"]
        cond = row.get("conditions", "") if include_conditions else ""
        if direction == "forward":
            src, tgt = reactants + cond, products
        else:
            src, tgt = products + cond, reactants
        pairs.append((src, tgt))
    return pairs


def build_vocab(token_lists, min_freq=1):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    tokens = SPECIAL_TOKENS + [t for t, c in counter.items() if c >= min_freq]
    stoi = {t: i for i, t in enumerate(tokens)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


def encode(
    text,
    stoi,
    merges=None,
    tokenizer="atom",
    Chem=None,
    roundtrip_check=False,
    selfies_input=False,
):
    bos = stoi["<bos>"]
    eos = stoi["<eos>"]
    unk = stoi["<unk>"]
    tokens = tokenize_text(
        text,
        tokenizer=tokenizer,
        merges=merges,
        Chem=Chem,
        roundtrip_check=roundtrip_check,
        selfies_input=selfies_input,
    )
    if tokens is None:
        raise ValueError("Tokenizer failed to produce tokens.")
    ids = [bos]
    for tok in tokens:
        ids.append(stoi.get(tok, unk))
    ids.append(eos)
    return ids


def decode(ids, itos, tokenizer="atom"):
    out = []
    for i in ids:
        tok = itos.get(int(i), "")
        if tok in ("<bos>", "<eos>", "<pad>"):
            continue
        out.append(tok)
    if tokenizer == "selfies":
        _require_selfies()
        sf = "".join(out)
        try:
            return _selfies.decoder(sf)
        except Exception:
            return ""
    return "".join(out)


class SmilesDataset(Dataset):
    def __init__(
        self,
        pairs,
        stoi,
        max_len=512,
        merges=None,
        tokenizer="atom",
        Chem=None,
        roundtrip_check=False,
        selfies_input=False,
    ):
        self.pairs = pairs
        self.stoi = stoi
        self.max_len = max_len
        self.merges = merges
        self.tokenizer = tokenizer
        self.Chem = Chem
        self.roundtrip_check = roundtrip_check
        self.selfies_input = selfies_input

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = encode(
            src,
            self.stoi,
            merges=self.merges,
            tokenizer=self.tokenizer,
            Chem=self.Chem,
            roundtrip_check=self.roundtrip_check,
            selfies_input=self.selfies_input,
        )[: self.max_len]
        tgt_ids = encode(
            tgt,
            self.stoi,
            merges=self.merges,
            tokenizer=self.tokenizer,
            Chem=self.Chem,
            roundtrip_check=self.roundtrip_check,
            selfies_input=self.selfies_input,
        )[: self.max_len]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
            tgt_ids, dtype=torch.long
        )


def split_pairs(pairs, val_ratio=0.05, seed=42):
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[n_val:], shuffled[:n_val]


def collate_batch(batch, pad_id):
    src_batch, tgt_batch = zip(*batch)
    src_len = max(x.size(0) for x in src_batch)
    tgt_len = max(x.size(0) for x in tgt_batch)
    src_out = torch.full((len(batch), src_len), pad_id, dtype=torch.long)
    tgt_out = torch.full((len(batch), tgt_len), pad_id, dtype=torch.long)
    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        src_out[i, : src.size(0)] = src
        tgt_out[i, : tgt.size(0)] = tgt
    return src_out, tgt_out
