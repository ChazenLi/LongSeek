import argparse
import csv
import json
import os
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def read_smiles_from_reactions(csv_path, max_rows=None):
    smiles = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if "rs>>ps" not in reader.fieldnames:
            raise ValueError("Missing 'rs>>ps' in CSV.")
        for row in reader:
            rxn = row.get("rs>>ps", "")
            if ">>" not in rxn:
                continue
            reactants, products = rxn.split(">>", 1)
            for side in (reactants, products):
                for part in side.split("."):
                    part = part.strip()
                    if part:
                        smiles.append(part)
            if max_rows and len(smiles) >= max_rows:
                break
    return smiles


def build_vocab(smiles_list):
    counter = {}
    for smi in smiles_list:
        for ch in smi:
            counter[ch] = counter.get(ch, 0) + 1
    tokens = SPECIAL_TOKENS + sorted(counter.keys())
    stoi = {t: i for i, t in enumerate(tokens)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


def encode(smiles, stoi):
    bos = stoi["<bos>"]
    eos = stoi["<eos>"]
    unk = stoi["<unk>"]
    ids = [bos]
    for ch in smiles:
        ids.append(stoi.get(ch, unk))
    ids.append(eos)
    return ids


def decode(ids, itos):
    out = []
    for idx in ids:
        tok = itos.get(int(idx), "")
        if tok in ("<bos>", "<eos>", "<pad>"):
            continue
        out.append(tok)
    return "".join(out)


def corrupt_smiles(smiles, rng):
    s = list(smiles)
    if not s:
        return smiles
    ops = ["delete", "insert", "swap", "drop_paren", "drop_ring", "dot"]
    op = rng.choice(ops)

    if op == "delete" and len(s) > 1:
        idx = rng.randrange(len(s))
        del s[idx]
    elif op == "insert":
        idx = rng.randrange(len(s) + 1)
        ch = rng.choice(s)
        s.insert(idx, ch)
    elif op == "swap" and len(s) > 1:
        idx = rng.randrange(len(s) - 1)
        s[idx], s[idx + 1] = s[idx + 1], s[idx]
    elif op == "drop_paren":
        if "(" in s or ")" in s:
            target = "(" if "(" in s else ")"
            idx = s.index(target)
            del s[idx]
    elif op == "drop_ring":
        digits = [i for i, ch in enumerate(s) if ch.isdigit()]
        if digits:
            del s[rng.choice(digits)]
    elif op == "dot":
        if "." in s:
            s = [ch for ch in s if ch != "."]
        else:
            idx = rng.randrange(len(s))
            s.insert(idx, ".")
    return "".join(s)


class RepairDataset(Dataset):
    def __init__(self, smiles_list, stoi, max_len=256, seed=42):
        self.smiles = smiles_list
        self.stoi = stoi
        self.max_len = max_len
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        clean = self.smiles[idx]
        corrupt = corrupt_smiles(clean, self.rng)
        src = encode(corrupt, self.stoi)[: self.max_len]
        tgt = encode(clean, self.stoi)[: self.max_len]
        return torch.tensor(src, dtype=torch.long), torch.tensor(
            tgt, dtype=torch.long
        )


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


class RepairModel(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=256, layers=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.encoder = nn.GRU(
            emb, hid, num_layers=layers, batch_first=True, dropout=dropout
        )
        self.decoder = nn.GRU(
            emb, hid, num_layers=layers, batch_first=True, dropout=dropout
        )
        self.out = nn.Linear(hid, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.emb(src)
        _, hidden = self.encoder(src_emb)
        tgt_emb = self.emb(tgt)
        out, _ = self.decoder(tgt_emb, hidden)
        return self.out(out)


class Repairer:
    def __init__(self, checkpoint_path, device=None):
        data = torch.load(checkpoint_path, map_location="cpu")
        self.stoi = data["vocab"]["stoi"]
        itos_raw = data["vocab"]["itos"]
        self.itos = {int(k): v for k, v in itos_raw.items()}
        cfg = data["config"]
        self.device = device or torch.device("cpu")
        self.model = RepairModel(
            vocab_size=len(self.stoi),
            emb=cfg["emb"],
            hid=cfg["hid"],
            layers=cfg["layers"],
            dropout=cfg["dropout"],
        ).to(self.device)
        self.model.load_state_dict(data["model"])
        self.model.eval()

    def repair(self, smiles, max_len=256):
        src = torch.tensor([encode(smiles, self.stoi)], device=self.device)
        bos = self.stoi["<bos>"]
        eos = self.stoi["<eos>"]
        ys = torch.tensor([[bos]], device=self.device)
        with torch.no_grad():
            src_emb = self.model.emb(src)
            _, hidden = self.model.encoder(src_emb)
            for _ in range(max_len - 1):
                tgt_emb = self.model.emb(ys)
                out, hidden = self.model.decoder(tgt_emb, hidden)
                logits = self.model.out(out[:, -1, :])
                next_id = torch.argmax(logits, dim=1).unsqueeze(1)
                ys = torch.cat([ys, next_id], dim=1)
                if int(next_id.item()) == eos:
                    break
        return decode(ys[0], self.itos)


def train(args):
    smiles = read_smiles_from_reactions(args.input, max_rows=args.max_rows)
    if not smiles:
        raise RuntimeError("No SMILES found.")
    stoi, itos = build_vocab(smiles)
    dataset = RepairDataset(smiles, stoi, max_len=args.max_len, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, stoi["<pad>"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RepairModel(
        vocab_size=len(stoi),
        emb=args.emb,
        hid=args.hid,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(args.output_dir, f"repair_{run_id}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_in = tgt[:, :-1]
            logits = model(src, tgt_in)
            tgt_out = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / max(len(loader), 1)
        print(f"epoch={epoch} train_loss={avg:.4f}")

    payload = {
        "model": model.state_dict(),
        "vocab": {"stoi": stoi, "itos": itos},
        "config": {
            "emb": args.emb,
            "hid": args.hid,
            "layers": args.layers,
            "dropout": args.dropout,
        },
    }
    torch.save(payload, ckpt_path)
    print(f"task=smiles_repair_train status=done checkpoint={ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="SMILES repair model (GRU).")
    parser.add_argument(
        "--input",
        default=os.path.join("MoleReact", "uspto_llm_clean.csv"),
        help="Input CSV with rs>>ps",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("MoleReact", "retro_transformer", "runs"),
    )
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--emb", type=int, default=128)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
