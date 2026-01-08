import argparse
import csv
import json
import math
import os
import random
import sys
import time
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import (
    SmilesDataset,
    build_vocab,
    collate_batch,
    decode,
    get_canon_stats,
    make_pairs,
    read_reactions,
    reset_canon_stats,
    split_pairs,
    tokenize_smiles,
    tokenize_text,
    train_bpe,
)
from model import (
    Seq2SeqTransformer,
    create_padding_mask,
    generate_square_subsequent_mask,
    greedy_decode,
    beam_search,
)


def train_epoch(
    model,
    optimizer,
    criterion,
    data_loader,
    pad_id,
    device,
    scaler,
    use_amp,
    scheduler,
    clip_norm,
):
    model.train()
    total_loss = 0.0
    for step, (src, tgt) in enumerate(data_loader, start=1):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]

        src_padding_mask = create_padding_mask(src, pad_id)
        tgt_padding_mask = create_padding_mask(tgt_input, pad_id)
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(
                src,
                tgt_input,
                src_mask=None,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
            )
            tgt_out = tgt[:, 1:]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)
            )
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        if step % 200 == 0:
            avg = total_loss / step
            print(f"step={step} train_loss={avg:.4f}")
    return total_loss / max(len(data_loader), 1)


def _load_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
    except Exception:
        return None
    return Chem


def build_scheduler(optimizer, scheduler_name, warmup_steps, total_steps):
    if scheduler_name == "none":
        return None
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    warmup_steps = min(warmup_steps, total_steps)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_name == "linear":
            return max(0.0, 1.0 - progress)
        if scheduler_name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _metric_improved(metric, best, higher_is_better, min_delta):
    if best is None:
        return True
    if higher_is_better:
        return metric > best + min_delta
    return metric < best - min_delta


def set_seed(seed, deterministic=False):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def estimate_truncation(
    pairs,
    max_len,
    merges,
    tokenizer,
    Chem,
    roundtrip_check,
    selfies_input,
):
    total = 0
    src_trunc = 0
    tgt_trunc = 0
    max_src = 0
    max_tgt = 0
    for src, tgt in pairs:
        src_tokens = tokenize_text(
            src,
            tokenizer=tokenizer,
            merges=merges,
            Chem=Chem,
            roundtrip_check=roundtrip_check,
            selfies_input=selfies_input,
        )
        tgt_tokens = tokenize_text(
            tgt,
            tokenizer=tokenizer,
            merges=merges,
            Chem=Chem,
            roundtrip_check=roundtrip_check,
            selfies_input=selfies_input,
        )
        if src_tokens is None or tgt_tokens is None:
            continue
        src_len = len(src_tokens) + 2
        tgt_len = len(tgt_tokens) + 2
        total += 1
        if src_len > max_len:
            src_trunc += 1
        if tgt_len > max_len:
            tgt_trunc += 1
        max_src = max(max_src, src_len)
        max_tgt = max(max_tgt, tgt_len)
    return {
        "total": total,
        "src_truncated": src_trunc,
        "tgt_truncated": tgt_trunc,
        "src_trunc_ratio": src_trunc / max(total, 1),
        "tgt_trunc_ratio": tgt_trunc / max(total, 1),
        "src_max_len": max_src,
        "tgt_max_len": max_tgt,
    }


def evaluate(
    model,
    criterion,
    data_loader,
    pad_id,
    device,
    bos_id,
    eos_id,
    itos,
    use_amp,
    top_k,
    max_samples,
    merges,
    Chem,
    tokenizer,
    eval_random,
    eval_seed,
):
    model.eval()
    max_samples_eval = max_samples
    if max_samples and eval_random and hasattr(data_loader, "dataset"):
        dataset = data_loader.dataset
        total_len = len(dataset)
        sample_size = min(max_samples, total_len)
        rng = random.Random(eval_seed)
        if sample_size < total_len:
            indices = rng.sample(range(total_len), sample_size)
        else:
            indices = list(range(total_len))
        subset = Subset(dataset, indices)
        data_loader = DataLoader(
            subset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            collate_fn=data_loader.collate_fn,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
        )
        max_samples_eval = 0
    total_loss = 0.0
    seen_batches = 0
    exact_matches = 0
    topk_matches = 0
    valid_top1 = 0
    valid_topk = 0
    total = 0
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            src_padding_mask = create_padding_mask(src, pad_id)
            tgt_padding_mask = create_padding_mask(tgt_input, pad_id)
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(
                    src,
                    tgt_input,
                    src_mask=None,
                    tgt_mask=tgt_mask,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                )
                tgt_out = tgt[:, 1:]
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)
                )
            total_loss += loss.item()
            seen_batches += 1

            for i in range(tgt.size(0)):
                if max_samples_eval and total >= max_samples_eval:
                    break
                src_i = src[i : i + 1]
                mask_i = src_padding_mask[i : i + 1]
                beams = beam_search(
                    model,
                    src_i,
                    mask_i,
                    bos_id,
                    eos_id,
                    max_len=tgt.size(1),
                    beam_size=top_k,
                    device=device,
                )
                pred_list = [decode(b[0][0], itos, tokenizer=tokenizer) for b in beams]
                pred = pred_list[0] if pred_list else ""
                gold = decode(tgt[i], itos, tokenizer=tokenizer)
                if pred == gold:
                    exact_matches += 1
                if gold in pred_list:
                    topk_matches += 1
                if Chem is not None:
                    if pred and Chem.MolFromSmiles(pred) is not None:
                        valid_top1 += 1
                    if any(Chem.MolFromSmiles(p) is not None for p in pred_list):
                        valid_topk += 1
                total += 1
            if max_samples_eval and total >= max_samples_eval:
                break
    acc = exact_matches / max(total, 1)
    acc_k = topk_matches / max(total, 1)
    valid1 = valid_top1 / max(total, 1)
    validk = valid_topk / max(total, 1)
    return total_loss / max(seen_batches, 1), acc, acc_k, valid1, validk


def main():
    warnings.filterwarnings(
        "ignore",
        message="The PyTorch API of nested tensors is in prototype stage",
    )
    parser = argparse.ArgumentParser(description="Minimal SMILES seq2seq transformer.")
    parser.add_argument(
        "--csv",
        default=os.path.join("MoleReact", "uspto_llm_selfies.csv"),
        help="Input CSV path",
    )
    parser.add_argument("--direction", choices=["retro", "forward"], default="retro")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--emb", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff", type=int, default=512)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "linear", "cosine"],
        default="none",
        help="Learning rate schedule with optional warmup",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Warmup steps for lr scheduler (per optimizer step)",
    )
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=0.0,
        help="Max grad norm (0 to disable)",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping based on validation metric",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Epochs without improvement before stopping",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=["val_loss", "val_exact", "val_topk"],
        default="val_loss",
        help="Metric for early stopping",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum change to count as improvement",
    )
    parser.add_argument(
        "--early-stop-max-epochs",
        type=int,
        default=20,
        help="Cap epochs when early stopping is enabled",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic cudnn settings (slower)",
    )
    parser.add_argument(
        "--tokenizer", choices=["atom", "bpe", "selfies"], default="selfies"
    )
    parser.add_argument("--bpe-merges", type=int, default=1000)
    parser.add_argument("--bpe-min-pair-freq", type=int, default=2)
    parser.add_argument("--bpe-max-samples", type=int, default=20000)
    parser.add_argument("--eval-top-k", type=int, default=5)
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=2000,
        help="Max samples to evaluate (0 to use full validation set)",
    )
    parser.add_argument(
        "--eval-random",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly sample eval subset when max_samples > 0",
    )
    parser.add_argument(
        "--use-conditions",
        action="store_true",
        help="Append solvents/catalyst/temp/time to src with [COND] tag",
    )
    parser.add_argument(
        "--selfies-roundtrip-check",
        action="store_true",
        help="Require SMILES->SELFIES->SMILES canonical consistency",
    )
    parser.add_argument(
        "--selfies-input",
        action="store_true",
        help="Input CSV already contains SELFIES reactions in selfies_rxn column",
    )
    parser.add_argument(
        "--selfies-column",
        default="selfies_rxn",
        help="Column name for SELFIES reactions",
    )
    parser.add_argument(
        "--selfies-progress-every",
        type=int,
        default=20000,
        help="Progress print interval for SELFIES preprocessing",
    )
    parser.add_argument(
        "--qc-pass-only",
        action="store_true",
        help="Only train on rows with qc_pass=true (mapped QC output)",
    )
    parser.add_argument(
        "--qc-field",
        default="qc_pass",
        help="Column name for qc pass flag",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("MoleReact", "retro_transformer", "runs"),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for key options (tokenizer/filters) in a console",
    )
    parser.add_argument(
        "--metrics-file",
        default="metrics.jsonl",
        help="Metrics filename inside run_dir",
    )
    parser.add_argument(
        "--plot-metrics",
        action="store_true",
        help="Save simple metrics plot to run_dir",
    )
    parser.add_argument(
        "--report-truncation",
        action="store_true",
        help="Report truncation ratio and max lengths for train/val",
    )
    args = parser.parse_args()

    interactive = args.interactive and sys.stdin.isatty()
    if interactive:
        if "--tokenizer" not in sys.argv:
            args.tokenizer = "selfies"
            choice = input("tokenizer (atom/bpe/selfies) [selfies]: ").strip().lower()
            if choice:
                args.tokenizer = choice
        if args.tokenizer == "bpe" and "--bpe-merges" not in sys.argv:
            val = input("bpe_merges [2000]: ").strip()
            if val:
                args.bpe_merges = int(val)
        if args.tokenizer == "selfies" and "--selfies-roundtrip-check" not in sys.argv:
            val = input("selfies_roundtrip_check (y/N): ").strip().lower()
            if val in ("y", "yes"):
                args.selfies_roundtrip_check = True
        if "--qc-pass-only" not in sys.argv:
            val = input("qc_pass_only (y/N): ").strip().lower()
            if val in ("y", "yes"):
                args.qc_pass_only = True

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    if args.tokenizer == "selfies":
        try:
            with open(args.csv, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, [])
        except Exception as exc:
            raise RuntimeError(f"Failed to read CSV header: {exc}")
        if args.selfies_column not in header:
            raise ValueError(
                f"Missing '{args.selfies_column}' column for SELFIES training."
            )
        if not args.selfies_input:
            args.selfies_input = True
            print(f"selfies_input=1 detected column={args.selfies_column}")

    rows = read_reactions(
        args.csv,
        max_rows=args.max_rows,
        include_conditions=args.use_conditions,
        qc_pass_only=args.qc_pass_only,
        qc_field=args.qc_field,
        selfies_input=args.selfies_input,
        selfies_column=args.selfies_column,
    )
    pairs = make_pairs(
        rows, direction=args.direction, include_conditions=args.use_conditions
    )

    if args.tokenizer == "selfies" and args.use_conditions:
        raise ValueError("SELFIES tokenizer does not support [COND] inputs.")
    if args.selfies_input and args.tokenizer != "selfies":
        raise ValueError("--selfies-input requires --tokenizer selfies.")
    train_pairs, val_pairs = split_pairs(
        pairs, val_ratio=args.val_ratio, seed=args.seed
    )

    set_seed(args.seed, deterministic=args.deterministic)

    token_lists = []
    merges = None
    Chem = None
    if args.tokenizer == "selfies":
        reset_canon_stats()
        Chem = _load_rdkit()
        if Chem is None:
            raise RuntimeError("RDKit is required for SELFIES tokenizer.")
        filtered_train = []
        filtered_val = []
        skipped_train = 0
        skipped_val = 0
        for idx, (src, tgt) in enumerate(train_pairs, start=1):
            src_tokens = tokenize_text(
                src,
                tokenizer="selfies",
                Chem=Chem,
                roundtrip_check=args.selfies_roundtrip_check,
                selfies_input=args.selfies_input,
            )
            tgt_tokens = tokenize_text(
                tgt,
                tokenizer="selfies",
                Chem=Chem,
                roundtrip_check=args.selfies_roundtrip_check,
                selfies_input=args.selfies_input,
            )
            if src_tokens is None or tgt_tokens is None:
                skipped_train += 1
                continue
            filtered_train.append((src, tgt))
            token_lists.append(src_tokens)
            token_lists.append(tgt_tokens)
            if args.selfies_progress_every and idx % args.selfies_progress_every == 0:
                print(f"selfies_progress train_seen={idx} kept={len(filtered_train)} skipped={skipped_train}")
        for idx, (src, tgt) in enumerate(val_pairs, start=1):
            src_tokens = tokenize_text(
                src,
                tokenizer="selfies",
                Chem=Chem,
                roundtrip_check=args.selfies_roundtrip_check,
                selfies_input=args.selfies_input,
            )
            tgt_tokens = tokenize_text(
                tgt,
                tokenizer="selfies",
                Chem=Chem,
                roundtrip_check=args.selfies_roundtrip_check,
                selfies_input=args.selfies_input,
            )
            if src_tokens is None or tgt_tokens is None:
                skipped_val += 1
                continue
            filtered_val.append((src, tgt))
            if args.selfies_progress_every and idx % args.selfies_progress_every == 0:
                print(f"selfies_progress val_seen={idx} kept={len(filtered_val)} skipped={skipped_val}")
        train_pairs = filtered_train
        val_pairs = filtered_val
        print(f"selfies_filter skipped_train={skipped_train} skipped_val={skipped_val}")
        stats = get_canon_stats()
        print(
            "selfies_stats "
            f"h_removed={stats['h_removed']} "
            f"sanitize_fail={stats['sanitize_fail']} "
            f"encode_fail={stats['encode_fail']}"
        )
        stoi, itos = build_vocab(token_lists, min_freq=args.min_freq)
    elif args.tokenizer == "bpe":
        for idx, (src, tgt) in enumerate(train_pairs):
            token_lists.append(tokenize_smiles(src))
            token_lists.append(tokenize_smiles(tgt))
            if args.bpe_max_samples and len(token_lists) >= args.bpe_max_samples:
                break
        merges, merged_lists = train_bpe(
            token_lists,
            num_merges=args.bpe_merges,
            min_pair_freq=args.bpe_min_pair_freq,
        )
        stoi, itos = build_vocab(merged_lists, min_freq=args.min_freq)
    else:
        for src, tgt in train_pairs:
            token_lists.append(tokenize_smiles(src))
            token_lists.append(tokenize_smiles(tgt))
        stoi, itos = build_vocab(token_lists, min_freq=args.min_freq)

    pad_id = stoi["<pad>"]
    bos_id = stoi["<bos>"]
    eos_id = stoi["<eos>"]

    train_data = SmilesDataset(
        train_pairs,
        stoi,
        max_len=args.max_len,
        merges=merges,
        tokenizer=args.tokenizer,
        Chem=Chem,
        roundtrip_check=args.selfies_roundtrip_check,
        selfies_input=args.selfies_input,
    )
    val_data = SmilesDataset(
        val_pairs,
        stoi,
        max_len=args.max_len,
        merges=merges,
        tokenizer=args.tokenizer,
        Chem=Chem,
        roundtrip_check=args.selfies_roundtrip_check,
        selfies_input=args.selfies_input,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        if not args.deterministic:
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"cuda=1 device={name} capability={cap}")
    else:
        print("cuda=0 device=cpu")
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if Chem is None:
        Chem = _load_rdkit()
    if Chem is None:
        print("rdkit=0 validity metrics disabled")
    else:
        print("rdkit=1 validity metrics enabled")

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "vocab.json"), "w", encoding="utf-8") as f:
        payload = {"stoi": stoi, "itos": itos, "tokenizer": args.tokenizer}
        if merges:
            payload["bpe_merges"] = merges
        json.dump(payload, f, ensure_ascii=True, indent=2)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=True, indent=2)

    if args.report_truncation:
        trunc_train = estimate_truncation(
            train_pairs,
            args.max_len,
            merges,
            args.tokenizer,
            Chem,
            args.selfies_roundtrip_check,
            args.selfies_input,
        )
        trunc_val = estimate_truncation(
            val_pairs,
            args.max_len,
            merges,
            args.tokenizer,
            Chem,
            args.selfies_roundtrip_check,
            args.selfies_input,
        )
        trunc_payload = {"train": trunc_train, "val": trunc_val}
        with open(os.path.join(run_dir, "truncation.json"), "w", encoding="utf-8") as f:
            json.dump(trunc_payload, f, ensure_ascii=True, indent=2)
        print(
            "truncation "
            f"train_src={trunc_train['src_trunc_ratio']:.4f} "
            f"train_tgt={trunc_train['tgt_trunc_ratio']:.4f} "
            f"val_src={trunc_val['src_trunc_ratio']:.4f} "
            f"val_tgt={trunc_val['tgt_trunc_ratio']:.4f}"
        )

    total_steps = len(train_loader) * max(1, args.epochs)
    scheduler = build_scheduler(
        optimizer, args.lr_scheduler, args.warmup_steps, total_steps
    )

    metrics_path = os.path.join(run_dir, args.metrics_file)
    metrics = []
    best_val = None
    best_exact = None
    early_best = None
    bad_epochs = 0
    max_epochs = args.epochs
    if args.early_stop and args.early_stop_max_epochs:
        max_epochs = min(max_epochs, args.early_stop_max_epochs)
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            pad_id,
            device,
            scaler,
            use_amp,
            scheduler,
            args.clip_norm,
        )
        val_loss, val_acc, val_acc_k, valid1, validk = evaluate(
            model,
            criterion,
            val_loader,
            pad_id,
            device,
            bos_id,
            eos_id,
            itos,
            use_amp,
            args.eval_top_k,
            args.eval_max_samples,
            merges,
            Chem,
            args.tokenizer,
            args.eval_random,
            args.seed + epoch,
        )
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        if best_exact is None or val_acc > best_exact:
            best_exact = val_acc
            torch.save(model.state_dict(), os.path.join(run_dir, "best_exact.pt"))
        trend = "up" if val_acc == best_exact else "down"
        gap = train_loss - val_loss
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            ## val_exact=0.0000：Top‑1 严格匹配率（预测 SMILES 与真实完全一致）为 0
            ## val_top5=0.0050：Top‑5 命中率（真实反应物出现在前 5 个候选中）0.5%
            ## valid@1=0.6650：Top‑1 预测中能被 RDKit 解析的比例 66.5%
            ## valid@5=0.7750：Top‑5 中至少有一个能解析的比例 77.5%
            ## gap(train-val)=0.8975：训练损失与验证损失差值（越大越可能过拟合或分布差异）
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_exact={val_acc:.4f} "
            f"val_top{args.eval_top_k}={val_acc_k:.4f} "
            f"valid@1={valid1:.4f} valid@{args.eval_top_k}={validk:.4f}"
        )
        print(
            f"analysis trend={trend} gap(train-val)={gap:.4f} "
            f"use_conditions={int(args.use_conditions)} tokenizer={args.tokenizer} "
            f"lr={lr_now:.6g}"
        )

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_exact": val_acc,
            "val_topk": val_acc_k,
            "valid1": valid1,
            "validk": validk,
            "gap": gap,
            "lr": lr_now,
        }
        metrics.append(entry)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")

        if args.early_stop:
            if args.early_stop_metric == "val_exact":
                current = val_acc
                higher_is_better = True
            elif args.early_stop_metric == "val_topk":
                current = val_acc_k
                higher_is_better = True
            else:
                current = val_loss
                higher_is_better = False
            improved = _metric_improved(
                current, early_best, higher_is_better, args.early_stop_min_delta
            )
            if improved:
                early_best = current
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                print(
                    "early_stop=1 "
                    f"metric={args.early_stop_metric} "
                    f"patience={args.early_stop_patience}"
                )
                break

    if args.plot_metrics and metrics:
        try:
            import matplotlib.pyplot as plt

            epochs = [m["epoch"] for m in metrics]
            train_losses = [m["train_loss"] for m in metrics]
            val_losses = [m["val_loss"] for m in metrics]
            val_exact = [m["val_exact"] for m in metrics]
            val_topk = [m["val_topk"] for m in metrics]

            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(epochs, train_losses, label="train_loss")
            ax1.plot(epochs, val_losses, label="val_loss")
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("loss")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(epochs, val_exact, label="val_exact", linestyle="--")
            ax2.plot(
                epochs,
                val_topk,
                label=f"val_top{args.eval_top_k}",
                linestyle="--",
            )
            ax2.set_ylabel("accuracy")

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="best")

            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, "metrics.png"), dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"plot_metrics=0 error={exc}")

    print(f"run_dir={run_dir}")
    print(f"task=train status=done run_dir={run_dir}")


if __name__ == "__main__":
    main()
