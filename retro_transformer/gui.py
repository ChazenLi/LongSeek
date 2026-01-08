import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch

from model import Seq2SeqTransformer
from predict import predict_one
from smiles_repair import Repairer

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    Draw = None
    RDLogger = None

try:
    import selfies as sf
except Exception:  # pragma: no cover - optional dependency
    sf = None

try:
    from PIL import ImageTk
except Exception:  # pragma: no cover - optional dependency
    ImageTk = None


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stoi = data["stoi"]
    itos = {int(k): v for k, v in data["itos"].items()}
    merges = data.get("bpe_merges")
    tokenizer = data.get("tokenizer", "atom")
    return stoi, itos, merges, tokenizer


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


class RetroGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Retro Transformer Inference")
        self.geometry("900x620")
        self.model = None
        self.stoi = None
        self.itos = None
        self.merges = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_preds = []
        self.last_pred_smiles = []
        self.last_pred_raw = []
        self.last_pred_fixed = []
        self.last_input_smiles = ""
        self.last_input_canon = ""
        self.last_input_selfies = ""
        self._img_ref = None
        self.repair_model = None
        self.tokenizer = "atom"

        if RDLogger is not None:
            RDLogger.DisableLog("rdApp.error")

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, **pad)

        path_frame = ttk.LabelFrame(main, text="Model Paths")
        path_frame.pack(fill=tk.X, **pad)

        ttk.Label(path_frame, text="Checkpoint:").grid(row=0, column=0, sticky=tk.W)
        self.ckpt_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.ckpt_var, width=70).grid(
            row=0, column=1, sticky=tk.EW
        )
        ttk.Button(path_frame, text="Browse", command=self._browse_ckpt).grid(
            row=0, column=2, sticky=tk.E
        )

        ttk.Label(path_frame, text="Vocab:").grid(row=1, column=0, sticky=tk.W)
        self.vocab_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.vocab_var, width=70).grid(
            row=1, column=1, sticky=tk.EW
        )
        ttk.Button(path_frame, text="Browse", command=self._browse_vocab).grid(
            row=1, column=2, sticky=tk.E
        )

        path_frame.columnconfigure(1, weight=1)

        cfg = ttk.LabelFrame(main, text="Model Config (match training)")
        cfg.pack(fill=tk.X, **pad)

        self.emb_var = tk.IntVar(value=256)
        self.heads_var = tk.IntVar(value=8)
        self.ff_var = tk.IntVar(value=512)
        self.enc_var = tk.IntVar(value=4)
        self.dec_var = tk.IntVar(value=4)
        self.drop_var = tk.DoubleVar(value=0.1)
        self.topk_var = tk.IntVar(value=5)
        self.maxlen_var = tk.IntVar(value=256)

        row = 0
        for label, var in [
            ("emb", self.emb_var),
            ("heads", self.heads_var),
            ("ff", self.ff_var),
            ("enc_layers", self.enc_var),
            ("dec_layers", self.dec_var),
            ("dropout", self.drop_var),
            ("top_k", self.topk_var),
            ("max_len", self.maxlen_var),
        ]:
            ttk.Label(cfg, text=label).grid(row=row // 4, column=(row % 4) * 2, sticky=tk.W)
            ttk.Entry(cfg, textvariable=var, width=8).grid(
                row=row // 4, column=(row % 4) * 2 + 1, sticky=tk.W
            )
            row += 1

        io_frame = ttk.LabelFrame(main, text="Input")
        io_frame.pack(fill=tk.X, **pad)
        ttk.Label(io_frame, text="Product SMILES:").grid(row=0, column=0, sticky=tk.W)
        self.smiles_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.smiles_var, width=80).grid(
            row=0, column=1, sticky=tk.EW
        )
        io_frame.columnconfigure(1, weight=1)

        ttk.Label(io_frame, text="Conditions (optional):").grid(
            row=1, column=0, sticky=tk.W
        )
        self.cond_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.cond_var, width=80).grid(
            row=1, column=1, sticky=tk.EW
        )
        ttk.Label(
            io_frame,
            text="Format: SOLV=...|CAT=...|TEMP=...|TIME=...",
        ).grid(row=2, column=1, sticky=tk.W)

        repair_frame = ttk.LabelFrame(main, text="Repair Model (optional)")
        repair_frame.pack(fill=tk.X, **pad)
        self.repair_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            repair_frame, text="Enable repair", variable=self.repair_enabled
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(repair_frame, text="Repair checkpoint:").grid(
            row=1, column=0, sticky=tk.W
        )
        self.repair_ckpt_var = tk.StringVar()
        ttk.Entry(repair_frame, textvariable=self.repair_ckpt_var, width=70).grid(
            row=1, column=1, sticky=tk.EW
        )
        ttk.Button(repair_frame, text="Browse", command=self._browse_repair).grid(
            row=1, column=2, sticky=tk.E
        )
        repair_frame.columnconfigure(1, weight=1)

        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, **pad)
        ttk.Button(btn_frame, text="Load Model", command=self._load_model).pack(
            side=tk.LEFT
        )
        ttk.Button(btn_frame, text="Predict", command=self._predict).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(btn_frame, text="Export SDF", command=self._export_sdf).pack(
            side=tk.LEFT
        )
        self.status_var = tk.StringVar(value=f"device={self.device.type}")
        ttk.Label(btn_frame, textvariable=self.status_var).pack(side=tk.RIGHT)

        out_frame = ttk.LabelFrame(main, text="Results")
        out_frame.pack(fill=tk.BOTH, expand=True, **pad)
        self.out_text = tk.Text(out_frame, height=12, wrap=tk.WORD)
        self.out_text.pack(fill=tk.BOTH, expand=False)
        self.img_label = ttk.Label(out_frame, text="RDKit image preview")
        self.img_label.pack(fill=tk.BOTH, expand=True)

    def _browse_ckpt(self):
        path = filedialog.askopenfilename(
            title="Select checkpoint", filetypes=[("Model", "*.pt"), ("All", "*.*")]
        )
        if path:
            self.ckpt_var.set(path)

    def _browse_vocab(self):
        path = filedialog.askopenfilename(
            title="Select vocab.json", filetypes=[("JSON", "*.json"), ("All", "*.*")]
        )
        if path:
            self.vocab_var.set(path)

    def _browse_repair(self):
        path = filedialog.askopenfilename(
            title="Select repair checkpoint",
            filetypes=[("Model", "*.pt"), ("All", "*.*")],
        )
        if path:
            self.repair_ckpt_var.set(path)

    def _load_model(self):
        ckpt = self.ckpt_var.get().strip()
        vocab = self.vocab_var.get().strip()
        if not ckpt or not os.path.exists(ckpt):
            messagebox.showerror("Error", "Checkpoint not found.")
            return
        if not vocab or not os.path.exists(vocab):
            messagebox.showerror("Error", "Vocab not found.")
            return
        self.stoi, self.itos, self.merges, self.tokenizer = load_vocab(vocab)
        self.model = Seq2SeqTransformer(
            num_encoder_layers=self.enc_var.get(),
            num_decoder_layers=self.dec_var.get(),
            emb_size=self.emb_var.get(),
            nhead=self.heads_var.get(),
            src_vocab_size=len(self.stoi),
            tgt_vocab_size=len(self.stoi),
            dim_feedforward=self.ff_var.get(),
            dropout=float(self.drop_var.get()),
        ).to(self.device)
        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.status_var.set(f"loaded device={self.device.type}")

    def _predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Load model first.")
            return
        raw_smi = self.smiles_var.get().strip()
        if not raw_smi:
            messagebox.showerror("Error", "Product SMILES is empty.")
            return
        self.last_input_smiles = raw_smi
        self.last_input_canon = ""
        self.last_input_selfies = ""
        smi = raw_smi
        cond = self.cond_var.get().strip()
        if cond:
            if self.tokenizer == "selfies":
                messagebox.showerror("Error", "SELFIES tokenizer does not support [COND].")
                return
            if not cond.startswith("[COND]"):
                cond = "[COND]" + cond
            smi = smi + cond.replace(" ", "_")
        src = smi
        if self.tokenizer == "selfies":
            try:
                _require_selfies_support()
                canon = _canonicalize_mixture(raw_smi)
            except Exception as exc:
                messagebox.showerror("Error", str(exc))
                return
            if not canon:
                messagebox.showerror(
                    "Error", "RDKit canonicalization failed for input SMILES."
                )
                return
            self.last_input_canon = canon
            src = _smiles_to_selfies(canon, roundtrip_check=True)
            if not src:
                messagebox.showerror(
                    "Error", "SELFIES conversion failed or roundtrip check failed."
                )
                return
            self.last_input_selfies = src

        def run():
            try:
                preds = predict_one(
                    self.model,
                    src,
                    self.stoi,
                    self.itos,
                    self.merges,
                    self.tokenizer,
                    self.device,
                    beam_size=self.topk_var.get(),
                    max_len=self.maxlen_var.get(),
                    length_penalty=0.6,
                    selfies_input=self.tokenizer == "selfies",
                )
                self.last_preds = preds
                self.last_pred_raw = [p[0] for p in preds]
                self.last_pred_smiles = self.last_pred_raw
                if self.repair_enabled.get():
                    self._ensure_repair_loaded()
                    self.last_pred_smiles = [
                        self.repair_model.repair(s, max_len=self.maxlen_var.get())
                        for s in self.last_pred_raw
                    ]
                self.out_text.delete("1.0", tk.END)
                self.out_text.insert(tk.END, f"product={smi}\n")
                if self.tokenizer == "selfies":
                    self.out_text.insert(
                        tk.END, f"product_canon={self.last_input_canon}\n"
                    )
                    self.out_text.insert(
                        tk.END, f"product_selfies={self.last_input_selfies}\n"
                    )
                for i, (text, score, norm) in enumerate(preds, start=1):
                    repaired = self.last_pred_smiles[i - 1]
                    if self.repair_enabled.get():
                        self.out_text.insert(
                            tk.END,
                            f"{i}\t{norm:.4f}\traw={text}\trepair={repaired}\n",
                        )
                    else:
                        self.out_text.insert(tk.END, f"{i}\t{norm:.4f}\t{text}\n")
                self._render_mols()
                self.status_var.set(f"done device={self.device.type}")
            except Exception as exc:
                self.status_var.set("error")
                messagebox.showerror("Error", str(exc))

        self.status_var.set("running...")
        threading.Thread(target=run, daemon=True).start()

    def _ensure_repair_loaded(self):
        if self.repair_model is not None:
            return
        ckpt = self.repair_ckpt_var.get().strip()
        if not ckpt or not os.path.exists(ckpt):
            raise RuntimeError("Repair checkpoint not found.")
        self.repair_model = Repairer(ckpt, device=self.device)

    def _render_mols(self):
        if Chem is None or Draw is None or ImageTk is None:
            self.img_label.configure(
                text="RDKit/Pillow not available for drawing.",
                image="",
            )
            self._img_ref = None
            return
        mols = []
        legends = []
        input_smi = self.last_input_canon or self.last_input_smiles
        if input_smi:
            mol = Chem.MolFromSmiles(input_smi)
            if mol is not None:
                mols.append(mol)
                legends.append("input")
        for idx, smi in enumerate(self.last_pred_smiles, start=1):
            fixed, reason = self._fix_smiles_if_needed(smi)
            mol = Chem.MolFromSmiles(fixed) if fixed else None
            if mol is None:
                continue
            mols.append(mol)
            suffix = "" if reason == "ok" else f"*"
            legends.append(f"{idx}{suffix}")
        if not mols:
            self.img_label.configure(text="No valid molecules to draw.", image="")
            self._img_ref = None
            return
        img = Draw.MolsToGridImage(
            mols,
            legends=legends,
            molsPerRow=3,
            subImgSize=(220, 200),
        )
        tk_img = ImageTk.PhotoImage(img)
        self._img_ref = tk_img
        self.img_label.configure(image=tk_img, text="")

    def _export_sdf(self):
        if Chem is None:
            messagebox.showerror("Error", "RDKit not available for SDF export.")
            return
        if not self.last_pred_smiles:
            messagebox.showerror("Error", "No predictions to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Save SDF",
            defaultextension=".sdf",
            filetypes=[("SDF", "*.sdf"), ("All", "*.*")],
        )
        if not path:
            return
        writer = Chem.SDWriter(path)
        for idx, smi in enumerate(self.last_pred_smiles, start=1):
            fixed, _ = self._fix_smiles_if_needed(smi)
            mol = Chem.MolFromSmiles(fixed) if fixed else None
            if mol is None:
                continue
            mol.SetProp("_Name", f"pred_{idx}")
            writer.write(mol)
        writer.close()
        messagebox.showinfo("Saved", f"SDF saved to {path}")

    def _fix_smiles_if_needed(self, smi):
        if Chem is None:
            return smi, "ok"
        if Chem.MolFromSmiles(smi) is not None:
            return smi, "ok"
        fixed = smi
        fixed, reason = self._fix_parentheses(fixed)
        fixed = self._fix_dots(fixed)
        if Chem.MolFromSmiles(fixed) is not None:
            return fixed, reason
        return "", "unfixed"

    @staticmethod
    def _fix_parentheses(smi):
        opens = smi.count("(")
        closes = smi.count(")")
        if opens == closes:
            return smi, "ok"
        if opens > closes:
            return smi + (")" * (opens - closes)), "fix_paren_append"
        # remove extra closing parens from the right
        diff = closes - opens
        out = smi
        while diff > 0 and ")" in out:
            idx = out.rfind(")")
            out = out[:idx] + out[idx + 1 :]
            diff -= 1
        return out, "fix_paren_trim"

    @staticmethod
    def _fix_dots(smi):
        out = smi.replace("..", ".")
        out = out.replace(".)", ")").replace("(.", "(")
        out = out.strip(".")
        return out


if __name__ == "__main__":
    RetroGUI().mainloop()
