import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        kwargs = dict(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        try:
            self.transformer = nn.Transformer(enable_nested_tensor=False, **kwargs)
        except TypeError:
            # Older PyTorch versions don't accept enable_nested_tensor.
            self.transformer = nn.Transformer(**kwargs)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_key_padding_mask=src_mask
        )

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask=tgt_mask,
        )


def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones((sz, sz), device=device), diagonal=1).bool()
    return mask


def create_padding_mask(seq, pad_id):
    return seq.eq(pad_id)


def greedy_decode(
    model,
    src,
    src_padding_mask,
    max_len,
    bos_id,
    eos_id,
    device,
):
    src_mask = None
    memory = model.encode(src, src_padding_mask)
    ys = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1), device)
        out = model.decode(ys, memory, tgt_mask)
        prob = model.generator(out[:, -1, :])
        next_token = torch.argmax(prob, dim=1).unsqueeze(1)
        ys = torch.cat([ys, next_token], dim=1)
        finished |= next_token.squeeze(1).eq(eos_id)
        if finished.all():
            break
    return ys


def beam_search(
    model,
    src,
    src_padding_mask,
    bos_id,
    eos_id,
    max_len,
    beam_size,
    device,
):
    model.eval()
    memory = model.encode(src, src_padding_mask)
    beams = [(torch.tensor([[bos_id]], device=device), 0.0, False)]

    for _ in range(max_len - 1):
        candidates = []
        for seq, score, finished in beams:
            if finished:
                candidates.append((seq, score, True))
                continue
            tgt_mask = generate_square_subsequent_mask(seq.size(1), device)
            out = model.decode(seq, memory, tgt_mask)
            logits = model.generator(out[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            topk = torch.topk(log_probs, k=beam_size)
            for token_id, token_score in zip(topk.indices.tolist(), topk.values.tolist()):
                next_seq = torch.cat(
                    [seq, torch.tensor([[token_id]], device=device)], dim=1
                )
                done = token_id == eos_id
                candidates.append((next_seq, score + token_score, done))
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]
        if all(finished for _, _, finished in beams):
            break
    return beams
