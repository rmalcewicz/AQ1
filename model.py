import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# BLOCK 1 — Stabilizer Embedding
# ─────────────────────────────────────────────
# At each round n, for each stabilizer i we have:
#   - m_ni  : raw measurement bit (0 or 1)
#   - d_ni  : detection event bit (did measurement flip vs previous round?)
# Both get embedded separately then summed into a D_MODEL vector.
# This vector is what gets added into the decoder hidden state each round.

class StabilizerEmbedding(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        # measurement bit: 0 or 1 -> d_model
        self.meas_embed = nn.Embedding(2, d_model)
        # detection event: 0 or 1 -> d_model
        self.event_embed = nn.Embedding(2, d_model)
        # learned position embedding: one per stabilizer
        # d=3 surface code has 8 stabilizers
        self.pos_embed = nn.Embedding(8, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, measurements, events, stabilizer_ids):
        """
        measurements:    (batch, n_stabilizers) int tensor, values 0/1
        events:          (batch, n_stabilizers) int tensor, values 0/1
        stabilizer_ids:  (n_stabilizers,) int tensor, values 0..7
        returns:         (batch, n_stabilizers, d_model)
        """
        m = self.meas_embed(measurements)          # (B, S, D)
        e = self.event_embed(events)               # (B, S, D)
        p = self.pos_embed(stabilizer_ids)         # (S, D) -> broadcast
        out = self.norm(m + e + p)
        return out

# ─────────────────────────────────────────────
# BLOCK 2 — Syndrome Transformer
# ─────────────────────────────────────────────
# At each round, after embedding, the 8 stabilizer vectors need to
# "talk to each other" — a stabilizer that flipped needs to know
# what its neighbours are doing.
#
# Two operations happen here:
#   1. Multi-head attention (global — every stabilizer attends to all others)
#      + attention bias (a learned per-pair offset based on spatial distance)
#   2. Conv layer (local — mixes information between adjacent stabilizers)
#
# The attention bias is the key AQ1 innovation vs a plain transformer —
# it encodes the 2D geometry of the surface code lattice directly
# into the attention scores, so the model knows which stabilizers
# are neighbours without having to learn it from scratch.

class SyndromeTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_stabilizers=8, dropout=0.1):
        super().__init__()
        self.n_stabilizers = n_stabilizers

        # Standard multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True   # (batch, seq, features)
        )

        # Learned attention bias: one scalar per (stabilizer_i, stabilizer_j) pair
        # Shape: (n_stabilizers, n_stabilizers) — added to raw attention logits
        self.attn_bias = nn.Parameter(
            torch.zeros(n_stabilizers, n_stabilizers)
        )

        # Conv layer for local spatial mixing
        # Treats the 8 stabilizers as a 1D sequence, kernel=3
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1   # same-size output
        )

        # Feed-forward after attention (standard transformer FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch, n_stabilizers, d_model)
        returns: (batch, n_stabilizers, d_model)
        """
        B, S, D = x.shape

        # 1. Multi-head attention with attention bias
        # attn_bias shape: (S, S) — broadcast across batch and heads
        # expand bias to (batch * n_heads, S, S)
        bias = self.attn_bias.unsqueeze(0).expand(B * self.attn.num_heads, -1, -1)
        attn_out, _ = self.attn(x, x, x, attn_mask=bias)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Conv layer (local mixing)
        # Conv1d expects (batch, channels, length) — transpose in/out
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x + self.dropout(conv_out))

        # 3. FFN
        x = self.norm3(x + self.dropout(self.ffn(x)))

        return x

# ─────────────────────────────────────────────
# BLOCK 3 — Recurrent Core (GRU)
# ─────────────────────────────────────────────
# After the syndrome transformer mixes information spatially,
# the GRU updates the hidden state temporally.
#
# Key idea: each stabilizer has its OWN hidden state vector h_i
# that persists across rounds. At round n:
#   h_i(n+1) = GRU(h_i(n), transformer_output_i(n))
#
# This means the model remembers the full history of each
# stabilizer — if stabilizer 3 flipped 5 rounds ago and just
# flipped back, the GRU hidden state captures that trajectory.
#
# In AQ1 the transformer computation happens INSIDE the recurrent
# block — the syndrome transformer output is fed directly into
# the GRU update at each step. This is what AQ2 changed (it moved
# the transformer outside for speed).

class RecurrentCore(nn.Module):
    def __init__(self, d_model=128, n_stabilizers=8):
        super().__init__()
        self.n_stabilizers = n_stabilizers
        self.d_model = d_model

        # One GRU cell per stabilizer — applied independently
        # GRU input: transformer output for stabilizer i this round
        # GRU hidden: hidden state for stabilizer i from last round
        self.gru_cell = nn.GRUCell(
            input_size=d_model,
            hidden_size=d_model
        )

    def forward(self, transformer_out, h_prev):
        """
        transformer_out: (batch, n_stabilizers, d_model)
        h_prev:          (batch, n_stabilizers, d_model)
        returns:
            h_next:      (batch, n_stabilizers, d_model)
        """
        B, S, D = transformer_out.shape

        # Reshape to (batch * n_stabilizers, d_model)
        # so we can apply GRUCell across all stabilizers at once
        x_flat = transformer_out.reshape(B * S, D)
        h_flat = h_prev.reshape(B * S, D)

        # GRU update
        h_next_flat = self.gru_cell(x_flat, h_flat)

        # Reshape back to (batch, n_stabilizers, d_model)
        h_next = h_next_flat.reshape(B, S, D)

        return h_next

    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state to zeros at the start of each experiment.
        Call this before processing round 0.
        """
        return torch.zeros(
            batch_size, self.n_stabilizers, self.d_model,
            device=device
        )
    
# ─────────────────────────────────────────────
# BLOCK 4 — Readout Network
# ─────────────────────────────────────────────
# After the final round, we have a hidden state vector for each
# of the 8 stabilizers. We need to collapse these into a single
# number: P(logical Z flipped).
#
# Two steps:
#   1. Pool across stabilizers — average all 8 hidden state vectors
#      into one vector of size d_model
#   2. MLP — two linear layers with GELU, ending in a single logit
#      sigmoid(logit) = probability of logical error
#
# The pooling is simple mean pooling — AQ1 doesn't use anything
# fancier here. The MLP then does the final classification.

class ReadoutNetwork(nn.Module):
    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, h_final):
        """
        h_final: (batch, n_stabilizers, d_model)
        returns: (batch,) — probability of logical error
        """
        # Mean pool across all stabilizers
        # (batch, n_stabilizers, d_model) -> (batch, d_model)
        pooled = h_final.mean(dim=1)

        # MLP -> single logit
        logit = self.mlp(pooled).squeeze(-1)  # (batch,)

        return logit  # raw logit — apply sigmoid outside for BCEWithLogitsLoss

# ─────────────────────────────────────────────
# FULL MODEL — AQ1Decoder
# ─────────────────────────────────────────────
# Wires all four blocks together.
# At each round n:
#   1. embed the current round's measurements + events
#   2. add embedding into hidden state
#   3. run syndrome transformer on hidden state
#   4. update hidden state with GRU
# After all T rounds:
#   5. run readout on final hidden state

class AQ1Decoder(nn.Module):
    def __init__(
        self,
        n_stabilizers=8,
        d_model=128,
        n_heads=4,
        n_transformer_layers=4,
        dropout=0.1
    ):
        super().__init__()
        self.n_stabilizers = n_stabilizers
        self.d_model = d_model

        self.embedding = StabilizerEmbedding(d_model)

        # Stack multiple transformer layers per round
        self.transformers = nn.ModuleList([
            SyndromeTransformer(d_model, n_heads, n_stabilizers, dropout)
            for _ in range(n_transformer_layers)
        ])

        self.recurrent = RecurrentCore(d_model, n_stabilizers)
        self.readout = ReadoutNetwork(d_model, dropout)

    def forward(self, measurements, events, stabilizer_ids):
        """
        measurements:   (batch, n_rounds, n_stabilizers) int tensor
        events:         (batch, n_rounds, n_stabilizers) int tensor
        stabilizer_ids: (n_stabilizers,) int tensor
        returns:        (batch,) logits
        """
        B, T, S = measurements.shape
        device = measurements.device

        # Initialize hidden state
        h = self.recurrent.init_hidden(B, device)

        # Process each round recurrently
        for t in range(T):
            # Embed this round's data
            emb = self.embedding(
                measurements[:, t, :],   # (B, S)
                events[:, t, :],          # (B, S)
                stabilizer_ids            # (S,)
            )

            # Add embedding into hidden state (residual injection)
            h = h + emb

            # Run transformer layers
            x = h
            for transformer in self.transformers:
                x = transformer(x)

            # Update hidden state with GRU
            h = self.recurrent(x, h)

        # Readout from final hidden state
        logit = self.readout(h)
        return logit
