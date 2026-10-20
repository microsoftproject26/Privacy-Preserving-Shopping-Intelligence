import ast
import pathlib

ROOT = pathlib.Path(r"F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\code")


def patch(fn, pairs):
    p = ROOT / fn
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert s.count(a) == 1, (fn, a[:70])
        s = s.replace(a, b)
    ast.parse(s)
    p.write_bytes(s.encode())
    print("patched", fn)


patch("sasrec_model.py", [
    # docstring: v3 recipe
    ('''Head: logits = LN_final(h_last) @ E[3:3+K]^T + b, with E the single item Parameter
(true tying; the product_idx_of_class == arange(K)+3 gate is enforced here).
"""''',
     '''Registry v3 recipe (Fable deep review, review/deep/FABLE_DEEP_REVIEW_RESPONSE.md), after the v2
failure (common-mode drift of the TIED item table under dense-CE Adam, amplified by the sqrt(d)
input scale into a Pre-LN stream with no input LN):
- input: z = LN_in(E[item] + P[pos] (+ rich side residual)), then input dropout (RecBole-style
  input LayerNorm, no sqrt(d) scaling);
- head: logits = LN_final(h_last) @ W_out^T + b with a SEPARATE output table W_out [K, d]
  (untied, as gSASRec / Transformers4Rec allow); class j <-> item token j+3 (arange gate kept);
- `recenter_output_()` removes the per-step common mode of W_out / b after each optimizer step:
  a per-query constant logit shift, exactly loss-neutral under softmax shift invariance.
"""'''),
    # constructor: input LN + separate output table
    ('''        self.final_norm = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.output_bias = nn.Parameter(torch.zeros(K))
''',
     '''        self.final_norm = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.input_norm = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.output_embed = nn.Parameter(torch.empty(K, d))    # untied output table (v3)
        self.output_bias = nn.Parameter(torch.zeros(K))
'''),
    # init: common scope, drawn from the common generator after the common modules
    ('''            self.final_norm.weight.fill_(1.0); self.final_norm.bias.zero_()
            self.output_bias.zero_()
''',
     '''            self.final_norm.weight.fill_(1.0); self.final_norm.bias.zero_()
            self.input_norm.weight.fill_(1.0); self.input_norm.bias.zero_()
            nn.init.normal_(self.output_embed, 0.0, 0.02, generator=g)
            self.output_bias.zero_()
'''),
    # unpadded input: no sqrt(d), input LN before dropout
    ('''            x = self.item_embed(tok) * scale + self.pos_embed(pos)
            if rich_on:
                tok_sel = {ch: side_kw[ch][:, :T].reshape(-1).index_select(0, flat_idx) for ch in SIDE_CHANNELS}
                num = side_kw["event_numeric_features"][:, :T].reshape(B * T, -1).index_select(0, flat_idx)
                flg = side_kw["event_quality_flags"][:, :T].reshape(B * T, -1).index_select(0, flat_idx)
                x = x + self._side_residual(tok_sel, num, flg)
            x = F.dropout(x, self.cfg.dropout, self.training)''',
     '''            x = self.item_embed(tok) + self.pos_embed(pos)
            if rich_on:
                tok_sel = {ch: side_kw[ch][:, :T].reshape(-1).index_select(0, flat_idx) for ch in SIDE_CHANNELS}
                num = side_kw["event_numeric_features"][:, :T].reshape(B * T, -1).index_select(0, flat_idx)
                flg = side_kw["event_quality_flags"][:, :T].reshape(B * T, -1).index_select(0, flat_idx)
                x = x + self._side_residual(tok_sel, num, flg)
            x = F.dropout(self.input_norm(x), self.cfg.dropout, self.training)'''),
    # padded reference input
    ('''        x = self.item_embed(item_tokens[:, :T]) * scale + self.pos_embed(pos)
        if rich_on:
            tok_sel = {ch: side_kw[ch][:, :T] for ch in SIDE_CHANNELS}
            x = x + self._side_residual(tok_sel, side_kw["event_numeric_features"][:, :T],
                                        side_kw["event_quality_flags"][:, :T])
        x = F.dropout(x * vm, self.cfg.dropout, self.training)''',
     '''        x = self.item_embed(item_tokens[:, :T]) + self.pos_embed(pos)
        if rich_on:
            tok_sel = {ch: side_kw[ch][:, :T] for ch in SIDE_CHANNELS}
            x = x + self._side_residual(tok_sel, side_kw["event_numeric_features"][:, :T],
                                        side_kw["event_quality_flags"][:, :T])
        x = F.dropout(self.input_norm(x) * vm, self.cfg.dropout, self.training)'''),
    ('''        scale = math.sqrt(d)
''', ''),
    # head: separate table + recenter + stats
    ('''    def output_weight(self) -> Tensor:
        return self.item_embed.weight[3:3 + self.K]          # a view of the single tied Parameter
''',
     '''    def output_weight(self) -> Tensor:
        return self.output_embed                              # untied output table [K, d] (v3)

    @torch.no_grad()
    def recenter_output_(self) -> None:
        """Remove the common mode of the output table and bias (after each optimizer step).
        Subtracting the row-mean vector from every W_out row and the mean from b shifts every
        logit of a query by the same constant: exactly loss-neutral (softmax shift invariance)."""
        self.output_embed.sub_(self.output_embed.mean(0, keepdim=True))
        self.output_bias.sub_(self.output_bias.mean())

    @torch.no_grad()
    def embedding_stats(self, sample_rows: Tensor) -> dict:
        """Cheap drift monitor: common-mode and median row norms of the input and output tables."""
        E = self.item_embed.weight[3:3 + self.K]
        W = self.output_embed
        return {"E_mean_row_norm": float(E.mean(0).norm()),
                "E_median_row_norm": float(E.index_select(0, sample_rows).norm(dim=1).median()),
                "W_mean_row_norm": float(W.mean(0).norm()),
                "W_median_row_norm": float(W.index_select(0, sample_rows).norm(dim=1).median()),
                "b_mean": float(self.output_bias.mean()), "b_std": float(self.output_bias.std())}
'''),
])
