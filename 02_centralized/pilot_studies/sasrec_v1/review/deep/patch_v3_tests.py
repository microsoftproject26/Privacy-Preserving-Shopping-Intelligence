import ast
import pathlib

CODE = pathlib.Path(r"F:\Queens University\Microsoft Project\eda\.local_ai\sasrec_v1\code")


def patch(fn, pairs):
    p = CODE / fn
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert s.count(a) == 1, (fn, a[:70])
        s = s.replace(a, b)
    ast.parse(s)
    p.write_bytes(s.encode())
    print("patched", fn)


# 1. independent reference builder: v3 input = LN_in(E + P), no sqrt(d)
patch("tests/test_attention.py", [
    ('''    scale = math.sqrt(d)
    pos = ar.unsqueeze(0).expand(B, T)
    x = model.item_embed(item_tokens) * scale + model.pos_embed(pos)
    x = x * vmask''',
     '''    pos = ar.unsqueeze(0).expand(B, T)
    # registry v3 input: LayerNorm_in(E[item] + P[pos]), no sqrt(d) scale
    x = F.layer_norm(model.item_embed(item_tokens) + model.pos_embed(pos), (d,),
                     model.input_norm.weight, model.input_norm.bias, model.input_norm.eps)
    x = x * vmask'''),
])

# 2. mutant: re-tie the head (negative control for the v3 untied-head test)
patch("tests/mutants/model_mutants.py", [
    ('''@contextmanager
def untie_head_weight(model):''',
     '''@contextmanager
def retie_head_weight(model):
    """Registry v3 negative control: `output_weight()` silently returns a VIEW of the input
    table item_embed.weight[3:3+K] again (the v2 tying that caused common-mode drift)."""
    orig = model.output_weight

    def mutant_output_weight(self):
        return self.item_embed.weight[3:3 + self.K]

    model.output_weight = types.MethodType(mutant_output_weight, model)
    try:
        yield model
    finally:
        model.output_weight = orig


@contextmanager
def recenter_by_row_(model):
    """Negative control for recentering: subtracts a PER-CLASS (non-constant) shift from the
    bias, which is NOT loss-neutral (breaks softmax shift invariance)."""
    orig = model.recenter_output_

    def mutant(self):
        with torch.no_grad():
            self.output_bias.sub_(torch.linspace(0.0, 1.0, self.output_bias.numel(), device=self.output_bias.device))

    model.recenter_output_ = types.MethodType(mutant, model)
    try:
        yield model
    finally:
        model.recenter_output_ = orig


@contextmanager
def untie_head_weight(model):'''),
])

# 3. head test: untied head (v3) + loss-neutral recentering, each with a negative control
patch("tests/test_head_objective.py", [
    ('''# ---------------------------------------------------------------------------
# 05 item 17 / Fable #3: tying by Parameter identity, not equal values.
# ---------------------------------------------------------------------------
def test_head_weight_is_a_view_of_item_embed_and_appears_once_in_optimizer_groups():
    cfg = make_config(group="item_only", d_model=12, blocks=1, heads=2, dropout=0.0)
    model = make_model(cfg, K=TINY_K, seed=4)

    ow = model.output_weight()
    # Shared storage, not merely equal values: mutate item_embed.weight in
    # place and confirm output_weight() reflects it immediately.
    with torch.no_grad():
        probe = model.item_embed.weight[5, 0].item()
        model.item_embed.weight[5, 0] += 1.2345
    assert abs(ow[5 - 3, 0].item() - (probe + 1.2345)) < 1e-6, \\
        "output_weight() must be a live view into item_embed.weight, not a copy"
    with torch.no_grad():
        model.item_embed.weight[5, 0] -= 1.2345  # restore

    groups = reuse_train.build_param_groups(model)
    all_params = [p for g in groups for p in g["params"]]
    n_identity_matches = sum(1 for p in all_params if p is model.item_embed.weight)
    assert n_identity_matches == 1, "the tied item table must appear exactly once across optimizer groups"
    ids = [id(p) for p in all_params]
    assert len(ids) == len(set(ids)), "optimizer groups must not contain a duplicated Parameter"

    # Negative control: an untied head weight is a genuinely different tensor.
    with mm.untie_head_weight(model) as (_, copy_w):
        assert copy_w is not model.item_embed.weight, "sanity: the mutant copy must not be the same object"
        with torch.no_grad():
            model.item_embed.weight[6, 0] += 9.0
        live_value = model.item_embed.weight[6, 0].item()
        mutant_output_value = model.output_weight()[6 - 3, 0].item()
        assert abs(live_value - mutant_output_value) > 1e-6, \\
            "negative control failed: the untied mutant must NOT reflect item_embed.weight mutations"
        with torch.no_grad():
            model.item_embed.weight[6, 0] -= 9.0
''',
     '''# ---------------------------------------------------------------------------
# Registry v3 (Fable deep review FIX 1): the output table is a SEPARATE Parameter
# (untied from the input item table), appearing exactly once in the optimizer groups.
# ---------------------------------------------------------------------------
def _is_untied(model) -> bool:
    ow = model.output_weight()
    with torch.no_grad():
        probe = ow[2, 0].item()
        model.item_embed.weight[5, 0] += 1.2345
        changed = abs(model.output_weight()[2, 0].item() - probe) > 1e-9
        model.item_embed.weight[5, 0] -= 1.2345
    return (ow is model.output_embed) and not changed and ow.data_ptr() != model.item_embed.weight.data_ptr() \\
        and tuple(ow.shape) == (model.K, model.d)


def test_output_head_is_untied_separate_parameter_once_in_optimizer_groups():
    cfg = make_config(group="item_only", d_model=12, blocks=1, heads=2, dropout=0.0)
    model = make_model(cfg, K=TINY_K, seed=4)
    assert _is_untied(model), "v3 head must be the separate output_embed Parameter, not a view of item_embed"
    groups = reuse_train.build_param_groups(model)
    all_params = [p for g in groups for p in g["params"]]
    assert sum(1 for p in all_params if p is model.output_embed) == 1
    assert sum(1 for p in all_params if p is model.item_embed.weight) == 1
    ids = [id(p) for p in all_params]
    assert len(ids) == len(set(ids)), "optimizer groups must not contain a duplicated Parameter"
    # the head really uses output_embed: a gradient reaches it through the logits
    lengths = torch.tensor([3, 1])
    batch = make_batch(lengths, T=3, K=TINY_K, seed=2)
    logits = model(**forward_kwargs(batch, model))
    logits[:, 0].sum().backward()
    assert model.output_embed.grad is not None and torch.count_nonzero(model.output_embed.grad[0]) > 0
    # Negative control: a re-tied head (the v2 bug) must be detected.
    with mm.retie_head_weight(model):
        assert not _is_untied(model), "negative control failed: a re-tied head was not detected"


def test_recenter_output_is_loss_neutral_and_zeroes_common_mode():
    cfg = make_config(group="item_only", d_model=12, blocks=1, heads=2, dropout=0.0)
    model = make_model(cfg, K=TINY_K, seed=9).eval()
    with torch.no_grad():   # give the table/bias a large common mode first
        model.output_embed.add_(torch.randn(1, model.d) * 3.0)
        model.output_bias.add_(5.0)
    lengths = torch.tensor([4, 2, 1])
    batch = make_batch(lengths, T=4, K=TINY_K, seed=5)
    kw = forward_kwargs(batch, model)
    with torch.no_grad():
        lp_before = torch.log_softmax(model(**kw).double(), dim=1)
        model.recenter_output_()
        lp_after = torch.log_softmax(model(**kw).double(), dim=1)
    assert torch.allclose(lp_before, lp_after, atol=1e-5), "recentering must not change log-probabilities"
    assert float(model.output_embed.mean(0).norm()) < 1e-5 and abs(float(model.output_bias.mean())) < 1e-5
    # Negative control: a non-constant (per-class) shift is NOT loss-neutral.
    with mm.recenter_by_row_(model):
        with torch.no_grad():
            model.recenter_output_()
            lp_bad = torch.log_softmax(model(**kw).double(), dim=1)
    assert not torch.allclose(lp_after, lp_bad, atol=1e-5), "negative control failed: per-class shift went undetected"
'''),
])

# 4. rich common params include the v3 modules
patch("tests/test_rich.py", [
    ('COMMON_PARAM_PREFIXES = ("item_embed", "pos_embed", "blocks", "final_norm", "output_bias")',
     'COMMON_PARAM_PREFIXES = ("item_embed", "pos_embed", "blocks", "final_norm", "output_bias",\n'
     '                         "input_norm", "output_embed")'),
])
