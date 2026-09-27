"""Synthetic H0 tasks: passkey, copying, MQAR (multi-query associative recall).

Token layout: 0-9 digits, 10-29 filler, 30=P, 31=Q, 32=SEP. vocab=34.
All batchers return (idx, tgt, mask, pos): idx/tgt are the LM-shifted pair
(model input length n), mask marks loss positions in tgt, pos is the needle
position per sample (None for non-passkey tasks) for depth-bucketed eval.

passkey: filler with a needle [P, d1..d5] at a random position and
[P, Q, d1..d5] at the end; loss on the final 5 digits.

copying: [pattern c tokens][SEP][filler][SEP][pattern]; loss on last c.

mqar: n_pairs key->value pairs [(k_i, v_i)] spaced through filler, then
[Q, k, v] x n_queries at the end; loss on the query values. Keys are 64
dedicated tokens (34-97) sampled WITHOUT replacement (a permutation per
sequence) -- an earlier version drew pairs with replacement from too few
keys, making targets contradictory (same key, different values) and
freezing the loss at ln(10). Fillers are 26-29 for this task only
(passkey/copying keep 18-29); values are digits.

mqar2 (two-hop MQAR): chains a_i -> b_i -> c_i. First half of the body
hosts (a_i, b_i) pairs, second half (b_i, c_i) pairs; tail is [Q, a_i, c_i]
x n_queries with loss on c_i. a-keys 34-97, b-mids 98-161 (same token is
hop-1 value and hop-2 key -- that IS the chain), c-values digits 0-9, vocab
192. Queries are sampled WITHOUT replacement per sequence so no answer is
ever visible earlier in the tail. A single-read model (1 layer, 1 round)
cannot express the chain; 2 rounds or 2 layers can.
"""
import torch

FILL0, FILL1 = 18, 30  # filler tokens 18..29 (passkey/copying)
P, Q, SEP = 30, 31, 32
VOCAB = 128  # 0-9 digits, 18-29 filler, 30-32 P/Q/SEP, 34-97 MQAR keys
KEY = 5
MQAR_KEYS = list(range(34, 98))  # 64 distinct keys (npairs <= 64)
MQAR_FILL0, MQAR_FILL1 = 26, 30  # mqar-only fillers (26..29)
MQAR2_BMIDS = list(range(98, 162))  # 64 distinct b-mids
MQAR2_VOCAB = 192


def _targets(seq, loss_len):
    idx, tgt = seq[:, :-1], seq[:, 1:]
    mask = torch.zeros(seq.shape[0], seq.shape[1] - 1, dtype=torch.bool)
    mask[:, -loss_len:] = True
    return idx, tgt, mask


def passkey_batch(bs, n, g, device):
    # seq has n+1 tokens so that idx = seq[:-1] keeps the model input at n.
    seq = torch.randint(FILL0, FILL1, (bs, n + 1), generator=g)
    key = torch.randint(0, 10, (bs, KEY), generator=g)
    tail = KEY + 2  # [P, Q, d1..d5] at the end
    hi = n + 1 - tail - (KEY + 1)  # needle must not overlap the tail
    pos = torch.randint(0, hi + 1, (bs,), generator=g)
    rows = torch.arange(bs)
    seq[rows, pos] = P
    seq[rows[:, None], pos[:, None] + torch.arange(1, KEY + 1)] = key
    seq[:, -tail] = P
    seq[:, -tail + 1] = Q
    seq[:, -KEY:] = key
    idx, tgt, mask = _targets(seq, KEY)
    return idx.to(device), tgt.to(device), mask.to(device), pos.to(device)


def copying_batch(bs, n, g, device, c=None):
    c = c or n // 8
    assert 2 * c + 2 <= n + 1
    seq = torch.randint(FILL0, FILL1, (bs, n + 1), generator=g)
    pat = torch.randint(FILL0, FILL1, (bs, c), generator=g)
    seq[:, :c] = pat
    seq[:, c] = SEP
    seq[:, -c - 1] = SEP
    seq[:, -c:] = pat
    idx, tgt, mask = _targets(seq, c)
    return idx.to(device), tgt.to(device), mask.to(device), None


def mqar_batch(bs, n, g, device, n_pairs=16, n_queries=4):
    assert n_pairs <= len(MQAR_KEYS), \
        f"n_pairs={n_pairs} > {len(MQAR_KEYS)} distinct keys"
    tail = 3 * n_queries
    seq = torch.randint(MQAR_FILL0, MQAR_FILL1, (bs, n + 1), generator=g)
    seg = (n + 1 - tail) // n_pairs
    assert seg >= 2, "sequence too short for n_pairs"
    ki = torch.argsort(torch.rand(bs, len(MQAR_KEYS), generator=g), dim=1)[:, :n_pairs]
    keys = MQAR_KEYS[0] + ki  # unique keys per sequence (no-replacement)
    vals = torch.randint(0, 10, (bs, n_pairs), generator=g)
    off = torch.randint(0, seg - 1, (bs, n_pairs), generator=g)
    p = torch.arange(n_pairs).unsqueeze(0) * seg + off  # (bs, n_pairs)
    rows = torch.arange(bs).unsqueeze(1)
    seq[rows, p] = keys
    seq[rows, p + 1] = vals
    qi = torch.randint(0, n_pairs, (bs, n_queries), generator=g)
    t = n + 1 - tail
    seq[:, t::3] = Q
    seq[:, t + 1::3] = keys.gather(1, qi)
    seq[:, t + 2::3] = vals.gather(1, qi)
    idx, tgt = seq[:, :-1], seq[:, 1:]
    mask = torch.zeros(bs, n, dtype=torch.bool)
    mask[:, t + 1::3] = True  # tgt positions predicting each value (after [Q,k])
    return idx.to(device), tgt.to(device), mask.to(device), None


def mqar2_batch(bs, n, g, device, n_pairs=16, n_queries=4):
    """Two-hop MQAR: a_i -> b_i in the first body half, b_i -> c_i in the
    second; tail queries [Q, a_i, c_i], loss on c_i. Answering requires
    chaining: read a->b, then use b to read b->c. Queries are a permutation
    (no replacement) so answers never leak into the visible tail."""
    assert n_pairs <= len(MQAR_KEYS), \
        f"n_pairs={n_pairs} > {len(MQAR_KEYS)} distinct keys"
    tail = 3 * n_queries
    body = n + 1 - tail
    half = body // 2
    seg1 = half // n_pairs
    seg2 = (body - half) // n_pairs
    assert seg1 >= 2 and seg2 >= 2, "sequence too short for n_pairs"
    seq = torch.randint(MQAR_FILL0, MQAR_FILL1, (bs, n + 1), generator=g)
    ai = torch.argsort(torch.rand(bs, len(MQAR_KEYS), generator=g), dim=1)[:, :n_pairs]
    bi = torch.argsort(torch.rand(bs, len(MQAR2_BMIDS), generator=g), dim=1)[:, :n_pairs]
    akeys = MQAR_KEYS[0] + ai                       # (bs, n_pairs) unique
    bmids = MQAR2_BMIDS[0] + bi                     # (bs, n_pairs) unique
    cvals = torch.randint(0, 10, (bs, n_pairs), generator=g)
    rows = torch.arange(bs).unsqueeze(1)
    off1 = torch.randint(0, seg1 - 1, (bs, n_pairs), generator=g)
    p1 = torch.arange(n_pairs).unsqueeze(0) * seg1 + off1
    seq[rows, p1] = akeys                           # a_i -> b_i, first half
    seq[rows, p1 + 1] = bmids
    off2 = torch.randint(0, seg2 - 1, (bs, n_pairs), generator=g)
    p2 = half + torch.arange(n_pairs).unsqueeze(0) * seg2 + off2
    seq[rows, p2] = bmids                           # b_i -> c_i, second half
    seq[rows, p2 + 1] = cvals
    qi = torch.argsort(torch.rand(bs, n_pairs, generator=g), dim=1)[:, :n_queries]
    t = n + 1 - tail
    seq[:, t::3] = Q
    seq[:, t + 1::3] = akeys.gather(1, qi)
    seq[:, t + 2::3] = cvals.gather(1, qi)
    idx, tgt = seq[:, :-1], seq[:, 1:]
    mask = torch.zeros(bs, n, dtype=torch.bool)
    mask[:, t + 1::3] = True  # tgt positions predicting each c (after [Q,a])
    return idx.to(device), tgt.to(device), mask.to(device), None
