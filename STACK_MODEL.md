# Stack model (attention push/pop)

`mga.stack_model.StackModel`: `forward([B,N], sup=None) -> [B,N,vocab]`.
Replaces the index model's machinery with a single exact mechanism.

## Doctrine

Text cannot be losslessly compressed, and a block that does not know the
query standard will be misjudged (attention scatter). So the block "summary"
is **written from the query, on demand, and never stored**: each supervised
position's query probes every visible block's raw keys, and the block's score
is its exact attention partition function

```
s_j(t) = logsumexp_{i in block j} (q_t · k_i / sqrt(hd))
```

— "how much attention mass would this block receive". Push = this per-block
reduction; pop = expand the per-head top-m blocks into raw tokens and run one
token-level softmax over [local window ++ selected blocks] with the harsh
`log_softmax(s)` gate as the selected blocks' logit bias.

- No compressor, no stored summary, no tree, no query-rewrite chain.
- The score path shares q and raw_k with the fine read, so selection is the
  EXACT block mass of the read semantics: a principled top-k approximation of
  full attention. At `topk >= visible blocks` it is exactly dense — small-n
  ignition behaves like a plain transformer.
- The harsh gate is load-bearing: it forces near-binary score margins, which
  are G-invariant and therefore survive zero-shot length extrapolation.
  (Relaxed gates train fine at small G and collapse at large G.)
- Gradient highways are dense: every visible block's keys receive gradient
  through the gate's logsumexp denominator; selected tokens receive content
  gradient through the read. The only discrete step is the top-m, which is
  fine (hard selection works once scores are exact).
- KV cache: raw K/V is written once at token granularity and never rewritten;
  the push pass only reads cached keys. All read/score paths are NoPE; RoPE
  lives only in the local halo encoder. No length-dependent parameters.

## Visibility / causality

Position t (block k) reads the local window `[(k-1)*b, t]` directly and may
select blocks `j <= k-2`. Block k-1 is covered by the local window. Suffix
perturbation and prefix truncation leave earlier logits bit-identical
(leak test variant `stack`: 0 diff).

## Notes

`sup` restricts push/pop to supervised positions (retrieval tasks: a handful
per sequence, so training cost is ~linear; sup=None for leak tests/LM).
CLI: `--model stack`, `--read_m` = topk blocks per head, `--local_layers`.
