# Misalignment Cartography

Maps where null results fall in the design space of an alignment-intervention experiment. Eleven
anti-sycophancy interventions were run on four Qwen2.5-Instruct models (0.5B-7B) over 240 items, then
analysed as a 704-cell multiverse of metric x sample size x intervention x model. Full write-up:
[REPORT.md](REPORT.md).

## Key findings

- Metric choice dominates: null rate is 0.81 (sampled answer), 0.80 (greedy answer), 0.52 (probability),
  0.25 (logit margin) for the same interventions on the same items (permutation p = 0.0002).
- Sample size matters as expected (0.77 at n=20 to 0.44 at n=200). Intervention granularity does not
  organise nulls (p = 0.88); model scale shows no monotone trend.
- Only 14% (95% CI 5-27%) of nulls from real interventions hide a demonstrable change in answers; 80% are
  logit shifts that do not flip answers; 6% are from an intervention with no detectable effect. The
  "blind spots rather than robustness" reading is not supported as pre-specified.
- Placebo interventions are not inert: on 7B, ablating random heads cut sycophancy more than ablating
  attribution-selected heads.
- Scope: one behaviour, one model family, forced-choice format.

## Reproduce

```bash
uv sync && source .venv/bin/activate          # torch cu130 build; needs a CUDA GPU (or pass --device cpu)
for m in 0.5B 1.5B 3B; do python src/run_interventions.py Qwen/Qwen2.5-$m-Instruct; done
python src/run_interventions.py Qwen/Qwen2.5-7B-Instruct --dtype bfloat16
python src/multiverse.py                      # tables in results/evaluations, figures in figures/
python src/supplementary.py                   # exploratory follow-ups
```

The sycophancy JSONL files go in `datasets/sycophancy/` (Anthropic model-written-evals, `sycophancy/`
folder); the sampled items are fixed in `datasets/processed/items.json`. Seed 42 throughout.
`multiverse.py` was run twice and produced byte-identical tables.

## Files

- `planning.md`: pre-registered plan, plus deviations
- `src/run_interventions.py`: builds interventions, records answer logits
- `src/multiverse.py`: simulated studies, clustering tests, null classification, figures
- `src/supplementary.py`: exploratory follow-ups
- `results/model_outputs/`: raw per-item outputs (`partial_cpu_run/` is an abandoned earlier run, unused)
- `results/evaluations/`, `figures/`, `logs/`
