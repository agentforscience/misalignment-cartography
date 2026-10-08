# Planning: Misalignment Cartography

Written before any experiment was run (2026-10-07). Deviations are logged at the end.

## Situation at start of session

- The resource-finder stage was skipped (`.neurico/pipeline_state.json`: `"skipped": true`), so there is
  no `literature_review.md`, `resources.md`, `papers/`, `datasets/` or `code/`. Background below is from
  my own knowledge of the literature, not from a gathered corpus.
- `OPENROUTER_KEY` is not set and no other LLM API key is present, so API models cannot be used.
- No GPU. 32 CPU cores (shared machine).
- Consequence: all experiments use real open-weight chat models run locally on CPU
  (Qwen2.5-Instruct 0.5B / 1.5B / 3B). The 7B model was benchmarked out of budget (~5 s per item on CPU).

## Motivation & Novelty Assessment

### Why This Research Matters
Alignment papers regularly report "the intervention did not change the behaviour" (a steering vector
that does nothing, a system prompt that fails to reduce sycophancy, an ablation with no effect). Such
nulls are read in two opposite ways: as evidence the behaviour is robust, or as a failed experiment.
Which reading is right matters for safety claims in both directions. If whether one gets a null is
largely fixed by design choices made before the experiment (metric, where the intervention is applied,
which model size), then nulls carry much less information about the model than they appear to.

### Gap in Existing Work
Multiverse / specification-curve analysis (Steegen et al. 2016; Simonsohn et al. 2020) is standard in
psychology but rarely applied to alignment interventions. Work on evaluation statistics (Miller 2024,
"Adding Error Bars to Evals") addresses sample size and variance but not the joint structure of
metric x intervention type x model. Steering-vector reliability work (e.g. Tan et al. 2024) shows
effects vary across inputs, but does not ask where nulls sit in design space. I know of no study that
runs one fixed set of interventions through a full grid of design choices and maps the null rate.

### Our Novel Contribution
A controlled multiverse: the same alignment interventions, on the same items, evaluated under every
combination of metric, sample size, intervention granularity and model scale. Because every design is
run on the same underlying item pool, we can also establish a reference effect for each intervention
and classify each null as "nothing there" or "something there that this design could not see".

### Experiment Justification
- Experiment 1 (intervention grid): 11 interventions x 3 models x 240 items, recording answer logits.
  Needed to obtain real model behaviour under each intervention; everything else is derived from it.
- Experiment 2 (multiverse of simulated studies): resample "studies" of size n from the pool and test
  each under each metric. Needed to turn raw effects into null rates per design cell.
- Experiment 3 (clustering analysis): variance decomposition and prediction of nulls from design
  features. This is the direct test of "nulls cluster along design features".
- Experiment 4 (null classification against reference effects): the direct test of "blind spots rather
  than genuine robustness".

## Research Question
For a fixed target behaviour (sycophancy), does the probability that an intervention experiment returns
a null depend systematically on (a) evaluation metric, (b) intervention granularity, (c) model scale —
and are those nulls mostly cases where a real effect exists but the design missed it?

## Hypothesis Decomposition
- H1 (metric): holding model, intervention and items fixed, null rate differs by metric; coarser
  metrics (sampled / greedy binary answer) give more nulls than continuous ones (probability, logit diff).
- H2 (granularity): null rate differs by intervention granularity (prompt-level, all-layer activation,
  single-layer activation, attention-head).
- H3 (scale): null rate differs by model size.
- H4 (blind spot vs robustness): among nulls from non-placebo interventions, the majority occur where
  the reference analysis shows a real effect.

Important asymmetry, fixed in advance: metric and n are *measurement-side* features — the underlying
effect is identical across their levels, so a null-rate difference is unambiguously methodological.
Granularity and scale are *effect-side* features — changing them changes the intervention or the model,
so a higher null rate there may reflect a truly smaller effect. H2/H3 clustering therefore does not by
itself imply a blind spot; only H4 speaks to that.

## Proposed Methodology

### Data
Anthropic model-written-evals sycophancy sets (Perez et al. 2022): NLP survey, PhilPapers 2020
(binary-choice items only), political typology. Each item is a user biography implying a view plus an
A/B question; `answer_matching_behavior` is the sycophantic choice, balanced between A and B.
Seeded stratified sample: 32 train items per domain (96) for building interventions, 80 evaluation
items per domain (240), disjoint, identical across models.

### Models
Qwen2.5-0.5B / 1.5B / 3B-Instruct, float32, CPU, default chat template. Single forward pass per item;
the assistant turn is prefilled with "(" so the next token is the answer letter.

### Interventions (11 + baseline), by granularity
- Prompt-level: `P_system` (anti-sycophancy system instruction), `P_user` (user-turn request to ignore
  their stated views), `P_placebo` (irrelevant formatting instruction).
- All-layer activation: `D_ablate` (project the sycophancy direction out of every layer's output),
  `D_ablate_rand` (same with a random direction; placebo).
- Single-layer activation: `S_0.1`, `S_0.3` (subtract the unit sycophancy direction scaled to
  0.1 / 0.3 of mean residual norm at the middle layer), `S_rand_0.3` (random direction; placebo).
- Attention-head: `H_top1`, `H_topk` (zero-ablate the 1 / top 2% of heads ranked by gradient x
  activation attribution to the sycophancy logit difference on train items), `H_randk` (placebo).

The sycophancy direction is a contrastive difference of means (Rimsky et al. 2023 style) on train items
at the middle layer (index n_layers // 2). Layer, doses and k are fixed here and not tuned on eval data.

### Metrics (per item, per condition)
- `sample`: one token sampled at T=1 from the full vocabulary equals the sycophantic letter (0/1).
- `greedy`: full-vocabulary argmax equals the sycophantic letter (0/1).
- `prob`: P(syc) / (P(A)+P(B)).
- `logit`: logit(syc) - logit(non-syc).

### Multiverse
A "study" = (model, intervention, metric, n). For each cell draw 1000 studies by resampling n items
with replacement from the 240-item pool, n in {20, 50, 100, 200}; paired test intervention vs baseline
(exact McNemar for binary metrics, paired t for continuous), two-sided, alpha = 0.05. Null = p >= 0.05.
Cell null rate = fraction of the 1000 studies that are null. 3 x 11 x 4 x 4 = 528 cells.

### Reference effects
Per model x intervention on the full pool (n=240): paired mean change with 95% CI and paired Cohen's d
for the logit metric; change in greedy sycophancy rate with CI. Holm correction across the 33
model x intervention pairs. Three classes:
- `none`: neither logit nor greedy effect is Holm-significant;
- `latent_only`: logit effect significant, greedy rate change not;
- `behavioural`: greedy rate change significant.
Nulls in `behavioural` cells are counted as missed effects. Nulls in `latent_only` cells are reported
separately and not counted as either, since a logit shift that flips no answers is arguably robustness
at the behavioural level.

### Baselines / controls
- Placebo interventions give the expected null rate when (nominally) nothing is done.
- "No clustering" baseline: null rate independent of design features; tested by permutation.
- Baseline sycophancy itself: rate of matching answers vs the 50% expected absent sycophancy.

### Statistical Analysis Plan
1. Marginal null rates by each feature, CIs by cluster bootstrap over the 33 model x intervention units.
2. Metric and n (within-unit factors): permutation test shuffling labels within model x intervention
   x (other factor); statistic = range of marginal null rates.
3. Granularity (11 interventions) and scale (3 models): descriptive, plus a permutation test over
   intervention labels for granularity. With 3 models from one family, scale gets no inferential
   claim beyond the within-item paired comparison.
4. Type-II ANOVA on cell null rate for share of variance by feature; and the same after adding the
   reference effect size, to see how much "granularity" and "scale" clustering is just effect size.
5. H4: share of non-placebo nulls falling in each reference class, with bootstrap CI.
6. Robustness: alpha = 0.01; excluding n = 20; per-domain breakdown.

## Expected Outcomes
Support: null rate varies strongly with metric at fixed effect; most non-placebo nulls sit in
`behavioural` cells. Refutation: null rates flat across metrics, or nulls concentrated in `none` cells
(i.e. nulls mostly reflect no effect). A mixed outcome is plausible — H1 holding while H4 fails — and
would mean nulls are design-dependent but still mostly "true".

## Timeline
Plan 20 min; code 45 min; intervention runs ~90 min (three models in parallel on CPU); analysis 30 min;
report 30 min.

## Potential Challenges
- Small models may show weak baseline sycophancy -> measured and reported; interventions cannot
  reduce what is not there (floor effect), which itself produces nulls and must be flagged.
- Steering may push probability mass off the answer letters -> record P(A)+P(B) per condition.
- Placebos may not be truly inert (random activation perturbations do change outputs) -> reference
  classes are determined empirically, not by label.
- One behaviour, one model family, three sizes: generalisation to "alignment experiments" at large is
  limited and will be stated as such.

## Success Criteria
All 528 cells populated from real model outputs; each of H1-H4 answered with an effect size and
uncertainty; nulls classified against reference effects; limitations stated.

## Deviations from this plan (logged after the fact)

1. **Two sessions.** The first session (CPU only) was killed at its 2-hour limit with 0.5B complete, 1.5B at
   8/12 and 3B at 5/12 conditions. Those partial files are kept in `results/model_outputs/partial_cpu_run/`
   and are not used in any result.
2. **GPU.** In the second session GPUs were available (RTX A6000), so torch was switched to the cu130 build
   and all models were rerun from scratch on GPU. The 0.5B GPU logits match the CPU run to within 1e-4 and
   every greedy answer agrees.
3. **Fourth model.** Qwen2.5-7B-Instruct was added (the plan dropped it only for CPU budget). Design grid is
   therefore 4 x 11 x 4 x 4 = 704 cells and Holm correction runs over 44 units, not 528 / 33.
4. **7B precision.** The GPUs were shared with other jobs, so 7B was run in bfloat16 (body) with the final
   unembedding computed in float32; the other three models are float32 throughout.
5. **Supplementary analyses** in `src/supplementary.py` (metric x model table, H4 sensitivity to the Holm
   threshold, sign disagreement, placebo direction) were added after seeing results and are exploratory.
