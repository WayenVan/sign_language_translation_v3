# Counterfactual Instruction-Switching Experiment

## Objective

Test whether the model selects the requested target language from the semantic
relation expressed by a natural-language instruction, rather than merely
detecting a language name or following its position in the prompt.

## Core design

For each unordered language pair, construct counterfactual prompt pairs that
contain exactly the same two language names but reverse which language is the
requested target.

### Template A: target first

```text
Translate the signing into {TARGET}, not {DISTRACTOR}.
```

Example pair:

```text
Translate the signing into English, not German.
Translate the signing into German, not English.
```

### Template B: target last

```text
Do not translate the signing into {DISTRACTOR}; use {TARGET}.
```

Example pair:

```text
Do not translate the signing into German; use English.
Do not translate the signing into English; use German.
```

The two templates balance the position of the correct target. Template A puts
the target first, whereas Template B puts it last. A model therefore cannot
succeed consistently by selecting either the first or last language name.

## Conditions

Use all three unordered language pairs:

- DE / EN
- DE / ZH
- EN / ZH

Each pair is evaluated in both target directions under both templates:

```text
2 templates x 3 language pairs x 2 directions = 12 conditions per video
```

The video is held fixed within every counterfactual comparison. Only the
semantic roles of the two language names change.

## Models

Evaluate the existing checkpoints without additional training:

- Qwen3-4B SingleFit
- Qwen3-4B MultiFit
- Qwen3-14B SingleFit
- Qwen3-14B MultiFit

## Metrics

### LAcc

Report LAcc separately for:

- canonical prompts;
- Template A (target first);
- Template B (target last);
- each target language and each language pair.

Also report a target-language confusion matrix.

### Bidirectional Switch Accuracy (BSA)

For a given video, template, and unordered language pair, count success only
when both counterfactual directions produce the requested language:

```text
English, not German -> English
German, not English -> German
```

Formally,

\[
\mathrm{BSA}
=
\frac{1}{N}
\sum_i
\mathbb{1}
\left[
\hat{\ell}_i^{A\rightarrow B}=B
\land
\hat{\ell}_i^{B\rightarrow A}=A
\right].
\]

BSA prevents a model with a fixed language preference from receiving credit
for one direction of a counterfactual pair.

BLEU-4 and ROUGE-L are secondary in this experiment and may be reported in the
appendix. The primary question is whether the requested output language is
controlled by instruction semantics.

## Evaluation scale

Preferred evaluation: all 642 PH14T test videos.

```text
642 videos x 12 conditions = 7,704 generations per model
```

Lower-cost alternative: select 200 test videos using a fixed seed before
evaluation.

```text
200 videos x 12 conditions = 2,400 generations per model
```

The subset must not be selected or modified after inspecting model outputs.

## Main result table

```latex
\begin{tabular}{llcccc}
  \toprule
  Backend & Prompt Fit
  & Canonical LAcc
  & Target-first LAcc
  & Target-last LAcc
  & BSA \\
  \midrule
  Qwen3-4B  & SingleFit & -- & -- & -- & -- \\
  Qwen3-4B  & MultiFit  & -- & -- & -- & -- \\
  Qwen3-14B & SingleFit & -- & -- & -- & -- \\
  Qwen3-14B & MultiFit  & -- & -- & -- & -- \\
  \bottomrule
\end{tabular}
```

Per-language LAcc and confusion matrices can be placed in the appendix.

## Supported conclusion

If MultiFit achieves high LAcc under both target positions and high BSA, the
results support the following claim:

> Because each counterfactual prompt pair contains the same language names,
> successful bidirectional switching cannot be explained by language-keyword
> detection alone. The model uses the semantic relation expressed by the
> instruction to identify the requested target.

This experiment does not establish unrestricted instruction following beyond
the tested target-language control constructions.
