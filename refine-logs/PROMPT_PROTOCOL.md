# Prompt Protocol for Instruction-Conditioned Multilingual SLT

**Status**: frozen design decision  
**Date**: 2026-08-24  
**Instruction language**: English only  
**Target languages**: German (`de`), English (`en`), Chinese (`zh`)

## Objective

Use one English natural-language instruction interface to control the target spoken language of a unified SLT model. The protocol deliberately avoids the extra claim that the model understands instructions written in multiple languages. The core questions are:

1. Does diverse prompt training preserve standard translation quality?
2. Does it improve generalization to unseen instruction wording?
3. Does the model follow instruction semantics under controlled distractors?

## Prompt Assets

### 1. Canonical prompts

File: `promts/canonical.jsonl`

Exactly one fixed prompt per target language (three records total):

```text
Translate the signing into German.
Translate the signing into English.
Translate the signing into Chinese.
```

Roles:

- the only training prompts for `Unified-FixedPrompt`;
- included in the training pool for `Unified-DiversePrompt`;
- the fixed prompts used by every model in standard BLEU/ROUGE/COMET evaluation.

These are fixed natural-language controls. They must not be described as literal language IDs. A true language-ID baseline would use dedicated tokens such as `<de>`, `<en>`, and `<zh>` and is optional.

### 2. Diverse training prompts

File: `promts/train_diverse.jsonl`

Contains several English paraphrases for each target language. They express the same translation intent through different wording. Training samples dynamically select from:

```text
canonical prompts + diverse training prompts
```

Roles:

- used only to train `Unified-DiversePrompt`;
- never randomly sampled during standard evaluation;
- may be traversed separately for a seen-prompt robustness analysis.

Recommended size: 5–10 diverse variants per target language, excluding the canonical record.

### 3. Held-out prompts

File: `promts/heldout.jsonl`

Contains English paraphrases for each target language that are absent from training and validation prompt pools.

Rules:

- never loaded during training;
- never used for hyperparameter selection;
- exact-template overlap with canonical/diverse train prompts is forbidden;
- used only for final unseen-instruction generalization evaluation.

Evaluation traverses all held-out variants per target language and reports target-language accuracy and translation metrics as mean ± standard deviation across prompt variants.

Recommended size: 4–5 held-out variants per target language.

### 4. Adversarial prompts

File: `promts/adversarial.jsonl`

Adversarial prompts are controlled instruction diagnostics, not an arbitrary collection of unrelated requests.

Scorable controls retain one unambiguous requested target language, for example:

- target-language name only;
- negated distractor: “Do not translate into English; translate into German”;
- explicit correction: “Translate into Chinese—correction: respond in German.”

For these records, report target-language accuracy and translation quality against the reference in the requested language.

Unrelated or target-unspecified prompts may be included only as diagnostic controls. Because they do not define a correct target language, they must not contribute to formal instruction accuracy or BLEU comparisons. Report output-language distribution and qualitative behavior instead.

## Training Configurations

| Model | Training prompt pool | Standard evaluation |
|---|---|---|
| `Unified-FixedPrompt` | canonical only | canonical only |
| `Unified-DiversePrompt` | canonical + train_diverse | canonical only |

Dataset examples are not physically duplicated for prompt diversity. The collator invokes a prompt sampler during training. The dataset continues to store video, translation, and target language facts only.

## Evaluation Protocol

### Standard translation quality

Every model uses the same three canonical prompts. Prompt selection is deterministic and never random. This is the condition used for the main Mono-versus-Unified and specialist-quality tables.

### Seen-prompt robustness

Optionally evaluate all diverse training variants after training. This analysis is separate from standard translation quality. Report variation across prompt wording.

### Held-out instruction generalization

Evaluate every held-out variant for each target language. Report:

- target-language accuracy;
- BLEU-4;
- ROUGE-L;
- COMET/XCOMET where supported;
- mean ± standard deviation across prompt variants.

### Adversarial instruction following

Evaluate scorable adversarial types separately. Do not merge target-name-only, negation, and correction into one opaque average. Target-unspecified prompts are appendix diagnostics only.

## Main Comparison Logic

Standard quality comparison:

| System | Training prompts | Eval prompts | # checkpoints | DE | EN | ZH | Macro |
|---|---|---|---:|---:|---:|---:|---:|
| Mono-Qwen | fixed canonical per target | canonical | 3 | — | — | — | — |
| Unified-FixedPrompt | canonical | canonical | 1 | — | — | — | — |
| Unified-DiversePrompt | canonical + diverse | canonical | 1 | — | — | — | — |

This table tests training choices under identical evaluation prompts. Prompt diversity in this table refers to the training condition, not random evaluation prompts.

Held-out instruction evaluation is reported separately and tests whether diverse training improves unseen-instruction robustness.

## Implementation Boundary

```text
Dataset: stable video/translation/target-language facts
PromptSampler: training-time prompt selection
Collator: connects sample and selected prompt
Processor: renders the selected template and tokenizes it
```

The initial implementation should remain backward compatible: absence of a sampler preserves the current fixed-prompt pipeline.
