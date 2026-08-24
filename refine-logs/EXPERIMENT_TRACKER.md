# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | freeze evaluation | existing Mono/Unified checkpoints | validation/test | BLEU, ROUGE, chrF, semantic metric | MUST | TODO | identical decode and cleaning |
| R002 | M0 | detector calibration | references + 50 outputs/language | test subset | language accuracy | MUST | TODO | require near-perfect reference classification |
| R003 | M0 | pipeline sanity | one language, 200–500 samples | tiny train/test | overfit loss/quality | MUST | TODO | rule out architecture-independent ceiling |
| R010 | M1 | mono anchor | Mono-DE/EN/ZH | test | per-language quality | MUST | TODO | reuse checkpoints if valid |
| R011 | M1 | simple control baseline | Unified-LID | test | quality + lang accuracy | MUST | TODO | seed 1 first |
| R012 | M1 | fixed text control | Unified-FixedPrompt | test | quality + lang accuracy | MUST | TODO | seed 1 first |
| R013 | M1 | final method | Unified-DiversePrompt | test | quality + lang accuracy | MUST | TODO | seed 1/current best |
| R020 | M2 | equivalence confirmation | Mono/LID/Diverse | test | mean±std, CI, retention | MUST | TODO | 3 seeds for decisive variants |
| R030 | M3 | seen prompt robustness | Unified-DiversePrompt | test × seen prompts | quality + lang accuracy | MUST | TODO | 3–5 prompts/video |
| R031 | M3 | held-out paraphrases | Unified-DiversePrompt | test × held-out prompts | quality + lang accuracy | MUST | TODO | no validation leakage |
| R032 | M3 | cross-lingual instruction matrix | Unified-DiversePrompt | all instruction×target cells | quality + lang accuracy | MUST | TODO | balanced cells |
| R033 | M3 | causal instruction controls | remove/target-name/conflict prompts | test | language behavior + quality | MUST | TODO | paired by video |
| R040 | M4 | quality anchor | selected reported SOTA | published protocol | BLEU, ROUGE | MUST | TODO | reported vs reproduced separated |
| R050 | M5 | visual grounding | normal/zero/wrong/shuffled video | test | delta quality + lang accuracy | MUST | TODO | evaluation only |
| R051 | M5 | qualitative evidence | same video, DE/EN/ZH instructions | curated + random samples | semantic consistency | MUST | TODO | predefine sampling rule |
