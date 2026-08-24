# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | metric sanity | best current mono checkpoint | validation/test | BLEU, ROUGE, language accuracy | MUST | TODO | repeat evaluation twice |
| R002 | M0 | small-set overfit | current architecture, one language | tiny train subset | loss, exact/semantic metrics | MUST | TODO | rule out pipeline ceiling |
| R010 | M1 | mono upper bound | Mono-DE | test | BLEU, ROUGE, COMET/XCOMET | MUST | TODO | seed 1 |
| R011 | M1 | mono upper bound | Mono-EN | test | BLEU, ROUGE, COMET/XCOMET | MUST | TODO | seed 1 |
| R012 | M1 | mono upper bound | Mono-ZH | test | BLEU, ROUGE, COMET/XCOMET | MUST | TODO | seed 1 |
| R020 | M2 | isolate multilingual interference | Unified-LID | test by language | quality + language accuracy | MUST | TODO | seed 1 |
| R021 | M2 | interface cost | Unified-FixedPrompt | test by language | quality + language accuracy | MUST | TODO | seed 1 |
| R022 | M2 | prompt diversity cost | Unified-DiversePrompt | test by language | quality + language accuracy | MUST | TODO | seed 1 |
| R030 | M3 | statistical confirmation | best Mono/LID/Instruct variants | test | mean±std | MUST | TODO | add seeds only after gate |
| R040 | M4 | visual grounding | normal/zero/wrong/shuffled video | test | delta quality, language accuracy | MUST | TODO | evaluation only |
| R041 | M4 | failure taxonomy | final unified vs mono | test | bucketed quality/errors | MUST | TODO | length/language/error type |
