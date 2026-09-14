跑接下来的 所有的14b 的消融任务

LLM_LORA_CKPT_LEAF="v5.0-qwen3-14b-cradio-l-nextframe-handroi-cls-31m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep80-de-0911.224x224/checkpoint-132000" LLM_LORA_CHECKPOINT_DIR="/users/2533494w/projects/sign_language_translation_v3/outputs/v5.0-qwen3-14b-cradio-l-nextframe-handroi-cls-31m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep80-de-0911.224x224/checkpoint-132000" LLM_LORA_CKPT_TAG="ckpt132k" LLM_LORA_MODEL_RUN_SLUG="qwen3-14b-cradio-l-nextframe-handroi-cls-31m" LLM_LORA_MODEL_WANDB_TAG="qwen3-14b" LLM_LORA_LAYER_COUNT=40 LLM_LORA_OUTPUT_DATE_TAG=0914 LLM_LORA_OUTPUT_ROOT="/users/2533494w/projects/sign_language_translation_v3/outputs/v5.0-llm-lora-de-ablation-qwen14b" bash scripts/run_llm_lora_ablation.sh qkvo 768 18

- [ ] 记得运行 Qwen3-32B dense 训练脚本：`sbatch scripts/run_cognition_scale_max.sh`（默认 de/en/zh 多语言 + diverse prompt，40 epochs）。
