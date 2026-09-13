
• TODO: 运行 bash scripts/lora_sweep/run_llm_lora_scaledup_sweep.sh --sbatch，
  提交 Qwen3-14B 的 qkvo LoRA rank=256/512 sweep。

● sbatch -J slt_llmlora_de_qkvo_r512 scripts/run_llm_lora_inject.sh
  /mnt/scratch/users/2533494w/slt_outputs/v5.0-qwen3-4b-cradio-l-nextframe-handr
  oi-cls-20m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep80-0907.224x2
  24/checkpoint-96000
