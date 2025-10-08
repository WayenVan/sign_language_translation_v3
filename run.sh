export WANDB_PROJECT=sign_language_translation_v3

export PYTHONPATH=./src:$PYTHONPATH

accelerate launch --num_processes=2 --mixed_precision=fp16 --debug -m csi_slt.commands.train \
	model.config.visual_backbone_kwargs.ckpt_path="outputs/ckpt/bacbone_pretrained/2025-10-05_20-29-56/checkpoint-21000" \
	engine.training_args.auto_output_root=./outputs/pretrain_adapter \
	engine.training_args.dataloader_num_workers=5 \
	engine.training_args.dataloader_prefetch_factor=1
# data.val.collator.video_token_scale=1.0 \
# model.config.visual_adapter_kwargs.use_temporal_shuffle=False # data.train.collator.video_token_scale=1.0 \
# data.test.collator.video_token_scale=1.0 \
# model.config.video_token_scale=1.0

# -m csi_slt.commands.train_ft # accelerate launch --num_processes=2 --mixed_precision=fp16 \
# 	engine.training_args.auto_output_root=./outputs/first_demo_ft
# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# 	-m csi_slt.commands.train_ft \
# 	engine.training_args.auto_output_root=./outputs/first_demo_ft
