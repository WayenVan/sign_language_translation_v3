export WANDB_PROJECT=sign_language_translation_v3

export PYTHONPATH=./src:$PYTHONPATH

accelerate launch --num_processes=2 --mixed_precision=bf16 --debug -m csi_slt.commands.train \
	model=base_model \
	engine.training_args.auto_output_root=./outputs/pretrain_adapter \
	engine.training_args.per_device_train_batch_size=8 \
	engine.training_args.per_device_eval_batch_size=4 \
	engine.training_args.dataloader_num_workers=10 \
	engine.training_args.eval_steps=1000 \
	engine.training_args.save_steps=1000 \
	engine.training_args.logging_steps=15
#
# data.train.collator.video_token_scale=1.0 \
# data.val.collator.video_token_scale=1.0 \
# data.test.collator.video_token_scale=1.0 \
# model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# model.config.video_token_scale=1.0

# -m csi_slt.commands.train_ft # accelerate launch --num_processes=2 --mixed_precision=fp16 \
# 	engine.training_args.auto_output_root=./outputs/first_demo_ft
# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# 	-m csi_slt.commands.train_ft \
# 	engine.training_args.auto_output_root=./outputs/first_demo_ft
