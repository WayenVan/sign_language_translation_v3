export WANDB_PROJECT=sign_language_translation_v3

export PYTHONPATH=./src:$PYTHONPATH

accelerate launch --num_processes=2 --mixed_precision=bf16 --debug -m csi_slt.commands.train \
	model=base_model \
	engine.training_args.auto_output_root=./outputs/pretrain_adapter \
	engine.training_args.per_device_train_batch_size=2 \
	engine.training_args.per_device_eval_batch_size=2 \
	engine.training_args.dataloader_num_workers=10 \
	engine.training_args.eval_steps=4000 \
	engine.training_args.save_steps=4000 \
	engine.training_args.logging_steps=15
# data.train.processor.video_token_scale=1.0 \
# data.val.processor.video_token_scale=1.0 \
# data.test.processor.video_token_scale=1.0 \
# model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# model.config.video_token_scale=1.0

# accelerate launch --num_processes=2 --mixed_precision=bf16 \
# 	-m csi_slt.commands.train_ft_peft \
# 	engine.training_args.dataloader_num_workers=10 \
# 	engine.training_args.auto_output_root=./outputs/peft_ft # accelerate launch --num_processes=2 --mixed_precision=fp16 \
