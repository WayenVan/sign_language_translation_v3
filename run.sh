export WANDB_PROJECT=sign_language_translation_v3

export PYTHONPATH=./src:$PYTHONPATH

# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# 	-m csi_slt.commands.train \
# 	engine.training_args.auto_output_root=./outputs/first_demo
# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# 	-m csi_slt.commands.train_ft \
# 	engine.training_args.auto_output_root=./outputs/first_demo_ft
accelerate launch --num_processes=2 --mixed_precision=fp16 \
	-m csi_slt.commands.train_ft \
	engine.training_args.auto_output_root=./outputs/first_demo_ft
