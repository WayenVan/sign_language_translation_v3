from transformers.trainer_callback import TrainerCallback, CallbackHandler
import os
import shutil
from transformers import logging
import torchinfo
import torch
from omegaconf import OmegaConf
from accelerate import Accelerator
from ..misc.git_utils import save_git_state
from transformers.modeling_utils import unwrap_model
from transformers.trainer import _is_peft_model


logger = logging.get_logger(__name__)


class SltTrainerCallbackHandler(CallbackHandler):
    """
    自定义 CallbackHandler，确保在回调中传递 trainer 实例。
    """

    def __init__(self, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = trainer

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                processing_class=self.processing_class,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                trainer=self.trainer,  # 传递 trainer 实例
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class SaveBestMetricCallback(TrainerCallback):
    """
    当指定 metric 达到新的最优值时，保存额外 checkpoint，并删除之前的。
    """

    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.best_metric = None
        self.last_checkpoint_path = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        trainer = kwargs.get("trainer", None)  # 通过 kwargs 获取 trainer
        if (
            trainer
            and state.global_step > 0
            and trainer.accelerator.is_local_main_process
        ):
            save_dir = os.path.join(args.output_dir, "best_checkpoint")

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            current_metric = metrics.get(self.metric_name)
            if current_metric is None:
                return

            if self.best_metric is None or current_metric > self.best_metric:
                self.best_metric = current_metric

                # 删除之前的 checkpoint
                if self.last_checkpoint_path and os.path.exists(
                    self.last_checkpoint_path
                ):
                    shutil.rmtree(self.last_checkpoint_path)

                # 保存新的 checkpoint
                checkpoint_path = os.path.join(
                    save_dir, f"best_{self.metric_name}={current_metric:.4f}"
                )
                trainer.save_model(checkpoint_path)
                self.last_checkpoint_path = checkpoint_path
                logger.info(
                    f"Saved new best checkpoint at {checkpoint_path} with {self.metric_name} = {current_metric}"
                )


class ModelInfoCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer and trainer.accelerator.is_local_main_process:
            print("Eval model ")
            model = kwargs.get("model", None)
            if model is not None and hasattr(model, "dummy_inputs"):
                with torch.no_grad():
                    try:
                        summary = torchinfo.summary(
                            model,
                            input_data=model.dummy_inputs,
                            col_names=[
                                "input_size",
                                "output_size",
                                "num_params",
                                "trainable",
                            ],
                            verbose=0,
                            depth=4,
                        )
                        print(summary)
                    except Exception as e:
                        print(f"无法生成模型摘要: {e}")
            else:
                print("模型不包含 dummy_inputs，无法生成摘要。")


class LogHydraConfigCallback(TrainerCallback):
    def __init__(self, hydra_config):
        super().__init__()
        self.hydra_config = hydra_config

    def on_train_begin(self, args, state, control, **kwargs):
        acc = Accelerator()
        if acc.is_main_process:
            is_wandb = False
            if isinstance(args.report_to, str):
                is_wandb = args.report_to == "wandb"
            elif isinstance(args.report_to, (list, tuple)):
                is_wandb = "wandb" in args.report_to

            if is_wandb:
                import wandb

                wandb.config.update(
                    {
                        "hydra_config": OmegaConf.to_container(
                            self.hydra_config, resolve=True
                        )
                    }
                )


class SaveGitInfoCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        acc = Accelerator()
        if acc.is_main_process:
            try:
                save_dir = os.path.join(args.output_dir, "git_info")

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_git_state(state_dir=save_dir)
                logger.info(f"Saved git info at {save_dir}")
            except Exception as e:
                logger.warning(f"Can not save git info: {e}")


class SaveBaseModelInPEFT(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        acc = Accelerator()
        if acc.is_local_main_process:
            model = kwargs.get("model", None)
            if model is not None:
                unwrapped_model = unwrap_model(model)
                if _is_peft_model(unwrapped_model):
                    # 保存基础模型
                    base_model = unwrapped_model.get_base_model()
                    save_dir = os.path.join(
                        args.output_dir, f"checkpoint-{state.global_step}"
                    )
                    base_model.save_pretrained(save_dir)
                    logger.info(
                        f"Saved base model of PEFT at {save_dir} for checkpoint-{state.global_step}"
                    )
                else:
                    logger.warn("Model is not a PEFT model, skipping base model save.")
            else:
                raise ValueError("Model is None, cannot save base model.")
