from transformers.trainer_callback import TrainerCallback, CallbackHandler
import os
import shutil
from transformers import logging

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

            if self.best_metric is None or current_metric < self.best_metric:
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
