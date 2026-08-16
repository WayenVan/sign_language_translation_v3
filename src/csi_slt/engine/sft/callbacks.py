from transformers.trainer_callback import (
    CallbackHandler,
    ExportableState,
    TrainerCallback,
)
import os
import shutil
from pathlib import Path
from transformers import logging
from omegaconf import OmegaConf
import time
from accelerate import Accelerator
from ...utils.git_state import save_git_state
from transformers.modeling_utils import unwrap_model
from transformers.trainer import _is_peft_model
from .scheduler import DSIDScheduler
from .information_visualization import render_llm_attention


logger = logging.get_logger(__name__)


class EvalInformationVisualizationCallback(TrainerCallback, ExportableState):
    """Periodically render information from the first evaluation samples."""

    def __init__(self, every_n_evaluations: int = -1, num_samples: int = 4):
        if isinstance(every_n_evaluations, bool) or not isinstance(
            every_n_evaluations, int
        ):
            raise TypeError("every_n_evaluations must be an integer")
        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer")
        if every_n_evaluations == 0 or every_n_evaluations < -1:
            raise ValueError("every_n_evaluations must be -1 or a positive integer")
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        self.every_n_evaluations = every_n_evaluations
        self.num_samples = num_samples
        self.evaluation_count = 0

    def state(self) -> dict:
        """Export the evaluation cadence for checkpoint continuation."""
        return {
            "args": {
                "every_n_evaluations": self.every_n_evaluations,
                "num_samples": self.num_samples,
            },
            "attributes": {"evaluation_count": self.evaluation_count},
        }

    def on_evaluate(self, args, state, control, **kwargs):
        self.evaluation_count += 1
        if self.every_n_evaluations == -1:
            return
        if self.evaluation_count % self.every_n_evaluations != 0:
            return

        trainer = kwargs.get("trainer")
        if trainer is None:
            raise RuntimeError(
                "EvalInformationVisualizationCallback requires the trainer"
            )

        records = trainer.collect_eval_information(self.num_samples)
        if not trainer.accelerator.is_main_process:
            return

        output_dir = Path(args.output_dir) / f"eval_info_step{state.global_step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            sample_index = record["sample_index"]
            attention = record["information"].llm_attentions[0][0]
            visual_mask = record["information"].llm_visual_mask[0]
            valid_mask = record["attention_mask"].bool()
            render_llm_attention(
                attention[valid_mask][:, valid_mask],
                visual_mask[valid_mask],
                output_dir / f"sample{sample_index:03d}_layer-1_llm_attention.png",
            )

        logger.info(
            "Saved %d evaluation information visualizations to %s",
            len(records),
            output_dir,
        )


class DSIDWeightSchedulerCallback(TrainerCallback, ExportableState):
    """Synchronize an engine-owned D-SID scheduler with Trainer global step."""

    def __init__(self, warmup_ratio: float = 0.1, decay_ratio: float = 0.3):
        self.warmup_ratio = float(warmup_ratio)
        self.decay_ratio = float(decay_ratio)
        self.scheduler: DSIDScheduler | None = None
        # Transformers restores these JSON-serializable attributes before
        # on_train_begin constructs the runtime scheduler.
        self.scheduler_state: dict[str, int] | None = None
        self.scheduler_parameters: dict[str, float | int] | None = None

    def state(self) -> dict:
        """Export the state needed for exact checkpoint continuation."""
        scheduler_state = (
            self.scheduler.state_dict()
            if self.scheduler is not None
            else self.scheduler_state
        )
        scheduler_parameters = (
            self._scheduler_parameters(self.scheduler)
            if self.scheduler is not None
            else self.scheduler_parameters
        )
        return {
            "args": {
                "warmup_ratio": self.warmup_ratio,
                "decay_ratio": self.decay_ratio,
            },
            "attributes": {
                "scheduler_state": scheduler_state,
                "scheduler_parameters": scheduler_parameters,
            },
        }

    @staticmethod
    def _scheduler_parameters(
        scheduler: DSIDScheduler,
    ) -> dict[str, float | int]:
        return {
            "max_weight": scheduler.max_weight,
            "total_steps": scheduler.total_steps,
        }

    @staticmethod
    def _unwrap_model(kwargs):
        trainer = kwargs.get("trainer")
        return (
            trainer.accelerator.unwrap_model(trainer.model)
            if trainer is not None
            else unwrap_model(kwargs["model"])
        )

    def _update_weight(self, state, kwargs, *, reset: bool = False) -> None:
        model = self._unwrap_model(kwargs)
        if not hasattr(model, "set_dsid_loss_weight"):
            return

        max_weight = float(model.config.dsid_loss_weight)
        if reset or self.scheduler is None:
            self.scheduler = DSIDScheduler(
                max_weight=max_weight,
                total_steps=state.max_steps,
                warmup_ratio=self.warmup_ratio,
                decay_ratio=self.decay_ratio,
            )
            if self.scheduler_state is not None:
                current_parameters = self._scheduler_parameters(self.scheduler)
                if self.scheduler_parameters != current_parameters:
                    raise RuntimeError(
                        "D-SID scheduler parameters changed while resuming: "
                        f"checkpoint={self.scheduler_parameters}, "
                        f"current={current_parameters}"
                    )
                checkpoint_step = int(self.scheduler_state["current_step"])
                if checkpoint_step != state.global_step:
                    raise RuntimeError(
                        "D-SID scheduler checkpoint step does not match Trainer "
                        f"global_step: {checkpoint_step} != {state.global_step}"
                    )
                self.scheduler.load_state_dict(self.scheduler_state)
                logger.info(
                    "Restored D-SID scheduler at step %d/%d with weight %.8f",
                    self.scheduler.current_step,
                    self.scheduler.total_steps,
                    self.scheduler.current_weight,
                )
                self.scheduler_state = None
                self.scheduler_parameters = None
        elif (
            self.scheduler.max_weight != max_weight
            or self.scheduler.total_steps != state.max_steps
        ):
            raise RuntimeError("D-SID scheduler arguments changed after training began")

        weight = self.scheduler.step(state.global_step)
        model.set_dsid_loss_weight(weight)

    def on_train_begin(self, args, state, control, **kwargs):
        self._update_weight(state, kwargs, reset=True)

    def on_step_begin(self, args, state, control, **kwargs):
        self._update_weight(state, kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        # Trainer increments global_step before this event. Synchronizing here
        # ensures a checkpoint saved after the update exports the same step as
        # TrainerState; on_step_begin will idempotently apply it to the next
        # optimizer update.
        self._update_weight(state, kwargs)


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

                # NOTE: check if peft model
                model = kwargs.get("model", None)
                if model is None:
                    model = trainer.model
                    logger.warning(
                        "Model is None in callback kwargs, using trainer.model instead."
                    )

                unwrapped_model = unwrap_model(model)
                if _is_peft_model(unwrapped_model):
                    # 保存基础模型
                    base_model = unwrapped_model.get_base_model()
                    base_model.save_pretrained(checkpoint_path)
                    logger.info(
                        f"Saved base model of PEFT at {checkpoint_path} for best {self.metric_name}"
                    )
                # NOTE: end peft model

                self.last_checkpoint_path = checkpoint_path
                logger.info(
                    f"Saved new best checkpoint at {checkpoint_path} with {self.metric_name} = {current_metric}"
                )


class ModelInfoCallback(TrainerCallback):
    """Print an aligned parameter table for the model on the main process."""

    _HEADERS = (
        "#",
        "Module",
        "Class",
        "Parameter",
        "Trainable",
        "DType",
        "Shape",
        "Parameters",
    )
    _ALIGNMENTS = ("right", "left", "left", "left", "center", "left", "right", "right")

    @staticmethod
    def _parameter_rows(model):
        """Return one row per unique parameter, attributed to its owner."""
        rows = []
        seen_parameters = set()
        for module_name, module in model.named_modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                parameter_id = id(parameter)
                if parameter_id in seen_parameters:
                    continue
                seen_parameters.add(parameter_id)

                shape = " x ".join(str(size) for size in parameter.shape) or "scalar"
                rows.append(
                    (
                        str(len(rows) + 1),
                        module_name or "<root>",
                        type(module).__name__,
                        parameter_name,
                        "yes" if parameter.requires_grad else "no",
                        str(parameter.dtype).removeprefix("torch."),
                        shape,
                        f"{parameter.numel():,}",
                    )
                )
        return rows

    @classmethod
    def _format_table(cls, rows):
        """Render a dependency-free table with stable column alignment."""
        widths = [len(header) for header in cls._HEADERS]
        for row in rows:
            for index, value in enumerate(row):
                widths[index] = max(widths[index], len(value))

        def border(fill="-"):
            return "+" + "+".join(fill * (width + 2) for width in widths) + "+"

        def format_value(value, width, alignment):
            if alignment == "right":
                return value.rjust(width)
            if alignment == "center":
                return value.center(width)
            return value.ljust(width)

        def format_row(row, alignments):
            cells = (
                format_value(value, width, alignment)
                for value, width, alignment in zip(row, widths, alignments)
            )
            return "| " + " | ".join(cells) + " |"

        lines = [
            border(),
            format_row(cls._HEADERS, ("center",) * len(cls._HEADERS)),
            border("="),
        ]
        lines.extend(format_row(row, cls._ALIGNMENTS) for row in rows)
        lines.append(border())
        return "\n".join(lines)

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer and trainer.accelerator.is_local_main_process:
            rows = self._parameter_rows(trainer.model)
            parameters = list(trainer.model.parameters())
            total = sum(parameter.numel() for parameter in parameters)
            trainable = sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            )
            frozen = total - trainable
            trainable_ratio = 100.0 * trainable / total if total else 0.0

            summary = (
                f"Parameter tensors: {len(rows):,} | "
                f"Total: {total:,} | "
                f"Trainable: {trainable:,} ({trainable_ratio:.2f}%) | "
                f"Frozen: {frozen:,}"
            )
            print(
                f"\nModel parameter information\n{self._format_table(rows)}\n{summary}\n"
            )


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


class SaveHydraConfigCallback(TrainerCallback):
    def __init__(self, hydra_config):
        super().__init__()
        self.hydra_config = hydra_config

    def on_save(self, args, state, control, **kwargs):
        acc = Accelerator()
        if acc.is_local_main_process:
            save_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 保存 hydra 配置
            config_path = os.path.join(save_dir, "hydra_config.yaml")
            with open(config_path, "w") as f:
                OmegaConf.save(self.hydra_config, f)
            logger.info(f"Saved Hydra config at {config_path}")


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


class ETACallback(TrainerCallback):
    def __init__(self, print_interval_seconds=300):
        self.print_interval = print_interval_seconds
        self.start_time = None
        self.last_print_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        self.start_time = time.time()
        self.last_print_time = self.start_time

        print(f"Total training steps: {state.max_steps}")
        print("Training started...")

    def on_log(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        if state.global_step == 0:
            return

        now = time.time()

        if now - self.last_print_time < self.print_interval:
            return

        elapsed = now - self.start_time
        steps_done = state.global_step
        total_steps = state.max_steps

        steps_per_sec = steps_done / elapsed
        remaining_steps = total_steps - steps_done

        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
        eta_hours = eta_seconds / 3600

        print(
            f"[Step {steps_done}/{total_steps}] "
            f"Elapsed: {elapsed / 3600:.2f}h | "
            f"ETA: {eta_hours:.2f}h"
        )

        self.last_print_time = now

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        total_time = time.time() - self.start_time
        print(f"Training finished. Total time: {total_time / 3600:.2f}h")
