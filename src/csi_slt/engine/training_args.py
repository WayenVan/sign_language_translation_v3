import os
from dataclasses import dataclass, field
from datetime import datetime

from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.utils import logging


logger = logging.get_logger(__name__)


@dataclass
class SltTrainingArguments(Seq2SeqTrainingArguments):
    auto_output_dir: bool = field(
        default=True,
        metadata={
            "help": "Whether to automatically set the output directory based on the model name."
            " If set to False, the output directory will be set to the value of `output_dir`."
        },
    )
    auto_output_root: str | None = field(
        default=None,
        metadata={
            "help": "The root directory to save the training results to when `auto_output_dir` is True."
            " The final output directory will be `<auto_output_root>/<model_name>`."
        },
    )

    @staticmethod
    def __init_output_base_name():
        now = datetime.now()
        return now.strftime("%Y-%m-%d_%H-%M-%S")

    def __post_init__(self):
        super().__post_init__()
        if self.auto_output_dir:
            if not self.auto_output_root:
                raise ValueError(
                    "auto_output_root must be specified when auto_output_dir is True."
                )

            base_name = self.__init_output_base_name()
            output_dir = os.path.join(self.auto_output_root, base_name)

            if self.output_dir:
                logger.warning(
                    "The `output_dir` argument is set, but `auto_output_dir` is True. "
                    f"The `output_dir` will be overridden from {self.output_dir} to {output_dir}."
                )
            self.output_dir = output_dir

            if self.run_name:
                logger.warning(
                    "The `run_name` argument is set, but `auto_output_dir` is True. "
                    f"The `run_name` will be overridden to the new output directory name: {output_dir}."
                )
            self.run_name = output_dir
        else:
            if self.auto_output_root:
                raise ValueError(
                    "auto_output_root must not be specified when auto_output_dir is False. "
                    "The `output_dir` will be set to the value of `output_dir`."
                )
