import hydra

from omegaconf import DictConfig, OmegaConf
import torch
import os
from accelerate import Accelerator
from ..engine.trainer import SltTrainer
from transformers import TrainingArguments
from ..data.datamodule import DataModule
from transformers import set_seed
from torch.utils.data import DataLoader, Subset
from hydra.utils import instantiate
from transformers import AutoTokenizer
from ..modeling_slt.slt import SltConfig, SltModel
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.generation.configuration_utils import GenerationConfig


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))

set_seed(42)


@hydra.main(
    version_base=None, config_path=DEFAULT_CONFIG_PATH, config_name="base_train"
)
def main(cfg: DictConfig):
    # create model
    slt_config = SltConfig(**OmegaConf.to_container(cfg.model.config, resolve=True))
    slt_model = SltModel(slt_config).cuda()

    # create datamodule
    llm_name = cfg.model.config.llm_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(llm_name)

    datamodule = DataModule(cfg.data, tokenizer=tokenizer)
    datamodule.setup("train")

    # create trainer
    training_args = Seq2SeqTrainingArguments(
        generation_config=GenerationConfig(
            **OmegaConf.to_container(cfg.engine.generation_config, resolve=True)
        ),
        **cfg.engine.training_args,
    )
    trainer = SltTrainer(
        model=slt_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=datamodule.train_dataset,
        eval_dataset=datamodule.val_dataset,
        train_data_collator=datamodule.train_collator,
        eval_data_collator=datamodule.val_collator,
    )

    # trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    main()
