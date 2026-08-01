import hydra

from omegaconf import DictConfig, OmegaConf
import os
from ..engine.trainer import SltTrainer
from ..engine.training_args import SltTrainingArguments
from ..data.datamodule import DataModule
from transformers import set_seed
from transformers import AutoTokenizer
from ..modeling_slt.slt import SltModel
from ..misc.utils import deep_merge
from transformers.generation.configuration_utils import GenerationConfig
import re
from accelerate import Accelerator


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))

set_seed(42)


@hydra.main(
    version_base=None, config_path=DEFAULT_CONFIG_PATH, config_name="train/ft"
)
def main(cfg: DictConfig):
    acc = Accelerator()
    # create model
    # slt_config = SltConfig(**OmegaConf.to_container(cfg.model.config, resolve=True))
    # slt_model = SltModel(slt_config).cuda()
    slt_model = SltModel.from_pretrained(
        cfg.model.checkpoint_dir,
    )

    # create datamodule
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint_dir)

    datamodule = DataModule(cfg.data, tokenizer=tokenizer)
    datamodule.setup("train")

    # generation config
    #
    generation_config_args = OmegaConf.to_container(
        cfg.engine.generation_config, resolve=True
    )
    model_generation_config = slt_model.generation_config.to_dict()

    # NOTE: enable training for visual adapter
    if cfg.model.ft_config.unfreeze_embedding:
        if acc.is_main_process:
            print("Unfreeze input embedding")
        for param in slt_model.get_input_embeddings().parameters():
            param.requires_grad = True

    # enable parameters by regex
    for name, param in slt_model.named_parameters():
        for regex in cfg.model.ft_config.target_params:
            if re.match(regex, name):
                param.requires_grad = True
                if acc.is_main_process:
                    print(f"Enable training for {name}")
                break

    # create trainer
    training_args = SltTrainingArguments(
        generation_config=GenerationConfig(
            **deep_merge(model_generation_config, generation_config_args)
        ),
        **cfg.engine.training_args,
    )
    trainer = SltTrainer(
        model=slt_model,
        args=training_args,
        hydra_config=cfg,
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
