import hydra

from omegaconf import DictConfig, OmegaConf
import os
from ..engine.trainer import SltTrainer
from ..engine.training_args import SltTrainingArguments
from ..data.datamodule import DataModule
from transformers import set_seed
from transformers import AutoTokenizer
from ..modeling_slt.slt import SltConfig, SltModel
from ..misc.utils import deep_merge
from transformers.generation.configuration_utils import GenerationConfig
import re
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
import accelerate as acc

from accelerate import Accelerator


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))

set_seed(42)


@hydra.main(
    version_base=None, config_path=DEFAULT_CONFIG_PATH, config_name="base_train_ft"
)
def main(cfg: DictConfig):
    # accelerate initialize
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

    # peft confg
    lora_args = OmegaConf.to_container(cfg.model.peft_config, resolve=True)

    target_modules = lora_args.pop("target_modules", None)
    target_modules_list = []
    if target_modules is not None:
        for name, _ in slt_model.named_modules():
            for regex in target_modules:
                if re.fullmatch(regex, name):
                    target_modules_list.append(name)
                    if acc.is_main_process:
                        print(f"Found target module for LoRA: {name}")
        if len(target_modules_list) == 0:
            raise ValueError(
                "No target modules found for LoRA. Please check the regex patterns."
            )

        lora_args["target_modules"] = target_modules_list
    else:
        lora_args["target_modules"] = []

    peft_config = LoraConfig(
        **lora_args,
        task_type=TaskType.CAUSAL_LM,
    )
    slt_model = get_peft_model(slt_model, peft_config)

    # NOTE: enable training for visual adapter
    for param in slt_model.base_model.visual_adapter.parameters():
        param.requires_grad = True

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
