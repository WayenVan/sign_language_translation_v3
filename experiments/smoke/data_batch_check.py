"""Manually build the real datamodule and inspect a collated batch."""

import hydra
import sys
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

sys.path.append("./src")
from csi_slt.modeling_slt.slt import SltConfig, SltModel
from csi_slt.data.datamodule import DataModule
from csi_slt.commands.config import instantiate_prompt_resolvers


def test_datamodule():
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(
            config_name="train/base",
            overrides=[
                # "data=ph14t_*x224x224_qwen_multiling",
                "data=ph14t_*x224x224_qwen_single_language",
                "data.language=zh",
                # "prompt=fixed_prompt",
                # "data=ph14t_*x224x224_gemma_multiling",
                # "datamodule=shared_subset",
                # "model=gemma3-1b-dno-base",
                "data.processor.video_processor.do_normalize=False",
            ],
        )

    llm_name = cfg.model.config.llm_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    print(tokenizer.convert_tokens_to_ids("\n"))

    datamodule = DataModule(
        cfg.data,
        cfg.datamodule,
        tokenizer=tokenizer,
        prompt_resolvers=instantiate_prompt_resolvers(
            cfg.prompt, ("train", "val", "test")
        ),
    )
    datamodule.setup("fit")
    train_dataset = datamodule.train_dataset
    collator = datamodule.train_collator
    collator.debug = True

    datamodule.print_batch(batch_size=16, num_workers=0, random=True, split="test")

    # loader = DataLoader(
    #     train_dataset,
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=0,
    #     collate_fn=collator,
    # )
    #
    # for batch in loader:
    #     # print(batch["input_ids"][0])
    #     # print(batch["labels"][0])
    #     # print(batch["input_ids"][0])
    #     print(batch)


if __name__ == "__main__":
    test_datamodule()
