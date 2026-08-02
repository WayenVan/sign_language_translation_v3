import hydra
import sys
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

sys.path.append("./src")
from csi_slt.modeling_slt.slt import SltConfig, SltModel
from csi_slt.data.datamodule import DataModule


def test_datamodule():
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(
            config_name="train/base",
            overrides=[
                "data=ph14t_*x224x224_qwen_multiling",
                # "model=gemma3-1b-dino-base",
                # "data.processor.video_processor.image_mean=[0.6, 0.6, 0.6]",
            ],
        )

    llm_name = cfg.model.config.llm_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    print(tokenizer.convert_tokens_to_ids("\n"))

    datamodule = DataModule(
        cfg.data,
        tokenizer=tokenizer,
    )
    datamodule.setup("train")
    train_dataset = datamodule.train_dataset
    collator = datamodule.train_collator
    collator.debug = True

    datamodule.print_batch(batch_size=16, num_workers=0, random=True)

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
