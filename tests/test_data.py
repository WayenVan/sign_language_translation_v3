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
        cfg = hydra.compose(config_name="base_train")

    llm_name = cfg.model.config.llm_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    print(tokenizer.convert_tokens_to_ids("\n"))

    datamodule = DataModule(
        cfg.data,
        tokenizer=tokenizer,
    )
    datamodule.setup("train")
    train_dataset = datamodule.train_dataset

    loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=datamodule.train_collator,
    )

    for batch in loader:
        # print(batch["input_ids"][0])
        # print(batch["labels"][0])
        print(batch["input_text"])


if __name__ == "__main__":
    test_datamodule()
