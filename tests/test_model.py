import hydra
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append("./src")
from csi_slt.modeling_slt.slt import SltConfig, SltModel
from csi_slt.data.datamodule import DataModule


def test_slt_model():
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="base_train")
        slt_config = SltConfig(**cfg.model.config)
        slt_model = SltModel(slt_config).cuda()

        # prepare datamodule
        llm_name = cfg.model.config.llm_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(llm_name)

        datamodule = DataModule(
            cfg.data,
            tokenizer=tokenizer,
        )
        datamodule.setup("train")
        train_dataset = datamodule.train_dataset
        val_dataset = datamodule.val_dataset

        loader = DataLoader(
            train_dataset,
            # val_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            # collate_fn=datamodule.val_collator,
            collate_fn=datamodule.train_collator,
        )
        slt_model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            for batch in loader:
                outputs = slt_model(
                    input_ids=batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    pixel_values=batch["pixel_values"].cuda(),
                    pixel_values_length=batch["pixel_values_length"].cuda(),
                    labels=batch["labels"].cuda(),
                )
                print(outputs.loss)
                # print("prompt_length:" + str(batch["input_ids"].shape[1]))
                # outputs = slt_model.generate(
                #     input_ids=batch["input_ids"].cuda(),
                #     pixel_values=batch["pixel_values"].cuda(),
                #     pixel_values_length=batch["pixel_values_length"].cuda(),
                #     attention_mask=batch["attention_mask"].cuda(),
                # )
                # print(tokenizer.batch_decode(outputs, skip_special_tokens=False)[0])


if __name__ == "__main__":
    test_slt_model()
