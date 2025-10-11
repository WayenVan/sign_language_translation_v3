import hydra
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from omegaconf import OmegaConf

sys.path.append("./src")
from csi_slt.modeling_slt.slt import SltConfig, SltModel
from csi_slt.data.datamodule import DataModule
import torchinfo
import re


def test_model_params():
    model = SltModel.from_pretrained(
        "outputs/first_demo/2025-09-24_01-05-28/best_checkpoint/best_eval_bleu4=0.0431"
    )

    for name, param in model.named_parameters():
        print(name, param.requires_grad)


def test_slt_model():
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="base_train")
        slt_config = SltConfig(**OmegaConf.to_container(cfg.model.config, resolve=True))
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
            # train_dataset,
            val_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=datamodule.val_collator,
            # collate_fn=datamodule.train_collator,
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
                    position_ids=batch["position_ids"].cuda(),
                )
                print(outputs.loss)
                # print("prompt_length:" + str(batch["input_ids"].shape[1]))
                # outputs = slt_model.generate(
                #     input_ids=batch["input_ids"].cuda(),
                #     pixel_values=batch["pixel_values"].cuda(),
                #     pixel_values_length=batch["pixel_values_length"].cuda(),
                #     attention_mask=batch["attention_mask"].cuda(),
                #     max_new_tokens=100,
                # )
                # print(tokenizer.batch_decode(outputs, skip_special_tokens=False)[0])


def test_model_save():
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="base_train")
        slt_config = SltConfig(**OmegaConf.to_container(cfg.model.config, resolve=True))
        slt_model = SltModel(slt_config).cuda()
    slt_model.save_pretrained("outputs/test_save")


def test_model_load():
    slt_model = SltModel.from_pretrained("outputs/test_save").cuda()
    print(slt_model)


def test_verify_gemma3():
    slt_model = SltModel.from_pretrained("outputs/test_save").cuda()
    model = slt_model.llm
    tokenizer = AutoTokenizer.from_pretrained(
        slt_model.config.llm_model_name_or_path,
        use_fast=True,
    )
    messages = [
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Write a poem on Hugging Face, the company",
                    },
                ],
            },
        ],
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs)

    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    print(outputs[0])


def test_dummy_inputs():
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="base_train")
        slt_config = SltConfig(**OmegaConf.to_container(cfg.model.config, resolve=True))
        slt_model = SltModel(slt_config).cuda()

    torchinfo.summary(
        slt_model,
        input_data=slt_model.dummy_inputs,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=4,
    )


def test_peft_model():
    from peft import get_peft_model, LoraConfig, TaskType

    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="base_train_ft")
        slt_config = SltConfig(**OmegaConf.to_container(cfg.model.config, resolve=True))
        slt_model = SltModel(slt_config)

    # for name, param in slt_model.named_parameters():
    #     print(name)

    lora_args = OmegaConf.to_container(cfg.engine.peft_config, resolve=True)

    target_modules = lora_args.pop("target_modules", None)
    target_modules_list = []
    if target_modules is not None:
        for name, _ in slt_model.named_modules():
            for regex in target_modules:
                if re.fullmatch(regex, name):
                    target_modules_list.append(name)
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

    peft_model = get_peft_model(slt_model, peft_config)
    print(peft_model.print_trainable_parameters())
    for param in peft_model.base_model.visual_adapter.parameters():
        param.requires_grad = True

    for name, param in peft_model.named_parameters():
        print(name, param.requires_grad)

    peft_model.save_pretrained("outputs/test_peft_save")


if __name__ == "__main__":
    # test_model_params()
    test_slt_model()
    # test_model_save()
    # test_model_load()
    # test_verify_gemma3()
    # test_dummy_inputs()
    # test_peft_model()
