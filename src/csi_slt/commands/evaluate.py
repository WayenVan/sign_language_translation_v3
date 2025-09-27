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
from transformers.trainer_utils import PredictionOutput

from accelerate import Accelerator


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))

set_seed(42)


@hydra.main(version_base=None, config_path=DEFAULT_CONFIG_PATH, config_name="base_eval")
def main(cfg: DictConfig):
    # accelerate initialize
    acc = Accelerator()

    # create model
    slt_model = SltModel.from_pretrained(
        cfg.model.checkpoint_dir,
    )

    # create datamodule
    llm_name = slt_model.config.llm_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(llm_name)

    datamodule = DataModule(cfg.data, tokenizer=tokenizer)
    datamodule.setup()

    # generation config
    #
    generation_config_args = OmegaConf.to_container(
        cfg.engine.generation_config, resolve=True
    )
    model_generation_config = slt_model.generation_config.to_dict()

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
        test_data_collator=datamodule.test_collator,
    )

    pred: PredictionOutput = trainer.predict(datamodule.test_dataset)

    if acc.is_main_process:
        preds_ids, pred_length, prompt_length = pred.predictions
        labels_ids = pred.label_ids

        full_prediction_texts = []
        predction_texts = []
        label_texts = []
        B = labels_ids.shape[0]
        for b in range(B):
            full_prediction = preds_ids[b][: pred_length[b]]
            prediction = full_prediction[prompt_length[b] :]
            label = labels_ids[b]
            # replace -100 in the labels as we can't decode them
            label = [l if l != -100 else tokenizer.pad_token_id for l in label]
            # decode
            full_pred_text = tokenizer.decode(
                full_prediction,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            pred_text = tokenizer.decode(
                prediction, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            label_text = tokenizer.decode(
                label, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            full_prediction_texts.append(full_pred_text)
            predction_texts.append(pred_text)
            label_texts.append(label_text)

        # save to file
        with open(os.path.join(training_args.output_dir, "predictions.txt"), "w") as f:
            for i in range(B):
                f.write(f"=== Example {i} ===\n")
                f.write(f"Full Prediction: {full_prediction_texts[i]}\n")
                f.write(f"Prediction: {predction_texts[i]}\n")
                f.write(f"Label: {label_texts[i]}\n")
                f.write("\n")


if __name__ == "__main__":
    main()

