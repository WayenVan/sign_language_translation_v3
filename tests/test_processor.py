import sys

sys.path.append("./src")

from csi_slt.data.processors.sign_video_processor import SignVideoProcessor
from csi_slt.data.processors.slt_processor import SignTranslationProcessor
from csi_slt.data.ph14t.ph14t_torch_dataset import Ph14TGeneralDataset
from csi_slt.data.collators.general_collator import GeneralSLTCollator
from transformers import AutoTokenizer, AutoVideoProcessor, AutoProcessor
from torch.utils.data import DataLoader


def test_processor_save_load():
    processor = SignVideoProcessor()
    processor.save_pretrained("outputs/processor_test")

    loaded_processor = AutoVideoProcessor.from_pretrained("outputs/processor_test")
    assert loaded_processor is not None


def test_slt_processor_save_load():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    prompt_paths_per_language = {
        "en": "jinjas/en_prompt.md.j2",
        "de": "jinjas/de_prompt.md.j2",
        "zh": "jinjas/zh_prompt.md.j2",
    }
    video_processor = SignVideoProcessor()
    processor = SignTranslationProcessor(
        video_processor=video_processor,
        tokenizer=tokenizer,
        prompt_paths_per_language=prompt_paths_per_language,
    )
    processor.save_pretrained("outputs/slt_processor_test")
    processor = AutoProcessor.from_pretrained(
        "outputs/slt_processor_test", trust_remote_code=True
    )
    assert processor is not None


def test_slt_processor():
    prompt_paths_per_language = {
        "en": "jinjas/en_prompt.md.j2",
        "de": "jinjas/de_prompt.md.j2",
        "zh": "jinjas/zh_prompt.md.j2",
    }
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    video_processor = SignVideoProcessor(
        image_mean=[0.6, 0.6, 0.6], image_std=[0.2, 0.2, 0.2]
    )
    processor = SignTranslationProcessor(
        video_processor=video_processor,
        tokenizer=tokenizer,
        prompt_paths_per_language=prompt_paths_per_language,
    )
    dataset = Ph14TGeneralDataset(
        data_root="/root/dataset/PHOENIX-2014-T-release-v3",
        mode="train",
    )
    collator = GeneralSLTCollator(processor=processor, mode="eval")
    # collator.debug = True
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collator
    )

    for batch in dataloader:
        print(batch)


def test_slt_processor_checkpoint():
    processor = AutoProcessor.from_pretrained(
        "/root/projects/sign_language_translation_v3/outputs/debug_outputs/2025-10-06_04-07-31/checkpoint-10",
        trust_remote_code=True,
    )
    assert processor.video_processor.train_transform is not None
    assert processor is not None


if __name__ == "__main__":
    # test_processor_save_load()
    # test_slt_processor()
    # test_slt_processor_checkpoint()
    test_slt_processor_save_load()
