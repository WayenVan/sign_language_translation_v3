import sys

sys.path.append("./src")

from csi_slt.data.processors.sign_video_processor import SignVideoProcessor
from csi_slt.data.processors.slt_processor import SignTranslationProcessor
from csi_slt.data.ph14t.ph14t_torch_dataset import Ph14TGeneralDataset
from csi_slt.data.collators.general_collator import GeneralSLTCollator
from transformers import AutoTokenizer, AutoVideoProcessor
from torch.utils.data import DataLoader


def test_processor_save_load():
    processor = SignVideoProcessor()
    processor.train_transform
    processor.prediction_transform
    processor.save_pretrained("outputs/processor_test")

    loaded_processor = AutoVideoProcessor.from_pretrained("outputs/processor_test")
    assert loaded_processor is not None


def test_slt_processor_save_load():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    video_processor = SignVideoProcessor()
    processor = SignTranslationProcessor(
        video_processor=video_processor,
        tokenizer=tokenizer,
    )
    processor.save_pretrained("outputs/slt_processor_test")
    processor = SignTranslationProcessor.from_pretrained(
        "outputs/slt_processor_test", trust_remote_code=True
    )
    assert processor is not None


def test_slt_processor():
    with open("jinjas/gemma_slt.jinja", "r", encoding="utf-8") as f:
        chat_template = f.read()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    video_processor = SignVideoProcessor(224, 224, downsample_rate=1)
    processor = SignTranslationProcessor(
        video_processor=video_processor,
        tokenizer=tokenizer,
        chat_template=chat_template,
        mode="train",
    )
    dataset = Ph14TGeneralDataset(
        data_root="/root/dataset/PHOENIX-2014-T-release-v3",
        mode="train",
    )
    collator = GeneralSLTCollator(processor=processor)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collator
    )

    for batch in dataloader:
        print(batch)


if __name__ == "__main__":
    # test_processor_save_load()
    test_slt_processor()
    # test_slt_processor_save_load()
