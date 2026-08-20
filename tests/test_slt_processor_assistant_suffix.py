from pathlib import Path

import pytest
from transformers import AutoTokenizer

from csi_slt.data.processors.sign_video_processor import SignVideoProcessor
from csi_slt.data.processors.slt_processor import SignTranslationProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    (
        "model_name",
        "video_soft_token",
        "video_start_token",
        "expected_suffix_id",
        "expected_suffix_text",
    ),
    [
        (
            "Qwen/Qwen3-1.7B",
            "<|video_pad|>",
            "<|vision_start|>",
            151645,
            "<|im_end|>",
        ),
        (
            "google/gemma-4-12b-it",
            "<unused0>",
            "<unused1>",
            106,
            "<turn|>",
        ),
    ],
)
def test_real_chat_template_suffix_preserves_decoded_target_text(
    model_name: str,
    video_soft_token: str,
    video_start_token: str,
    expected_suffix_id: int,
    expected_suffix_text: str,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                video_soft_token,
                video_start_token,
            ]
        },
        replace_extra_special_tokens=False,
    )
    processor = SignTranslationProcessor(
        video_processor=SignVideoProcessor(),
        tokenizer=tokenizer,
        prompt_paths_per_language={
            "de": str(PROJECT_ROOT / "jinjas/de_prompt.md.j2"),
        },
        video_soft_token=video_soft_token,
        video_start_token=video_start_token,
    )
    target_text = "Das Wetter ist heute schön."

    suffix_ids = processor._get_assistant_suffix_ids()
    _, target_ids, _ = processor._encode_prompts_and_targets(
        [target_text],
        ["de"],
        add_bos_token=False,
        add_eos_token=True,
    )

    assert suffix_ids == [expected_suffix_id]
    assert target_ids[0][-len(suffix_ids) :] == suffix_ids
    assert tokenizer.decode(
        target_ids[0],
        skip_special_tokens=False,
    ) == target_text + expected_suffix_text
    assert tokenizer.decode(
        target_ids[0],
        skip_special_tokens=True,
    ) == target_text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
