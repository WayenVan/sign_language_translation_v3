import torch

from csi_slt.data.datamodule import DataModule


class _ReadableTokenizer:
    def batch_decode(self, rows, *, skip_special_tokens):
        assert skip_special_tokens is False
        return [" ".join(str(token_id) for token_id in row) for row in rows]


def _make_datamodule() -> DataModule:
    datamodule = object.__new__(DataModule)
    datamodule.tokenizer = _ReadableTokenizer()
    return datamodule


def test_text_fields_as_strings_decodes_every_processor_text_path():
    datamodule = _make_datamodule()
    batch = {
        "input_ids": torch.tensor([[0, 10, 11], [20, 21, 22]]),
        "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        "labels": torch.tensor([[-100, -100, 11], [-100, 21, 22]]),
        "pseudo_gloss_input_ids": torch.tensor([[0, 30], [40, 41]]),
        "pseudo_gloss_attention_mask": torch.tensor([[0, 1], [1, 1]]),
        "pseudo_gloss_teacher_input_ids": torch.tensor(
            [[0, 10, 30, 11], [20, 40, 41, 22]]
        ),
        "pseudo_gloss_teacher_attention_mask": torch.tensor(
            [[0, 1, 1, 1], [1, 1, 1, 1]]
        ),
        "pseudo_gloss_teacher_labels": torch.tensor(
            [[-100, -100, -100, 11], [-100, -100, -100, 22]]
        ),
        "empty_source_teacher_input_ids": torch.tensor(
            [[0, 10, 11], [0, 20, 22]]
        ),
        "empty_source_teacher_attention_mask": torch.tensor(
            [[0, 1, 1], [0, 1, 1]]
        ),
        "empty_source_teacher_labels": torch.tensor(
            [[-100, -100, 11], [-100, -100, 22]]
        ),
        "pixel_values": torch.zeros(2, 3, 1, 1),
        "names": ("sample-a", "sample-b"),
        "lang": ("en", "de"),
        "input_text": None,
    }

    text_batch = datamodule._text_fields_as_strings(batch)

    assert text_batch == {
        "input_ids": ["10 11", "20 21 22"],
        "labels": ["11", "21 22"],
        "pseudo_gloss_input_ids": ["30", "40 41"],
        "pseudo_gloss_teacher_input_ids": ["10 30 11", "20 40 41 22"],
        "pseudo_gloss_teacher_labels": ["11", "22"],
        "empty_source_teacher_input_ids": ["10 11", "20 22"],
        "empty_source_teacher_labels": ["11", "22"],
        "names": ["sample-a", "sample-b"],
        "lang": ["en", "de"],
    }


def test_text_field_decoding_rejects_a_misaligned_attention_mask():
    datamodule = _make_datamodule()
    batch = {
        "input_ids": torch.tensor([[10, 11]]),
        "attention_mask": torch.tensor([[1]]),
    }

    try:
        datamodule._text_fields_as_strings(batch)
    except ValueError as error:
        assert "attention_mask shape" in str(error)
    else:
        raise AssertionError("Expected a mismatched attention mask to be rejected")
