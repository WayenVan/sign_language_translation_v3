from omegaconf import OmegaConf

from csi_slt.data.datamodule import DataModule


class _RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def add_special_tokens(self, special_tokens_dict, **kwargs):
        self.calls.append((special_tokens_dict, kwargs))
        return 0


def _make_datamodule(data_cfg, tokenizer):
    return DataModule(
        data_cfg=OmegaConf.create(data_cfg),
        datamodule_cfg=OmegaConf.create({}),
        tokenizer=tokenizer,
        prompt_resolvers={},
    )


def test_registers_video_tokens_from_processor_config_without_replacing_existing():
    tokenizer = _RecordingTokenizer()

    _make_datamodule(
        {
            "processor": {
                "video_soft_token": "<unused0>",
                "video_start_token": "<unused1>",
            }
        },
        tokenizer,
    )

    assert tokenizer.calls == [
        (
            {
                "additional_special_tokens": ["<unused0>", "<unused1>"],
            },
            {"replace_extra_special_tokens": False},
        )
    ]


def test_does_not_register_tokens_absent_from_processor_config():
    tokenizer = _RecordingTokenizer()

    _make_datamodule({"processor": {"_target_": "example.Processor"}}, tokenizer)

    assert tokenizer.calls == []
