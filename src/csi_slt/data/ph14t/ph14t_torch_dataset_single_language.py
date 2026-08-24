"""Single-language views of the multilingual PHOENIX14T dataset."""

from .ph14t_torch_dataset_multiling import Ph14TMultiLinglDataset


def _has_target_language(example, target_language: str) -> bool:
    """Keep one language; named arguments make Datasets' cache hash explicit."""

    return example["lang"] == target_language


class Ph14TSingleLanguageDataset(Ph14TMultiLinglDataset):
    """One-language view of PHOENIX14T multilingual data.

    Filtering uses Hugging Face Datasets' fingerprinted Arrow cache. It runs
    before PyTorch DataLoader workers are created, so workers only receive the
    already-filtered table and do not repeat the scan.
    """

    SUPPORTED_LANGUAGES = frozenset({"zh", "en", "de"})

    def __init__(
        self,
        data_root: str,
        language: str,
        mode: str = "train",
        pseudo_gloss_column: str = "orig_pseudo_gloss_strict",
        pipline=None,
    ):
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"language must be one of {sorted(self.SUPPORTED_LANGUAGES)}, "
                f"got {language!r}"
            )

        self.language = language
        super().__init__(
            data_root=data_root,
            mode=mode,
            pseudo_gloss_column=pseudo_gloss_column,
            pipline=pipline,
        )
        self.hg_dataset = self.hg_dataset.filter(
            _has_target_language,
            fn_kwargs={"target_language": language},
            load_from_cache_file=True,
            desc=f"Selecting {language} samples from {mode}",
        )
        if len(self.hg_dataset) == 0:
            raise ValueError(
                f"No {language!r} samples found in PHOENIX14T split {mode!r}."
            )

    @property
    def cache_namespace(self) -> str:
        return f"ph14t_single_language/{self.language}"
