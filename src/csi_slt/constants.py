from enum import Enum
from bidict import bidict


LANGUAGE_MAP = bidict(
    {
        "en": 1,
        "de": 2,
        "zh": 3,
    }
)

LANGUAGE_NAME_MAP = bidict(
    {
        "en": "english",
        "de": "Deutsch",
        "zh": "中文",
    }
)
