from transformers.processing_utils import ProcessorMixin
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.models.gemma3.processing_gemma3 import Gemma3Processor
from transformers.image_processing_utils import (
    ImageProcessingMixin,
    BaseImageProcessor,
    BatchFeature,
)
from transformers.utils import TensorType, filter_out_non_signature_kwargs
import numpy as np

from typing import Union

from albumentations import (
    CenterCrop,
    Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    Resize,
    ColorJitter,
)
from 


class SignTranslationProcessor(ProcessorMixin):
    attributes = ["video_processor", "tokenizer"]
    video_processor_class = 
    tokenizer_class = "AutoTokenizer"
