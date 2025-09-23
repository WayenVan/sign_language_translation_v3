from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
import torch
from torch import nn
from typing import Optional
from transformers.cache_utils import DynamicCache, Cache

from transformers import logging


from ..configuration_slt.configuration import SltConfig
from .registry import VISUAL_ADAPTERS, VISUAL_BACKBONES

from .output_utils import (
    VisualBackboneOutput,
    VisualAdapterOutput,
    PrepareForCausalLMOutput,
)

logger = logging.get_logger(__name__)


class SltModel(PreTrainedModel, GenerationMixin):
    config_class = SltConfig
    MAX_TOKEN_LENGTH = 1024
    _tied_weights_keys = ["llm.lm_head.weight"]

    def __init__(self, config: SltConfig):
        super().__init__(config)
        self._init_llm()
        self._init_visual_backbone()
        self._init_visual_adapter()

        self.start_video_embds = nn.Parameter(
            torch.randn(
                1, self.config.hidden_size, dtype=torch.float32, device=self.device
            ),
            requires_grad=True,
        )
        self.end_video_embeds = nn.Parameter(
            torch.randn(
                1, self.config.hidden_size, dtype=torch.float32, device=self.device
            ),
            requires_grad=True,
        )
        self.visual_position_embedding = nn.Embedding(
            self.MAX_TOKEN_LENGTH, self.config.hidden_size
        )
        self.config.num_extra_tokens = 2  # start and end of vlideo

        self.config.is_encoder_decoder = False
        self.config.is_decoder = True

        self._post_init()

    @property
    def dummy_inputs(self):
        V_TOKEN = self.config.video_soft_token_id
        V_TOKEN_NUM = (
            int(self.config.video_token_scale * 4) + 2
        )  # NOTE: 2 extra tokens for start and end of video

        # fmt: off
        input_ids= torch.tensor(
            [[ 0, 0, 0, 1, 2, 3,] + [V_TOKEN] * V_TOKEN_NUM + [ 4, 5, 9, 7, ]],
            dtype=torch.long,
            device=self.device,
        )
        seq_len = input_ids.shape[1]
        return {
            "input_ids": input_ids,
            "pixel_values": torch.ones(
                (4, 3, 224, 224), dtype=torch.float32, device=self.device
            ),
            "pixel_values_length": torch.tensor(
                [4], dtype=torch.long, device=self.device
            ),
            "attention_mask": torch.ones(1, seq_len , dtype=torch.long, device=self.device),
            "labels": torch.ones( 1, seq_len, dtype=torch.long, device=self.device)
        }
        # fmt: on

    def _init_llm(self):
        self.llm_config = AutoConfig.from_pretrained(self.config.llm_model_name_or_path)

        attn_implementation = self.config.llm_init_kwargs.pop(
            "attn_implementation",
            "eager",  # NOTE: default to eager since spda produce nan
        )
        if self.config.llm_model_name_or_path.startswith("google/gemma-3-1b"):
            from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

            self.llm = Gemma3ForCausalLM.from_pretrained(
                self.config.llm_model_name_or_path,
                attn_implementation=attn_implementation,
                **self.config.llm_init_kwargs,
            )

        else:
            self.llm = AutoModel.from_pretrained(
                self.config.llm_model_name_or_path,
                attn_implementation=attn_implementation,
                **self.config.llm_init_kwargs,
            )

        self.config.bos_token_id = self.llm_config.bos_token_id
        self.config.eos_token_id = self.llm_config.eos_token_id
        self.config.pad_token_id = self.llm_config.pad_token_id

        generation_config = self.llm.generation_config
        generation_config.do_sample = False
        generation_config.max_length = self.MAX_TOKEN_LENGTH
        generation_config.top_k = None
        generation_config.top_p = None
        self.generation_config = generation_config  # NOTE: we copy genertion config from llm's original config

        for param in self.llm.parameters():
            param.requires_grad = False

    def _init_visual_backbone(self):
        backbone_cls = VISUAL_BACKBONES.get(self.config.visual_backbone_type)
        if backbone_cls is None:
            raise ValueError(
                f"Unsupported visual backbone type: {self.config.visual_backbone_type}"
            )
        self.visual_backbone = backbone_cls(**self.config.visual_backbone_kwargs)

        for param in self.visual_backbone.parameters():
            param.requires_grad = False

    def _init_visual_adapter(self):
        adapter_cls = VISUAL_ADAPTERS.get(self.config.visual_adapter_type)
        if adapter_cls is None:
            raise ValueError(
                f"Unsupported visual adapter type: {self.config.visual_adapter_type}"
            )
        self.visual_adapter = adapter_cls(**self.config.visual_adapter_kwargs)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    @torch.no_grad()
    def _post_init(self):
        # init the start and end video embeddings with the mean of the word embeddings
        mean = self.llm.get_input_embeddings().weight.data.mean(dim=0, keepdim=True)
        self.start_video_embds.copy_(mean)
        self.end_video_embeds.copy_(mean)

        # init the visual position embedding
        torch.nn.init.trunc_normal_(self.visual_position_embedding.weight, std=0.02)

    def visual_position_embedding_forward(
        self, video_feats: torch.Tensor, video_length: torch.Tensor
    ):
        """
        Forward pass through the visual position embedding.
        args:
            video_feats: Tensor, shape [BT, D], video features
            video_length: Tensor, shape [B], length of each video in the batch
        """
        B = video_length.shape[0]
        position_ids = torch.cat(
            [torch.arange(video_length[b], device=video_feats.device) for b in range(B)]
        )
        position_embeddings = self.visual_position_embedding(position_ids)
        return video_feats + position_embeddings  # [BT, D]

    def get_visual_feats(
        self, video: torch.Tensor, video_length: torch.Tensor
    ) -> VisualAdapterOutput:
        """
        Forward pass through the visual encoder.
        args:
            video: Tensor, shape [BT, C, H, W], concated video frames across batch
            video_length: Tensor, shape [B], length of each video in the batch
        """

        _, C, H, W = video.shape
        B = video_length.shape[0]

        visual_backbone_output: VisualBackboneOutput = self.visual_backbone(
            video, video_length
        )  # [BT, CLS+HW+REGISTIRY, C]
        visual_adapter_output: VisualAdapterOutput = self.visual_adapter(
            visual_backbone_output.visual_features, visual_backbone_output.visual_length
        )  # [BT,  D]

        return visual_adapter_output

    def prepare_for_casual_lm(
        self,
        text_input_ids: torch.Tensor,  # [B, L] [<pad>, ..., <bos>, .... <start_of_image>, ...]
        video: torch.Tensor,  # [BT, C, H, W]
        video_length: torch.Tensor,  # [B], length of each video in the batch
    ):
        B = video_length.shape[0]

        visual_output = self.get_visual_feats(video, video_length)

        visual_feats = self.visual_position_embedding_forward(
            visual_output.visual_features, visual_output.visual_length
        )  # [BT, D]

        _, D = visual_feats.shape

        t_length = (
            visual_output.visual_length
        )  # [B], number of video tokens in visual feats

        visual_feats = torch.split(
            visual_feats, t_length.tolist(), dim=0
        )  # list of [T, D]

        visual_mask_text = text_input_ids.eq(
            self.config.video_soft_token_id
        ).long()  # [B, L]
        t_length_text = visual_mask_text.sum(
            dim=1
        )  # [B], number of video tokens in text

        assert (t_length_text == t_length + 2).all(), (
            "The length of text and video must be the same."
        )  # NOTE: 2 extra tokens for video was added

        extened_visual_feats = []
        for b in range(B):
            start_video_pos = visual_mask_text[b].nonzero(as_tuple=True)[0][0]
            end_video_pos = visual_mask_text[b].nonzero(as_tuple=True)[0][-1]
            _ex_visual_feat = torch.cat(
                [
                    torch.zeros(start_video_pos, D, device=self.device),  # before
                    self.start_video_embds,
                    visual_feats[b],
                    self.end_video_embeds,
                    torch.zeros(
                        text_input_ids.shape[1] - end_video_pos - 1,
                        D,
                        device=self.device,
                    ),  # after
                ]
            )
            extened_visual_feats.append(_ex_visual_feat)

        inputs_embeds = torch.where(
            visual_mask_text.bool().unsqueeze(-1),  # [B, L, 1]
            torch.stack(extened_visual_feats, dim=0).contiguous(),  # [B, L, D]
            self.llm.get_input_embeddings()(text_input_ids).contiguous(),  # [B, L, D]
        )

        return PrepareForCausalLMOutput(
            input_ids=text_input_ids,  # [B, L]
            inputs_embeds=inputs_embeds,  # [B, L, D]
            visual_mask=visual_mask_text,  # [B, L]
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L] [<pad>, ..., <bos>, .... <video_soft_token>, ...]
        pixel_values: torch.Tensor,  # [BT, C, H, W]
        pixel_values_length: torch.Tensor,  # [B], length of each video in the batch
        attention_mask: Optional[torch.Tensor] = None,  # [B, L]
        labels: Optional[torch.Tensor] = None,  # [B, L]
        **llm_forward_kwargs: dict,
    ):
        use_cache = llm_forward_kwargs.pop("use_cache", None)

        if pixel_values_length is not None:
            assert (pixel_values_length % 4 == 0).all(), (
                "The length of pixel_values_length must be a multiple of 4."
            )

        past_key_values: Cache | None = llm_forward_kwargs.pop("past_key_values", None)
        inputs_embeds = llm_forward_kwargs.pop("inputs_embeds", None)

        # normal forward
        if not past_key_values or past_key_values[0][0] is None:
            prepare_output = self.prepare_for_casual_lm(
                input_ids, pixel_values, pixel_values_length
            )
            inputs_embeds = prepare_output.inputs_embeds
        else:
            assert use_cache is True, (
                "use_cache must be True when past_key_values is provided"
            )
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(
                self.llm.config,
            )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **llm_forward_kwargs,
        )

        loss = None
        if labels is not None:
            # shift so that tokens < n predict n
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(
        cls,
        *args,
        **kwargs,
    ):
        model = super().from_pretrained(*args, **kwargs)
        logger.warn(
            "NOTE: the `lm_head.weight` will be tied properly, ignore the warning above"
        )
        return model
