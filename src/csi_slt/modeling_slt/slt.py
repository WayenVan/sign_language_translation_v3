from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel, AutoConfig
from transformers.utils import ModelOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
from dataclasses import dataclass
import torch
from torch import nn
from typing import Optional, NamedTuple


from ..configuration_slt.configuration import SltConfig
from .registry import VISUAL_ADAPTERS, VISUAL_BACKBONES

from .output_utils import (
    VisualBackboneOutput,
    VisualAdapterOutput,
    PrepareForCausalLMOutput,
)


class SltModel(PreTrainedModel, GenerationMixin):
    config_class = SltConfig

    def __init__(self, config: SltConfig):
        super().__init__(config)
        self._init_llm()
        self._init_visual_backbone()
        self._init_visual_adapter()

        self.start_video_embds = nn.Parameter(
            torch.zeros(
                1, self.config.hidden_size, dtype=torch.float32, device=self.device
            ),
            requires_grad=True,
        )
        self.end_video_embeds = nn.Parameter(
            torch.zeros(
                1, self.config.hidden_size, dtype=torch.float32, device=self.device
            ),
            requires_grad=True,
        )

        self.config.is_encoder_decoder = False
        self.config.is_decoder = True

    def _init_llm(self):
        self.llm_config = AutoConfig.from_pretrained(self.config.llm_model_name_or_path)

        if self.config.llm_model_name_or_path.startswith("google/gemma-3-1b"):
            from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

            Gemma3ForCausalLM.from_pretrained(
                self.config.llm_model_name_or_path, **self.config.llm_init_kwargs
            )

        else:
            self.llm = AutoModel.from_pretrained(
                self.config.llm_model_name_or_path, **self.config.llm_init_kwargs
            )

    def _init_visual_backbone(self):
        backbone_cls = VISUAL_BACKBONES.get(self.config.visual_backbone_type)
        if backbone_cls is None:
            raise ValueError(
                f"Unsupported visual backbone type: {self.config.visual_backbone_type}"
            )
        self.visual_backbone = backbone_cls(**self.config.visual_backbone_kwargs)

    def _init_visual_adapter(self):
        adapter_cls = VISUAL_ADAPTERS.get(self.config.visual_adapter_type)
        if adapter_cls is None:
            raise ValueError(
                f"Unsupported visual adapter type: {self.config.visual_adapter_type}"
            )
        self.visual_adapter = adapter_cls(**self.config.visual_adapter_kwargs)

    @torch.no_grad()
    def _post_init(self):
        # init the start and end video embeddings with the mean of the word embeddings
        mean = self.gemma.get_input_embeddings().weight.data.mean(dim=0, keepdim=True)
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
            torch.stack(extened_visual_feats, dim=0),  # [B, L, D]
            self.gemma.get_input_embeddings()(text_input_ids).contiguous(),  # [B, L, D]
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
        use_cache = llm_forward_kwargs.get("use_cache", None)

        inputs_embeds = None
        if not use_cache:
            prepare_output = self.prepare_for_casual_lm(
                input_ids, pixel_values, pixel_values_length
            )
            inputs_embeds = prepare_output.inputs_embeds

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
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
