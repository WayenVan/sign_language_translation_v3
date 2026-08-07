"""
The slt version specific for Qwen-VL, to utilize the off-the-shelf Qwen-VL visual backbone and LLM.
"""

from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
import torch
from torch import nn
from typing import Optional
from transformers.cache_utils import DynamicCache, Cache
from transformers.generation.configuration_utils import GenerationConfig
from transformers import PretrainedConfig

from transformers import logging
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from typing import Callable
from einops import rearrange


from ..configuration_slt.configuration import SltConfig
from .registry import VISUAL_ADAPTERS, VISUAL_BACKBONES

from .output_utils import (
    VisualBackboneOutput,
    VisualAdapterOutput,
    PrepareForCausalLMOutput,
)

logger = logging.get_logger(__name__)


def is_meta_model(model):
    for name, param in model.named_parameters():
        # 如果发现一个参数在 meta device，就认为这是 meta 模型
        return param.device == torch.device("meta")
    return False


def token_type_ids_mask_function(
    token_type_ids: Optional[torch.Tensor],
) -> Optional[Callable]:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """
    # Do not return an additional mask in this case
    if token_type_ids is None:
        return None

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        is_video_block_q = token_type_ids[batch_idx, q_idx] == 1
        is_video_block_kv = token_type_ids[batch_idx, kv_idx] == 1

        return is_video_block_q & is_video_block_kv

    return inner_mask


class SltQwenVLModel(PreTrainedModel, GenerationMixin):
    config_class = SltConfig
    MAX_TOKEN_LENGTH = 512
    _tied_weights_keys = {
        "llm.lm_head.weight": "llm.model.language_model.embed_tokens.weight"
    }

    def __init__(self, config: SltConfig):
        super().__init__(config)
        self._init_visual_adapter()
        self._init_llm()

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

        assert self.config.video_token_scale == 1 / (
            self.llm_vision_config.temporal_patch_size
            * self.visual_adapter.get_temporal_downsample_factor()
        ), (
            "The video_token_scale must be the inverse of the product of temporal_patch_size and temporal downsample factor of visual adapter."
        )

        self._init_embedding_weights()
        self.post_init()

    @property
    def dummy_inputs(self):
        N_FRAMES = 32 * int(
            1.0 / self.config.video_token_scale
        )  # NOTE: make sure the number of video tokens is a integer
        V_TOKEN = self.config.video_soft_token_id
        V_TOKEN_NUM = (
            int(self.config.video_token_scale * N_FRAMES) + 2
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
                (N_FRAMES, 3, 224, 224), dtype=torch.float32, device=self.device
            ),
            "pixel_values_length": torch.tensor(
                [N_FRAMES], dtype=torch.long, device=self.device
            ),
            "attention_mask": torch.ones(1, seq_len , dtype=torch.long, device=self.device),
            "labels": torch.ones( 1, seq_len, dtype=torch.long, device=self.device),
            "position_ids": torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0),
        }
        # fmt: on

    def _init_llm(self):
        self.llm_config = AutoConfig.from_pretrained(
            self.config.llm_model_name_or_path, **self.config.llm_init_kwargs
        )

        if is_meta_model(self):
            self.llm = Qwen3VLForConditionalGeneration._from_config(
                self.llm_config,
                **self.config.llm_init_kwargs,
            )

        else:
            self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
                self.config.llm_model_name_or_path,
                **self.config.llm_init_kwargs,
            )

        self.llm_text_config = self.llm.config.get_text_config()
        self.llm_vision_config = self.llm.config.vision_config

        self.config.bos_token_id = self.llm_text_config.bos_token_id

        # NOTE: fix the eos_token_id issue for gemma3, it could be a list, but not supported in huggingface
        if isinstance(self.llm_text_config.eos_token_id, list):
            self.config.eos_token_id = self.llm_text_config.eos_token_id[0]
        else:
            self.config.eos_token_id = self.llm_text_config.eos_token_id

        self.config.pad_token_id = self.llm_text_config.pad_token_id

        generation_config = self.llm.generation_config
        if generation_config is None:
            generation_config = GenerationConfig()

        generation_config.do_sample = False
        generation_config.top_k = None
        generation_config.top_p = None
        generation_config.temperature = None

        self.generation_config = generation_config  # NOTE: we copy genertion config from llm's original config

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
    def _init_embedding_weights(self):
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

    def get_video_grid(self, video_length: torch.Tensor):
        """
        Get the video grid size for each video in the batch.
        args:
            video_length: Tensor, shape [B], length of each video in the batch
        returns:
            video_grid: Tensor, shape [B, 3], the grid size for each video in the batch, including temporal and spatial dimensions
        """
        B = video_length.shape[0]
        video_grid_t = video_length / self.llm_vision_config.temporal_patch_size
        _hw = 224 / self.llm_vision_config.patch_size
        video_grid_hw = (
            torch.tensor([_hw, _hw], device=video_length.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1)
        )  # [B, 2]
        video_grid = torch.cat(
            [video_grid_t.unsqueeze(1), video_grid_hw], dim=1
        ).long()  # [B, 3]
        return video_grid

    def get_placeholder_mask(self, text_input_ids: torch.Tensor):
        """
        Get the placeholder mask for the input text, which indicates the positions of video tokens in the input text.
        args:
            text_input_ids: Tensor, shape [B, L], input text ids
        returns:
            placeholder_mask: Tensor, shape [B, L], the placeholder mask for the input text
        """
        placeholder_mask = text_input_ids.eq(
            self.config.video_soft_token_id
        ).long()  # [B, L]
        return placeholder_mask

    def get_visual_feats(
        self, video: torch.Tensor, video_length: torch.Tensor
    ) -> VisualAdapterOutput:
        """
        Forward pass through the visual encoder.
        args:
            video: Tensor, shape [BT, C, H, W], concated video frames across batch
            video_length: Tensor, shape [B], length of each video in the batch
        """
        video_grid = self.get_video_grid(video_length)  # [B, 3], [B]

        visual_feats = self.llm.model.get_video_features(
            video, video_grid, return_dict=True
        )  # [BT, CLS+HW, C]

        visual_feats = torch.cat(visual_feats.pooler_output).contiguous()  # [BTHW, C]
        visual_feats = rearrange(
            visual_feats,
            "(f hw) c -> f hw c",
            hw=video_grid[0, 1]
            * video_grid[0, 2]
            // (self.llm_vision_config.spatial_merge_size**2),
        )

        visual_backbone_output = VisualBackboneOutput(
            visual_features=visual_feats,  # [BT, H W, C]
            visual_length=video_grid[:, 0],
        )

        _, C, H, W = video.shape
        B = video_length.shape[0]

        visual_adapter_output: VisualAdapterOutput = self.visual_adapter(
            visual_backbone_output
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

        if t_length is None:
            raise ValueError("video_length is required for prepare_for_casual_lm")

        visual_feats = torch.split(
            visual_feats, t_length.tolist(), dim=0
        )  # list of [T, D]
        visual_feats = [
            torch.cat(
                [
                    self.start_video_embds,
                    visual_feats[b],
                    self.end_video_embeds,
                ],
                dim=0,
            )
            for b in range(B)
        ]  # add start and end video embeddings for each video in the batch
        visual_feats = torch.cat(visual_feats, dim=0)

        visual_mask_text = self.get_placeholder_mask(text_input_ids)  # [B, L]

        t_length_text = visual_mask_text.sum(
            dim=1
        )  # [B], number of video tokens in text

        assert (t_length_text == t_length + 2).all(), (
            "The length of text and video must be the same."
        )  # NOTE: 2 extra tokens for video was added

        inputs_embeds = self.llm.get_input_embeddings()(
            text_input_ids
        ).contiguous()  # [B, L, D]
        visual_feats = visual_feats.to(inputs_embeds.dtype)

        inputs_embeds = inputs_embeds.masked_scatter(
            visual_mask_text.unsqueeze(-1).bool(), visual_feats
        )  # [B, L, D]

        return PrepareForCausalLMOutput(
            input_ids=text_input_ids,  # [B, L]
            inputs_embeds=inputs_embeds,  # [B, L, D]
            visual_mask=visual_mask_text,  # [B, L]
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L] [<pad>, ..., <bos>, .... <video_soft_token>, ...]
        pixel_values: Optional[torch.Tensor] = None,  # [BT, C, H, W]
        pixel_values_length: Optional[
            torch.Tensor
        ] = None,  # [B], length of each video in the batch
        attention_mask: Optional[torch.Tensor] = None,  # [B, L]
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,  # [B, L]
        labels: Optional[torch.Tensor] = None,  # [B, L]
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **llm_forward_kwargs: dict,
    ):
        # if pixel_values is provided, pixel_values_length is not provcided, we assume there is only one video in the batch
        if pixel_values_length is None and pixel_values is not None:
            assert input_ids.shape[0] == 1, (
                "When pixel_values_length is not provided, input_ids batch size must be 1."
            )
            pixel_values_length = torch.tensor(
                [pixel_values.shape[0]], dtype=torch.long, device=pixel_values.device
            )
        # length must be a multiple of (1/video_token_scale)
        if pixel_values_length is not None:
            assert (
                pixel_values_length % int(1.0 / self.config.video_token_scale) == 0
            ).all(), (
                "The length of pixel_values_length must be a multiple of (1/video_token_scale)."
            )

        past_key_values: Cache | None = llm_forward_kwargs.pop("past_key_values", None)
        inputs_embeds = llm_forward_kwargs.pop("inputs_embeds", None)

        if inputs_embeds is None:
            if pixel_values is not None:
                prepare_output = self.prepare_for_casual_lm(
                    input_ids, pixel_values, pixel_values_length
                )
                inputs_embeds = prepare_output.inputs_embeds
            else:
                assert input_ids.shape[1] == 1, (
                    "When inputs_embeds is None, input_ids sequence length must be 1."
                )
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(
                self.llm.config,
            )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Prepare mask arguments
        mask_kwargs = {
            "config": self.llm.config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        if (
            hasattr(self.llm.config, "use_bidirectional_attention")
            and self.llm.config.use_bidirectional_attention
        ):
            logger.warn(
                "The LLM is configured to use bidirectional, which is not fully supported by our current implementation. The causal mask will be disabled, but the model may still not work as expected."
            )
        # NOTE: this `is_prefill` logic is not flawless, it fails when we're using a cache eagerly initialized
        # (e.g. compiled prefill) AND `pixel_values` are not provided. Determining prefill in that case requires
        # checking data values, which is not compile-compatible.
        is_prefill = (
            not use_cache
            or past_key_values is None
            or not past_key_values.is_initialized
            or pixel_values is not None
        )

        # apply bidirectional atteninon for video tokens if needed
        if token_type_ids is not None and is_prefill:
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                token_type_ids.to(cache_position.device),
            )

        attention_mask = create_causal_mask(**mask_kwargs)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            past_key_values=past_key_values,
            pixel_values=None,  # NOTE: we are embeeding video features intoo inputs_embeds, so we don't need to pass
            pixel_values_videos=None,
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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        pixel_values_length=None,
        cache_position=None,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_length"] = pixel_values_length

        return model_inputs
