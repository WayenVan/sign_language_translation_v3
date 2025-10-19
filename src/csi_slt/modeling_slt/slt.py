from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
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
    create_masks_for_generate,
)
from typing import Callable


from ..configuration_slt.configuration import SltConfig
from .registry import VISUAL_ADAPTERS, VISUAL_BACKBONES

from .output_utils import (
    VisualBackboneOutput,
    VisualAdapterOutput,
    PrepareForCausalLMOutput,
)

logger = logging.get_logger(__name__)


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


class SltModel(PreTrainedModel, GenerationMixin):
    config_class = SltConfig
    MAX_TOKEN_LENGTH = 512
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
            "labels": torch.ones( 1, seq_len, dtype=torch.long, device=self.device),
            "position_ids": torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0),
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
        elif self.config.llm_model_name_or_path.startswith("google/gemma-3"):
            from transformers.models.gemma3.modeling_gemma3 import (
                Gemma3ForConditionalGeneration,
            )

            self.llm = Gemma3ForConditionalGeneration.from_pretrained(
                self.config.llm_model_name_or_path,
                attn_implementation=attn_implementation,
                **self.config.llm_init_kwargs,
            )
        else:
            self.llm = AutoModel.from_pretrained(
                self.config.llm_model_name_or_path,
                **self.config.llm_init_kwargs,
            )

        self.config.bos_token_id = self.llm_config.bos_token_id

        # NOTE: fix the eos_token_id issue for gemma3, it could be a list, but not supported in huggingface
        if isinstance(self.llm_config.eos_token_id, list):
            self.config.eos_token_id = self.llm_config.eos_token_id[0]
        else:
            self.config.eos_token_id = self.llm_config.eos_token_id

        self.config.eos_token_id = self.llm_config.eos_token_id
        self.config.pad_token_id = self.llm_config.pad_token_id

        generation_config = self.llm.generation_config
        if generation_config is None:
            generation_config = GenerationConfig()

        generation_config.do_sample = False
        generation_config.max_length = self.MAX_TOKEN_LENGTH
        generation_config.top_k = None
        generation_config.top_p = None

        self.generation_config = generation_config  # NOTE: we copy genertion config from llm's original config

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
        # length must be a multiple of 4
        if pixel_values_length is not None:
            assert (pixel_values_length % 4 == 0).all(), (
                "The length of pixel_values_length must be a multiple of 4."
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

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.llm.config.get_text_config(),
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # NOTE: this `is_prefill` logic is not flawless, it fails when we're using a cache eagerly initialized
            # (e.g. compiled prefill) AND `pixel_values` are not provided. Determining prefill in that case requires
            # checking data values, which is not compile-compatible.
            is_prefill = (
                not use_cache
                or past_key_values is None
                or not past_key_values.is_initialized
                or pixel_values is not None
            )
            if token_type_ids is not None and is_prefill:
                mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                    token_type_ids.to(cache_position.device),
                )

            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
            # import matplotlib.pyplot as plt
            #
            # plt.imshow(causal_mask_mapping["full_attention"][0, 0].cpu().numpy())
            # plt.savefig("outputs/causal_mask.png")

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            # token_type_ids=token_type_ids,
            cache_position=cache_position,
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

    @staticmethod
    def create_masks_for_generate(
        config: PretrainedConfig,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        position_ids: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # Prepare mask arguments
        mask_kwargs = {
            "config": config.get_text_config(),
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Add the token type ids mask for generate as well
        if token_type_ids is not None and input_embeds.shape[1] != 1:
            # We need to pass an additional mask function to account for token type ids, and it needs to be an `or`
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                token_type_ids.to(cache_position.device),
            )
        return create_masks_for_generate(**mask_kwargs)

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
