# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os,sys
import logging
from typing import Dict, List, Optional, Tuple, OrderedDict

import numpy as np

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from fairseq.modules import GradMultiply, LayerNorm
from copy import deepcopy
# from .encoder import TransformerEncoder_prompt

DBG=True if len(sys.argv) == 1 else False

if DBG:
    from hubert_pretraining import (
        AVHubertPretrainingConfig,
        AVHubertPretrainingTask,
    )
    from resnet import ResEncoder
    # logging.basicConfig(
    #     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
    #     stream=sys.stdout,
    # )
    from utils import compute_mask_indices
    from decoder import TransformerDecoder

else:
    from .hubert_pretraining import (
        AVHubertPretrainingConfig,
        AVHubertPretrainingTask,
    )
    from .resnet import ResEncoder
    from .utils import compute_mask_indices
    from .decoder import TransformerDecoder

from omegaconf import II

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)


@dataclass
class AVHubertConfig(FairseqDataclass):
    label_rate: int = II("task.label_rate")
    input_modality: str = II("task.input_modality")
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={
            "help": "dropout to apply to the features (after feat extr)"
        },
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length_audio: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_audio: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_length_image: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_image: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={
            "help": "number of filters for convolutional positional embeddings"
        },
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={
            "help": "number of groups for convolutional positional embedding"
        },
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    resnet_relu_type: str = field(default='prelu', metadata={"help": 'relu type for resnet'})
    resnet_weights: Optional[str] = field(default=None, metadata={"help": 'resnet weights'})
    sim_type: str = field(default='cosine', metadata={"help": 'similarity type'})

    sub_encoder_layers: int = field(default=0, metadata={'help': 'number of transformer layers for single modality'})
    audio_feat_dim: int = field(default=-1, metadata={'help': 'audio feature dimension'})
    modality_dropout: float = field(default=0, metadata={'help': 'drop one modality'})
    audio_dropout: float = field(default=0, metadata={'help': 'drop audio feature'})
    modality_fuse: str = field(default='concat', metadata={'help': 'fusing two modalities: add,concat'})
    selection_type : str = field(default='same_other_seq', metadata={'help': 'type of selectig images, same_other_seq: replace masked span with span from another sequence, same_seq: repace masked span with span of the same sequence'})
    masking_type : str = field(default='input', metadata={'help': 'input or feature masking'})

    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num of decoder layers"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings "
            "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.1,
        metadata={
            "help": "dropout probability for attention weights "
            "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

class MultiModalPromptLearner(nn.Module):
    def __init__(self, prompt_length, prompt_depth, dtype=torch.float32):
        super().__init__()
        prompt_length_half = prompt_length // 3 # use half length for generating static prompts, and the other for generating dynamic prompts
        # Default is 1, which is compound shallow prompting
        embed_dim_audio = 768
        embed_dim_video = 768
        embed_dim = embed_dim_audio + embed_dim_video
        
        self.prompt_depth = prompt_depth  # max=12, but will create 11 such shared prompts
        self.video_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, embed_dim_video, dtype=dtype), std=0.02))
        self.video_prompt_missing = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, embed_dim_video, dtype=dtype), std=0.02))
        self.audio_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, embed_dim_audio, dtype=dtype), std=0.02))
        self.audio_prompt_missing = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, embed_dim_audio, dtype=dtype), std=0.02))
        self.common_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, embed_dim_audio, dtype=dtype), std=0.02))
        self.common_prompt_video = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, embed_dim_audio, dtype=dtype), std=0.02))
        self.common_prompt_audio = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, embed_dim_audio, dtype=dtype), std=0.02))
        # Also make corresponding projection layers, for each prompt
        r = 16
        single_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//r),
                nn.GELU(),
                nn.Linear(embed_dim//r, embed_dim_audio),
                )
        self.compound_prompt_projections_audio = _get_clones(single_layer, self.prompt_depth) # modal-common prompts
        self.layernorm_audio = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])
        
        single_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//r),
                nn.GELU(),
                nn.Linear(embed_dim//r, embed_dim_video),
                )
        self.compound_prompt_projections_video = _get_clones(single_layer, self.prompt_depth) # modal-common prompts
        self.layernorm_video = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])
        self.common_prompt_projection_video = nn.Sequential(
                nn.Linear(embed_dim_audio, embed_dim_audio//r),
                nn.GELU(),
                nn.Linear(embed_dim_audio//r, embed_dim_video),
                )
        self.common_prompt_projection_audio = nn.Sequential(
                nn.Linear(embed_dim_audio, embed_dim_audio//r),
                nn.GELU(),
                nn.Linear(embed_dim_audio//r, embed_dim_audio),
                )

    def forward(self, missing_type):

        # Before returning, need to transform
        # prompts to 768 for the video side   
        
        # Prompts of prompt_depth layers (combination of 3 kinds)
        all_prompts_video = [ [] for _ in range(self.prompt_depth)]
        all_prompts_audio = [ [] for _ in range(self.prompt_depth)]
        
        for i in range(len(missing_type)):
            # set initial prompts for each modality
            if missing_type[i] == 0:  # modality complete
                initial_prompt_video = self.video_prompt_complete
                initial_prompt_audio = self.audio_prompt_complete
                common_prompt = self.common_prompt_complete
            elif missing_type[i] == 1:  # missing video 
                initial_prompt_video = self.video_prompt_missing
                initial_prompt_audio = self.audio_prompt_complete
                common_prompt = self.common_prompt_audio
            elif missing_type[i] == 2:  # missing audio 
                initial_prompt_video = self.video_prompt_complete
                initial_prompt_audio = self.audio_prompt_missing
                common_prompt = self.common_prompt_video

            # generate the prompts of the first layer
            all_prompts_video[0].append(self.compound_prompt_projections_video[0](self.layernorm_video[0](torch.cat([initial_prompt_video, initial_prompt_audio], -1))))
            all_prompts_audio[0].append(self.compound_prompt_projections_audio[0](self.layernorm_audio[0](torch.cat([initial_prompt_video, initial_prompt_audio], -1))))
            # generate the prompts of the rest layers
            for index in range(1, self.prompt_depth):
                all_prompts_video[index].append(
                    self.compound_prompt_projections_video[index](self.layernorm_video[index](torch.cat([all_prompts_video[index-1][-1], all_prompts_audio[index-1][-1]], -1))))
                all_prompts_audio[index].append(
                    self.compound_prompt_projections_audio[index](self.layernorm_audio[index](torch.cat([all_prompts_video[index-1][-1], all_prompts_audio[index-1][-1]], -1))))
            all_prompts_video[0][i] = torch.cat([
                    all_prompts_video[0][i], 
                    self.common_prompt_projection_video(common_prompt)]
                    ,0)
            # assert all_prompts_video[0][i].isnan().any(), "NaN detected in self.common_prompt_projection_video(common_prompt)"
            all_prompts_audio[0][i] = torch.cat([
                    all_prompts_audio[0][i], 
                    self.common_prompt_projection_audio(common_prompt)]
                    ,0)
        # generate the prompts in each layer as a tensor [B, L, C]
        all_prompts_video = [torch.stack(prompts) for prompts in all_prompts_video]
        all_prompts_audio = [torch.stack(prompts) for prompts in all_prompts_audio]
        
        for i in range(len(all_prompts_video)):
            all_prompts_video[i] = all_prompts_video[i].transpose(0, 1)
        for i in range(len(all_prompts_audio)):
            all_prompts_audio[i] = all_prompts_audio[i].transpose(0, 1)
        
        return all_prompts_video, all_prompts_audio   

class SubModel(nn.Module):
    def __init__(self, resnet=None, input_dim=None, cfg=None):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, cfg.encoder_embed_dim)
        self.encoder = TransformerEncoder(cfg) if cfg.encoder_layers > 0 else None

    def forward(self, x):
        if self.resnet is not None:
            x = self.resnet(x)
        x = self.proj(x.transpose(1, 2))
        if self.encoder is not None:
            x = self.encoder(x)[0].transpose(1, 2)
        else:
            x = x.transpose(1, 2)
        return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, prompt_length=16,
                 i=0, prompt_depth=0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # For the first iteration i, we do not need to add the learnable parameters here
        # as it will be added in the beginning, for both audio and the vision branch
        self.attn_mask = attn_mask
        self.prompt_length_half = prompt_length // 3
        if i == 0 and i <= prompt_depth:
            self.attn_prompt = nn.MultiheadAttention(d_model, 1)
            self.prompts_dynamic_complete = nn.init.normal_(torch.empty(self.prompt_length_half, 1, d_model), std=0.02)
            self.prompts_dynamic_video = nn.init.normal_(torch.empty(self.prompt_length_half, 1, d_model), std=0.02)
            self.prompts_dynamic_audio = nn.init.normal_(torch.empty(self.prompt_length_half, 1, d_model), std=0.02)
        # This must be consistent with the config file prompt
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):
        # For the first layer, we do not need to add any duplicate, as it is already added
        # as the shallow version
        x = inputs[0]  # [length, batch, channel]
        compound_prompts_deeper = inputs[1]
        # print("x:", x.shape)
        # print("compound_prompts_deeper[0]:", compound_prompts_deeper[0].shape)
        counter = inputs[2]
        missing_type = inputs[3]
        if len(compound_prompts_deeper) > 0:
            # This means that deeper compound prompts are turned on
            # Here it behaves differently for audio and visual side
            # Forward function is same for both

            # First check if the ith layer needs compound prompts or not
            if counter == 0:
                # First check if the ith layer needs compound prompts or not
                if counter < len(compound_prompts_deeper):
                    # Remove the outputs produced by learnable tokens of previous layer
                    if not self.first_layer:
                        visual_features = x[self.prompt_length_half * 3:, :, :]
                    else:
                        visual_features = x
                    prompts_dynamic = []
                    for i in range(len(missing_type)):
                        if missing_type[i] == 0:  # modality complete
                            prompts_dynamic.append(self.prompts_dynamic_complete)
                        elif missing_type[i] == 1:  # missing video 
                            prompts_dynamic.append(self.prompts_dynamic_audio)
                        elif missing_type[i] == 2:  # missing audio
                            prompts_dynamic.append(self.prompts_dynamic_video)
                    # print("prompts_dynamic[0] shape: ", prompts_dynamic[0].shape)
                    # print("prompts_dynamic length: ", len(prompts_dynamic))
                    prompts_dynamic = torch.cat(prompts_dynamic, 1)
                    visual_features = visual_features.transpose(0, 1)
                    
                    # print("prompts_dynamic shape: ", prompts_dynamic.shape)
                    # print("visual_features shape: ", visual_features.shape)
                    prompts_dynamic = self.attn_prompt(prompts_dynamic.to(x.get_device()).to(x.dtype), visual_features, visual_features, need_weights=False, attn_mask=None)[0]
                    # Create/configure learnable tokens of this layer
                    prompts_staged_and_common = compound_prompts_deeper[counter]  # extract the correct index
                    # assert prompts_staged_and_common.isnan().any(), f"NaN detected in prompts_staged_and_common 1"
                    # prompts_staged_and_common = prompts_staged_and_common.permute(1, 0, 2)
                    # assert prompts_staged_and_common.isnan().any(), f"NaN detected in prompts_staged_and_common 2"
                    
                    # assert prompts_dynamic.isnan().any(), f"NaN detected in prompts_dynamic"
                    
                    # Add the learnable tokens of this layer with the input, by replacing previous
                    # layer learnable tokens
                    
                    x = torch.cat([prompts_staged_and_common, prompts_dynamic, visual_features], dim=0)

                    # Once done, update the counter, so that the next time, it does not use same learnable tokens
                    counter += 1
                    # assert x.isnan().any(), f"NaN detected in x 1"
            else:
                # First check if the ith layer needs compound prompts or not
                if counter < len(compound_prompts_deeper):
                    # Remove the outputs produced by learnable tokens of previous layer
                    if not self.first_layer:
                        features = x[self.prompt_length_half*3:, :, :]
                    else:
                        features = x
                    prompts_dynamic_and_common = x[self.prompt_length_half:self.prompt_length_half*3, :, :]
                    # Create/configure learnable tokens of this layer
                    prompts = compound_prompts_deeper[counter]  # extract the correct index
                    # prompts = prompts.permute(1, 0, 2)
                    # Add the learnable tokens of this layer with the input, by replacing previous
                    # layer learnable tokens
                    
                    # logger.debug("counter: %d, prompts shape: %s", counter, prompts.shape)
                    # logger.debug("prompts_dynamic_and_common shape: %s", prompts_dynamic_and_common.shape)
                    # logger.debug("features shape: %s", features.shape)
                    
                    x = torch.cat([prompts, prompts_dynamic_and_common, features], dim=0)
                    # Once done, update the counter, so that the next time, it does not use same learnable tokens
                    counter += 1
                    # assert x.isnan().any(), f"NaN detected in x 1"
        x = x + self.attention(self.ln_1(x))
        # assert x.isnan().any(), f"NaN detected in x 2"
        x = x + self.mlp(self.ln_2(x))
        # assert x.isnan().any(), f"NaN detected in x 3"
        # print("x2:", x.shape)
        return [x, compound_prompts_deeper, counter, missing_type]  # return again as a list, so that nn.seq can work

@register_model("av_hubert", dataclass=AVHubertConfig)
class AVHubertModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: AVHubertConfig,
        task_cfg: AVHubertPretrainingConfig,
        dictionaries: List[Dictionary],
        **kwargs
    ) -> None:
        super().__init__()
        logger.info(f"HubertModel Config: {cfg}")

        self.prompt_length = 12
        self.prompt_depth = cfg.encoder_layers // 3
        self.modal_prompt_learner = MultiModalPromptLearner(self.prompt_length, self.prompt_depth)

        feature_ds_rate = 1
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate
        sub_cfg = deepcopy(cfg)
        sub_cfg.encoder_layers = sub_cfg.sub_encoder_layers
        resnet = ResEncoder(relu_type=cfg.resnet_relu_type, weights=cfg.resnet_weights)
        self.feature_extractor_audio = SubModel(resnet=None, input_dim=cfg.audio_feat_dim, cfg=sub_cfg)
        self.feature_extractor_video = SubModel(resnet=resnet, input_dim=resnet.backend_out, cfg=sub_cfg)
        self.modality_dropout, self.audio_dropout = cfg.modality_dropout, cfg.audio_dropout
        self.modality_fuse = cfg.modality_fuse
        self.encoder_embed_dim = cfg.encoder_embed_dim
        if self.modality_fuse == 'concat':
            self.embed = cfg.encoder_embed_dim * 2
        elif self.modality_fuse == 'add':
            self.embed = cfg.encoder_embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob_image, self.mask_prob_audio = cfg.mask_prob_image, cfg.mask_prob_audio
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length_image, self.mask_length_audio = cfg.mask_length_image, cfg.mask_length_audio
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        self.sim_type = cfg.sim_type
        self.selection_type = cfg.selection_type
        self.masking_type = cfg.masking_type

        final_dim = (
            cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        )

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.audio_feat_dim).uniform_() if self.masking_type == 'input' else torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        
        self.resblocks_audio = nn.Sequential(*[
            ResidualAttentionBlock(cfg.encoder_embed_dim, cfg.encoder_attention_heads, None, self.prompt_length, i, self.prompt_depth)
            for i in range(cfg.sub_encoder_layers)
        ])
        proj_std = (cfg.encoder_embed_dim ** -0.5) * ((2 * cfg.encoder_layers) ** -0.5)
        attn_std = cfg.encoder_embed_dim ** -0.5
        fc_std = (2 * cfg.encoder_embed_dim) ** -0.5
        for block in self.resblocks_audio:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        self.resblocks_video = nn.Sequential(*[
            ResidualAttentionBlock(cfg.encoder_embed_dim, cfg.encoder_attention_heads, None, self.prompt_length, i, self.prompt_depth)
            for i in range(cfg.sub_encoder_layers)
        ])
        for block in self.resblocks_video:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)
        self.ln_final = LayerNorm(cfg.encoder_embed_dim)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info(
                "cannot find dictionary. assume will be used for fine-tuning"
            )
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: AVHubertConfig, task: AVHubertPretrainingTask):
        """Build a new model instance."""

        kwargs = {}
        model = AVHubertModel(cfg, task.cfg, task.dictionaries, **kwargs)
        return model

    def apply_input_mask(self, x, padding_mask, target_list):
        B, C, T = x.shape[:3]
        is_audio = True if len(x.shape) == 3 else False
        if is_audio:
            mask_prob, mask_length = self.mask_prob_audio, self.mask_length_audio
        else:
            mask_prob, mask_length = self.mask_prob_image, self.mask_length_image
        if mask_prob > 0:

            mask_indices, starts, ends, batch_indexes = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices_np = mask_indices
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = x.transpose(1, 2).contiguous() # [B, T, C, H, W]
            if B == 1:
                x[mask_indices] = 0
            elif is_audio:
                x[mask_indices] = self.mask_emb
            elif self.selection_type == 'same_other_seq':
                perm = (torch.arange(B) + torch.randint(low=1, high=B, size=(1,))) % B
                x_perm = x[perm]
                x[mask_indices] = x_perm[mask_indices]
            elif self.selection_type == 'same_seq':
                batch_indexes_, other_indexes = [], []
                for batch_index, start, end in zip(batch_indexes, starts, ends):
                    length = end-start
                    other_start = np.setdiff1d(np.arange(T), np.arange(max(0, start-length), end))
                    if len(other_start) > 0:
                        other_start = np.random.choice(other_start, size=1)
                    else:
                        other_start = 0
                    other_end = other_start + length
                    other_indexes.append(np.arange(other_start, other_end).clip(max=T-1))
                    batch_indexes_.append(np.zeros([length], dtype=np.int64)+batch_index)
                batch_indexes, other_indexes = np.concatenate(batch_indexes_), np.concatenate(other_indexes)
                x[mask_indices] = x[batch_indexes, other_indexes]

            x = x.transpose(1, 2).contiguous()
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            logger.info(f"No mask channel prob for input masking")
        return x, mask_indices

    def apply_feature_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        assert self.mask_prob_audio == self.mask_prob_image and self.mask_length_audio == self.mask_length_image, f"masking prob/length for image/audio be same for feature masking"
        mask_prob, mask_length = self.mask_prob_audio, self.mask_length_image
        if mask_prob > 0:
            mask_indices, _, _, _ = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices, _, _, _ = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_features(self, source: torch.Tensor, modality: str) -> torch.Tensor:
        extractor = eval(f"self.feature_extractor_{modality}")
        if self.feature_grad_mult > 0:
            features = extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features

    def forward_targets(
            self, features: torch.Tensor, mask_indices: torch.Tensor, target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
            if mask_indices is not None:
                mask_indices = mask_indices[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, mask_indices, target_list

    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def compute_logits(self, feats, emb_mat):
        # feats: [B, T, F], emb_mat: [V, F]
        if self.sim_type == 'dot':
            logits = torch.matmul(feats, emb_mat.transpose(0, 1))
        elif self.sim_type == 'cosine':
            batch_size, timesteps, emb_dim = feats.size()
            feats_ = feats.view(-1, emb_dim)
            nom = (feats_.unsqueeze(dim=1) * emb_mat.unsqueeze(dim=0)).sum(dim=-1) # [B*T, V]
            denom = (feats_**2).sum(dim=-1).sqrt().unsqueeze(dim=1) * (emb_mat**2).sum(dim=-1).sqrt().unsqueeze(dim=0) # [B*T, V]
            logits = (nom/denom.clamp(min=1e-6)).view(batch_size, timesteps, -1)
        else:
            raise NotImplementedError
        logits = logits / self.logit_temp
        return logits

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        src_audio, src_video = source['audio'], source['video']
        if mask and self.masking_type == 'input':
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video)
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None

        features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
        features_video = self.forward_features(src_video, modality='video')
        modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()
        if self.training:
            if modality_drop_prob < self.modality_dropout:
                if audio_drop_prob < self.audio_dropout:
                    features_audio = 0 * features_audio
                else:
                    features_video = 0 * features_video
        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        if target_list is not None:
            features, mask_indices, target_list = self.forward_targets(features, mask_indices, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        if self.masking_type == 'feature' and mask:
            x, mask_indices = self.apply_feature_mask(features, padding_mask, target_list)
        else:
            x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
        proj_x = self.final_proj(x)
        if self.untie_final_proj:
            proj_x_list = proj_x.chunk(len(self.num_classes), dim=-1)
        else:
            proj_x_list = [proj_x for _ in self.num_classes]
        logit_list = [self.compute_logits(proj, emb).view(-1, num_class) for proj, emb, num_class in zip(proj_x_list, label_embs_list, self.num_classes)] # [[B*T, V]]
        mask, unmask = torch.logical_and(mask_indices, ~padding_mask).view(-1), torch.logical_and(~mask_indices, ~padding_mask).view(-1) # [B*T]
        logit_m_list, logit_u_list = [logit[mask] for logit in logit_list], [logit[unmask] for logit in logit_list]
        target_m_list, target_u_list = [target.view(-1)[mask].long() for target in target_list], [target.view(-1)[unmask].long() for target in target_list]
        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "target_m_list": target_m_list,
            "target_u_list": target_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def extract_finetune(self, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None):
        
        # for param in self.parameters():
        #     if param.grad is None:
        #         continue
        #     if torch.isnan(param.grad).any():
        #         print(f"NaN detected in {param} gradients in AVHubertModel!")
        #         self.zero_grad()
        #         print("NaN detected and set grad zero")
        #         return
        
        src_audio, src_video, missing_type = source['audio'], source['video'], source['type']
        if mask and self.masking_type == 'input':
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list=None)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list=None)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video) # mask_indices not used in fine-tuning
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None

        prompts_video, prompts_audio = self.modal_prompt_learner(missing_type)
        # print("prompts length: ", len(prompts_video))
        # print("prompts[0] shape: ", prompts_video[0].shape)
        # for p in prompts_video:
        #     assert p.isnan().any(), f"NaN detected in prompts_video"
        # for p in prompts_audio:
        #     assert p.isnan().any(), f"NaN detected in prompts_audio"
        if src_audio is not None and src_video is None:
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
            features_video = features_audio.new_zeros(features_audio.size(0), self.encoder_embed_dim, features_audio.size(-1))
        elif src_audio is None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = features_video.new_zeros(features_video.size(0), self.encoder_embed_dim, features_video.size(-1))
        elif src_audio is not None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]

        features_video = features_video.transpose(1, 2)
        features_audio = features_audio.transpose(1, 2)
        # print("features_video shape1: ", features_video.shape)
        # print("features_audio shape1: ", features_audio.shape)
        
        features_video = self.resblocks_video([features_video, prompts_video, 0, missing_type])[0].transpose(0, 1)#[:, self.prompt_length:, :]
        features_audio = self.resblocks_audio([features_audio, prompts_audio, 0, missing_type])[0].transpose(0, 1)#[:, self.prompt_length:, :]
        # print("features_video shape2: ", features_video.shape)
        # print("features_audio shape2: ", features_audio.shape)
        
        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=2)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        # print("features shape:", features.shape)
        features = features.transpose(0, 1)
        features_pen = features.float().pow(2).mean()

        # features = features.transpose(1, 2)
        
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)
        # logger.debug(f"padding_mask shape: {padding_mask.shape}")

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)
        x = features
        mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        # print("features_video shape: ", features_video.shape)
        # print("features_audio shape: ", features_audio.shape)
        # print("x shape: ", x.shape)
        # print("prompts audio[0] shape: ", prompts_audio[0].shape)
        # print("prompts video[0] shape: ", prompts_video[0].shape)
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )
        # x = self.ln_final(x)

        return x, padding_mask


    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []
        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None

    def get_logits(self, net_output, is_masked=True):
        raise NotImplementedError

    def get_targets(self, net_output, is_masked=True):
        raise NotImplementedError

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(
            x.float(), targets.float(), dim=-1
        ).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits
