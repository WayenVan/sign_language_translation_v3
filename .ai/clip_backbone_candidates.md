# CLIP 风格 visual backbone 候选

目标是寻找获得广泛认可、视觉编码器约 1–2B 参数的 CLIP 风格模型，同时
提供适合 SLT visual adapter 的 dense patch features 和全局图像表示。

| 模型 | 视觉塔规模（约） | 分辨率 / patch 数 | 优点 | 建议 |
| --- | ---: | ---: | --- | --- |
| [SigLIP 2 Giant 256](https://huggingface.co/google/siglip2-giant-opt-patch16-256) | 1B | 256px / 256 | Transformers 原生支持、Apache-2.0，训练目标增强 localization 和 dense features | **首选** |
| [OpenCLIP ViT-bigG/14](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) | ~1.8B | 224px / 256 | 成熟的大型 CLIP；OpenCLIP 报告 ImageNet zero-shot 80.1% | 很强但更重的备选 |
| [EVA01-CLIP-g/14-plus](https://huggingface.co/QuanSun/EVA-CLIP) | ~1B（完整模型 1.3B） | 224px / 256 | 成熟的 EVA-CLIP checkpoint、MIT、ImageNet zero-shot 79.3% | 可靠的旧模型，但接入不如 HF 原生方便 |
| [MetaCLIP 2 Worldwide Huge](https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu) | ~0.63B（完整模型 1.86B） | 224px / 256 | 多语言对比训练强，Transformers 原生支持 | CC-BY-NC-4.0 且视觉塔小于目标，不作为默认 |
| [SigLIP 2 Giant 384](https://huggingface.co/google/siglip2-giant-opt-patch16-384) | 1B | 384px / 576 | 相同 SigLIP 2 recipe，空间细节更多 | 仅在算力充足时使用；每帧 patch 数是 256 版的 2.25 倍 |

## 结论

采用 `google/siglip2-giant-opt-patch16-256`。其视觉配置为 40 层、hidden
size 1536、MLP size 6144、16 个 attention heads。完整图文 checkpoint 约
2B 参数，视觉塔是官方发布的 1B 级 ViT-g。除了 classification、retrieval
和 VLM transfer，SigLIP 2 还明确增强了 localization 和 dense prediction，
比只依据全局 zero-shot accuracy 选择模型更适合手部和身体 patch features。

SLT 接口使用 `last_hidden_state` 作为 packed per-frame patch features，使用
`pooler_output` 作为每帧 global feature。256px 输入配合 16px patch size，
每帧产生 256 个 patch tokens，两种输出的维度均为 1536。

数据管线需要使用 checkpoint 对应的 256px preprocessing 和官方
normalization，不能直接沿用现有的 224px ImageNet-normalized 配置。

References: [SigLIP 2 paper](https://arxiv.org/abs/2502.14786),
[OpenCLIP](https://github.com/mlfoundations/open_clip), and
[EVA-CLIP](https://huggingface.co/QuanSun/EVA-CLIP).
