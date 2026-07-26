# DINOFrameAdapterCross 结构图生成提示词

生成一张适合机器学习论文/技术报告的横向模型架构图，白色背景、扁平化矢量信息图风格、清晰的深色无衬线字体、细线箭头、无装饰性 3D 渲染、无人物、无真实照片、无水印。画面比例 16:9，分辨率高，所有公式与张量形状必须清晰可读。

图标题：`DINOFrameAdapterCross: Cross-Frame Patch Aggregation for Sign Language Translation`

展示从左到右的数据流，分为四个有浅色背景的阶段，并以颜色区分模块：DINO 特征为蓝色，跨帧聚合为橙色，池化为绿色，投影输出为紫色。

1. 左侧输入：一个 `Packed variable-length video batch`。画出两个由竖向虚线分开的连续视频片段，例如 `Video 1: frames 1...T₁` 和 `Video 2: frames 1...T₂`，强调帧在 batch 维度连续拼接。标注：`F = Σᵢ Tᵢ`，`visual_length = [T₁, T₂, ...]`。

2. DINOv2 feature extraction：每一帧经过冻结的 `Frozen DINOv2 with registers`，输出两类特征：
   - 蓝色 `CLS token [F, D]`
   - 蓝色 patch token grid `Patch features [F, P, D]`，其中 `P = H × W`，`D = 768`
   明确标注 register tokens are discarded。

3. Cross-frame patch aggregation：
   - 主分支为当前帧 patch grid：`Current patches [F, P, D]`
   - 另一分支将 patch grid 做 `right shift within each video only`，使用虚线边界清楚表示绝不能跨 Video 1 / Video 2 边界；第一帧 `copy itself`。得到 `Previous-frame patches [F, P, D]`。
   - 两个分支进入 `L2 normalize → cosine similarity`，画出一个橙色小矩阵并标注 `Similarity [F, P, P]`。
   - 接着 `softmax over previous-frame patches`，然后 `weighted aggregation`，输出 `Aggregated previous features [F, P, D]`。
   - 将 current patch 与 aggregated previous feature concatenate，标注 `Concat [F, P, 2D]`。

4. Learned patch pooling and projection：
   - `LayerNorm(2D) → Linear(2D, 1) → Softmax over P`，输出绿色 `Patch weights [F, P]`；用小热力图/权重条显示“每帧对 P 个 spatial patches 做内容相关加权”。
   - `Weighted sum over patches`，输出 `Pooled local feature [F, 2D]`。
   - 与来自第 2 步的 CLS token concatenate，标注 `Concat CLS + pooled local [F, 3D]`。
   - 紫色投影模块：`LayerNorm(3D) → Linear(3D, hidden_dim) → GELU → Linear(hidden_dim, output_dim)`。
   - 最终输出为紫色 `One visual token per frame [F, output_dim]`，旁注 `output_dim = 2048` 和 `visual_length preserved`。

在图底部添加一个简洁注释栏：

- `Temporal constraint: all frame shifts and interactions stay within each video segment.`
- `Spatial attention complexity: O(F · P²).`
- `Output: one temporally enriched visual token for each video frame.`

要求：严格按上述逻辑绘制；清楚呈现 packed variable-length batch、视频边界、右移操作、P×P patch correspondence、softmax pooling 和每帧一个输出 token。不要画语言模型、文本 token、contrastive loss、训练曲线或无关组件。
