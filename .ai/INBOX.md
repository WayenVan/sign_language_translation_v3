# 📥 Inbox

> Capture first. Organize later.
> 条目按时间顺序排列,最新在底部。

---

## 2026-08-10 13:23

接下来要测试 DINOv3 H+ 在 decoder last 对比学习模式下的性能。

#todo #experiment

---

## 2026-08-12 06:43

在基础训练上加入 LoRA 一起训练，训练交给 Cognition。

#todo #training #lora

---

## 2026-08-22 12:45

有空的时候给 huggingface/accelerate 提 issue + PR：FSDP2 下持久化 buffer 会让
`fsdp2_load_full_state_dict` 崩在 `'Tensor' object has no attribute 'device_mesh'`。
最新的 1.14.0 仍未修复。完整的复现、根因、补丁草稿和测试方案见
[accelerate_fsdp2_persistent_buffer_bug.md](accelerate_fsdp2_persistent_buffer_bug.md)。
注意笔记第 5 节——有个自然但会造成静默权重破坏的错误修法。

#todo #upstream #fsdp #accelerate

---

## 2026-08-31 02:15

C-RADIOv4 逐层 patch 可分性诊断跑完了，三条结论推翻了之前的两个假设：末层特征**没有**被
背景污染（`output_layer: [-1]` 不用改）；**CLS attention 不能当 token 筛选的打分器**——
在「手 vs 脸」上 AUC 只有 0.034，是个反向的人脸检测器，用它做 top-k 会优先丢掉手；
两个探针任务全部饱和，说明 patch 特征不是瓶颈，真正该怀疑的是 224×224 下手型细节根本
没被编码。完整数据、图和复跑方式见
[cradio_patch_separability_diagnosis.md](cradio_patch_separability_diagnosis.md)。
下一步：手型回归探针（现有 features.npz 就能跑，不用重跑 backbone）。

#experiment #visual-adapter #diagnosis

---

## 2026-09-04 20:12

CTC blank 槽位稀释视觉序列的问题：软 collapse 是刻意不做的（保梯度），但代价是高比例
blank 预测原样变成 LLM token，长 blank run 浪费上下文预算。发现一个硬约束——视觉占位
符数量在 collator 阶段就按 `video_token_scale` 定死、与 CTC 预测内容无关，所以任何
「按内容动态合并 blank」的方案都要先把数据管线改成两阶段（先跑 CTC 再定占位符数量），
目前不具备。三档缓解方案（blank 比例正则 / 调小 video_token_scale / 两阶段 collator
重构）和取舍记在 [slt_ctc_design.md](slt_ctc_design.md) 第 8 节风险 5。暂缓未动，先记录。

#todo #ctc #codebook #architecture
