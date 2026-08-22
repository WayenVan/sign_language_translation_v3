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
