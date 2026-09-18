


- [ ] 三种语言 lora sweep， 两个 多语言 的lorasweep，分别在不同的节点上


- [ ] 跑 eval：每个 final ckpt 依次跑 5 个 suite（fixed / diverse / unseen / wrong_task / unrelated），本地跑，细节见 .ai/eval.md

```bash
bash scripts/eval/run_eval_fixed_prompt.sh      <CKPT>     # 单语言 ckpt 加 de|en|zh
bash scripts/eval/run_eval_diverse_prompt.sh    <CKPT>     # 多语言 ckpt only
bash scripts/eval/run_eval_unseen_prompt.sh     <CKPT>
bash scripts/eval/run_eval_wrong_task_prompt.sh <CKPT>
bash scripts/eval/run_eval_unrelated_prompt.sh  <CKPT>
```

- [ ] 14b 两个多语言 run 各取最好 ckpt（macro bleu4）跑 diverse sweep：`bash scripts/eval/sweep14b/run_multilang_diverse_sweep.sh`（输出 outputs/eval/14b-multilang-diverse-eval/）
      注：09-17 那次跑出的 11 个变体因 RoPE buffer bug 全部作废、已删除；bug 已于 09-18 修复（根因见 .ai/eval.md），重跑即可
