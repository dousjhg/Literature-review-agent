# 聚类质量验收

`run_eval.py` 是整条 pipeline 的旧评测框架；`topics.yaml` 仍是待填写模板。
新增 `cluster_pairs.py` 专门检查真实论文的子主题分组，不能用文件所属大领域替代专家标签。

## 生成盲审材料

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/review_clusters.py \
  --report logs/clusters-<run_id>.json --output evals/cluster-review.json
```

主文件只有论文原文、论文对和空白判断，不显示当前分组、相似度和抽样层。
`.selection.json` 另存这些诊断信息，标注时不要展示给审核者。
脚本不会覆盖已有文件。抽样含组内较弱对、组间较强对以及确定性随机背景对；
因此结果只是问题定位指标，**不是整个文献库的无偏准确率**。

论文先按 ID 哈希分成 development / holdout；同标题或同 DOI 的可能重复记录会作为
整体分到一侧，禁止跨侧论文对。`--split holdout` 可以生成另一侧的材料。
目前保存的语料已经用于探索，不能因此把 holdout 宣称为从未查看过的最终测试集。
增加/修改语料后应重新冻结划分；源文本指纹不匹配时评分会拒绝旧标注。

## 填写判断

审核者编辑主文件的 `reviews`，每项填写：

- `judgment`：`same_subtopic`、`different_subtopic` 或 `uncertain`。
- `provenance`：人工填写 `human`；模型建议必须保留 `model`。
- `reviewer`：审核者标识；`rationale`：具体任务、方法或评价设置上的判断理由。
- `evidence`：明确的同组/异组判断需要双方原文引文，格式为 `paper_id`、`quote`。

共享“低资源语言”“LLM”等宽泛领域词不等于同一研究子方向。资料不足就选 uncertain。
可以引用原文标题，但仅凭标题无法确认的推断不应强行标注；建议第二位审核者复核争议对。
系统可以校验来源和逐字一致性，无法证明声明为 human 的标签确实由人作出。

```bash
python scripts/review_clusters.py --report logs/clusters-<run_id>.json \
  --annotations evals/cluster-review.json
```

只对具备审核者、理由、双方有效引文的 human 判断计算 co-membership precision/recall/F1。
读取 main/side 地图时，保留角色字段，抽样和评分均排除涉及 main 的论文对，单独报告排除数。
主组是问题相关性池；其中两篇论文不必属于同一研究子主题。主组应另按问题直接性和证据判断。
模型建议、未填记录不纳入分数；uncertain 单独计数，无有效标签时分数为 null。
同一份标注可以检查不同策略的分组，只要语料完全一致；禁止用已查看的评测对反复调参后
再把结果称为独立验证。

## 标签对照

```bash
python scripts/compare_cluster_labels.py \
  --baseline logs/clusters-<old>.json --candidate logs/clusters-<new>.json \
  --output logs/cluster-label-comparison.json
```

该工具要求两个报告成员划分完全一致；评分器使用同一 rubric，隐藏生产者 rationale 与
版本名，每次调用隔离 Nightshift 缓存及 H-MEM 记忆范围，核验回执，交替版本顺序。默认重复两次。
LLM 诊断与人类论文对评价是不同证据；改名不能被报告为分组准确率提升。

分组策略对比可加 `--compare-partitions`；此时允许成员划分变化，两边的尺寸规则都只作诊断，
使用相同的来源评分提示词。该实验比较整个策略组合，不能把效果单独归因于某一个参数。
