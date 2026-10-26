# 聚类质量验收

`run_eval.py` 对 `topics.yaml` 中每个题目跑完整 pipeline，并在代码中计算流程指标（引文逐字率、首次评分通过率、用量、耗时）；survey 答案键为空时，聚类一致性与 Gap 召回保持 null。结果见 [docs/evaluation.md](../docs/evaluation.md)。
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

推荐用终端标注工具逐对完成（可随时 `q` 退出，再次运行从未完成处继续）：

```bash
python scripts/annotate_cluster_pairs.py evals/cluster-review-low-resource.json --reviewer "你的名字"
python scripts/review_clusters.py --report logs/clusters-20261025-071713-a7873c.json \
  --annotations evals/cluster-review-low-resource.json
```

工具把两篇摘要拆成编号句子；选择 same / different / uncertain、写理由，再输入句子编号作为
证据，工具逐字复制原句，因此引文一定与原文一致。每一项都必须由审核者在交互终端中输入，
非交互环境会拒绝运行；只写入 `provenance: human`。上面的 report 是当前默认分组
（complete linkage）运行；同一份标注也可对其他报告评分，只要语料指纹一致。

也可以手工编辑主文件的 `reviews`，每项填写：

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

## Connection / Gap 来源验收

`finding-review-main-side.json` 冻结了真实地图 `20261025-230430-c3fbbf` 的七项材料：
一条保留的 Connection、三对组内和三对主簇–侧簇 Gap 候选。**七项均待人工判断，
annotation 全部为 null**。队列包含模型弃权的候选，便于检查漏召回；它不是无偏全库抽样。
默认隐藏模型决定和理由，原文使用冻结语料，不在导出时调用模型。

```bash
python scripts/review_findings.py --run-id YOUR_MAP_ID --output evals/your-review.json
```

脚本拒绝覆盖文件。导出的 JSON 是审核材料；保存判断请使用地图的 **人工验收** 页面或
`POST /cluster/{id}/review/{item_id}`，不会自动导入离线文件中的标签。
先阅读双方原文，再选择 `supported`、`unsupported` 或 `uncertain`。
明确判断要求审核者、理由和双方至少 20 字符的逐字原文引文；无法确认就选 uncertain。
“有依据”的假设也不能证明全球新颖性。若查看了模型意见，应保留相应自报字段。

审核保存到独立 sidecar，绑定版本 token 和原文指纹，修改保留历史，不改动原始地图。
审核者身份是自行声明，系统不能验证 human 的真实性。无有效明确判断时 support_rate
为 null；该材料不报告 semantic_accuracy。现有 23 对聚类子主题材料也仍待人工完成。

页面现在支持 [独立 Review Agent](../docs/agents/review_agent.md) 自动给出原文支持、问题相关性、
可检验性三项分数。AI 报告保留 model 来源，既不填入人工标签，也不改变人工已保存的判断。
没有具体假设的论文对显示“待明确假设”，分数为 null。查看 AI 建议后再主动提交人工判断，
会记录已查看模型意见；模型评分不计入人类准确率。

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
