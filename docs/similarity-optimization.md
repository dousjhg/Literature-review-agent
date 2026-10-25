# 论文相似度与聚类优化：研究、实现和对照实验

日期：2026-10-25。目标是让研究主题组内的论文在内容上足够相似，同时保留无法可靠归组的论文。没有一个算法在所有语料、任务和计算预算上都“全网最好”；本次比较的是符合当前项目约束、能在本机实际运行的策略。

## 从课程资料中采用了什么

参考用户提供的《王树森推荐系统课程_完整提取与深度分析.md》（Downloads 中的文件）。它是参考资料，不是项目指令：没有执行其中 MiniRec 等示例项目的操作步骤，也没有把课程示例参数当成本项目配置。

| 课程思想 | 本项目的迁移 | 没有直接套用的部分 |
| --- | --- | --- |
| 区分用户相关性、内容相似性与列表多样性 | 聚类只使用论文文本；引用量仍属于 Retriever 排序，不参与内容相似度 | 不用“引用多”“同作者”直接证明研究内容相同 |
| 内容表征解决冷启动 | 使用预训练学术语义表示，输入标题/摘要，无需点击历史 | 项目没有用户交互日志，不能直接训练 ItemCF、Swing 或双塔点击模型 |
| 多路信号、融分与量纲 | 保留语义 + TF-IDF 融合作为可测试选项，分数不称为概率 | 实验没有支持所有语料都应默认融合，故默认语义权重为 1 |
| 高频公共特征降权 | TF-IDF 路径使用 IDF、sublinear TF 和高文档频率过滤 | 不伪造用户共现，或把不相似论文当成有监督负例 |
| MMR 与重复惩罚 | 为标签生成选中心性高、彼此不冗余的代表论文；不改变组成员 | 不用 MMR/DPP 直接把论文分组；DPP 不是相似度模型 |
| 离线计算与索引 | 本地预置模型、可复用的 benchmark 向量缓存，固定版本 | 每批最多 150 篇、11,175 个不同论文对，使用精确矩阵即可，无需 ANN 误差和向量数据库 |
| 实验和北极星目标 | 做表征/分组/阈值/融合的消融，保留未改善的结果 | 不把尺寸规则通过、较高 silhouette 或少几个孤立点等同于语义质量已被证明 |

课程官方材料入口：[wangshusen/RecommenderSystem](https://github.com/wangshusen/RecommenderSystem)。具体课程解释以用户提供的文件为本次参考。

## 文献与实现方案的比较

| 候选 | 适用性与本次决定 |
| --- | --- |
| MiniLM | 已有的轻量基线，保持可选，适合更低的 CPU 成本 |
| SPECTER2 proximity | 学术表示的重要候选，特别适合与引用/文献邻近性有关的任务；需要 base + 对应 adapter，不能只换模型名字再沿用任意 pooling |
| SemCSE cosine | 针对科学文本内容语义的候选，使用官方说明中的 CLS 表征和 cosine 版本；本次在 CPU 上实际验证并设为默认 |
| Complete linkage | 直接控制组内最差论文对，符合“同组论文内容相似”的可解释要求，替代只检查路径连通的旧方法 |
| Average linkage、密度/邻接图分组 | 可作为将来研究候选，但不具有同样的组内最差相似度约束；当前批量较小，优先采用可审查的约束 |

[SemCSE 官方模型卡](https://huggingface.co/CLAUSE-Bielefeld/SemCSE_cosine)区分 cosine 与 Euclidean 两个 checkpoint，并给出 CLS 用法；其作者报告的语义基准支持把它作为科学内容的候选，不能直接证明它对本项目所有子主题最好。[SPECTER2 官方模型卡](https://huggingface.co/allenai/specter2)说明了任务 adapter 的区别。二者都没有被当成跨任务通用最优证明。

## 当前默认算法

1. **表示**：`CLAUSE-Bielefeld/SemCSE_cosine`，固定 revision `abc74650cfd087c38d27c480778ac41907a08367`。官方指定 CLS pooling；不误用一般 token mean pooling。长输入按模型窗口编码，窗口向量再平均并归一化，达到配置上限时记日志。这个长输入扩展不等同于原论文完整评估协议，也不是尚待实现的 Markovian 引擎。
2. **相似度**：单位向量点积得到 cosine。结果代表内容相似性，不代表“同一主题”的概率，更不表示论文互相支持或因果关联。
3. **分组**：complete-linkage 层次聚类，以 `distance = 1 - similarity` 切树。默认 cutoff 为 0.7，无需先指定组数。每组内任意两个成员均达到实际 cutoff；不会只因为 A 接近 B、B 接近 C，就把相距很远的 A、C 自动合并。
4. **稳定性**：先按 paper id 排序，避免相同距离的并列结果随检索返回顺序改变。每篇论文恰好归属一组，孤立论文单独保留。
5. **标签样本**：MMR 以组内中心性为 relevance、与已选样本的相似度为重复惩罚，最多选 8 篇供 LLM 命名。标签输入会声明这是代表子集；全部真实 paper ids 仍由代码维护。
6. **审核与重试**：尺寸规则和真实 LLM grader 保留。一次重试可按配置调整 cutoff，并将 fix 传入 labeller；每次实际 cutoff、组大小、最差相似度与所用模型写入 trace。失败不会通过强行合并来掩盖。

完整链接的“最大组间距离”定义见 [SciPy linkage 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)。这是分组直径的约束，不是对论文结论真实一致的证明。

## 可选策略为何没有默认开启

### TF-IDF 融合

可用 `cluster_dense_weight=0.85`，让语义 cosine 与词项 cosine 融合。title 和 abstract 分别向量化并按权重组合；词项使用 unigram/bigram、IDF 和次线性词频。没有可用稀疏向量的论文对回退为纯语义，不把缺失当成零相似度证据。算法依据参见 [scikit-learn TF-IDF 文档](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)。

默认关闭：下表中，它略改善某些 broad-topic pair AUC，却降低相应聚类 ARI；不能因为课程介绍多路融合就强行上线。它还会依赖当前批次的词频分布，换批次后同一论文对的融合分数可能变化。

### 无监督自适应 cutoff

设置 `cluster_threshold_mode="adaptive"` 后，比较配置中的候选 cutoff，目标为：

`silhouette × 非单论文组覆盖率`。

全体一个组、全体单点不被赋予虚假的好 silhouette。所有选择只看距离矩阵，不读取主题标签；重试不低于配置的候选下限。silhouette 的定义/适用范围参见 [官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)。这是本项目的工程启发式，不是有监督校准，也不是经过证明的最优切分。

默认仍用固定 cutoff：自适应方法在混合大领域样本上更容易恢复粗主题，但在一个窄主题内部可能选择 0.4–0.45 并合并多数论文，无法证明它能区分研究子方向。默认 0.7 更便于审查组内相似度要求；未来需要有专家子主题标签的语料进一步校准。

## 可复现实验与结果

三份保存数据先过滤无摘要记录，并按 id、DOI、规范化标题去重，剩 **154 篇**。按 paper id 的 SHA-256 奇偶划成不重叠的 78 / 76 篇批次。每篇的“领域”标签仅来自它所在样本文件，属于弱标签，并非专家标注的研究子方向。

两批均已用于探索和检查，不把第二批称为未触碰的最终测试集。自适应 cutoff 没有读取这些弱标签；下面的指标只用它们评价“粗领域分离”。不同策略的 silhouette 不能代替真实语义标注。

76 篇混合批次的对照：

| 编码器 / 分组 | 实际 cutoff | 最大组占比 | 非单点覆盖率 | 粗领域 ARI |
| --- | --- | --- | --- | --- |
| MiniLM + 旧连通分量 | 0.70 | 52.63% | 65.79% | 0.2606 |
| MiniLM + complete，固定 | 0.70 | 14.47% | 51.32% | 0.1044 |
| MiniLM + complete，自适应 | 0.40 | 28.95% | 97.37% | 0.6002 |
| **SemCSE + complete，固定（默认）** | **0.70** | **18.42%** | **82.89%** | **0.2011** |
| SemCSE + complete，自适应 | 0.40 | 34.21% | 98.68% | 0.9033 |
| SemCSE + TF-IDF + complete，自适应 | 0.40 | 32.89% | 97.37% | 0.8371 |

同一批次中，旧默认最大组内部最小 cosine 只有 **0.2519**；新默认所有非单点组的最小 cosine 为 **0.7013**。这直接验证链式误合并被消除。新默认粗领域 ARI 低于旧默认，不能隐去；固定高阈值的任务是保留较细且紧密的内容组，而不是尽量恢复三个大领域。

一次观察到的冷启动 CPU 耗时（含加载与 154 篇编码）：MiniLM 约 **7.7 秒**，SemCSE 约 **31.8 秒**。这不是 P95 或严格控制硬件负载的性能基准，SemCSE 的成本更高；更注重速度时可选择 MiniLM。完整分组、参数、两批以及每个单领域的结果保存在 [JSON 对照报告](benchmarks/similarity-ablation.json)。

## 配置与复现

```bash
pip install -r requirements.txt
python scripts/setup_embeddings.py --profile semcse
python scripts/setup_embeddings.py --profile minilm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/benchmark_similarity.py --profiles minilm semcse
python scripts/check_threshold.py --profile semcse --output logs/thresholds-semcse.json
python scripts/check_threshold.py --profile minilm --legacy --output logs/thresholds-baseline.json
python scripts/check_clusters.py
pytest -q
```

默认 `EMBEDDING_PROFILE=semcse`。切换快模型用 `EMBEDDING_PROFILE=minilm`，不需要修改 Nightshift/NVIDIA Key。Docker 构建时预置所选模型，`--build-arg EMBEDDING_PROFILE=minilm` 可改用小模型；容器运行时不下载权重。测试保留 sample-mode 的输出契约，新增覆盖链式样本、边界阈值、排列不变性、空词表、模型/profile 不匹配和 MMR 代表选择。

实际 30 篇低资源语言论文的 Nightshift 运行：起始 0.7 得到 18 组，尺寸规则因单点组过多失败；一次重试到约 0.65 后得到 13 组，组内最小 cosine 为 0.6623。尺寸规则通过，但真实 LLM grader 仅给出 **6/9（未达 7/9）**，指出多个组仍共享过于相近的主题。原样保留警告和全部论文，图谱写入成功；没有为了拿到高分而执行会破坏组内阈值的强制合并。该运行 17 次 LLM 调用、35,485 tokens。故本次可以确认算法约束与链路改善，不能宣称研究主题质量已经完全验收。

上一阶段本机与 `litagent-similarity:local` 断网容器中均为 **181 项测试通过**。另外在无网络、无 GPU 的容器中加载预置的 SemCSE CLS 模型，对八篇真实论文生成 `(8, 768)` 向量，并验证 complete linkage 的 0.7 组内约束；镜像中没有 `.env`。模型预置层已与业务代码复制层分离，后续修改分组代码可复用模型层缓存。

后续最有价值的是收集专家标注的“同一研究子方向 / 易混淆不同方向”论文对和分组，按论文隔离划分开发与最终测试，并评估 pair precision/recall、组内语义一致性和孤立覆盖率。引用、共引、共同参考文献可以成为单独的关系信号，但当前字段不足，不能用 citation_count 冒充这些关系。

## 第二阶段：主题解释与质量验收

### 针对当前场景的研究选择（2026-10-25）

本阶段检查了近期原始论文、官方模型说明和真实项目输出，采用**相邻组对照、按任务/方法/数据区分、原文片段定位、独立评审与人工论文对评价**。这是基于这些方向的工程实现，不声称完整复现下列算法，也没有跨所有候选进行排名。

| 研究方向与原始来源 | 与本项目的关系 |
| --- | --- |
| [Context-Aware Hierarchical Taxonomy Generation，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.788/) | 科学论文包含方法、数据、评价等多个方面；采用这些方面帮助识别真正的子主题区别。当前没有实现其完整层次树和多方面训练流程。 |
| [Using language models to label clusters of scientific documents，2025](https://arxiv.org/abs/2511.02601) | 科学文献的可读标签本身需要评价，不能将标签好听等同于成员正确；分开检查标签与分组。 |
| [TLSA，ACL 2026](https://aclanthology.org/2026.acl-long.869/) | 研究相近类别的文本/标签对齐与消歧。采用邻近类别对照这一设计方向；本项目没有足够部分标注数据，不声称训练了 TLSA 双编码器或复现其指标。 |
| [LLM-generated constraints，AAAI 2026](https://arxiv.org/abs/2601.11118) | 生成的约束可能出错，需要处理噪声。本阶段先生成可审查的诊断与人工论文对，不把模型建议当真值或强制 must-link。 |
| [RAILS，2026-09 预印本](https://arxiv.org/abs/2609.24464) | 提出检索候选标签与有界批处理以适应增量大规模文本。当前每批最多 150 篇，采用有界批处理和邻组上下文即可；没有引入其在线标签池，也不把客服工单上的结果视为学术文献上的证明。 |
| [Qwen3 Embedding 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 是指令化、多语言表示的重要候选；本轮主要问题已有证据指向标签/粒度/缺失文本。保持上阶段测过的 SemCSE 作为控制变量，没有仅凭综合排行榜换模型或宣称优于 Qwen3。 |

### 已落实的改动

1. `litagent/cluster_labels.py`：每组保留全部标题，最多 8 个 MMR 原文示例，检索两个最近邻组作为对照。邻组提供最小、平均和最大跨组相似度，避免把一个很像的论文对误当成整个组都应合并。
2. 每批最多 4 个目标组统一命名；邻组上下文由原始论文计算，不读取前一批生成的标签，避免批次先后反馈。名称按照任务、方法、数据/领域、评价设置来区分，不能为了唯一而编造区别。
3. 模型选择 `paper_id + span_id`；代码依据源文本的连续偏移取回完整引文，并校验组归属、源文本包含关系和引用覆盖。真实测试发现模型直接复制引文会改变空格/标点，因此采用片段引用而非放宽逐字校验。
4. `coherent`、`mixed`、`insufficient_evidence` 明确区分。只有标题的组由代码直接拒绝认定一致；不会消耗模型调用。多组看起来很相似，也不允许通过改名隐藏成员混杂。
5. 有重试机会时，确定性的尺寸检查在命名前执行，避免给随后必然被重做的分组消耗预算。阈值和最终成员约束不变；`sample_mode` 输出契约不变。
6. Grader 不再读取生成器 rationale；原代码虽有“独立评分”的注释，实际仍把 rationale 发给模型。Rubric 现在要求检查来源来区分命名重复与成员问题，不能只因标签相似就要求合并。新旧结果必须在**同一新 rubric**下比较，不能把旧 6/9 与新的单次分数直接当成因果证据。
7. `litagent/cluster_audit.py`：输出每组最弱论文对、组间最近论文对、合并是否满足实际阈值、重复名称、可能重复记录和缺失摘要。不会据此自动删除论文或强制合并。
8. `evals/cluster_pairs.py` 与 `scripts/review_clusters.py`：生成隐藏模型预测的人工审阅包；阻止重复/未知论文对、源文件变化、开发/保留集跨论文泄漏。明确的人工判断要求审核者、理由和双方原文证据；模型建议不计入人工指标，未标注时分数保持 null。

入口和复现命令见 [聚类说明](agents/clusterer.md) 与 [人工验收说明](../evals/README.md)。`scripts/compare_cluster_labels.py` 在成员完全相同的条件下，用同一个提示词重复评判新旧标签，隐藏版本名和 rationale，且逐次隔离 Nightshift 缓存。其结果仍然只是 LLM 诊断，不是专家准确率。

**范围说明：**以上标签对照不改变表示和分组成员。原文匹配只证明引用存在，并不能证明它支持模型的语义判断。真正的子主题聚类改进仍需要人工论文对验证；Connection finder / Gap finder 的真实分支仍是后续工作。

### 发现并修正的评测污染：Nightshift H-MEM

同规则对照中，模型反复引用当前输入没有包含的旧 rationale 和旧 rubric。检查本机 Nightshift 0.92.3 随附源码后发现：

- `web/js/engine/proxy.js` 接收 `meterless-scope`，缺省为 `user`。
- `web/js/engine/hmem.js` 的 `retrieve` 以 scope 作硬隔离，但不以 app 作硬隔离。
- `web/js/engine/orchestrator.js` 会把同一 scope 下相似请求的旧答案作为系统上下文注入；这与返回缓存不同，**cache=miss 不表示没有旧答案注入**。

因此原来只有 `x-title: litagent-<run_id>` 的隔离不足。`integrations/llm.py` 现在为每次请求和 JSON 修复生成独立的 `meterless-scope`，记录 `receipt.hmem` 的注入数，拒绝明确返回了旧记忆注入或错误 scope 的响应。调用之间仍由项目的 Pydantic 数据和 World Model 传递上下文。没有修改用户的 Nightshift 全局记忆设置或删除其他记录。

隔离前的对照保存在 [已失效的诊断记录](benchmarks/cluster-label-quality-pre-isolation.json)，明确标注不能作为独立评测证据。隔离后的 [标签对照](benchmarks/cluster-label-quality.json) 中，每次评分回执均为零旧答案注入：旧标签两次为 **1/9、3/9**，新标签为 **4/9、4/9**。样本少、同一模型评分，且仍未达标，不能据此宣称人工语义准确率提高。

隔离后的完整聚类阶段运行 `20261025-064853-65ce71`：30 篇、13 组、3 次模型调用、28,794 tokens；21 条引文通过当时的拼接输入包含检查，3 次请求均报告零旧记忆注入；后续原始字段复核发现其中 3 条跨字段，修复与最终验收见下文。该次 grader 为 **6/9**，与重复对照的 4/9 不一致，也说明 LLM 评分存在波动。模型诊断为 3 个混杂组、4 个主题一致组、1 个证据不足组，另有 5 个单点组；这不是专家标签。这次历史结果见 [拼接输入阶段的审计快照](benchmarks/cluster-quality-audit-context-only.json)。

### 候选：原文支持的任务分组 + 硬相似度约束

根据上面的混杂问题，增加 `cluster_grouping_strategy="semantic"` 候选路径（默认仍是 `complete`）：

1. SemCSE 表示不变。模型读取有界原文片段与相似邻居，按具体任务、方法、数据/领域提出子主题，允许重新组合先前被层次聚类切开的论文。
2. 输出必须完整覆盖输入、无额外/重复 ID；模型遗漏或重复分配的记录保守地变成单点，并记录原因，不任意选择其中一个主题。多论文组必须引用每一个成员的有效原文片段。标题缺失证据和无兼容邻居的记录由代码单独保留，未知 ID 和无效引文仍然拒绝。
3. 每个建议组再经 complete linkage 投影到固定的原始相似度阈值。任何不满足 0.7 两两约束的建议都会被拆分，模型不能强行合并。
4. 这一路径把组尺寸作为诊断，始终请 grader 根据原文评价；不会因为单点多而降低阈值，也不会把尺寸合格当语义正确。对照时两种策略都使用相同的原文评分规则。

这是推理阶段的有约束语义分组，没有训练新模型，也没有把模型输出转成人工真值。原文截断、模型分组不稳定、词义相近却任务不同、投影造成同一标签多个紧密子组，仍是需要检查的局限。

```bash
python scripts/check_clusters.py --grouping-strategy semantic --max-redos 0
python scripts/compare_cluster_labels.py --compare-partitions \
  --baseline logs/clusters-<baseline>.json --candidate logs/clusters-<candidate>.json \
  --output logs/cluster-partition-comparison.json
```

`--max-redos 0` 用于一次性候选实验，避免用同一 grader 不断重做挑选高分。是否更改默认分组策略应由多主题测试和人工论文对质量决定，不只看某一次 LLM 得分。

### 候选实测与最终决定

相同的 30 条记录，在已隔离 H-MEM 的环境下得到以下结果；两次 semantic 运行都只做一次生成和一次来源评分，没有反复生成挑选高分：

| 路径 | 实际 cutoff | 组数 | 单点论文占比 | 本次 grader | 模型调用 / tokens |
| --- | --- | --- | --- | --- | --- |
| 默认 complete + 相邻对照命名 | 约 0.65（尺寸重试后） | 13 | 16.67% | 6/9 | 3 / 28,794 |
| semantic 候选 | 0.70 | 23 | 66.67% | 1/9 | 2 / 21,541 |
| semantic 候选，同基线 cutoff | 0.65 | 28 | 90.00% | 1/9 | 2 / 25,993 |

完整结果见 [分组候选报告](benchmarks/cluster-partition-candidates.json)。三次都验证了全部论文 ID 恰好保留一次、组内最小相似度满足各自 cutoff、模型回执显示零旧答案注入。引用当时通过的是拼接输入包含检查，后来严格原文字段复核分别发现 3、1、3 条跨字段引用，已在报告中披露，不能将这些历史运行称为原始字段级证据全部合格。两次候选是独立生成，不是固定模型提案后只改变 cutoff 的因果消融；不能据此声称“降低 cutoff 会增加单点”。候选生成不稳定、保守证据约束和几何投影共同造成了碎片化。

**决定：保留 `complete` 默认分组。** 新的任务分组只保留为可复现实验选项，没有因为论文更新或技术名称更先进就切换默认。候选开发时还发现遗漏、重复分配、缺失成员引文、格式错误和一次服务超时，均有日志；当前代码把遗漏、歧义和缺失引用的成员单独保留，未知 ID/无效片段仍须通过有界修复或明确失败。模型给出了一个方向名称，不等于已经解决精细子主题划分。

本阶段保留的默认改进是相邻对照命名、原文片段证据、独立记忆范围、先尺寸检查再命名，以及质量审计/盲审工具。聚类的语义质量仍未通过验收，不能宣称已经达到“全网最优”。下一项有价值的输入是对 [23 个待审论文对](../evals/cluster-review-low-resource.json) 的人工判断，明确目标子主题粒度；这份标注当前全部 pending，没有伪造准确率。

最终 **221 项测试在本机及断网 CPU 容器通过**。容器使用已有 `litagent-similarity:local` 的依赖/模型层，并只读挂载本轮最新的 `litagent`、`prompts`、`tests`、`evals`；没有把旧镜像误称为已重建的新镜像。测试覆盖记忆范围隔离、异常记忆注入、来源片段、跨组借用引文、覆盖完整性、重叠/缺失/无引文成员的保守保留、投影后两两约束、人工标注来源及隔离、原有 sample/API/World Model 契约。`git diff --check` 和依赖检查通过。


### 原始字段级证据的最终复核

最终检查区分了“出现在模型输入中”和“出现在原始摘要/全文中”：嵌入需要的标题与摘要拼接可能人为插入标点，不能将跨越该分隔符的字符串当论文原文。现在 `evidence_source` 先选择单独的 `abstract` 或 `full_text` 字段，再生成连续偏移片段；标题作为上下文另行展示。只有标题时仍不认定主题一致。

修复后的默认运行 `20261025-071713-a7873c`：**30 篇全部保留、13 组、3 次模型调用、29,516 tokens**。23 条引文全部在对应论文的原始摘要或全文内逐字匹配，零跨字段引用；3 次回执均为零旧记忆注入。最终 grader **6/9，仍未达标**，失败警告保留。模型自身标记 5 个混杂组、2 个一致组、1 个证据不足组，另有 5 个单点组；这些标签仍不是人工真值。

最终可审查数据在 [质量审计快照](benchmarks/cluster-quality-audit.json)。历史对照与候选失败结果没有被覆盖成成功结果；默认分组算法没有切换，人工论文对指标仍等待真实标注。
