import { useEffect, useState } from "react";
import { ArrowUpRight, ArrowRight, BookOpen } from "lucide-react";
import { request, messageOf } from "../api";
import type {
  Candidate,
  Evidence,
  MapResult,
  Move,
  Paper,
  Selection,
} from "../types";
function SourceQuotes({
  evidence,
  papers,
  showTitles = true,
}: {
  evidence: Evidence[];
  papers: Paper[];
  showTitles?: boolean;
}) {
  return (
    <div className="source-quotes">
      {evidence.map((e, i) => (
        <div key={i}>
          {showTitles && (
            <p className="source-caption">
              {papers.find((p) => p.id === e.paper_id)?.title ?? e.paper_id}
            </p>
          )}
          <blockquote>{e.quote}</blockquote>
        </div>
      ))}
    </div>
  );
}
export function Inspector({
  map,
  papers,
  selection,
  onSelect,
  moves,
  onMove,
  canEdit,
  onReview,
}: {
  map: MapResult;
  papers: Paper[];
  selection: Selection | null;
  onSelect: (s: Selection) => void;
  moves: Move[];
  onMove: (paper: string, target: string) => void;
  canEdit: boolean;
  onReview: () => void;
}) {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [error, setError] = useState("");
  const [full, setFull] = useState<Paper | null>(null);
  const paper =
    selection?.type === "paper"
      ? papers.find((p) => p.id === selection.id)
      : undefined;
  const cluster = map.clusters.find((c) =>
    c.paper_ids.includes(paper?.id ?? ""),
  );
  const detail = cluster?.paper_details?.find((d) => d.paper_id === paper?.id);
  useEffect(() => {
    setCandidates([]);
    setError("");
    setFull(null);
    if (!paper || cluster?.kind === "main") return;
    const abort = new AbortController();
    request<{ candidates: Candidate[] }>(
      `/cluster/${map.run_id}/regenerate/candidates?paper_id=${encodeURIComponent(paper.id)}`,
      { signal: abort.signal },
    )
      .then((r) => {
        if (!abort.signal.aborted) setCandidates(r.candidates);
      })
      .catch((e) => {
        if (!abort.signal.aborted) setError(messageOf(e));
      });
    return () => abort.abort();
  }, [paper?.id, cluster?.kind, map.run_id]);
  const connection =
    selection?.type === "connection"
      ? map.connections.find((c) => c.id === selection.id)
      : null;
  const gap =
    selection?.type === "gap"
      ? map.gaps.find((g) => g.id === selection.id)
      : null;
  const topic =
    selection?.type === "cluster"
      ? map.clusters.find((c) => c.id === selection.id)
      : null;
  return (
    <aside className="inspector" aria-label="详情与证据">
      <h2 className="inspector-heading">
        {paper
          ? "论文详情"
          : connection
            ? "关联证据"
            : gap
              ? "研究假设"
              : topic
                ? "主题详情"
                : "原文与研究证据"}
      </h2>
      {paper ? (
        <>
          <h3 className="paper-title">{paper.title}</h3>
          <p className="paper-meta">
            {paper.year ?? "年份未知"}
            <span>·</span>
            {paper.source === "openalex" ? "OpenAlex" : paper.source}
          </p>
          {!!detail?.keywords.length && (
            <div className="keywords">
              {detail.keywords.map((k) => (
                <span key={k}>{k}</span>
              ))}
            </div>
          )}
          <section className="inspector-section">
            <h3>原文证据</h3>
            {detail?.evidence.length ? (
              <div className="paper-evidence">
                <SourceQuotes
                  evidence={detail.evidence}
                  papers={papers}
                  showTitles={false}
                />
              </div>
            ) : paper.abstract ? (
              <blockquote>{paper.abstract}</blockquote>
            ) : (
              <p className="muted">当前记录没有可供核验的摘要。</p>
            )}
            {paper.abstract && (
              <details>
                <summary>展开完整摘要</summary>
                <p className="original-text">{paper.abstract}</p>
              </details>
            )}
            <button
              className="text-button small"
              onClick={async () => {
                try {
                  setFull(
                    await request<Paper>(
                      `/cluster/${map.run_id}/papers/${encodeURIComponent(paper.id)}`,
                    ),
                  );
                } catch (e) {
                  setError(messageOf(e));
                }
              }}
            >
              查看全文提供状态
            </button>
            {full && (
              <p className="muted small">
                {full.full_text
                  ? `已提供全文，${full.full_text.length.toLocaleString()} 个字符。`
                  : "未提供全文；本次核验使用可用摘要。"}
              </p>
            )}
            {full?.full_text && (
              <details>
                <summary>展开已提供全文</summary>
                <p className="original-text">{full.full_text}</p>
              </details>
            )}
          </section>
          <section className="inspector-section">
            <h3>归属与备选</h3>
            <div className="membership">
              <span className="muted">当前归属</span>
              <strong>
                {cluster?.kind === "main" ? "主簇" : cluster?.label}
              </strong>
            </div>
            {cluster?.kind === "main" ? (
              <p className="muted small">主簇成员保持固定</p>
            ) : (
              <>
                {(detail?.also_related ?? []).map((a) => (
                  <div key={a.cluster_id} className="alternative-row">
                    <span>
                      {map.clusters.find((c) => c.id === a.cluster_id)?.label ??
                        a.cluster_id}
                    </span>
                    <span>{a.similarity.toFixed(2)}</span>
                  </div>
                ))}
                <p className="muted small">
                  相似度是相关性提示，不代表额外归属或已核验的科学关联。
                </p>
                <label className="form-field small">
                  选择移动目标
                  <select
                    aria-label="选择移动目标"
                    disabled={!canEdit}
                    value={
                      moves.find((m) => m.paper_id === paper.id)
                        ?.target_cluster_id ?? ""
                    }
                    onChange={(e) => onMove(paper.id, e.target.value)}
                  >
                    <option value="">保持当前归属</option>
                    {candidates.map((c) => (
                      <option
                        key={c.target_cluster_id}
                        value={c.target_cluster_id}
                        disabled={!c.eligible}
                      >
                        {
                          map.clusters.find((x) => x.id === c.target_cluster_id)
                            ?.label
                        }{" "}
                        · {c.target_similarity.toFixed(2)}
                        {c.eligible ? "" : " · 不满足数值约束"}
                      </option>
                    ))}
                  </select>
                </label>
                {!canEdit && (
                  <p className="muted small">
                    打开当前版本并匹配服务模式后可预览移动。
                  </p>
                )}
              </>
            )}
          </section>
          {/^(W\d+)$/.test(paper.id) && (
            <a
              className="source-link"
              href={`https://openalex.org/${paper.id}`}
              target="_blank"
              rel="noreferrer"
            >
              <ArrowUpRight size={17} />在 OpenAlex 查看
            </a>
          )}
        </>
      ) : topic ? (
        <>
          <h3 className="paper-title">{topic.label}</h3>
          <p className="paper-meta">
            {topic.kind === "main" ? "主簇" : "侧簇"} · {topic.paper_ids.length}{" "}
            篇论文
          </p>
          <p className="muted small">{topic.rationale}</p>
          <div className="paper-list">
            {topic.paper_ids.map((id) => (
              <button key={id} onClick={() => onSelect({ type: "paper", id })}>
                <BookOpen size={16} />
                <span>{papers.find((p) => p.id === id)?.title ?? id}</span>
                <ArrowRight size={15} />
              </button>
            ))}
          </div>
        </>
      ) : connection ? (
        <>
          <h3 className="finding-title">{connection.description}</h3>
          <p className="muted small">
            {connection.kind === "within_cluster" ? "簇内关联" : "跨簇关联"} ·{" "}
            {map.sample_mode
              ? "样例占位结果"
              : "通过独立原文核验，仍需人工审阅"}
          </p>
          <SourceQuotes evidence={connection.evidence} papers={papers} />
          <button className="secondary-button" onClick={onReview}>
            进入人工验收
          </button>
        </>
      ) : gap ? (
        <>
          <h3 className="finding-title">{gap.description}</h3>
          <p className="hypothesis-label">
            {map.sample_mode ? "样例占位" : "待验证研究假设"} ·{" "}
            {gap.scope === "main_side" ? "主簇与侧簇" : "簇内比较"}
          </p>
          <p>{gap.why_it_matters}</p>
          {gap.proposed_test && (
            <section className="inspector-section">
              <h3>建议实验</h3>
              <p>{gap.proposed_test}</p>
            </section>
          )}
          <SourceQuotes evidence={gap.evidence} papers={papers} />
          {gap.coverage && (
            <section className="inspector-section">
              <h3>已有研究覆盖检查</h3>
              <p>
                {(
                  {
                    unavailable: "检索不可用",
                    inconclusive: "覆盖情况尚不确定",
                    no_address_found_in_checked_sources:
                      "在所查来源中未找到明确答案",
                    addressed: "已有来源回答此问题",
                  } as Record<string, string>
                )[gap.coverage.status] ?? gap.coverage.status}
              </p>
              <p className="muted small">{gap.coverage.scope}</p>
              {gap.coverage.searches.map((s, i) => (
                <p className="muted small" key={i}>
                  {s.query} · {s.papers.length} 条结果
                </p>
              ))}
            </section>
          )}
          <button className="secondary-button" onClick={onReview}>
            进入人工验收
          </button>
        </>
      ) : (
        <div className="inspector-placeholder">
          <BookOpen size={28} />
          <p>点击论文、主题或关联，查看原文与核验结果。</p>
          <button className="text-button" onClick={onReview}>
            查看候选验收
          </button>
        </div>
      )}
      {error && (
        <p className="inline-error" role="alert">
          {error}
        </p>
      )}
    </aside>
  );
}
