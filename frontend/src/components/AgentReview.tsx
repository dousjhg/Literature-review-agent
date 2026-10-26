import { useEffect, useRef, useState } from "react";
import { Bot, LoaderCircle } from "lucide-react";
import { request, messageOf } from "../api";
import type { AgentReviewReport, MapResult, ReviewItem } from "../types";

export function AgentReview({
  map,
  token,
  item,
  report,
  onReport,
  onSeen,
  timeout,
}: {
  map: MapResult;
  token: string;
  item: ReviewItem;
  report: AgentReviewReport | null;
  onReport: (report: AgentReviewReport) => void;
  onSeen: () => void;
  timeout: number;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const running = useRef(false);
  const assessment = report?.items.find((i) => i.item_id === item.id);
  useEffect(() => {
    if (assessment) onSeen();
  }, [assessment, onSeen]);
  const load = async (run: boolean) => {
    if (running.current) return;
    running.current = true;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const out = await request<{
        report: AgentReviewReport | null;
        reused?: boolean;
      }>(`/cluster/${map.run_id}/review-agent`, {
        method: run ? "POST" : "GET",
        body: run ? { expected_version_token: token } : undefined,
        timeout,
      });
      if (out.report) {
        onReport(out.report);
        setMessage(
          out.reused ? "已读取本版本的已有评审，未重复调用模型。" : "",
        );
      } else setMessage("当前版本暂无 AI 评分，可点击「AI 自动评分」生成。");
    } catch (e) {
      setError(messageOf(e));
    } finally {
      running.current = false;
      setBusy(false);
    }
  };
  return (
    <section className="agent-review" aria-label="AI 评审建议">
      <div className="agent-review-header">
        <h3>
          <Bot size={18} />
          AI 评审建议
        </h3>
        <div className="agent-review-actions">
          <button
            className="quiet-button"
            disabled={busy}
            onClick={() => void load(false)}
          >
            查看已有评分
          </button>
          <button
            className="secondary-button"
            disabled={busy}
            onClick={() => void load(true)}
          >
            {busy ? (
              <LoaderCircle size={15} className="spin" />
            ) : (
              <Bot size={15} />
            )}
            {busy ? "正在评审…" : "AI 自动评分"}
          </button>
        </div>
      </div>
      <p className="muted small">
        对本版本的具体关联与研究假设逐项评分，保留独立原文引文。人工记录单独保存。
      </p>
      {error && (
        <p className="inline-error" role="alert">
          {error}
        </p>
      )}
      {message && (
        <p className="muted small" role="status">
          {message}
        </p>
      )}
      {assessment && (
        <div className="agent-review-result">
          <strong>
            {assessment.status === "sample"
              ? "样例流程 · 未评分"
              : assessment.status === "needs_hypothesis"
                ? "待明确具体假设 · 未评分"
                : assessment.status === "insufficient_sources"
                  ? "缺少原文 · 未评分"
                  : `AI 建议：${({ supported: "有依据", unsupported: "不支持", uncertain: "资料不足" } as const)[assessment.verdict!]} · ${assessment.total}/9`}
          </strong>
          {assessment.scores && (
            <dl className="agent-review-scores">
              <div>
                <dt>原文支持</dt>
                <dd>{assessment.scores.source_support}/3</dd>
              </div>
              <div>
                <dt>问题相关性</dt>
                <dd>{assessment.scores.relevance}/3</dd>
              </div>
              <div>
                <dt>可检验性</dt>
                <dd>{assessment.scores.testability}/3</dd>
              </div>
            </dl>
          )}
          <p>{assessment.reason}</p>
          {assessment.limitations.length > 0 && (
            <ul>
              {assessment.limitations.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          )}
          {assessment.evidence.length > 0 && (
            <details>
              <summary>
                查看 Agent 引用的原文（{assessment.evidence.length} 段）
              </summary>
              {assessment.evidence.map((e, i) => (
                <figure key={i}>
                  <figcaption>
                    {item.papers.find((p) => p.id === e.paper_id)?.title ??
                      e.paper_id}
                  </figcaption>
                  <blockquote>{e.quote}</blockquote>
                </figure>
              ))}
            </details>
          )}
          <p className="muted small">
            AI 建议仅用于辅助审核，评分不代表准确率，也不确认全球新颖性。
            {report?.model && `模型：${report.model}`}
          </p>
        </div>
      )}
    </section>
  );
}
