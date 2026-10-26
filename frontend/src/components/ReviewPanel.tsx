import { useCallback, useEffect, useState } from "react";
import { Check, Eye, Quote } from "lucide-react";
import { request, messageOf } from "../api";
import type {
  AgentReviewReport,
  Evidence,
  MapResult,
  ReviewItem,
  ReviewPack,
} from "../types";
import { AgentReview } from "./AgentReview";
export function ReviewPanel({
  map,
  timeout,
}: {
  map: MapResult;
  timeout: number;
}) {
  const [pack, setPack] = useState<ReviewPack | null>(null);
  const [selected, setSelected] = useState("");
  const [error, setError] = useState("");
  const [agentReport, setAgentReport] = useState<AgentReviewReport | null>(
    null,
  );
  const [seenItems, setSeenItems] = useState<Set<string>>(new Set());
  const recordSeen = useCallback((id: string) => {
    setSeenItems((old) => (old.has(id) ? old : new Set([...old, id])));
  }, []);
  useEffect(() => {
    const abort = new AbortController();
    request<ReviewPack>(`/cluster/${map.run_id}/review`, {
      signal: abort.signal,
    })
      .then((p) => {
        if (!abort.signal.aborted) {
          setPack(p);
          setSelected(p.items[0]?.id ?? "");
        }
      })
      .catch((e) => {
        if (!abort.signal.aborted) setError(messageOf(e));
      });
    return () => abort.abort();
  }, [map.run_id]);
  const item = pack?.items.find((i) => i.id === selected);
  return (
    <section className="review-workspace" aria-label="人工验收工作区">
      <aside className="review-queue">
        <h2>验收材料</h2>
        <p className="muted small">
          {pack
            ? `${pack.summary.reviewed} / ${pack.summary.total} 已完成`
            : "正在准备原文…"}
        </p>
        <p className="muted small">
          先查看原文，再填写判断。模型评分默认隐藏。
        </p>
        {pack?.items.map((i, n) => (
          <button
            className={`review-item ${selected === i.id ? "selected" : ""}`}
            onClick={() => {
              setSelected(i.id);
              setError("");
            }}
            key={i.id}
          >
            <span>
              {i.annotation ? (
                <Check size={15} />
              ) : (
                String(n + 1).padStart(2, "0")
              )}
            </span>
            <div>
              <strong>
                {i.kind === "gap"
                  ? i.scope === "main_side"
                    ? "主簇 ↔ 侧簇候选"
                    : "簇内缺口候选"
                  : "关联核验"}
              </strong>
              <small>{i.papers.map((p) => p.title).join(" / ")}</small>
            </div>
          </button>
        ))}
        {pack && !pack.items.length && (
          <p className="muted">当前版本没有可核验的论文对。</p>
        )}
      </aside>
      <div className="review-detail">
        {item && pack ? (
          <ReviewForm
            key={item.id}
            item={item}
            map={map}
            token={pack.version_token}
            agentReport={agentReport}
            onAgentReport={setAgentReport}
            wasSeen={seenItems.has(item.id)}
            recordSeen={recordSeen}
            timeout={timeout}
            onSaved={(annotation, summary) =>
              setPack((old) =>
                old
                  ? {
                      ...old,
                      summary,
                      items: old.items.map((i) =>
                        i.id === item.id ? { ...i, annotation } : i,
                      ),
                    }
                  : old,
              )
            }
          />
        ) : (
          <div className="center-message">
            {error || "请选择一项验收材料。"}
          </div>
        )}
      </div>
    </section>
  );
}
function ReviewForm({
  item,
  map,
  token,
  onSaved,
  agentReport,
  onAgentReport,
  wasSeen,
  recordSeen,
  timeout,
}: {
  item: ReviewItem;
  map: MapResult;
  token: string;
  agentReport: AgentReviewReport | null;
  onAgentReport: (report: AgentReviewReport) => void;
  wasSeen: boolean;
  recordSeen: (id: string) => void;
  timeout: number;
  onSaved: (
    annotation: NonNullable<ReviewItem["annotation"]>,
    summary: ReviewPack["summary"],
  ) => void;
}) {
  const [verdict, setVerdict] = useState(item.annotation?.verdict ?? "");
  const [reviewer, setReviewer] = useState(item.annotation?.reviewer ?? "");
  const [rationale, setRationale] = useState(item.annotation?.rationale ?? "");
  const [evidence, setEvidence] = useState<Evidence[]>(
    item.annotation?.evidence ?? [],
  );
  const [revealed, setRevealed] = useState(
    item.annotation?.saw_model_assessment ?? false,
  );
  const [assessment, setAssessment] =
    useState<ReviewItem["model_assessment"]>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  const markSeen = useCallback(
    () => recordSeen(item.id),
    [item.id, recordSeen],
  );
  const quote = (id: string, source: string) => {
    const selected = window.getSelection()?.toString();
    const text =
      selected && source.includes(selected) ? selected : source.slice(0, 400);
    setEvidence((old) => [
      ...old.filter((e) => e.paper_id !== id),
      { paper_id: id, quote: text },
    ]);
    setSaved(false);
  };
  const submit = async () => {
    setError("");
    setSaved(false);
    if (!verdict) {
      setError("请选择一项判断：有依据、不支持或资料不足。");
      return;
    }
    if (!reviewer.trim()) {
      setError("请填写审核者姓名或审核标识，可使用单字或缩写。");
      return;
    }
    if (rationale.trim().length < 10) {
      setError("判断理由至少需要 10 个字符，请补充具体依据。");
      return;
    }
    if (
      verdict !== "uncertain" &&
      item.paper_ids.some(
        (id) =>
          !evidence.some(
            (e) => e.paper_id === id && e.quote.trim().length >= 20,
          ),
      )
    ) {
      setError("明确判断需要分别引用两篇论文的原文，每段至少 20 个字符。");
      return;
    }
    setBusy(true);
    try {
      const out = await request<{
        annotation: NonNullable<ReviewItem["annotation"]>;
        summary: ReviewPack["summary"];
      }>(`/cluster/${map.run_id}/review/${encodeURIComponent(item.id)}`, {
        method: "POST",
        body: {
          expected_version_token: token,
          verdict,
          reviewer,
          rationale,
          evidence,
          saw_model_assessment: revealed || wasSeen,
        },
      });
      onSaved(out.annotation, out.summary);
      setSaved(true);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <div className="review-title">
        <h2>{item.kind === "gap" ? "核验可检验的研究假设" : "核验论文关联"}</h2>
        <button
          className="quiet-button"
          onClick={async () => {
            try {
              const all = await request<ReviewPack>(
                `/cluster/${map.run_id}/review?reveal=true`,
              );
              setAssessment(
                all.items.find((i) => i.id === item.id)?.model_assessment,
              );
              setRevealed(true);
              markSeen();
            } catch (e) {
              setError(messageOf(e));
            }
          }}
        >
          <Eye size={15} />
          查看模型判断
        </button>
      </div>
      <p className="review-question">
        {item.claim || `根据两篇原文，是否能提出贴合以下研究问题的可检验假设？`}
      </p>
      {!item.claim && <p className="muted">{map.question}</p>}
      {item.proposed_test && (
        <p className="muted">建议实验：{item.proposed_test}</p>
      )}
      <p className="muted small">
        请判断两篇论文之间的具体关联或研究假设，而不是分别评价论文质量。“有依据”不等同于确认全球研究空白。
      </p>
      {!item.claim && (
        <p className="notice">
          当前只有候选论文对，尚未提出具体假设。可以记录探索意见；确认科学结论前需要先明确可检验的问题。
        </p>
      )}
      <AgentReview
        map={map}
        token={token}
        item={item}
        report={agentReport}
        onReport={onAgentReport}
        onSeen={markSeen}
        timeout={timeout}
      />
      {assessment && (
        <div className="notice">
          <strong>模型判断：{assessment.decision}</strong>
          <p>
            {assessment.reason ||
              "该条为已保留关联，具体判断由原文与人工核验。"}
          </p>
        </div>
      )}
      <div className="review-sources">
        {item.papers.map((p) => (
          <article key={p.id}>
            <p className="source-caption">
              {p.year ?? ""} ·{" "}
              {p.source_field === "full_text" ? "已提供全文" : "原始摘要"}
            </p>
            <h3>{p.title}</h3>
            <div className="review-source-text">
              {p.source_text || "无可核验正文"}
            </div>
            <button
              className="text-button"
              onClick={() => quote(p.id, p.source_text)}
              disabled={!p.source_text}
            >
              <Quote size={14} />
              引用选中文字或首段原文
            </button>
            {evidence.find((e) => e.paper_id === p.id) && (
              <label className="form-field small">
                已选引文
                <textarea
                  aria-label={`引文 ${p.id}`}
                  rows={3}
                  value={evidence.find((e) => e.paper_id === p.id)?.quote}
                  onChange={(e) =>
                    setEvidence((old) =>
                      old.map((q) =>
                        q.paper_id === p.id
                          ? { ...q, quote: e.target.value }
                          : q,
                      ),
                    )
                  }
                />
              </label>
            )}
          </article>
        ))}
      </div>
      <form
        className="review-form"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <fieldset>
          <legend>你的判断</legend>
          <div className="judgment-options">
            {[
              ["supported", "有依据"],
              ["unsupported", "不支持"],
              ["uncertain", "资料不足"],
            ].map(([v, label]) => (
              <label key={v}>
                <input
                  type="radio"
                  name="verdict"
                  value={v}
                  checked={verdict === v}
                  onChange={() => {
                    setVerdict(v);
                    setSaved(false);
                  }}
                />
                {label}
              </label>
            ))}
          </div>
        </fieldset>
        <label className="form-field">
          审核者
          <input
            value={reviewer}
            onChange={(e) => {
              setReviewer(e.target.value);
              setSaved(false);
              setError("");
            }}
            placeholder="填写你的姓名或审核标识"
            maxLength={80}
          />
        </label>
        <label className="form-field">
          判断理由
          <textarea
            value={rationale}
            onChange={(e) => {
              setRationale(e.target.value);
              setSaved(false);
              setError("");
            }}
            rows={3}
            placeholder="说明具体任务、方法、数据或评测设置上的依据"
          />
        </label>
        {error && (
          <p className="inline-error" role="alert">
            {error}
          </p>
        )}
        <div className="review-submit">
          <span className="muted small">
            {saved ? "判断已保存至当前版本。" : "记录仅在主动提交后保存。"}
          </span>
          <button className="primary-button" disabled={busy} type="submit">
            {busy ? "正在保存…" : "保存人工判断"}
          </button>
        </div>
      </form>
    </>
  );
}
