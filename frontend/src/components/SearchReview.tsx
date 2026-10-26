import { useState } from "react";
import { RefreshCw } from "lucide-react";
import { request, messageOf } from "../api";
import type { Paper, SearchResult } from "../types";
export function useSearchReview(query: string, effort: string, timeout: number) {
  const [result, setResult] = useState<SearchResult | null>(null);
  const [kept, setKept] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const run = async (feedback?: string) => {
    setLoading(true);
    setError("");
    try {
      const replan = feedback && result?.plan;
      const out = await request<SearchResult>("/search", {
        method: "POST",
        body: {
          input: {
            query: replan ? result.plan!.question : query.trim(),
            effort,
            ...(replan ? { feedback, previous_plan: result.plan } : {}),
          },
        },
        timeout,
      });
      setResult(out);
      setKept(new Set(out.papers.map((p) => p.id)));
      return true;
    } catch (e) {
      setError(messageOf(e));
      return false;
    } finally {
      setLoading(false);
    }
  };
  const toggle = (id: string) =>
    setKept((old) => {
      const next = new Set(old);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const selected: Paper[] = (result?.papers ?? []).filter((p) => kept.has(p.id));
  return { result, kept, selected, loading, error, run, toggle, reset: () => setResult(null) };
}
const describe = (plan: NonNullable<SearchResult["plan"]>) => {
  const c = plan.criteria;
  const defaulted = new Set(plan.used_defaults.map((d) => d.split("=")[0]));
  const mark = (name: string) => (defaulted.has(name) ? "（默认）" : "");
  const years = `${c.year_from ?? "…"}–${c.year_to ?? "至今"}${mark("year_from")}`;
  const types =
    c.paper_types.length === 1
      ? c.paper_types[0] === "review"
        ? "仅综述"
        : "仅原始研究"
      : "综述 + 原始研究";
  return [
    `主题：${c.topic}`,
    `年份：${years}`,
    `类型：${types}${mark("paper_types")}`,
    ...(c.min_citations ? [`引用 ≥ ${c.min_citations}`] : []),
  ];
};
export function SearchReview({
  search,
}: {
  search: ReturnType<typeof useSearchReview>;
}) {
  const [feedback, setFeedback] = useState("");
  const { result, kept, loading } = search;
  if (!result) return null;
  if (result.status !== "ok")
    return (
      <p className="search-message" role="status">
        {result.message_to_user}
      </p>
    );
  const all = kept.size === result.papers.length;
  return (
    <section className="search-review" aria-label="检索结果审核">
      <p className="search-summary">{result.summary}</p>
      {result.plan && (
        <div className="search-plan">
          <div className="plan-criteria">
            {describe(result.plan).map((text) => (
              <span key={text}>{text}</span>
            ))}
          </div>
          <p className="muted small">
            检索词：{result.plan.queries.join(" · ")}
          </p>
        </div>
      )}
      {result.warnings.map((w) => (
        <p className="inline-error small" key={w}>
          {w}
        </p>
      ))}
      <div className="search-list-heading">
        <strong>
          保留 {kept.size} / {result.papers.length} 篇
        </strong>
        <button
          type="button"
          className="text-button small"
          onClick={() =>
            result.papers.forEach(
              (p) => kept.has(p.id) === all && search.toggle(p.id),
            )
          }
        >
          {all ? "全部取消" : "全部保留"}
        </button>
      </div>
      <ul className="search-papers">
        {result.papers.map((p) => (
          <li key={p.id}>
            <label>
              <input
                type="checkbox"
                checked={kept.has(p.id)}
                onChange={() => search.toggle(p.id)}
              />
              <span>
                <strong>{p.title}</strong>
                <small>
                  {[
                    p.year,
                    p.is_review ? "综述" : null,
                    p.citation_count != null ? `${p.citation_count} 次引用` : null,
                    p.abstract ? null : "无摘要",
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </small>
              </span>
            </label>
          </li>
        ))}
      </ul>
      <div className="search-feedback">
        <label className="form-field">
          {result.follow_up_question}
          <textarea
            rows={2}
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="例如：只要 2023 年以后的原始研究"
          />
        </label>
        <button
          type="button"
          className="secondary-button"
          disabled={loading || !feedback.trim()}
          onClick={() =>
            void search.run(feedback.trim()).then((ok) => ok && setFeedback(""))
          }
        >
          <RefreshCw size={14} />
          按反馈重新检索
        </button>
      </div>
    </section>
  );
}
