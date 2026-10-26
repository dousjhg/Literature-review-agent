import { useState } from "react";
import { Check, RefreshCw, Upload } from "lucide-react";
import { Dialog } from "./Dialog";
import { SearchReview, useSearchReview } from "./SearchReview";
import type { CatalogItem, MapResult, Paper, Plan, Versions } from "../types";
export function MovePreview({
  plan,
  map,
  busy,
  canApply,
  onApply,
  onClose,
}: {
  plan: Plan;
  map: MapResult;
  busy: boolean;
  canApply: boolean;
  onApply: () => void;
  onClose: () => void;
}) {
  return (
    <Dialog title="预览分组变更" onClose={onClose} busy={busy}>
      <div className="dialog-body">
        <p className="muted">
          检查整批移动后的论文归属。原版本将保留，可从版本记录中恢复。
        </p>
        {plan.changes.map((c) => (
          <div className="change-row" key={c.cluster_id}>
            <strong>
              {map.clusters.find((x) => x.id === c.cluster_id)?.label ??
                c.cluster_id}
            </strong>
            <span>
              {c.before.length} 篇 → {c.after.length} 篇
              {!c.after.length ? " · 将移除空簇" : ""}
            </span>
          </div>
        ))}
        <p className="muted small">
          将更新 {plan.relabel_cluster_ids.length} 个侧簇标签，复用{" "}
          {plan.reused_label_cluster_ids.length}{" "}
          个标签。已启用的关联与假设将按最终归属重新核验。
        </p>
        {!plan.applicable && (
          <div className="inline-error" role="alert">
            {plan.errors.map((e, i) => (
              <p key={i}>{e}</p>
            ))}
          </div>
        )}
      </div>
      <footer className="dialog-actions">
        <button className="secondary-button" disabled={busy} onClick={onClose}>
          取消
        </button>
        <button
          className="primary-button"
          disabled={!plan.applicable || busy || !canApply}
          onClick={onApply}
        >
          {busy ? "正在生成新版本…" : "应用并生成新版本"}
        </button>
      </footer>
    </Dialog>
  );
}
export function VersionDialog({
  map,
  versions,
  catalog,
  busy,
  onView,
  onActivate,
  onRefresh,
  onClose,
}: {
  map: MapResult;
  versions: Versions;
  catalog: CatalogItem[];
  busy: boolean;
  onView: (id: string) => void;
  onActivate: (id: string) => void;
  onRefresh: () => void;
  onClose: () => void;
}) {
  return (
    <Dialog title="版本记录" onClose={onClose} wide busy={busy}>
      <div className="dialog-body">
        <div className="version-intro">
          <p className="muted">
            查看历史不会改变当前版本。只有主动设为当前版本才会切换。
          </p>
          <button className="quiet-button" onClick={onRefresh} disabled={busy}>
            <RefreshCw size={15} />
            刷新
          </button>
        </div>
        {[...versions.versions].reverse().map((v) => (
          <div className="version-row" key={v.run_id}>
            <div>
              <strong>
                v{v.revision} ·{" "}
                {v.status === "partial"
                  ? "待完成"
                  : catalog.find((m) => m.run_id === v.run_id)?.stage ===
                      "findings"
                    ? "已分析"
                    : "分组地图"}
                {versions.active_head.run_id === v.run_id && (
                  <span className="active-version">
                    <Check size={13} />
                    当前
                  </span>
                )}
              </strong>
              <code>{v.run_id}</code>
            </div>
            <div className="version-buttons">
              <button
                className="quiet-button"
                disabled={busy || map.run_id === v.run_id}
                onClick={() => onView(v.run_id)}
              >
                查看
              </button>
              <button
                className="secondary-button"
                disabled={
                  busy ||
                  v.status !== "done" ||
                  v.run_id === versions.active_head.run_id
                }
                onClick={() => onActivate(v.run_id)}
              >
                设为当前版本
              </button>
            </div>
          </div>
        ))}
      </div>
    </Dialog>
  );
}
export function NewMapDialog({
  papers,
  question,
  busy,
  sample,
  timeout,
  onCreate,
  onClose,
}: {
  papers: Paper[];
  question: string;
  busy: boolean;
  sample: boolean;
  timeout: number;
  onCreate: (q: string, effort: string, p: Paper[]) => void;
  onClose: () => void;
}) {
  const [mode, setMode] = useState<"search" | "import">("search");
  const [query, setQuery] = useState(question);
  const [effort, setEffort] = useState("normal");
  const [source, setSource] = useState<Paper[]>(papers);
  const [error, setError] = useState("");
  const search = useSearchReview(query, effort, timeout);
  const locked = busy || search.loading;
  const [name, setName] = useState(
    papers.length ? `使用当前 ${papers.length} 篇论文` : "尚未选择论文记录",
  );
  const upload = async (file: File | undefined) => {
    if (!file) return;
    setError("");
    try {
      const raw = JSON.parse(await file.text());
      const ps = Array.isArray(raw)
        ? raw
        : (raw.input?.papers ?? raw.output?.papers ?? raw.papers);
      if (
        !Array.isArray(ps) ||
        !ps.length ||
        !ps.every(
          (p) =>
            typeof p.id === "string" &&
            typeof p.title === "string" &&
            typeof p.source === "string",
        )
      )
        throw Error("需要包含 id、title、source 的论文记录 JSON。");
      setSource(ps);
      setName(`${file.name} · ${ps.length} 篇`);
      const importedQuery =
        raw.input?.query ??
        raw.query ??
        raw.output?.plan?.question ??
        raw.output?.question;
      if (typeof importedQuery === "string" && importedQuery.trim())
        setQuery(importedQuery);
      const importedEffort = raw.input?.effort ?? raw.effort;
      if (["normal", "medium", "high", "ultra"].includes(importedEffort))
        setEffort(importedEffort);
    } catch (e) {
      setError(e instanceof Error ? e.message : "无法读取文件");
    }
  };
  const reviewed = mode === "search" && search.result?.status === "ok";
  const submit = () => {
    setError("");
    if (!query.trim()) {
      setError("请填写研究问题。");
      return;
    }
    if (mode === "search" && !reviewed) {
      void search.run();
      return;
    }
    const chosen = mode === "search" ? search.selected : source;
    const cap = (
      { normal: 30, medium: 60, high: 100, ultra: 150 } as Record<string, number>
    )[effort];
    if (!chosen.length) {
      setError(mode === "search" ? "请至少保留一篇论文。" : "请选择论文记录。");
      return;
    }
    if (chosen.length > cap) {
      setError(`当前级别最多支持 ${cap} 篇，请更换级别或论文记录。`);
      return;
    }
    onCreate(search.result?.plan?.question ?? query.trim(), effort, chosen);
  };
  return (
    <Dialog
      title="新建研究地图"
      onClose={onClose}
      busy={locked}
      wide={reviewed}
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <div className="dialog-body">
          <div className="segmented mode-switch" role="group" aria-label="论文来源">
            {(
              [
                ["search", "在线检索"],
                ["import", "导入记录"],
              ] as const
            ).map(([value, label]) => (
              <button
                key={value}
                type="button"
                className={mode === value ? "active" : ""}
                aria-pressed={mode === value}
                disabled={locked}
                onClick={() => {
                  setMode(value);
                  setError("");
                }}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="form-field">
            研究问题
            <textarea
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                search.reset();
              }}
              rows={3}
              placeholder="例如：LoRA 如何改善低资源语言机器翻译？可写明年份、引用数或论文类型。"
              required
              disabled={locked}
            />
          </label>
          <label className="form-field">
            研究投入
            <select
              value={effort}
              disabled={locked}
              onChange={(e) => {
                setEffort(e.target.value);
                search.reset();
              }}
            >
              <option value="normal">Normal · 最多 30 篇</option>
              <option value="medium">Medium · 最多 60 篇</option>
              <option value="high">High · 最多 100 篇</option>
              <option value="ultra">Ultra · 最多 150 篇</option>
            </select>
          </label>
          {mode === "search" ? (
            <SearchReview search={search} />
          ) : (
            <label className="upload-field">
              <Upload size={22} />
              <span>{name}</span>
              <small>选择已有检索结果或论文记录 JSON</small>
              <input
                type="file"
                accept=".json,application/json"
                onChange={(e) => void upload(e.target.files?.[0])}
              />
            </label>
          )}
          <p className="muted small">
            {sample
              ? "当前为样例模式：检索使用保存的样例论文，分组与评分为占位结果。"
              : mode === "search"
                ? "检索会调用 Planner 与 OpenAlex，论文经 Zetaris 保存；审核通过前不会建立地图。"
                : "新建地图会调用已配置的模型。High / Ultra 使用记录中已提供的全文，缺失时使用摘要。"}
          </p>
          {(error || search.error) && (
            <p className="inline-error" role="alert">
              {error || search.error}
            </p>
          )}
        </div>
        <footer className="dialog-actions">
          <button
            className="secondary-button"
            type="button"
            onClick={onClose}
            disabled={locked}
          >
            取消
          </button>
          <button className="primary-button" disabled={locked} type="submit">
            {search.loading
              ? "正在检索…"
              : busy
                ? "正在建立地图…"
                : mode === "search" && !reviewed
                  ? "检索论文"
                  : mode === "search"
                    ? `用保留的 ${search.selected.length} 篇建立地图`
                    : "建立地图"}
          </button>
        </footer>
      </form>
    </Dialog>
  );
}
