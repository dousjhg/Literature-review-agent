import { Search, Info, Plus, MoreHorizontal } from "lucide-react";
import { useState } from "react";
import type { Cluster, Paper, Selection } from "../types";
export const colors = [
  "#86bda9",
  "#92bce4",
  "#b5a5df",
  "#dfad8d",
  "#e1c16b",
  "#cb9aa8",
];
export const topicColor = (c: Cluster, i: number) =>
  c.kind === "main" ? "#28584b" : colors[i % colors.length];
export function TopicSidebar({
  clusters,
  papers,
  selection,
  onSelect,
  search,
  onSearch,
  onNew,
  mobileOpen,
}: {
  clusters: Cluster[];
  papers: Paper[];
  selection: Selection | null;
  onSelect: (s: Selection) => void;
  search: string;
  onSearch: (s: string) => void;
  onNew: () => void;
  mobileOpen: boolean;
}) {
  const [all, setAll] = useState(false);
  const searchText = search.trim().toLocaleLowerCase();
  const matches = clusters.filter(
    (c) =>
      !searchText ||
      c.label.toLocaleLowerCase().includes(searchText) ||
      c.paper_ids.some((id) =>
        papers.some(
          (p) =>
            p.id === id && p.title.toLocaleLowerCase().includes(searchText),
        ),
      ),
  );
  const shown = all || searchText ? matches : matches.slice(0, 7);
  return (
    <aside
      className={`topic-sidebar ${mobileOpen ? "mobile-open" : ""}`}
      aria-label="研究主题导航"
    >
      <div className="sidebar-heading">
        <h2>研究主题</h2>
        <button
          className="icon-button"
          aria-label="新建地图"
          title="从论文记录新建地图"
          onClick={onNew}
        >
          <Plus size={18} />
        </button>
      </div>
      <label className="search-field">
        <Search size={17} />
        <input
          value={search}
          onChange={(e) => onSearch(e.target.value)}
          placeholder="搜索论文或主题"
          aria-label="搜索论文或主题"
        />
      </label>
      <nav className="topic-list">
        {shown.map((c) => {
          const index = clusters.indexOf(c);
          const selected =
            selection?.type === "cluster"
              ? selection.id === c.id
              : selection?.type === "paper" &&
                c.paper_ids.includes(selection.id);
          return (
            <button
              key={c.id}
              className={`topic-row ${selected ? "selected" : ""}`}
              onClick={() => onSelect({ type: "cluster", id: c.id })}
              title={c.label}
              aria-pressed={selected}
            >
              <span
                className="topic-dot"
                style={{ background: topicColor(c, index) }}
              />
              <span className="topic-name">
                {c.kind === "main" ? "主簇 · " : ""}
                {c.kind === "main"
                  ? "直接回答研究问题"
                  : c.label.replace(/^Unclustered:\s*/, "")}
              </span>
              <span className="topic-count">{c.paper_ids.length}</span>
            </button>
          );
        })}
        {!shown.length && <p className="muted small">没有匹配的主题。</p>}
      </nav>
      {matches.length > 7 && (
        <button className="show-topics" onClick={() => setAll((v) => !v)}>
          <MoreHorizontal size={20} />
          {all ? "收起主题" : `查看全部 ${matches.length} 个主题`}
        </button>
      )}
      <div className="sidebar-note">
        <Info size={16} />
        <span>研究假设仍需人工核验</span>
      </div>
    </aside>
  );
}
