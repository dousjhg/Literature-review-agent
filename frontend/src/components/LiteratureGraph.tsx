import { useMemo, useRef, useState, type PointerEvent } from "react";
import { Minus, Plus, RotateCcw } from "lucide-react";
import type { Filter, MapResult, Paper, Selection } from "../types";
import { topicColor } from "./TopicSidebar";
function wrap(text: string) {
  const words = text.replace(/^Unclustered:\s*/, "").split(/\s+/);
  const lines = [""];
  for (const w of words) {
    if (lines.at(-1)!.length + w.length > 18 && lines.length < 2) lines.push(w);
    else lines[lines.length - 1] += (lines.at(-1) ? " " : "") + w;
  }
  return lines.map((line) =>
    line.length > 20 ? line.slice(0, 18) + "…" : line,
  );
}
export function LiteratureGraph({
  map,
  papers,
  selection,
  onSelect,
  filter,
  onFilter,
  search,
  onReview,
}: {
  map: MapResult;
  papers: Paper[];
  selection: Selection | null;
  onSelect: (s: Selection) => void;
  filter: Filter;
  onFilter: (f: Filter) => void;
  search: string;
  onReview: () => void;
}) {
  const [view, setView] = useState({ x: 0, y: 0, scale: 1 });
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(
    null,
  );
  const svg = useRef<SVGSVGElement>(null);
  const layout = useMemo(() => {
    const positions = new Map<
      string,
      { x: number; y: number; cluster: string }
    >();
    let side = 0;
    const sideCount = map.clusters.filter((c) => c.kind !== "main").length;
    const hasMain = map.clusters.some((c) => c.kind === "main");
    const dense = sideCount > 18;
    let cols = Math.ceil(Math.sqrt((sideCount + 1) * 1.3));
    if (cols % 2 === 0) cols++;
    let rows = Math.ceil((sideCount + 1) / cols);
    if (rows % 2 === 0) rows++;
    const groups = map.clusters.map((c, i) => {
      let x = 500,
        y = 380;
      const halo = dense
        ? Math.min(
            45,
            Math.min(840 / (cols - 1), 600 / Math.max(rows - 1, 1)) / 2 - 7,
          )
        : 66;
      if (c.kind !== "main") {
        const n = side++;
        if (dense) {
          const middle = Math.floor(rows / 2) * cols + Math.floor(cols / 2);
          const slot = hasMain && n >= middle ? n + 1 : n;
          x = 80 + ((slot % cols) * 840) / (cols - 1);
          y = 90 + (Math.floor(slot / cols) * 600) / Math.max(rows - 1, 1);
        } else if (!hasMain && sideCount <= 8) {
          const angle = -Math.PI / 2 + (n * 2 * Math.PI) / sideCount;
          x += sideCount > 1 ? Math.cos(angle) * 250 : 0;
          y += sideCount > 1 ? Math.sin(angle) * 190 : 0;
        } else {
          const ring = n < 12 ? 0 : 1;
          const angle =
            -Math.PI / 2 +
            (ring ? ((n - 12) * Math.PI) / 3 + Math.PI / 6 : (n * Math.PI) / 6);
          x += Math.cos(angle) * (ring ? 202 : 385);
          y += Math.sin(angle) * (ring ? 155 : 283);
        }
      }
      c.paper_ids.forEach((id, j) => {
        const angle =
          (j * Math.PI * 2) / Math.max(c.paper_ids.length, 1) - Math.PI / 2;
        const r = c.paper_ids.length === 1 ? 0 : Math.min(25, halo * 0.55);
        positions.set(id, {
          x: x + Math.cos(angle) * r,
          y: y + Math.sin(angle) * r * 0.7,
          cluster: c.id,
        });
      });
      return { c, x, y, halo, color: topicColor(c, i) };
    });
    return { groups, positions };
  }, [map.clusters]);
  const paperMap = useMemo(
    () => new Map(papers.map((p) => [p.id, p])),
    [papers],
  );
  const matches = (id: string) =>
    !search ||
    (paperMap.get(id)?.title.toLowerCase().includes(search.toLowerCase()) ??
      false) ||
    map.clusters
      .find((c) => c.paper_ids.includes(id))
      ?.label.toLowerCase()
      .includes(search.toLowerCase());
  const point = (id: string) => layout.positions.get(id);
  const selectedGroup = map.clusters.find((c) =>
    selection?.type === "cluster"
      ? selection.id === c.id
      : selection?.type === "paper" && c.paper_ids.includes(selection.id),
  );
  const hints =
    selection?.type === "paper"
      ? (map.clusters
          .flatMap((c) => c.paper_details ?? [])
          .find((d) => d.paper_id === selection.id)?.also_related ?? [])
      : [];
  const edgeEnter = (event: React.KeyboardEvent, s: Selection) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(s);
    }
  };
  const activeIds =
    filter === "connections"
      ? new Set(map.connections.flatMap((c) => c.paper_ids))
      : filter === "gaps"
        ? new Set(map.gaps.flatMap((g) => g.between_paper_ids))
        : null;
  const panStart = (e: PointerEvent<SVGSVGElement>) => {
    if ((e.target as Element).closest("[data-interactive]")) return;
    drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const panMove = (e: PointerEvent<SVGSVGElement>) => {
    if (drag.current && svg.current) {
      const ratio = 1000 / svg.current.getBoundingClientRect().width;
      setView((v) => ({
        ...v,
        x: drag.current!.vx + (e.clientX - drag.current!.x) * ratio,
        y: drag.current!.vy + (e.clientY - drag.current!.y) * ratio,
      }));
    }
  };
  const noFindings =
    (filter === "gaps" && !map.gaps.length) ||
    (filter === "connections" && !map.connections.length);
  return (
    <section className="graph-panel" aria-label="交互论文地图">
      <div className="graph-toolbar">
        <div className="segmented" aria-label="地图筛选">
          {(
            [
              ["all", "全部"],
              ["main_side", "主簇与侧簇"],
              ["connections", "关联"],
              ["gaps", "缺口"],
            ] as [Filter, string][]
          ).map(([key, label]) => (
            <button
              key={key}
              aria-pressed={filter === key}
              className={filter === key ? "active" : ""}
              onClick={() => onFilter(key)}
            >
              {label}
            </button>
          ))}
        </div>
        <button
          className="quiet-button reset"
          onClick={() => setView({ x: 0, y: 0, scale: 1 })}
        >
          <RotateCcw size={15} />
          重置视图
        </button>
      </div>
      {!noFindings && (
        <div className="graph-mobile-caption">
          <span>{selectedGroup?.label ?? "点选主题或论文查看原文"}</span>
          <small>可拖动地图 · 使用＋放大查看</small>
        </div>
      )}
      <div className="graph-canvas">
        {noFindings ? (
          <div className="graph-empty">
            <h2>
              {filter === "gaps"
                ? "暂无通过核验的研究假设"
                : "暂无通过核验的关联"}
            </h2>
            <p>当前版本保留了核验结果与弃权记录。</p>
            <button className="secondary-button" onClick={onReview}>
              查看候选验收
            </button>
          </div>
        ) : (
          <svg
            ref={svg}
            viewBox="0 0 1000 790"
            className="map-svg"
            aria-label={`论文地图，${papers.length} 篇论文`}
            onPointerDown={panStart}
            onPointerMove={panMove}
            onPointerUp={() => {
              drag.current = null;
            }}
            onPointerCancel={() => {
              drag.current = null;
            }}
          >
            <g
              transform={`translate(${500 + view.x},${380 + view.y}) scale(${view.scale}) translate(-500,-380)`}
            >
              {layout.groups.map(({ c, x, y, halo, color }) => {
                const main = c.kind === "main";
                const dim =
                  activeIds && !c.paper_ids.some((id) => activeIds.has(id));
                const selected =
                  selection?.type === "cluster" && selection.id === c.id;
                return (
                  <g key={c.id} opacity={dim ? 0.14 : 1}>
                    <ellipse
                      data-interactive="true"
                      role="button"
                      tabIndex={0}
                      aria-label={`主题：${c.label}`}
                      cx={x}
                      cy={y}
                      rx={main ? 43 : halo}
                      ry={main ? 38 : halo * 0.59}
                      fill={color}
                      fillOpacity={main ? 0.1 : 0.11}
                      stroke={selected ? color : "none"}
                      strokeWidth={1.5}
                      onClick={() => onSelect({ type: "cluster", id: c.id })}
                      onKeyDown={(e) =>
                        edgeEnter(e, { type: "cluster", id: c.id })
                      }
                    />
                    {c.paper_ids.length > 1 &&
                      c.paper_ids.map((id) => {
                        const p = point(id)!;
                        return (
                          <line
                            key={id}
                            x1={x}
                            y1={y}
                            x2={p.x}
                            y2={p.y}
                            stroke={color}
                            strokeOpacity={0.28}
                          />
                        );
                      })}
                    <text
                      x={x}
                      y={y + (main ? 55 : 57)}
                      textAnchor="middle"
                      className={
                        main ? "main-graph-label" : "topic-graph-label"
                      }
                    >
                      {(main
                        ? ["主簇 · 直接回答问题"]
                        : map.clusters.length > 19 &&
                            !selected &&
                            !(
                              selection?.type === "paper" &&
                              c.paper_ids.includes(selection.id)
                            )
                          ? [`${c.paper_ids.length} 篇`]
                          : wrap(c.label)
                      ).map((line, j) => (
                        <tspan key={j} x={x} dy={j ? 16 : 0}>
                          {line}
                        </tspan>
                      ))}
                    </text>
                  </g>
                );
              })}
              {(filter === "all" || filter === "main_side") &&
                selection?.type === "paper" &&
                hints.map((h) => {
                  const start = point(selection.id);
                  const end = layout.groups.find(
                    (g) => g.c.id === h.cluster_id,
                  );
                  return start && end ? (
                    <g key={h.cluster_id}>
                      <line
                        x1={start.x}
                        y1={start.y}
                        x2={end.x}
                        y2={end.y}
                        stroke="#a8b7b2"
                        strokeDasharray="5 6"
                        strokeWidth={1.2}
                      />
                      <text
                        x={(start.x + end.x) / 2}
                        y={(start.y + end.y) / 2 - 7}
                        className="hint-label"
                      >
                        {h.similarity.toFixed(2)}
                      </text>
                    </g>
                  ) : null;
                })}
              {filter !== "gaps" &&
                map.connections.map((c) => {
                  const a = point(c.paper_ids[0]),
                    b = point(c.paper_ids[1]);
                  if (
                    !a ||
                    !b ||
                    (filter === "main_side" && c.kind !== "between_clusters")
                  )
                    return null;
                  return (
                    <g key={c.id}>
                      <line
                        x1={a.x}
                        y1={a.y}
                        x2={b.x}
                        y2={b.y}
                        stroke="#28584b"
                        strokeWidth={2.5}
                      />
                      <line
                        data-interactive="true"
                        role="button"
                        tabIndex={0}
                        aria-label={`查看关联：${c.description}`}
                        x1={a.x}
                        y1={a.y}
                        x2={b.x}
                        y2={b.y}
                        stroke="transparent"
                        strokeWidth={14}
                        onClick={() =>
                          onSelect({ type: "connection", id: c.id })
                        }
                        onKeyDown={(e) =>
                          edgeEnter(e, { type: "connection", id: c.id })
                        }
                      />
                    </g>
                  );
                })}
              {(filter === "all" ||
                filter === "gaps" ||
                filter === "main_side") &&
                map.gaps.map((g) => {
                  const a = point(g.between_paper_ids[0]),
                    b = point(g.between_paper_ids[1]);
                  return a && b ? (
                    <g
                      key={g.id}
                      data-interactive="true"
                      role="button"
                      tabIndex={0}
                      aria-label={`查看研究假设：${g.description}`}
                      onClick={() => onSelect({ type: "gap", id: g.id })}
                      onKeyDown={(e) => edgeEnter(e, { type: "gap", id: g.id })}
                    >
                      <line
                        x1={a.x}
                        y1={a.y}
                        x2={b.x}
                        y2={b.y}
                        stroke="#a86e3c"
                        strokeDasharray="3 4"
                        strokeWidth={2}
                      />
                      <circle
                        cx={(a.x + b.x) / 2}
                        cy={(a.y + b.y) / 2}
                        r={11}
                        fill="#fff9ef"
                        stroke="#a86e3c"
                      />
                      <text
                        x={(a.x + b.x) / 2}
                        y={(a.y + b.y) / 2 + 4}
                        textAnchor="middle"
                        fontSize={13}
                        fill="#82562e"
                      >
                        ?
                      </text>
                    </g>
                  ) : null;
                })}
              {layout.groups.flatMap(({ c, color }) =>
                c.paper_ids.map((id) => {
                  const p = point(id)!;
                  const selected =
                    selection?.type === "paper" && selection.id === id;
                  const dim = !matches(id) || (activeIds && !activeIds.has(id));
                  return (
                    <g
                      key={id}
                      data-interactive="true"
                      role="button"
                      tabIndex={0}
                      className="paper-node"
                      aria-label={`论文：${paperMap.get(id)?.title ?? id}`}
                      opacity={dim ? 0.15 : 1}
                      onClick={() => onSelect({ type: "paper", id })}
                      onKeyDown={(e) => edgeEnter(e, { type: "paper", id })}
                    >
                      <title>{paperMap.get(id)?.title ?? id}</title>
                      {selected && (
                        <circle
                          cx={p.x}
                          cy={p.y}
                          r={c.kind === "main" ? 27 : 15}
                          fill="none"
                          stroke={color}
                          strokeWidth={1}
                          opacity={0.4}
                        />
                      )}
                      <circle
                        cx={p.x}
                        cy={p.y}
                        r={c.kind === "main" ? 20 : 8}
                        fill={color}
                        stroke="white"
                        strokeWidth={1.5}
                      />
                    </g>
                  );
                }),
              )}
            </g>
          </svg>
        )}
      </div>
      <div className="graph-bottom">
        <div
          className="graph-legend"
          title="节点位置用于分组浏览，不代表语义距离；数值提示才是相似度。簇内浅色细线表示归属。"
        >
          <span>
            <i style={{ background: "#28584b" }} />
            主簇
          </span>
          <span>
            <i style={{ background: "#d9e8e0" }} />
            侧簇
          </span>
          <span>
            <b className="legend-line" />
            关联
          </span>
          <span>
            <b className="legend-line dashed" />
            相似度提示
          </span>
        </div>
        <div className="zoom-controls">
          <button
            aria-label="缩小地图"
            onClick={() =>
              setView((v) => ({ ...v, scale: Math.max(0.55, v.scale - 0.15) }))
            }
          >
            <Minus size={18} />
          </button>
          <button
            aria-label="放大地图"
            onClick={() =>
              setView((v) => ({ ...v, scale: Math.min(2.5, v.scale + 0.15) }))
            }
          >
            <Plus size={18} />
          </button>
        </div>
      </div>
    </section>
  );
}
