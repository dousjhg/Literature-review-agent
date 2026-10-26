import { useRef, useState } from "react";
import { ChevronDown, LoaderCircle, Menu, Play, X } from "lucide-react";
import { request, messageOf, ApiError } from "./api";
import { DialogNotice } from "./components/Dialog";
import { useWorkspace } from "./useWorkspace";
import type { Filter, MapResult, Move, Paper, Plan, Selection } from "./types";
import { TopicSidebar } from "./components/TopicSidebar";
import { LiteratureGraph } from "./components/LiteratureGraph";
import { Inspector } from "./components/Inspector";
import { ReviewPanel } from "./components/ReviewPanel";
import {
  MovePreview,
  NewMapDialog,
  VersionDialog,
} from "./components/MapDialogs";
export default function App() {
  const ws = useWorkspace();
  const [tab, setTab] = useState<"map" | "review">("map");
  const [dialog, setDialog] = useState<"versions" | "new" | null>(null);
  const [busy, setBusy] = useState("");
  const busyRef = useRef(false);
  const [notice, setNotice] = useState("");
  const operate = async (label: string, action: () => Promise<void>) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(label);
    setNotice("");
    try {
      await action();
    } catch (e) {
      setNotice(messageOf(e));
      if (e instanceof ApiError && e.status === 409 && ws.data) {
        try {
          await ws.open(ws.data.map.run_id);
        } catch {
          /* Keep the original conflict visible. */
        }
      }
    } finally {
      busyRef.current = false;
      setBusy("");
    }
  };
  const data = ws.data;
  const editable =
    !!data &&
    data.map.run_id === data.versions.active_head.run_id &&
    data.map.sample_mode === ws.config.sample_mode;
  const create = (q: string, effort: string, papers: Paper[]) =>
    void operate("正在建立地图", async () => {
      let selected = papers;
      if (
        (effort === "high" || effort === "ultra") &&
        papers === data?.papers
      ) {
        selected = (
          await request<{ papers: Paper[] }>(
            `/cluster/${data.map.run_id}/papers?include_full_text=true`,
          )
        ).papers;
      }
      const map = await request<MapResult>("/cluster", {
        method: "POST",
        body: { input: { query: q, effort, papers: selected } },
        timeout: (ws.config.run_timeout_s + 120) * 1000,
      });
      setDialog(null);
      setTab("map");
      await ws.open(map.run_id);
      if (map.status === "partial")
        setNotice("地图已保存为待完成版本，可继续分析。");
    });
  const analyze = () =>
    data &&
    void operate("正在分析关联与研究假设", async () => {
      const map = await request<MapResult>(
        `/cluster/${data.map.run_id}/analyze`,
        {
          method: "POST",
          body: {
            expected_version_token: data.map.version_token,
            request_id: crypto.randomUUID().replaceAll("-", ""),
          },
          timeout: (ws.config.run_timeout_s + 120) * 1000,
        },
      );
      await ws.open(map.run_id);
      if (map.status === "partial")
        setNotice(
          "已保留完成结果。原版本保持为当前版本，可打开这个待完成版本继续分析。",
        );
    });
  return (
    <DialogNotice.Provider value={notice}>
      <div className="app-shell">
        <header className="app-header">
          <a className="wordmark" href="/">
            Literature Map
          </a>
          <nav className="top-nav" aria-label="主导航">
            <button
              className={tab === "map" ? "active" : ""}
              aria-pressed={tab === "map"}
              onClick={() => setTab("map")}
            >
              论文地图
            </button>
            <button
              className={tab === "review" ? "active" : ""}
              aria-pressed={tab === "review"}
              onClick={() => setTab("review")}
              disabled={!data}
            >
              人工验收
            </button>
          </nav>
          <div className="header-actions">
            <button
              className="text-button history-button"
              onClick={() => setDialog("versions")}
              disabled={!data || !!busy}
            >
              版本记录
            </button>
            <button
              className="primary-button analyze-button"
              disabled={
                !data ||
                !!busy ||
                data.map.sample_mode !== ws.config.sample_mode ||
                (!editable && data.map.status !== "partial")
              }
              onClick={analyze}
            >
              {busy ? <LoaderCircle size={16} className="spin" /> : null}
              {busy
                ? "处理中…"
                : data?.map.status === "partial"
                  ? "继续分析"
                  : "运行分析"}
            </button>
          </div>
        </header>
        {data && (
          <section className="question-bar">
            <h1 title={data.map.question}>{data.map.question}</h1>
            <span className="map-counts">
              {data.papers.length} 篇论文 ·{" "}
              {data.map.clusters.filter((c) => c.kind === "main").length} 个主簇
              · {data.map.clusters.filter((c) => c.kind !== "main").length}{" "}
              个侧簇
            </span>
          </section>
        )}
        {(notice ||
          ws.error ||
          data?.map.sample_mode ||
          (data && data.map.sample_mode !== ws.config.sample_mode) ||
          busy) && (
          <div
            className={`notice-bar ${notice || ws.error ? "error" : ""}`}
            role={notice || ws.error ? "alert" : "status"}
          >
            {notice ||
              ws.error ||
              busy ||
              (data?.map.sample_mode
                ? "样例模式：分组、相似度和评分为占位结果。"
                : "当前服务为样例模式；真实地图可阅读，切换真实模式后可执行分析或移动。")}
            {notice && (
              <button
                className="icon-button"
                aria-label="关闭提示"
                onClick={() => setNotice("")}
              >
                <X size={16} />
              </button>
            )}
          </div>
        )}
        {ws.loading ? (
          <main className="center-message">
            <LoaderCircle className="spin" size={24} />
            <p>正在读取地图与原文…</p>
          </main>
        ) : !data ? (
          <main className="empty-workspace">
            <h1>从论文出发，连接研究问题。</h1>
            <p>导入已有论文记录，建立可查看原文证据的研究地图。</p>
            <button className="primary-button" onClick={() => setDialog("new")}>
              新建地图
            </button>
            {ws.error && <p className="inline-error">{ws.error}</p>}
          </main>
        ) : (
          <WorkspaceView
            key={data.map.run_id}
            data={data}
            tab={tab}
            setTab={setTab}
            editable={editable}
            busy={!!busy}
            onNew={() => setDialog("new")}
            onOperation={operate}
            onOpen={ws.open}
            timeout={(ws.config.run_timeout_s + 120) * 1000}
          />
        )}
        {data && (
          <footer className="workspace-footer">
            <div className="version-selector">
              <strong>当前视图</strong>
              <label>
                <span className="sr-only">选择地图版本</span>
                <select
                  value={ws.runId}
                  disabled={!!busy}
                  onChange={(e) => ws.setRunId(e.target.value)}
                >
                  {ws.catalog.map((m) => (
                    <option value={m.run_id} key={m.run_id}>
                      {m.sample_mode ? "样例 · " : ""}v{m.revision}{" "}
                      {m.active ? "· 当前 " : ""}· {m.run_id}
                    </option>
                  ))}
                </select>
                <ChevronDown size={13} />
              </label>
            </div>
            {data.map.warnings.length > 0 && (
              <details className="quality-notes">
                <summary>质量提示 {data.map.warnings.length}</summary>
                <div>
                  {data.map.warnings.map((w, i) => (
                    <p key={i}>{w}</p>
                  ))}
                </div>
              </details>
            )}
          </footer>
        )}
        {dialog === "new" && (
          <NewMapDialog
            papers={data?.papers ?? []}
            question={data?.map.question ?? ""}
            sample={ws.config.sample_mode}
            busy={!!busy}
            onCreate={create}
            onClose={() => setDialog(null)}
          />
        )}
        {dialog === "versions" && data && (
          <VersionDialog
            map={data.map}
            versions={data.versions}
            catalog={ws.catalog}
            busy={!!busy}
            onClose={() => setDialog(null)}
            onRefresh={() =>
              void operate("正在刷新版本", async () => {
                await ws.open(data.map.run_id);
              })
            }
            onView={(id) => {
              setDialog(null);
              void ws.open(id);
            }}
            onActivate={(id) =>
              void operate("正在切换当前版本", async () => {
                await request(`/cluster/${data.map.root_run_id}/activate`, {
                  method: "POST",
                  body: {
                    target_run_id: id,
                    expected_head_run_id: data.versions.active_head.run_id,
                    expected_head_version_token:
                      data.versions.active_head.version_token,
                  },
                });
                setDialog(null);
                await ws.open(id);
              })
            }
          />
        )}
      </div>
    </DialogNotice.Provider>
  );
}
function WorkspaceView({
  data,
  tab,
  setTab,
  editable,
  busy,
  onNew,
  onOperation,
  onOpen,
  timeout,
}: {
  data: NonNullable<ReturnType<typeof useWorkspace>["data"]>;
  tab: "map" | "review";
  setTab: (v: "map" | "review") => void;
  editable: boolean;
  busy: boolean;
  onNew: () => void;
  onOperation: (label: string, action: () => Promise<void>) => Promise<void>;
  onOpen: (id: string) => Promise<void>;
  timeout: number;
}) {
  const { map, papers } = data;
  const [selection, setSelection] = useState<Selection | null>({
    type: "paper",
    id:
      map.clusters.find((c) => c.kind === "main")?.paper_ids[0] ??
      papers[0]?.id,
  });
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const [moves, setMoves] = useState<Move[]>([]);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [mobile, setMobile] = useState(false);
  const changeFilter = (f: Filter) => {
    setFilter(f);
    if (f === "connections")
      setSelection(
        map.connections[0]
          ? { type: "connection", id: map.connections[0].id }
          : null,
      );
    if (f === "gaps")
      setSelection(map.gaps[0] ? { type: "gap", id: map.gaps[0].id } : null);
    if (f === "main_side")
      setSelection({
        type: "paper",
        id:
          map.clusters.find((c) => c.kind === "main")?.paper_ids[0] ??
          papers[0].id,
      });
  };
  return (
    <>
      {tab === "review" ? (
        <ReviewPanel map={map} timeout={timeout} />
      ) : (
        <main className="workspace-grid">
          <button
            className="mobile-topic-toggle"
            onClick={() => setMobile((v) => !v)}
            aria-label="切换主题导航"
          >
            <Menu size={17} />
            研究主题
          </button>
          <TopicSidebar
            clusters={map.clusters}
            papers={papers}
            selection={selection}
            search={search}
            onSearch={setSearch}
            onSelect={(s) => {
              setSelection(s);
              setMobile(false);
            }}
            onNew={onNew}
            mobileOpen={mobile}
          />
          <LiteratureGraph
            map={map}
            papers={papers}
            selection={selection}
            onSelect={setSelection}
            filter={filter}
            onFilter={changeFilter}
            search={search}
            onReview={() => setTab("review")}
          />
          <Inspector
            map={map}
            papers={papers}
            selection={selection}
            onSelect={setSelection}
            moves={moves}
            canEdit={editable && !busy}
            onMove={(pid, target) => {
              setMoves((old) => [
                ...old.filter((m) => m.paper_id !== pid),
                ...(target
                  ? [{ paper_id: pid, target_cluster_id: target }]
                  : []),
              ]);
              setPlan(null);
            }}
            onReview={() => setTab("review")}
          />
        </main>
      )}
      {tab === "map" && (
        <div className="move-bar">
          {moves.length > 0 && (
            <>
              <span>{moves.length} 篇论文待移动</span>
              <button
                className="text-button small"
                disabled={busy}
                onClick={() => setMoves([])}
              >
                清空
              </button>
            </>
          )}
          <button
            className="secondary-button"
            disabled={!moves.length || busy || !editable}
            onClick={() =>
              void onOperation("正在检查移动约束", async () =>
                setPlan(
                  await request<Plan>(
                    `/cluster/${map.run_id}/regenerate/preview`,
                    { method: "POST", body: { moves } },
                  ),
                ),
              )
            }
          >
            <Play size={13} fill="currentColor" />
            预览移动
          </button>
        </div>
      )}
      {plan && (
        <MovePreview
          plan={plan}
          map={map}
          canApply={editable}
          busy={busy}
          onClose={() => setPlan(null)}
          onApply={() =>
            void onOperation("正在生成新版本", async () => {
              const child = await request<MapResult>(
                `/cluster/${map.run_id}/regenerate`,
                {
                  method: "POST",
                  body: { preview_id: plan.preview_id },
                  timeout,
                },
              );
              setPlan(null);
              setMoves([]);
              await onOpen(child.run_id);
            })
          }
        />
      )}
    </>
  );
}
