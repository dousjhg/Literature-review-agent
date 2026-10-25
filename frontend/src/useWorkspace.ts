import { useCallback, useEffect, useState } from "react";
import { request, messageOf } from "./api";
import type { CatalogItem, MapResult, Paper, Versions } from "./types";
export function useWorkspace() {
  const [catalog, setCatalog] = useState<CatalogItem[]>([]);
  const [config, setConfig] = useState({
    sample_mode: false,
    run_timeout_s: 600,
  });
  const [runId, setRunId] = useState("");
  const [data, setData] = useState<{
    map: MapResult;
    papers: Paper[];
    versions: Versions;
  } | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const refreshCatalog = useCallback(async (signal?: AbortSignal) => {
    const c = await request<{ maps: CatalogItem[] }>("/maps", { signal });
    setCatalog(c.maps);
    return c.maps;
  }, []);
  useEffect(() => {
    const abort = new AbortController();
    Promise.all([
      refreshCatalog(abort.signal),
      request<typeof config>("/ui/config", { signal: abort.signal }),
    ])
      .then(([maps, cfg]) => {
        if (abort.signal.aborted) return;
        setConfig(cfg);
        const initial = new URLSearchParams(location.search).get("run");
        setRunId(
          maps.find((m) => m.run_id === initial)?.run_id ??
            maps[0]?.run_id ??
            "",
        );
        if (!maps.length) setLoading(false);
      })
      .catch((e) => {
        if (!abort.signal.aborted) {
          setError(messageOf(e));
          setLoading(false);
        }
      });
    return () => abort.abort();
  }, [refreshCatalog]);
  useEffect(() => {
    if (!runId) return;
    const abort = new AbortController();
    setLoading(true);
    setData(null);
    setError("");
    Promise.all([
      request<MapResult>(`/cluster/${runId}`, { signal: abort.signal }),
      request<{ papers: Paper[] }>(`/cluster/${runId}/papers`, {
        signal: abort.signal,
      }),
      request<Versions>(`/cluster/${runId}/versions`, { signal: abort.signal }),
    ])
      .then(([map, p, versions]) => {
        if (abort.signal.aborted) return;
        setData({ map, papers: p.papers, versions });
        setLoading(false);
        const url = new URL(location.href);
        url.searchParams.set("run", map.run_id);
        history.replaceState(null, "", url);
      })
      .catch((e) => {
        if (!abort.signal.aborted) {
          setError(messageOf(e));
          setLoading(false);
        }
      });
    return () => abort.abort();
  }, [runId]);
  const open = async (id: string) => {
    await refreshCatalog();
    if (id === runId) {
      const versions = await request<Versions>(`/cluster/${id}/versions`);
      setData((old) => (old ? { ...old, versions } : old));
    } else setRunId(id);
  };
  return {
    catalog,
    config,
    data,
    runId,
    error,
    loading,
    open,
    refreshCatalog,
    setRunId,
  };
}
