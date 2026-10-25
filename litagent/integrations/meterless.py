"""Meterless: World Model (shared findings graph) and Markovian engine (chunked reading).

ResearchWorld records the agents' findings in the Meterless World Model
(litagent/integrations/world_model/, our Python implementation of the Meterless spec):

    run, planning attempt  -> contexts (an attempt's parent is the run)
    one attempt's search   -> entity `search`
    paper                  -> entity `paper:<source>:<id>`;  paper -found-by-> search
    author                 -> name-keyed entity: the same normalized name is one author ("José García" =
                              "Jose Garcia"); near-matches only queue for review, never auto-merge,
                              because short names collide ("Liang Li" vs "Xiang Li" scored 0.875);
                              paper -authored-by-> author
    cluster                -> entity;  paper -member-of-> cluster
    connection             -> edge paper -connects-to-> paper, with description and evidence quotes
    gap                    -> entity;  gap -between-> paper,  gap -in-cluster-> cluster
    grader score           -> fact `grade` about the stage (most recent wins, history kept)

Every write carries provenance (agent, run, time) and every relationship sits in the attempt's
context, so a re-plan never mixes attempts. The event log is saved to logs/world-<run_id>.json:
/run returns `paper_map` (to_graph_json) and GET /world/{run_id} serves the saved world.
Later agents read from it (read_clusters, cluster_context) instead of passing data around.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from litagent.config import WORLD_MODEL_MAX_AUTHORS
from litagent.integrations.world_model import WorldModel, iso
from litagent.schemas import Cluster, Connection, Gap, Grade, Paper, QuestionCluster
from litagent.trace import current_trace

SYSTEM = "litagent"
MULTI_VALUED = {"authored-by", "connects-to", "between", "also-related-to"}  # membership remains single-valued
# Authors: never auto-merge near-matches (different people with short names score >= 0.85); queue them
RESOLVER_BANDS = {"author": (float("inf"), 0.82)}


class ResearchWorld:
    def __init__(self, run_id: str = "default", directory: str | None = None):
        self.run_id = run_id
        if directory:
            if any((Path(directory) / f"{prefix}-{run_id}.json").exists()
                   for prefix in ("cluster-map", "cluster-state")):
                raise ValueError("Cannot replace the World Model of an immutable cluster version; use a new run ID")
            # A run id names one run: running it again starts a fresh world instead of extending the old one
            (Path(directory) / f"world-{run_id}.json").unlink(missing_ok=True)
        self.world = WorldModel(namespace=run_id, storage="local" if directory else "memory", directory=directory,
                                multi_valued_types=MULTI_VALUED, resolver_bands=RESOLVER_BANDS)
        self.run_context = self.world.upsert_context(
            type="run", external_key={"system": SYSTEM, "id": run_id}, source=self._src("orchestrator"))["id"]
        self.attempt = 0
        self.context = self.run_context
        self._search: str | None = None
        self._papers: dict[str, str] = {}  # paper ref (e.g. W123) -> world id, for this attempt
        self._clusters: dict[str, str] = {}

    # ----- writes (one per agent) -----

    def start_attempt(self, attempt: int, question: str = "", queries: list[str] | None = None) -> None:
        """Each planning attempt gets its own context, so a re-plan starts a clean map."""
        self.attempt = attempt
        key = {"system": SYSTEM, "id": f"{self.run_id}:{attempt}"}
        src = self._src("planner")
        self.context = self.world.upsert_context(type="attempt", external_key=key, attrs={"attempt": attempt},
                                                 parent=self.run_context, source=src)["id"]
        self._search = self.world.upsert_entity(type="search", external_key=key, source=src,
                                                attrs={"question": question, "queries": queries or []})["id"]
        self._papers, self._clusters = {}, {}

    def add_papers(self, papers: list[Paper]) -> None:
        if self._search is None:
            self.start_attempt(1)
        src = self._src("retriever")
        merges_before = sum(1 for a in self.world.audit() if a["kind"] == "auto-merge")
        queued_before = len(self.world.merge_queue())
        authors: set[str] = set()
        for p in papers:
            pid = self.world.upsert_entity(
                type="paper", external_key={"system": p.source, "id": p.id}, source=src,
                attrs={"ref": p.id, "title": p.title, "year": p.year, "doi": p.doi,
                       "citation_count": p.citation_count, "is_review": p.is_review},
            )["id"]
            self._papers[p.id] = pid
            self.world.relate(pid, "found-by", self._search, source=src, context=self.context)
            for name in p.authors[:WORLD_MODEL_MAX_AUTHORS]:
                author = self.world.upsert_entity(type="author", name=name, source=src)
                authors.add(author["id"])
                self.world.relate(pid, "authored-by", author["id"], source=src, context=self.context)
        self.world.flush()
        merged = sum(1 for a in self.world.audit() if a["kind"] == "auto-merge") - merges_before
        queued = len(self.world.merge_queue()) - queued_before
        _log("write papers", f"{len(papers)} papers, {len(authors)} authors",
             f"similar author names: {merged} merged, {queued} queued for operator review")

    def write_clusters(self, clusters: list[Cluster]) -> None:
        if any(isinstance(c, QuestionCluster) for c in clusters):
            members = [pid for c in clusters for pid in c.paper_ids]
            if (len(members) != len(set(members)) or not set(members) <= self._papers.keys()
                    or len({c.id for c in clusters}) != len(clusters)
                    or sum(isinstance(c, QuestionCluster) and c.kind == "main" for c in clusters) > 1):
                raise ValueError("Question map requires unique, known members and cluster IDs, with at most one main")
        src = self._src("clusterer")
        for c in clusters:
            attrs = {"ref": c.id, "label": c.label, "rationale": c.rationale}
            if isinstance(c, QuestionCluster):
                attrs.update(kind=c.kind, question=c.question)
            cid = self.world.upsert_entity(
                type="cluster", external_key={"system": SYSTEM, "id": f"{self.run_id}:{self.attempt}:{c.id}"},
                source=src, attrs=attrs,
            )["id"]
            self._clusters[c.id] = cid
            for ref in c.paper_ids:
                if ref in self._papers:
                    self.world.relate(self._papers[ref], "member-of", cid, source=src, context=self.context)
        # All target cluster nodes must exist before alternative edges are written.
        for c in clusters:
            if not isinstance(c, QuestionCluster):
                continue
            for detail in c.paper_details:
                if detail.paper_id not in self._papers:
                    raise ValueError("Question-map details reference an unknown paper")
                self.world.assert_fact(self._papers[detail.paper_id], "cluster_detail", detail.model_dump(mode="json"),
                                       source=src, context=self.context)
                for alternative in detail.also_related:
                    if alternative.cluster_id not in self._clusters:
                        raise ValueError("Alternative references an unknown cluster")
                    attrs = alternative.model_dump(exclude={"cluster_id"})
                    self.world.relate(self._papers[detail.paper_id], "also-related-to", self._clusters[alternative.cluster_id],
                                      attrs=attrs, source=src, context=self.context)
        self.world.flush()
        _log("write clusters", f"{len(clusters)} clusters", "paper -member-of-> cluster edges")

    def write_connections(self, connections: list[Connection]) -> None:
        src = self._src("connection_finder")
        written = 0
        for conn in connections:
            refs = [r for r in conn.paper_ids if r in self._papers]
            if len(refs) < 2:
                continue
            self.world.relate(
                self._papers[refs[0]], "connects-to", self._papers[refs[1]], source=src, context=self.context,
                attrs={"ref": conn.id, "kind": conn.kind, "description": conn.description,
                       "cluster_refs": conn.cluster_ids, "evidence": [e.model_dump() for e in conn.evidence],
                       **conn.model_dump(mode="json", exclude={"id", "kind", "description", "cluster_ids", "paper_ids", "evidence"})},
            )
            written += 1
        self.world.flush()
        _log("write connections", f"{written} connections", "paper -connects-to-> paper edges with evidence")

    def write_gaps(self, gaps: list[Gap]) -> None:
        src = self._src("gap_finder")
        for g in gaps:
            gid = self.world.upsert_entity(
                type="gap", external_key={"system": SYSTEM, "id": f"{self.run_id}:{self.attempt}:{g.id}"}, source=src,
                attrs={"ref": g.id, "description": g.description, "why_it_matters": g.why_it_matters,
                       "cluster_ref": g.cluster_id, "evidence": [e.model_dump() for e in g.evidence],
                       **g.model_dump(mode="json", exclude={"id", "description", "why_it_matters", "cluster_id", "between_paper_ids", "evidence"})},
            )["id"]
            for ref in g.between_paper_ids:
                if ref in self._papers:
                    self.world.relate(gid, "between", self._papers[ref], source=src, context=self.context)
            if g.cluster_id in self._clusters:
                self.world.relate(gid, "in-cluster", self._clusters[g.cluster_id], source=src, context=self.context)
        self.world.flush()
        _log("write gaps", f"{len(gaps)} gaps", "gap entities with -between-> paper edges")

    def record_grade(self, stage: str, grade: Grade) -> None:
        """Grades are facts about the stage: a redo's newer grade wins, older ones stay as history."""
        stage_id = self.world.upsert_entity(
            type="stage", external_key={"system": SYSTEM, "id": f"{self.run_id}:{self.attempt}:{stage}"},
            attrs={"stage": stage}, source=self._src("grader"))["id"]
        self.world.assert_fact(stage_id, "grade", grade.model_dump(mode="json"), source=self._src("grader"),
                               confidence=grade.total / 9, context=self.context)
        self.world.flush()

    # ----- reads -----

    def read_clusters(self) -> dict[str, list[str]]:
        """Cluster ref -> paper refs for the current attempt, as the next agent sees them."""
        out: dict[str, list[str]] = {ref: [] for ref in self._clusters}
        for rel in self._edges("member-of"):
            out.setdefault(self._ref(rel["to"]), []).append(self._ref(rel["from"]))
        return {cluster: sorted(papers) for cluster, papers in out.items()}

    def read_connections(self) -> list[dict[str, Any]]:
        return [self._edge_json(rel) for rel in self._edges("connects-to")]

    def shared_authors(self, ref_a: str, ref_b: str) -> list[str]:
        """Canonical names of authors two papers share (near-duplicate spellings already merged)."""
        def author_ids(ref: str) -> set[str]:
            return {r["to"] for r in self._edges("authored-by") if r["from"] == self._papers.get(ref)}

        graph = self.world.view("graph")
        return sorted(graph.entity(i)["name"] for i in author_ids(ref_a) & author_ids(ref_b))

    def cluster_context(self, cluster_ref: str) -> dict[str, Any]:
        """What the World Model knows about one cluster: its papers and the connections among them.
        Structured, not pre-rendered, so each agent formats it for its own prompt (spec §8.2)."""
        papers = set(self.read_clusters().get(cluster_ref, []))
        graph = self.world.view("graph")
        cluster = graph.entity(self._clusters[cluster_ref]) if cluster_ref in self._clusters else None
        return {
            "cluster": {k: v for k, v in (cluster or {}).get("attrs", {}).items() if k != "ref"},
            "papers": [self._node_json(graph.entity(self._papers[r])) for r in sorted(papers)],
            "connections": [c for c in self.read_connections() if c["source"] in papers and c["target"] in papers],
        }

    def to_graph_json(self) -> dict[str, Any]:
        """The paper map for the current attempt: papers, clusters and gaps as nodes; member-of,
        connects-to and between as edges. Node ids are our refs (W123, C1); world_id is the World Model id."""
        graph = self.world.view("graph")
        nodes = [self._node_json(graph.entity(wid)) for wid in self._papers.values()]
        nodes += [self._node_json(graph.entity(wid)) for wid in self._clusters.values()]
        gap_ids = {rel["from"] for rel in self._edges("between")} | {rel["from"] for rel in self._edges("in-cluster")}
        nodes += [self._node_json(graph.entity(wid)) for wid in sorted(gap_ids)]
        edges = [self._edge_json(rel) for t in ("member-of", "also-related-to", "connects-to", "between") for rel in self._edges(t)]
        return {
            "nodes": nodes,
            "edges": edges,
            "world": {"run_id": self.run_id, "attempt": self.attempt, "events": len(self.world.events()),
                      "pending_merges": sum(1 for p in self.world.merge_queue() if p["status"] == "pending")},
        }

    # ----- helpers -----

    def _src(self, agent: str) -> dict:
        return {"kind": "agent", "by": agent, "at": iso(), "runId": self.run_id}

    def _edges(self, type_: str) -> list[dict]:
        return self.world.view("graph").relationships(type=type_, in_context=self.context)

    def _ref(self, world_id: str) -> str:
        entity = self.world.view("graph").entity(world_id)
        return (entity or {}).get("attrs", {}).get("ref", world_id)

    def _node_json(self, entity: dict) -> dict[str, Any]:
        attrs = dict(entity["attrs"])
        attrs.pop("cluster_detail", None)  # Reattach only a fact from the current planning attempt.
        node = {"id": attrs.pop("ref", entity["id"]), "world_id": entity["id"], "type": entity["type"], **attrs}
        if entity["type"] == "paper":
            graph = self.world.view("graph")
            node["authors"] = [graph.entity(r["to"])["name"] for r in self._edges("authored-by") if r["from"] == entity["id"]]
            # Facts are context-scoped and latest-wins; the paper entity itself stays unique.
            facts = self.world.view("facts").about(entity["id"])
            detail = next((f for f in reversed(facts) if f["predicate"] == "cluster_detail"
                           and f.get("context") == self.context and not f.get("supersededBy")), None)
            if detail:
                node["cluster_detail"] = detail["value"]
        return node

    def _edge_json(self, rel: dict) -> dict[str, Any]:
        attrs = dict(rel.get("attrs") or {})
        attrs.pop("ref", None)
        return {"source": self._ref(rel["from"]), "target": self._ref(rel["to"]), "type": rel["type"],
                "by": rel["source"].get("by"), **attrs}


def load_world(run_id: str, directory: str) -> WorldModel | None:
    """A saved run's World Model (logs/world-<run_id>.json), or None if there is none."""
    if not (Path(directory) / f"world-{run_id}.json").exists():
        return None
    return WorldModel(namespace=run_id, storage="local", directory=directory,
                      multi_valued_types=MULTI_VALUED, resolver_bands=RESOLVER_BANDS)


def read_in_chunks(text: str, task: str) -> str:
    """Markovian engine: read a long full-text paper in chunks, carrying compressed state forward.

    Used at High / Ultra effort. TODO: replace with the Meterless Markovian engine.
    """
    raise NotImplementedError("TODO: connect the Meterless Markovian engine")


def _log(action: str, decision: str, why: str) -> None:
    if trace := current_trace():
        trace.log("meterless_world_model", action, decision, why=why)
