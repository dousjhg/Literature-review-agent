"""Input/output contracts between agents.

Every agent returns one of these models instead of free text, so the next agent,
the grading agent and the front end can all rely on the same fields.
For Track 02, findings carry Evidence: every claim points back to a paper and a quote.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from litagent.config import Effort

# ---------- Planner ----------


class SearchCriteria(BaseModel):
    topic: str
    year_from: int | None = None
    year_to: int | None = None
    min_citations: int | None = None  # only when the user gives a number, e.g. "at least 50 citations"
    min_citations_per_year: float | None = None
    paper_types: list[Literal["primary", "review"]] = Field(
        default_factory=list, description="Empty = user didn't say; the Planner's defaults fill it"
    )


class SearchPlan(BaseModel):
    question: str
    criteria: SearchCriteria
    queries: list[str] = Field(description="Search strings to send to the scholarly APIs")
    effort: Effort
    max_papers: int
    token_budget: int
    used_defaults: list[str] = Field(default_factory=list, description="Which criteria fell back to defaults")
    rationale: str | None = None  # one sentence from the Planner: why these criteria and queries


class PlannerResult(BaseModel):
    status: Literal["ok", "need_topic"]
    plan: SearchPlan | None = None
    message_to_user: str | None = None  # friendly prompt when status == "need_topic"


# ---------- Retriever ----------


class Paper(BaseModel):
    id: str
    title: str
    abstract: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    citation_count: int | None = None
    is_review: bool | None = None
    is_open_access: bool = False
    pdf_url: str | None = None
    doi: str | None = None
    source: str  # e.g. "openalex", "semantic_scholar", "arxiv"
    full_text: str | None = None  # only filled at High / Ultra effort


class RetrievalSummary(BaseModel):
    """Shown to the user at Human review 1."""

    papers: list[Paper]
    summary: str
    follow_up_question: str = "Are these relevant? Any preferences on years, citations or paper type?"


# ---------- Clusterer / Connection finder / Gap finder ----------


class Evidence(BaseModel):
    paper_id: str
    quote: str  # verbatim text from the paper's abstract or full text


class ClusterLabel(BaseModel):
    """The labeller names a fixed group; it cannot change membership or invent paper ids."""

    model_config = ConfigDict(str_strip_whitespace=True)
    label: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class Cluster(BaseModel):
    id: str
    label: str
    paper_ids: list[str]
    rationale: str  # why these papers belong together


class ClusterAssessment(ClusterLabel):
    """Evidence-grounded naming of a fixed cluster, with an explicit abstention path."""

    cluster_id: str
    coherence: Literal["coherent", "mixed", "insufficient_evidence"]
    contrast: str = Field(min_length=1, description="Supported distinction from neighbors, or an explicit inability to distinguish")
    evidence: list[Evidence] = Field(default_factory=list)


class ClusterAssessmentBatch(BaseModel):
    assessments: list[ClusterAssessment]


class EvidenceReference(BaseModel):
    """An LLM selects a supplied source span; code copies the original bytes into Evidence."""

    paper_id: str
    span_id: str


class ClusterAssessmentProposal(ClusterLabel):
    cluster_id: str
    coherence: Literal["coherent", "mixed", "insufficient_evidence"]
    contrast: str = Field(min_length=1)
    evidence: list[EvidenceReference] = Field(default_factory=list)


class ClusterAssessmentProposals(BaseModel):
    assessments: list[ClusterAssessmentProposal]


class TopicProposal(ClusterLabel):
    paper_ids: list[str] = Field(min_length=1)
    evidence: list[EvidenceReference] = Field(default_factory=list)


class TopicProposals(BaseModel):
    topics: list[TopicProposal]


class Connection(BaseModel):
    id: str
    kind: Literal["within_cluster", "between_clusters"]
    paper_ids: list[str]
    cluster_ids: list[str]
    description: str
    evidence: list[Evidence]


class Gap(BaseModel):
    id: str
    cluster_id: str
    description: str
    between_paper_ids: list[str]
    why_it_matters: str
    evidence: list[Evidence]


# ---------- Grading agents ----------


class Grade(BaseModel):
    stage: Literal["clusters", "connections", "gaps"]
    accuracy: int = Field(ge=0, le=3)
    appropriateness: int = Field(ge=0, le=3)
    actionability: int = Field(ge=0, le=3)
    fix: str | None = None  # one concrete fix when the grade fails
    reasoning: str = ""

    @property
    def total(self) -> int:
        return self.accuracy + self.appropriateness + self.actionability


# ---------- Human review ----------


class HumanFeedback(BaseModel):
    approved: bool
    comment: str | None = None


# ---------- Final output ----------


class TokenUsage(BaseModel):
    """LLM tokens used by one run, checked against the effort level's token budget."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    estimated_calls: int = 0  # calls where the provider reported no usage, so tokens were estimated
    budget: int | None = None

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RunResult(BaseModel):
    run_id: str
    # partial = time limit or token budget hit; no_results = the searches ran but found no papers
    status: Literal["done", "need_topic", "no_results", "max_replans_reached", "partial"]
    message_to_user: str | None = None
    plan: SearchPlan | None = None
    papers: list[Paper] = Field(default_factory=list)
    clusters: list[Cluster] = Field(default_factory=list)
    connections: list[Connection] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)  # e.g. a stage that never passed grading
    paper_map: dict[str, Any] | None = None  # Meterless world model graph (nodes + edges) for the front end
    usage: TokenUsage | None = None
