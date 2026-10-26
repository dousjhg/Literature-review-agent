# Grading agent

Code: `litagent/agents/grader.py` · Prompt: `prompts/grader.md`

1. **Goal:** Check the Clusterer, Connection finder and Gap finder before their output moves on.
2. **Plan:**
   1. Run cheap cluster-size checks; question maps exempt main and use side size as a diagnostic.
   2. Score the output against the source papers with the stage's rubric.
   3. Pass at 7/9; otherwise return exactly one fix.
3. **Tools:** LLM.
4. **Memory to keep:** grades are recorded as World Model facts and in the trace.
5. **Rubric (0–3 × 3):** accuracy · appropriateness · actionability, with stage-specific details in the prompt.
6. **Stop conditions:** each stage gets one redo (`max_redo_per_stage`), then continues with a warning. TODO: or ask the user?
7. **Run log:** score per criterion, pass/fail, fix.
8. **Guardrails:** sees the output and the evidence, not the producing agent's reasoning. TODO: try a different model from the producing agents.

Question maps include the research question, cluster roles and selected source evidence.
The producer's per-paper reasons and alternate scores are excluded from the grader input.
Main is judged by directness to the question, while sides are judged by topic coherence.
`POST /cluster` is a single-pass checkpoint and retains a failed grade with warnings;
full pipeline runs keep their bounded grading retry. Each model call uses an isolated
Nightshift H-MEM scope. See [main/side behavior](../main-side-clusters.md).

**Input:** stage, output, papers · **Output:** `Grade`
**Owner:** TODO · **Status:** Source-based grading, size diagnostics, sample mode and World Model recording implemented.

For independent per-claim scoring in the review UI, see the [Review Agent](review_agent.md).
Its model-only reports are separate from these stage grades and from human annotations.
