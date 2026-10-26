# NusaMT / LSFTL review explanation source audit

Checked 26 October 2026. Source map: `20261025-230430-c3fbbf`.
Affected item: `gap-f8faa00ccde14c57`. Historical AI report: `20261026-002900-f3baf5`.

## What was wrong

The old explanation mixed paper identities and metrics. The frozen NusaMT abstract
(`W4403364915`) explicitly names LLaMA2-7B, Balinese/Minangkabau, FLORES-200 and spBLEU.
The frozen LSFTL abstract (`W4408280927`) reports COMET improvements; it does not specify
those same base-model, language and benchmark details. Neither paper has full text in
this map. The proposed future test names both metrics, which is not evidence that either
published study reported both. The exact origin of the model's confusion is not observable.

The original quotes were valid, but valid quotes alone did not establish that the prose
attributed their contents correctly. The old report also spoke about omissions from a
paper when it had only received an abstract.

## External primary-source check

[NusaMT-7B, arXiv:2410.07830v1](https://arxiv.org/html/2410.07830v1):

- Section 4.3 and Table 2 define/use spBLEU; Section 4.4 and Table 3 report spBLEU benchmarks.
- Section 6 explicitly limits the findings to spBLEU. COMET does not appear in this HTML
  version. A shared future COMET evaluation may be proposed, but is not a NusaMT result.
- Section 4.2 says the SFT stage uses LoRA, rank 16, with approximately 0.1% trainable
  parameters. Thus the candidate's description of NusaMT as a full-parameter SFT baseline
  is also inaccurate for this version. SFT alone does not imply full-parameter tuning.

An appropriate revised research direction is to compare controlled adaptation pipelines,
with base model, data, compute, LoRA configuration and evaluation explicitly controlled.
Continued pre-training, cleaning and self-learning should be separate ablations if the
intent is to isolate the contribution of the adaptation method. This is a proposed design,
not a demonstrated performance ranking or a confirmed novel gap.

## Correction and provenance

The old AI cache, original hypothesis/map and all human annotations are preserved. Updated
prompt/code fingerprints cause a fresh AI report to be generated. The app reads the report
for the current implementation; it does not silently rewrite old reports. The new explanation
audit checks paper identity and distinguishes original observations from proposed tests.

Live development checks exposed additional model errors: treating SFT as full-parameter
tuning, contradicting a reason in its limitations, and claiming LSFTL's abstract names no
metrics despite COMET. A summary-only model audit missed errors. The final implementation
therefore requires per-sentence attribution/evidence coverage and adds a deterministic guard
for unqualified metric-absence assertions. Intermediate caches remain historical artifacts;
they are not evidence of successful correction and their fingerprints are superseded.

The refreshed AI run uses the **same frozen abstracts**. The external full-text facts above
are documented separately and are not silently inserted into that map or its AI inputs.
Its source-bound explanation can identify full-parameter tuning as unsubstantiated by the
abstract; the stronger LoRA correction comes from this external full-text check. Updating
the underlying candidate/source corpus requires a new version, not editing human judgments.

## Final live validation

Current report: `20261026-005323-26afbd`, fingerprint
`8fd6ee073a543c340d43d797b41789f6ee659c0be12b176aac12ddc6d56f78e9`.
The Gap reason now assigns spBLEU to NusaMT and COMET to LSFTL. Its verdict remains
`uncertain` (source support 1/3, relevance 3/3, testability 3/3). Scores remain model
judgments; this is not a performance ranking or confirmation of the original full-parameter
baseline premise. The external full-text correction above still applies to that premise.

The final run used two model calls / 10,114 tokens including the assertion audit. Cache
read/reuse was verified; checksums of the original map, human annotations and every earlier
AI report were unchanged. Backend suite: 315 passing tests. No frontend code changed.
