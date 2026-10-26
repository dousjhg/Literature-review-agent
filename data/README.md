# Data

Hackathon rule: use open data you can use legally, keep samples small, add download scripts, and avoid sensitive or restricted data.

## Sources

| Source | What we use | Licence / terms |
| --- | --- | --- |
| OpenAlex | Paper metadata and abstracts | CC0 (check https://openalex.org) |
| Semantic Scholar | Metadata (only when `SEMANTIC_SCHOLAR_API_KEY` is set) | [Semantic Scholar API License Agreement](https://www.semanticscholar.org/product/api/license); used per run, not redistributed |
| arXiv | Preprint metadata and open-access PDFs | Metadata via the [arXiv API terms](https://info.arxiv.org/help/api/tou.html); each PDF keeps its author's licence. PDF text is extracted for analysis and kept only with the run's local records (paper store, `logs/`); PDF files are not stored, and nothing is redistributed in this repo |

## In this folder

- `sample/`: real OpenAlex papers (CC0), about 60 per topic, used by sample mode (`SAMPLE_MODE=true`) and the tests. The Retriever picks the file whose name shares the most words with the question. Add a topic with
  `python scripts/download_sample_data.py "your topic" --limit 60 --year-from 2021 --out data/sample --name your_topic_words`.
- `raw/`, `pdfs/`, `cache/`: downloaded data; git-ignored. Recreate with `python scripts/download_sample_data.py`.
