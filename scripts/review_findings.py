"""Freeze a source-bound human finding review pack; never manufacture judgments."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litagent.config import Settings  # noqa: E402
from litagent.finding_review import prepare_review  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output exists; choose a new filename to preserve previous reviews.')
    pack = prepare_review(args.run_id, Settings(sample_mode=False))
    pack['format'] = 'finding-human-review-v1'
    pack['review_note'] = 'Pending items have annotation=null. Use the UI to submit a reviewer, supported/unsupported/uncertain verdict, rationale and exact original quotes from both papers. No model judgments count as human labels. Semantic accuracy is not established by this diagnostic set.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pack, ensure_ascii=False, indent=2)+'\n')
    print(f"Prepared {len(pack['items'])} items; {pack['summary']['pending']} pending; {args.output}")


if __name__ == '__main__':
    main()
