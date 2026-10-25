"""Ephemeral sample API for browser regression tests; never writes real reviewer labels."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import uvicorn
from litagent import api
from litagent.cluster_preview import run_cluster_map
from litagent.config import Effort, Settings
from litagent.schemas import Paper

settings = Settings(sample_mode=True, logs_dir=os.environ['LITAGENT_QA_LOGS'], gap_similarity_max=1.0)
papers = [Paper.model_validate(p) for p in json.loads((ROOT/'data/sample/low_resource_languages_llm.json').read_text())[:6]]
run_cluster_map(papers, '样例测试：如何适配低资源语言？', Effort.NORMAL, settings, 'qa-root')
api.app.dependency_overrides[api.get_settings] = lambda: settings
uvicorn.run(api.app, host='127.0.0.1', port=int(os.environ['LITAGENT_QA_PORT']), log_level='warning')
