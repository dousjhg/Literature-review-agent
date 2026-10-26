"""Cross-pool recall, correct graph scope, and genuine user-submitted review records."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from litagent import api
from litagent.agents.gap_finder import find_gaps
from litagent.cluster_regeneration import invalidated_findings
from litagent.cluster_state import load_version, path_for, save_version, set_head
from litagent.finding_engine import FindingContext, choose_pairs, sources
from litagent.finding_review import prepare_review
from litagent.integrations.meterless import ResearchWorld
from litagent.schemas import GroundedGap
from tests.test_findings import model  # shared offline model fixture
from tests.test_cluster_regeneration import snapshot


def cross_snapshot(tmp_path):
    vectors = [[1,0,0],[.7,.71414,0],[.7,.364,.6144],[.75,.6614,0],[0,0,1],[0,.1,.995]]
    return snapshot(tmp_path, vectors=vectors)


def test_singleton_main_has_cross_pool_gap_candidates_without_side_side_leakage(model, tmp_path):
    root,cfg,papers,arrays=cross_snapshot(tmp_path)
    ctx=FindingContext(question=root.question,similarity=arrays.similarity)
    pairs,_=choose_pairs('gaps',papers,root.clusters,cfg,ctx,set(sources(papers,cfg,ctx)))
    cross=[p for p in pairs if p['scope']=='main_side']
    assert cross and all('M' in p['paper_ids'] and len(p['cluster_ids'])==2 for p in cross)
    assert all(len(p['cluster_ids'])==1 or p['scope']=='main_side' for p in pairs)
    assert len(cross)<=cfg.gap_main_side_max_candidates
    legacy,_=choose_pairs('gaps',papers,root.clusters,replace(cfg,gap_main_side_max_candidates=0),ctx,set(sources(papers,cfg,ctx)))
    assert all(p['scope']=='within_cluster' for p in legacy)
    assert {p['pair_id'] for p in legacy}<={p['pair_id'] for p in pairs}


def test_main_side_gap_anchors_main_and_is_attached_to_both_clusters(model,tmp_path):
    root,cfg,papers,arrays=cross_snapshot(tmp_path)
    ctx=FindingContext(question=root.question,similarity=arrays.similarity)
    gaps=find_gaps(papers,root.clusters,[],cfg,context=ctx)
    gap=next(g for g in gaps if g.scope=='main_side')
    assert gap.cluster_id=='MAIN' and len(gap.cluster_ids)==2 and gap.hypothesis_status=='provisional'
    world=ResearchWorld('cross-gap-world',str(tmp_path));world.start_attempt(1,root.question);world.add_papers(papers);world.write_clusters(root.clusters);world.write_gaps(gaps)
    graph=world.to_graph_json()
    assert {e['target'] for e in graph['edges'] if e['source']==gap.id and e['type']=='in-cluster'}==set(gap.cluster_ids)
    side=next(cid for cid in gap.cluster_ids if cid!='MAIN')
    result=root.model_copy(deep=True,update={'gaps':gaps})
    _,invalid=invalidated_findings(result,SimpleNamespace(finding_dependencies=ctx.dependencies,papers=papers),{side},set())
    assert gap.id in invalid
    assert len([e for e in graph['edges'] if e['type']=='member-of'])==len(papers)


def test_gap_scope_validator_rejects_inconsistent_multi_cluster_metadata(model,tmp_path):
    root,cfg,papers,arrays=cross_snapshot(tmp_path)
    gaps=find_gaps(papers,root.clusters,[],cfg,context=FindingContext(similarity=arrays.similarity))
    gap=next(g for g in gaps if g.scope=='main_side')
    with pytest.raises(ValueError,match='Gap scope'):
        GroundedGap.model_validate({**gap.model_dump(),'cluster_ids':['MAIN']})


@pytest.fixture
def client(model,tmp_path):
    root,cfg,papers,arrays=snapshot(tmp_path)
    api.app.dependency_overrides[api.get_settings]=lambda:cfg
    try:
        with TestClient(api.app) as browser:
            yield browser,root,cfg,papers,arrays
    finally:
        api.app.dependency_overrides.clear()


def test_review_pack_starts_pending_and_does_not_run_models_or_create_labels(client,model):
    c,root,cfg,_,_=client
    out=c.get('/cluster/root/review').json()['output']
    assert out['summary']['total']>0 and out['summary']['reviewed']==0 and out['summary']['semantic_accuracy'] is None
    assert out['summary']['support_rate'] is None and all(i['annotation'] is None and 'model_assessment' not in i for i in out['items'])
    assert not path_for(cfg,'finding-review','root').exists() and not model['calls']
    revealed=c.get('/cluster/root/review?reveal=true').json()['output']
    assert all('model_assessment' in i for i in revealed['items'])


def test_review_requires_own_quotes_reviewer_reason_and_matching_version(client):
    c,root,cfg,_,_=client
    item=c.get('/cluster/root/review').json()['output']['items'][0]
    path=path_for(cfg,'cluster-map','root');before=path.read_bytes()
    body={'expected_version_token':root.version_token,'verdict':'supported','reviewer':'X','rationale':'Both original sources explicitly support the stated scientific comparison.',
          'evidence':[{'paper_id':p['id'],'quote':p['source_text'][:100]} for p in item['papers']]}
    url=f"/cluster/root/review/{item['id']}"
    assert c.post(url,json={**body,'evidence':body['evidence'][:1]}).status_code==422
    assert c.post(url,json={**body,'reviewer':''}).status_code==422
    assert c.post(url,json={**body,'reviewer':'   '}).status_code==422
    assert c.post(url,json={**body,'expected_version_token':'0'*32}).status_code==409
    assert c.post(url,json={**body,'evidence':[{'paper_id':'M','quote':'invented quote outside supplied source'}]}).status_code==422
    response=c.post(url,json=body)
    assert response.status_code==200,response.json()
    result=response.json()['output']
    assert result['annotation']['provenance']=='human' and result['annotation']['source']=='user_submission'
    assert result['summary']['reviewed']==1 and result['summary']['semantic_accuracy'] is None
    assert path.read_bytes()==before
    assert c.get('/cluster/root/review').json()['output']['items'][0]['annotation']['reviewer']=='X'
    assert c.post(url,json={**body,'verdict':'uncertain','evidence':[]}).status_code==200
    stored=json.loads(path_for(cfg,'finding-review','root').read_text())
    assert len(stored['history'])==1


def test_catalog_sources_and_targeted_alternatives_are_read_only(client):
    c,root,cfg,papers,arrays=client
    data=c.get('/maps').json()['output']['maps']
    assert data[0]['run_id']=='root' and data[0]['active']
    assert c.get('/ui/config').json()['output']['sample_mode'] is False
    metadata=c.get('/cluster/root/papers').json()['output']['papers']
    assert len(metadata)==len(papers) and all('full_text' not in p for p in metadata)
    assert 'full_text' in c.get('/cluster/root/papers?include_full_text=true').json()['output']['papers'][0]
    assert c.get('/cluster/root/papers/A').json()['output']['id']=='A'
    assert c.get('/cluster/root/papers/unknown').status_code==404
    targeted=c.get('/cluster/root/regenerate/candidates?paper_id=A').json()['output']
    assert targeted['candidates'] and all(x['paper_id']=='A' for x in targeted['candidates'])
    assert c.get('/cluster/root/regenerate/candidates?paper_id=unknown').status_code==404
    assert not path_for(cfg,'finding-review','root').exists()


def test_corrupt_manifest_is_not_presented_as_a_working_map(client):
    c,_,cfg,_,_=client
    p=path_for(cfg,'cluster-state','root');data=json.loads(p.read_text());data['checksum']='corrupt';p.write_text(json.dumps(data))
    out=c.get('/maps').json()['output']
    assert out['maps']==[] and out['warnings']
    assert c.get('/cluster/root/review').status_code==409
