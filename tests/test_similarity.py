"""Regressions for chaining, sparse fallback, cut selection and content-only scoring."""

import numpy as np
import pytest

from litagent.config import Settings
from litagent.schemas import Paper
from litagent.similarity import (complete_linkage_groups, cosine_matrix, paper_similarity,
                                 partition_metrics, representative_indices, select_partition)


def test_complete_linkage_does_not_merge_a_similarity_chain():
    similarity = np.array([[1, .8, .2], [.8, 1, .8], [.2, .8, 1]])
    groups = complete_linkage_groups(similarity, .7)
    assert sorted(map(len, groups)) == [1, 2]
    assert partition_metrics(similarity, groups)["min_within_similarity"] >= .7


@pytest.mark.parametrize("threshold", [0, .4, .7, 1])
def test_all_member_pairs_meet_the_threshold(threshold):
    vectors = np.random.default_rng(42).normal(size=(35, 16))
    similarity = cosine_matrix(vectors)
    groups = complete_linkage_groups(similarity, threshold)
    assert sorted(i for group in groups for i in group) == list(range(35))
    for group in groups:
        assert np.min(similarity[np.ix_(group, group)]) >= threshold


def test_equal_scores_at_the_boundary_are_included():
    similarity = np.array([[1, .7], [.7, 1]])
    assert complete_linkage_groups(similarity, .7) == [[0, 1]]
    assert complete_linkage_groups(np.ones((2, 2)), 1) == [[0, 1]]


def test_a_score_just_below_the_boundary_is_not_rounded_into_a_cluster():
    below = np.nextafter(.4, 0)
    assert complete_linkage_groups(np.array([[1, below], [below, 1]]), .4) == [[0], [1]]


def test_empty_and_single_document_batches():
    assert complete_linkage_groups(np.empty((0, 0)), .7) == []
    assert complete_linkage_groups(np.ones((1, 1)), .7) == [[0]]
    groups, threshold, _ = select_partition(np.ones((1, 1)), Settings())
    assert groups == [[0]]
    assert threshold == .7  # no fabricated silhouette for one paper


@pytest.mark.parametrize("bad", [np.array([[1, .2], [.8, 1]]), np.array([[1, np.nan], [np.nan, 1]]),
                                 np.array([[1, 1.1], [1.1, 1]]), np.ones((2, 3))])
def test_malformed_similarity_is_rejected(bad):
    with pytest.raises(ValueError, match="symmetric square"):
        complete_linkage_groups(bad, .7)


def test_lexical_terms_disambiguate_a_shared_generic_semantic_direction():
    ps = [Paper(id="G1", title="Graph neural networks", abstract="Molecular graphs message passing drug discovery", source="test"),
          Paper(id="G2", title="Graph neural network models", abstract="Graph message passing molecular drug discovery", source="test"),
          Paper(id="P", title="Protein structure transformers", abstract="Protein folding amino acid sequence geometry", source="test")]
    vectors = np.array([[1, 0], [.999, .01], [.999, -.01]])
    scores = paper_similarity(ps, vectors, Settings(cluster_dense_weight=.85))
    assert scores.lexical[0, 1] > scores.lexical[0, 2]
    assert scores.combined[0, 1] > scores.combined[0, 2]
    np.testing.assert_allclose(np.diag(scores.combined), 1)
    np.testing.assert_allclose(scores.combined, scores.combined.T)


def test_empty_vocabulary_falls_back_to_semantics_instead_of_penalizing_papers():
    ps = [Paper(id=str(i), title="the and of", abstract="and the", source="test") for i in range(2)]
    scores = paper_similarity(ps, np.array([[1, 0], [.8, .6]]), Settings(cluster_dense_weight=.85))
    assert scores.lexical_missing == 2
    np.testing.assert_allclose(scores.combined, scores.dense)


def test_citations_and_author_popularity_do_not_change_content_similarity():
    ps = [Paper(id="A", title="Graph molecule prediction", abstract="Molecular graph neural networks", source="test"),
          Paper(id="B", title="Graph based molecules", abstract="Message passing neural network prediction", source="test")]
    vectors = np.eye(2)
    before = paper_similarity(ps, vectors, Settings(cluster_dense_weight=.85)).combined
    ps[0].citation_count = 1_000_000
    ps[0].authors = ["Famous Author"]
    np.testing.assert_allclose(before, paper_similarity(ps, vectors, Settings(cluster_dense_weight=.85)).combined)


def test_adaptive_cut_avoids_trivial_all_in_one_and_all_singleton_objectives():
    similarity = np.array([[1, .85, .3, .3], [.85, 1, .3, .3], [.3, .3, 1, .85], [.3, .3, .85, 1]])
    settings = Settings(cluster_threshold_mode="adaptive", cluster_threshold_candidates=(.2, .7, .9))
    groups, threshold, rows = select_partition(similarity, settings)
    assert groups == [[0, 1], [2, 3]]
    assert threshold == .7
    assert next(row for row in rows if row["threshold"] == .2)["objective"] is None
    assert next(row for row in rows if row["threshold"] == .9)["objective"] is None


def test_retry_cannot_relax_below_the_candidate_floor():
    similarity = np.array([[1, .3, .1], [.3, 1, .2], [.1, .2, 1]])
    _, threshold, _ = select_partition(similarity, Settings(cluster_threshold_mode="adaptive"), "merge singletons")
    assert threshold >= min(Settings().cluster_threshold_candidates)


def test_fixed_mode_does_not_silently_calibrate():
    similarity = np.array([[1, .6], [.6, 1]])
    groups, threshold, rows = select_partition(similarity, Settings(cluster_threshold_mode="fixed"))
    assert groups == [[0], [1]] and threshold == .7 and len(rows) == 1


def test_mmr_selects_diverse_examples_without_reassigning_group_members():
    similarity = np.array([[1, .99, .5], [.99, 1, .5], [.5, .5, 1]])
    group = [0, 1, 2]
    chosen = representative_indices(similarity, group, limit=2, relevance_weight=.3)
    assert chosen == [0, 2]
    assert group == [0, 1, 2]


def test_invalid_similarity_weights_and_non_partitions_are_rejected():
    with pytest.raises(ValueError, match="weights"):
        paper_similarity([], np.empty((0, 2)), Settings(cluster_dense_weight=2))
    with pytest.raises(ValueError, match="partition"):
        partition_metrics(np.eye(3), [[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="fixed or adaptive"):
        select_partition(np.eye(3), Settings(cluster_threshold_mode="guess"))
