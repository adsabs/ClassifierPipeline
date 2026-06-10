from ClassifierPipeline import models


def _index_names(table):
    return {index.name for index in table.indexes}


def test_score_table_has_priority4_indexes():
    assert _index_names(models.ScoreTable.__table__) == {
        "ix_scores_bibcode",
        "ix_scores_scix_id",
        "ix_scores_run_id",
        "ix_scores_created",
        "ix_scores_run_scix_override_created",
        "ix_scores_run_bibcode_override_created",
    }


def test_override_table_has_priority4_indexes():
    assert _index_names(models.OverrideTable.__table__) == {
        "ix_overrides_bibcode",
        "ix_overrides_scix_id",
        "ix_overrides_bibcode_created",
        "ix_overrides_scix_id_created",
    }


def test_final_collection_table_has_priority4_indexes():
    assert _index_names(models.FinalCollectionTable.__table__) == {
        "ix_final_collection_bibcode",
        "ix_final_collection_scix_id",
        "ix_final_collection_validated_created",
        "ix_final_collection_bibcode_created",
        "ix_final_collection_scix_id_created",
    }
