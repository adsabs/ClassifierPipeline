"""add priority 4 indexes

Revision ID: 5c4fbe0e8f2a
Revises: 90672303d3f6
Create Date: 2026-03-17 17:30:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '5c4fbe0e8f2a'
down_revision: Union[str, None] = '90672303d3f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index('ix_scores_bibcode', 'scores', ['bibcode'], unique=False)
    op.create_index('ix_scores_scix_id', 'scores', ['scix_id'], unique=False)
    op.create_index('ix_scores_run_id', 'scores', ['run_id'], unique=False)
    op.create_index('ix_scores_created', 'scores', ['created'], unique=False)
    op.create_index(
        'ix_scores_run_scix_override_created',
        'scores',
        ['run_id', 'scix_id', 'overrides_id', 'created'],
        unique=False,
    )
    op.create_index(
        'ix_scores_run_bibcode_override_created',
        'scores',
        ['run_id', 'bibcode', 'overrides_id', 'created'],
        unique=False,
    )

    op.create_index('ix_final_collection_bibcode', 'final_collection', ['bibcode'], unique=False)
    op.create_index('ix_final_collection_scix_id', 'final_collection', ['scix_id'], unique=False)
    op.create_index(
        'ix_final_collection_validated_created',
        'final_collection',
        ['validated', 'created'],
        unique=False,
    )
    op.create_index(
        'ix_final_collection_bibcode_created',
        'final_collection',
        ['bibcode', 'created'],
        unique=False,
    )
    op.create_index(
        'ix_final_collection_scix_id_created',
        'final_collection',
        ['scix_id', 'created'],
        unique=False,
    )

    op.create_index('ix_overrides_bibcode', 'overrides', ['bibcode'], unique=False)
    op.create_index('ix_overrides_scix_id', 'overrides', ['scix_id'], unique=False)
    op.create_index('ix_overrides_bibcode_created', 'overrides', ['bibcode', 'created'], unique=False)
    op.create_index('ix_overrides_scix_id_created', 'overrides', ['scix_id', 'created'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_overrides_scix_id_created', table_name='overrides')
    op.drop_index('ix_overrides_bibcode_created', table_name='overrides')
    op.drop_index('ix_overrides_scix_id', table_name='overrides')
    op.drop_index('ix_overrides_bibcode', table_name='overrides')

    op.drop_index('ix_final_collection_scix_id_created', table_name='final_collection')
    op.drop_index('ix_final_collection_bibcode_created', table_name='final_collection')
    op.drop_index('ix_final_collection_validated_created', table_name='final_collection')
    op.drop_index('ix_final_collection_scix_id', table_name='final_collection')
    op.drop_index('ix_final_collection_bibcode', table_name='final_collection')

    op.drop_index('ix_scores_run_bibcode_override_created', table_name='scores')
    op.drop_index('ix_scores_run_scix_override_created', table_name='scores')
    op.drop_index('ix_scores_created', table_name='scores')
    op.drop_index('ix_scores_run_id', table_name='scores')
    op.drop_index('ix_scores_scix_id', table_name='scores')
    op.drop_index('ix_scores_bibcode', table_name='scores')
