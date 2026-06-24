"""add final collection validated_at

Revision ID: 7a1c0d4e9b2f
Revises: 5c4fbe0e8f2a
Create Date: 2026-06-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import Column

from adsputils import UTCDateTime


# revision identifiers, used by Alembic.
revision: str = '7a1c0d4e9b2f'
down_revision: Union[str, None] = '5c4fbe0e8f2a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('final_collection', Column('validated_at', UTCDateTime))
    op.create_index(
        'ix_final_collection_validated_validated_at',
        'final_collection',
        ['validated', 'validated_at'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_final_collection_validated_validated_at', table_name='final_collection')
    op.drop_column('final_collection', 'validated_at')
