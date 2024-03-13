"""initialize database

Revision ID: 74a83030b18d
Revises: 
Create Date: 2024-02-23 11:24:57.637919

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import Column, Integer, ARRAY, String, Text
from adsputils import get_date, UTCDateTime

# revision identifiers, used by Alembic.
revision: str = '74a83030b18d'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:

    # Overrides table
    op.create_table('overrides',
                    Column('id',Integer, primary_key=True),
                    Column('override', ARRAY(String)),
                    Column('created', UTCDateTime, default=get_date()),
                    )

    # Models table
    op.create_table('models',
                    Column('id', Integer, primary_key=True),
                    Column('model', Text),
                    Column('revision', Text),
                    Column('tokenizer', Text),
                    Column('postprocessing', Text),
                    # Column('labels', Text),
                    Column('created', UTCDateTime, default=get_date()),
                    )

    #Scores table
    op.create_table('scores',
                    Column('id', Integer, primary_key=True),
                    Column('bibcode',String(19)),
                    Column('scores', Text),
                    Column('overrides_id', Integer, nullable=True),
                    Column('models_id', Integer, nullable=True),
                    Column('created', UTCDateTime, default=get_date()),
                    )
    op.create_foreign_key('fk_overrides_id_scores',
                          'scores',
                          'overrides',
                          ['overrides_id'],
                          ['id'])
    op.create_foreign_key('fk_models_id_scores',
                          'scores',
                          'models',
                          ['models_id'],
                          ['id'])


    # Final Collection table
    op.create_table('final_collection',
                    Column('id', Integer, primary_key=True),
                    Column('score_id', Integer),
                    Column('collection', ARRAY(String)),
                    Column('created', UTCDateTime, default=get_date()),
                    )
    op.create_foreign_key('fk_score_id_final_collection',
                          'final_collection',
                          'scores',
                          ['score_id'],
                          ['id'])


def downgrade() -> None:
    op.drop_constraint('fk_overrides_id_scores', 'scores', type_='foreignkey')
    op.drop_constraint('fk_models_id_scores', 'scores', type_='foreignkey')
    op.drop_constraint('fk_score_id_final_collection', 'final_collection', type_='foreignkey')
    op.drop_table('scores')
    op.drop_table('overrides')
    op.drop_table('final_collection')
    op.drop_table('models')

