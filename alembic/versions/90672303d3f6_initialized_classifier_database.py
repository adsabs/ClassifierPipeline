"""initialized classifier database

Revision ID: 90672303d3f6
Revises: None
Create Date: 2024-11-21 09:26:21.363885

"""
from alembic import op
from sqlalchemy import Column, Integer, ARRAY, String, Text, Boolean, Numeric, BigInteger

from typing import Sequence, Union

from adsputils import get_date, UTCDateTime


# revision identifiers, used by Alembic.
revision: str = '90672303d3f6'
down_revision = None
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:

    # Overrides table
    op.create_table('overrides',
                    Column('id',Integer, primary_key=True),
                    Column('bibcode',String(19)),
                    Column('scix_id',String(19)),
                    Column('override', ARRAY(String)),
                    Column('created', UTCDateTime, default=get_date()),
                    )

    # Models table
    op.create_table('models',
                    Column('id', Integer, primary_key=True),
                    Column('model', Text),
                    # Column('revision', Text),
                    # Column('tokenizer', Text),
                    Column('postprocessing', Text),
                    # Column('labels', Text),
                    Column('created', UTCDateTime, default=get_date()),
                    )

    # Run table
    op.create_table('run',
                    Column('id', Integer, primary_key=True),
                    Column('run',String(20)),
                    Column('model_id', Integer, nullable=True),
                    Column('created', UTCDateTime, default=get_date()),
                    )
    op.create_foreign_key('fk_model_id_run',
                          'run',
                          'models',
                          ['model_id'],
                          ['id'])

    #Scores table
    op.create_table('scores',
                    Column('id', Integer, primary_key=True),
                    Column('bibcode',String(19)),
                    Column('scix_id',String(19)),
                    Column('scores', Text),
                    Column('run_id', Integer),
                    Column('overrides_id', BigInteger, nullable=True),
                    Column('models_id', Integer, nullable=True),
                    Column('created', UTCDateTime, default=get_date()),
                    )
    op.create_foreign_key('fk_overrides_id_scores',
                          'scores',
                          'overrides',
                          ['overrides_id'],
                          ['id'])
    op.create_foreign_key('fk_run_id_scores',
                          'scores',
                          'run',
                          ['run_id'],
                          ['id'])



    # Final Collection table
    op.create_table('final_collection',
                    Column('id', Integer, primary_key=True),
                    Column('bibcode',String(19)),
                    Column('scix_id',String(19)),
                    Column('score_id', Integer),
                    Column('collection', ARRAY(String)),
                    Column('validated', Boolean, default=False),
                    Column('created', UTCDateTime, default=get_date()),
                    )
    op.create_foreign_key('fk_score_id_final_collection',
                          'final_collection',
                          'scores',
                          ['score_id'],
                          ['id'])


def downgrade() -> None:
    op.drop_constraint('fk_overrides_id_scores', 'scores', type_='foreignkey')
    # op.drop_constraint('fk_models_id_scores', 'scores', type_='foreignkey')
    op.drop_constraint('fk_model_id_run', 'run', type_='foreignkey')
    op.drop_constraint('fk_score_id_final_collection', 'final_collection', type_='foreignkey')
    op.drop_table('scores')
    op.drop_table('overrides')
    op.drop_table('final_collection')
    op.drop_table('models')
    op.drop_table('run')

