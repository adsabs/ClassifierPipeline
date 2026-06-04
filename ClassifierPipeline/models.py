# -*- coding: utf-8 -*-

from builtins import str
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import ARRAY, BigInteger, Boolean, Column, ForeignKey, Index, Integer, Numeric, String, Text, TIMESTAMP
from sqlalchemy.types import Enum
import json
import sys
from adsputils import get_date, UTCDateTime

Base = declarative_base()


class ScoreTable(Base):
    __tablename__ = 'scores'
    __table_args__ = (
        Index('ix_scores_bibcode', 'bibcode'),
        Index('ix_scores_scix_id', 'scix_id'),
        Index('ix_scores_run_id', 'run_id'),
        Index('ix_scores_created', 'created'),
        Index('ix_scores_run_scix_override_created', 'run_id', 'scix_id', 'overrides_id', 'created'),
        Index('ix_scores_run_bibcode_override_created', 'run_id', 'bibcode', 'overrides_id', 'created'),
    )
    id = Column(Integer, primary_key=True)
    bibcode = Column(String(19))
    scix_id = Column(String(19))
    scores = Column(Text)
    created = Column(UTCDateTime, default=get_date)
    overrides_id = Column(Integer, ForeignKey('overrides.id'))
    run_id = Column(BigInteger, ForeignKey('run.id'))

class ModelTable(Base):
    __tablename__ = 'models'
    id = Column(Integer, primary_key=True)
    model = Column(Text)
    postprocessing = Column(Text)
    created = Column(UTCDateTime, default=get_date)

class OverrideTable(Base):
    __tablename__ = 'overrides'
    __table_args__ = (
        Index('ix_overrides_bibcode', 'bibcode'),
        Index('ix_overrides_scix_id', 'scix_id'),
        Index('ix_overrides_bibcode_created', 'bibcode', 'created'),
        Index('ix_overrides_scix_id_created', 'scix_id', 'created'),
    )
    id = Column(Integer, primary_key=True)
    bibcode = Column(String(19))
    scix_id = Column(String(19))
    override = Column(ARRAY(String))
    created = Column(UTCDateTime, default=get_date)

class FinalCollectionTable(Base):
    __tablename__ = 'final_collection'
    __table_args__ = (
        Index('ix_final_collection_bibcode', 'bibcode'),
        Index('ix_final_collection_scix_id', 'scix_id'),
        Index('ix_final_collection_validated_created', 'validated', 'created'),
        Index('ix_final_collection_validated_validated_at', 'validated', 'validated_at'),
        Index('ix_final_collection_bibcode_created', 'bibcode', 'created'),
        Index('ix_final_collection_scix_id_created', 'scix_id', 'created'),
    )
    id = Column(Integer, primary_key=True)
    bibcode = Column(String(19))
    scix_id = Column(String(19))
    score_id = Column(Integer, ForeignKey('scores.id'))
    collection = Column(ARRAY(String))
    validated = Column(Boolean, default=False)
    validated_at = Column(UTCDateTime)
    created = Column(UTCDateTime, default=get_date)

class RunTable(Base):
    __tablename__ = 'run'
    id = Column(Integer, primary_key=True)
    run = Column(String(20))
    created = Column(UTCDateTime, default=get_date)
    model_id = Column(Integer, ForeignKey('models.id'))
