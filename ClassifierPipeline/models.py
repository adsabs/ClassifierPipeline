# -*- coding: utf-8 -*-

from builtins import str
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, ARRAY, ForeignKey
from sqlalchemy.types import Enum
import json
import sys
from adsputils import get_date, UTCDateTime

Base = declarative_base()


class ScoreTable(Base):
    __tablename__ = 'scores'
    id = Column(Integer, primary_key=True)
    bibcode = Column(String(19))
    scores = Column(Text)
    created = Column(UTCDateTime, default=get_date)
    overrides_id = Column(Integer, ForeignKey('overrides.id'))
    models_id = Column(Integer, ForeignKey('models.id'))

class ModelTable(Base):
    __tablename__ = 'models'
    id = Column(Integer, primary_key=True)
    model = Column(Text)
    # revision = Column(Text)
    # tokenizer = Column(Text)
    postprocessing = Column(Text)
    # labels = Column(Text)
    created = Column(UTCDateTime, default=get_date)

class OverrideTable(Base):
    __tablename__ = 'overrides'
    id = Column(Integer, primary_key=True)
    # score_id = Column(Integer, ForeignKey('scores.id'))
    bibcode = Column(String(19))
    override = Column(ARRAY(String))
    created = Column(UTCDateTime, default=get_date)

class FinalCollectionTable(Base):
    __tablename__ = 'final_collection'
    id = Column(Integer, primary_key=True)
    bibcode = Column(String(19))
    score_id = Column(Integer, ForeignKey('scores.id'))
    collection = Column(ARRAY(String))
    created = Column(UTCDateTime, default=get_date)

