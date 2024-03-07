import json
import pickle
import zlib

# from builtins import str
# from .models import ClaimsLog, Records, AuthorInfo, ChangeLog
# from ClassiferPipeline.models import ScoreTable, OverrideTable, FinalCollectionTable
import ClassifierPipeline.models as models
# import ClassifierPipeline.app as app_module
from adsputils import get_date, ADSCelery, u2asc
# from ADSOrcid import names
# from ADSOrcid.exceptions import IgnorableException
# from celery import Celery
from contextlib import contextmanager
# from dateutil.tz import tzutc
# from sqlalchemy import and_
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import scoped_session, sessionmaker
# import cachetools
# import datetime
# import os
# import random
# import time
# import traceback

# global objects; we could make them belong to the app object but it doesn't seem necessary
# unless two apps with a different endpint/config live along; TODO: move if necessary
# cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# orcid_cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# ads_cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# bibcode_cache = cachetools.TTLCache(maxsize=2048, ttl=3600, timer=time.time, missing=None, getsizeof=None)

# ALLOWED_STATUS = set(['claimed', 'updated', 'removed', 'unchanged', 'forced', '#full-import'])

def get_checksum(text):
    """Compute CRC (integer) of a string"""
    #return hex(zlib.crc32(text.encode('utf-8')) & 0xffffffff)
    return zlib.crc32(text.encode('utf-8'))


def clear_caches():
    """Clears all the module caches."""
    cache.clear()
    classifier_cache.clear()
    ads_cache.clear()
    bibcode_cache.clear()


class SciXClassifierCelery(ADSCelery):


    # def __init__(self, *args, **kwargs):
    #     pass

    def index_record(self, record):
        """
        Sasves a record into a database

        :param: record- dictionar
        :return: boolean - whether record successfuly added
                to the database
        """
        print('Indexing record in index_record')
    # id = Column(Integer, primary_key=True)
    # bibcode = Column(String(19), unique=True)
    # scores = Column(Text)
    # created = Column(UTCDateTime, default=get_date)

        # Scores Table

        scores = {'scores': {cat:score for cat, score in zip(record['categories'], record['scores'])},
                  'earth_science_adjustment': record['earth_science_adjustment'],
                  'collections': record['collections']}

        # Just for initialy working out the logic
        overrides_id = None
        models_id = None
        
        score_row = models.ScoreTable(bibcode=record['bibcode'], 
                                         # scores=json.dumps(scores))
                                        scores=json.dumps(scores),
                                        overrides_id = overrides_id,
                                        models_id = models_id
                                        ) 


        # Models Table


        labels_dict = {'labels': record['model']['labels'],
                      'id2label': record['model']['id2label'],
                      'label2id': record['model']['label2id']}


        # Note that bot model and tokenizer are objects and need to be
        # converted to strings
                      
        model_dictionary_string = {k: str(v) for k, v in record['model']['model'].__dict__.items()}
        tokenizer_dictionary_string = {k: str(v) for k, v in record['model']['tokenizer'].__dict__.items()}

        model_row = models.ModelTable(model=model_dictionary_string,
                                      tokenizer=tokenizer_dictionary_string,
                                      postprocessing=json.dumps(record['postprocessing']),
                                      labels=json.dumps(labels_dict)
                                      )
        
        import pdb; pdb.set_trace()

        # for k, v in rmd.items():
        #     print('-----------------')
        #     print(k, v)
        #     print('-----------------')

        # import pdb; pdb.set_trace()

        # Overrides Table
            # This will be used only if 
            #  1) If an override already exists for a record and
            #       The same model and postprocessing is the same
            #  2) If a Curator adds an override during the validation step



        # Final Collection Table
        # final_collection = record['collections']
        final_collections_row = models.FinalCollectionTable(collection = record['collections']
                                                            )



        # res = []
        with self.session_scope() as session:

            # import pdb; pdb.set_trace()
            # session.add(score_table)
            # session.commit()
            
            # First check if the model exists
            # check_model_query = session.query(models.ModelTable).filter(models.ModelTable.model == pickle.dumps(record['model']['model']) and models.ModelTable.postprocessing == json.dumps(record['postprocessing'])).order_by(models.ModelTable.created.desc()).first()

            # check_model_query = session.query(models.ModelTable).filter(models.ModelTable.model == model_dictionary_string).order_by(models.ModelTable.created.desc()).first()
            # check_model_query = session.query(models.ModelTable).filter(models.ModelTable.model == model_dictionary_string).order_by(models.ModelTable.created.desc()).first()
            session.add(model_row)
            session.commit()
            print('checkpoint001')
            import pdb; pdb.set_trace()

            check_model_query = session.query(ModelTable).filter(ModelTable.model == record['model'] and ModelTable.postprocessing == record['postprocessing']).order_by(ModelTable.created.desc()).first()

            if check_model_query is None:
                session.add(model_row)
                session.commit()
                # do a commit then can try model_row.id for the scores column below
            
            

            # Check if record is already in database
        

            check_query = session.query(ScoreTable).filter(ScoreTable.bibcode == record['bibcode']).order_by(ScoreTable.created.desc()).first()

            # Process 1
            # If the record does not exist
            # 1) Add record to Scores Table
            # 2) Add scores to Final Collections Table
            # 3) Check if model is in model table
            #    a) If so link scores to model
            #    b) Add model to Model Table and the link to Score Table
            if check_query is None:

                # Add record to score table
                session.add(score_row)
                session.add(final_collection_row)





            # Process 2
            # If the record exists
            #   1) check if there is an ovverride
            #       a) True - override exists
            #               add scores to Score Table and link to
            #               existing override
            #          Then update Final Collection Table to point to latest


            #       b) False
            #       a) If not add scores and use for Final Collection Table 
            #           following process 1


            print('checkpoint002')
            import pdb; pdb.set_trace()
            


            # for c in claims:
            #     if isinstance(c, ClaimsLog):
            #         claim = c
            #     else:
            #         claim = self.create_claim(**c)
            #     if claim:
            #         session.add(claim)
            #         res.append(claim)
            # session.commit()
            # res = [x.toJSON() for x in res]
        # return res


    def score_record_collections(self, record, classifier):
        """
        Given a record and a classifier, score the record
        and return a list of scores

        :param: record - Records object
        :param: classifier - Classifier object

        :return: list of scores
        """
        pass


    def postprocess_classifier_scores(self, record, scores):
        """
        Given a record and a list of scores, postprocess
        the scores and return a list of collections

        :param: record - Records object
        :param: scores - list of scores

        :return: list of scores
        """
        pass



