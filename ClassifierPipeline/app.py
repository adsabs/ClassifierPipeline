import os
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
from sqlalchemy import create_engine, desc, and_
from sqlalchemy.orm import scoped_session, sessionmaker
from ClassifierPipeline import tasks
# import cachetools
# import datetime
# import os
# import random
# import time
# import traceback

proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
# global objects; we could make them belong to the app object but it doesn't seem necessary
# unless two apps with a different endpint/config live along; TODO: move if necessary
# cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# orcid_cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# ads_cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# bibcode_cache = cachetools.TTLCache(maxsize=2048, ttl=3600, timer=time.time, missing=None, getsizeof=None)

# ALLOWED_STATUS = set(['claimed', 'updated', 'removed', 'unchanged', 'forced', '#full-import'])

ALLOWED_CATEGORIES = set(['astronomy', 'planetary science', 'heliophysics', 'earth science', 'physics', 'other physics', 'other'])

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

        # Start with Model Table and Overrides Table then link to Scores Table

        # Just for initialy working out the logic
        # overrides_id = None
        # models_id = None
        
            # Process 1
            # If the record does not exist
            # 1) Check if model is in model table
            #    a) Add model to Model Table and the link to Score Table
            # 2) Add record to Scores Table
            # 3) Add scores to Final Collections Table

            # Process 2
                #check if there is an ovverride
            #       a) True - override exists
            #               add scores to Score Table and link to
            #               existing override
            #          Then update Final Collection Table to point to latest


            #       b) False
            #       a) If not add scores and use for Final Collection Table 
            #           following process 1

        # print('checkpoint001')
        # import pdb; pdb.set_trace()


        # scores = {'scores': {cat:score for cat, score in zip(record['categories'], record['scores'])},
        #                   'earth_science_adjustment': record['earth_science_adjustment'],
        #                   'collections': record['collections']}

        with self.session_scope() as session:

            if record['validate'] is False:

                # Create model table
                model_row = models.ModelTable(model=json.dumps(record['model']),
                                              # revision=record['model']['revision'],
                                              # tokenizer=record['model']['tokenizer'],
                                              postprocessing=json.dumps(record['postprocessing'])
                                              )

                # Check if model is already in the database
                check_model_query = session.query(models.ModelTable).filter(and_(models.ModelTable.model == json.dumps(record['model']), models.ModelTable.postprocessing == json.dumps(record['postprocessing']))).order_by(models.ModelTable.created.desc()).first()

                # import pdb; pdb.set_trace()
                if check_model_query is None:
                    session.add(model_row)
                    session.commit()
                    models_id = model_row.id
                else:
                    models_id = check_model_query.id

                # Run Table
                run_row = models.RunTable(model_id=models_id#,
                                          # run=record['run_name']
                                          )

                # Check if run is already in the database
                # check_run_query = session.query(models.RunTable).filter(models.RunTable.run == record['run_name'] and models.RunTable.model_id == models_id).order_by(models.RunTable.created.desc()).first()

                # if check_run_query is not None:
                #     run_id = check_run_query.id
                # else:
                #     session.add(run_row)
                #     session.commit()
                #     run_id = run_row.id

                # import pdb; pdb.set_trace()
                if record['bibcode'] == record['run_bibcode']:
                    session.add(run_row)
                    session.commit()
                    run_id = run_row.id
                else:
                    run_query = session.query(models.ScoreTable).filter(models.ScoreTable.bibcode == record['run_bibcode']).order_by(models.ScoreTable.created.desc()).first()
                    run_id = run_query.run_id

                record['run_id'] = run_id

                
                # import pdb; pdb.set_trace()
                # Check if there is an override
                check_overrides_query = session.query(models.OverrideTable).filter(models.OverrideTable.bibcode == record['bibcode']).order_by(models.OverrideTable.created.desc()).first()

                if check_overrides_query is not None:
                    final_collections = check_overrides_query.override
                    overrides_id = check_overrides_query.id
                else:
                    final_collections = record['collections']
                    overrides_id = None

                # print('checkpoint001')
                # import pdb; pdb.set_trace()
            
                # Now create the score table
                scores = {'scores': {cat:score for cat, score in zip(record['categories'], record['scores'])},
                          'earth_science_adjustment': record['earth_science_adjustment'],
                          'collections': record['collections']}

                score_row = models.ScoreTable(bibcode=record['bibcode'], 
                                            scores=json.dumps(scores),
                                            overrides_id = overrides_id,
                                            # models_id = models_id,
                                            # run_id = record['run_id']
                                            run_id = run_id
                                            ) 

                # Check if EXACT record is already in the database
                # check_scores_query = session.query(models.ScoreTable).filter(models.ScoreTable.bibcode == record['bibcode'] and models.ScoreTable.scores == json.dumps(scores) and models.ScoreTable.overrides_id == overrides_id and models.ScoreTable.models_id == models_id).order_by(models.ScoreTable.created.desc()).first()
                check_scores_query = session.query(models.ScoreTable).filter(and_(models.ScoreTable.bibcode == record['bibcode'], models.ScoreTable.scores == json.dumps(scores), models.ScoreTable.overrides_id == overrides_id, models.ScoreTable.run_id == run_id)).order_by(models.ScoreTable.created.desc()).first()
                # check_scores_query = session.query(models.ScoreTable).filter(and_(models.ScoreTable.bibcode == record['bibcode'], models.ScoreTable.scores == json.dumps(scores))).order_by(models.ScoreTable.created.desc()).first()

                # check_scores_query = session.query(models.ScoreTable).filter( and_(models.ScoreTable.bibcode == record['bibcode'], models.ScoreTable.run_id == run_id)).order_by(models.ScoreTable.created.desc()).first()

            
                # print('checkpoint001')
                # import pdb; pdb.set_trace()

                # import pdb; pdb.set_trace()
                if check_scores_query is None:
                    # Add record to score table
                    session.add(score_row)
                    session.commit()
                    score_id = score_row.id

                # import pdb; pdb.set_trace()
                final_collections_row = models.FinalCollectionTable(bibcode = record['bibcode'], 
                                                                        collection = final_collections,
                                                                        score_id = score_row.id
                                                                        )


                # Check if there is an override
                # check_overrides_query = session.query(models.OverrideTable).filter(models.OverrideTable.bibcode == record['bibcode']])).order_by(models.OverrideTable.created.desc()).first()
                # Check if EXACT final_collection record is already in the database
                # check_final_collection_query = session.query(models.FinalCollectionTable).filter(and_(models.FinalCollectionTable.bibcode == record['bibcode'], models.FinalCollectionTable.collection == final_collections, models.FinalCollectionTable.score_id == score_id)).order_by(models.FinalCollectionTable.created.desc()).first()
                check_final_collection_query = session.query(models.FinalCollectionTable).filter(models.FinalCollectionTable.bibcode == record['bibcode']).order_by(models.FinalCollectionTable.created.desc()).first()

                if check_final_collection_query is None:
                    session.add(final_collections_row)
                    session.commit()
                if check_final_collection_query is not None:
                    check_final_collection_query.final_collection = final_collections
                    session.commit()

                # print('checkpoint002')
                # import pdb; pdb.set_trace()
                tasks.task_output_results(record)
                
            else:
                print('Record is validated')
                # pass

                # Check if there is an override
                check_overrides_query = session.query(models.OverrideTable).filter(and_(models.OverrideTable.bibcode == record['bibcode'], models.OverrideTable.override == record['override'])).order_by(models.OverrideTable.created.desc()).first()

                if check_overrides_query is None:
                    override_row = models.OverrideTable(bibcode=record['bibcode'], override=record['override'])
                    # import pdb; pdb.set_trace()
                    session.add(override_row)
                    session.commit()
                    overrides_id = override_row.id

                    # update_scores_query = session.query(models.ScoreTable).filter(and_(models.ScoreTable.bibcode == record['bibcode'], models.ScoreTable.scores == json.dumps(scores))).order_by(models.ScoreTable.created.desc()).first()
                    update_scores_query = session.query(models.ScoreTable).filter(models.ScoreTable.bibcode == record['bibcode']).order_by(models.ScoreTable.created.desc()).all()
                    # import pdb; pdb.set_trace()
                    for element in update_scores_query:
                        element.overrides_id = overrides_id
                        session.commit()
                    # update_scores_query.overrides_id = overrides_id
                    # session.commit()

                    # update_final_collection_query = session.query(models.FinalCollectionTable).filter(and_(models.FinalCollectionTable.bibcode == record['bibcode'], models.FinalCollectionTable.collection == final_collections, models.FinalCollectionTable.score_id == score_id)).order_by(models.FinalCollectionTable.created.desc()).first()
                    update_final_collection_query = session.query(models.FinalCollectionTable).filter(models.FinalCollectionTable.bibcode == record['bibcode']).order_by(models.FinalCollectionTable.created.desc()).first()

                    update_final_collection_query.collection = record['override']
                    update_final_collection_query.validated = record['validate']
                    session.commit()


                print('checkpoint003')
                # import pdb; pdb.set_trace()
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


    # def update_validated_records(self, run_name):
    def update_validated_records(self, run_id):
        """
        Updates validated records in the database

        :param: run_id- Boolean
        :return: boolean - whether update successful
        """
        print(f'Updating run_id: {run_id}')

        # import pdb; pdb.set_trace()
        with self.session_scope() as session:

            # run_query = session.query(models.RunTable).filter(models.RunTable.run == run_name).first()
            run_query = session.query(models.RunTable).filter(models.RunTable.id == run_id).first()

            if run_query is not None:

                run_id_query = session.query(models.ScoreTable).filter(models.ScoreTable.run_id == run_query.id).all()

                for record in run_id_query:

                    update_final_collection_query = session.query(models.FinalCollectionTable).filter(and_(models.FinalCollectionTable.bibcode == record.bibcode, models.FinalCollectionTable.validated == False)).order_by(models.FinalCollectionTable.created.desc()).first()

                    if update_final_collection_query is not None:
                        update_final_collection_query.validated = True
                        session.commit()

 
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



