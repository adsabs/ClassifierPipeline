# print('app - 00')
import os
import json
import pickle
import zlib
import csv

# print('app - 01')
import ClassifierPipeline.models as models
from adsputils import get_date, ADSCelery, u2asc
from contextlib import contextmanager
from sqlalchemy import create_engine, desc, and_, or_
from sqlalchemy.orm import scoped_session, sessionmaker
from adsputils import load_config, setup_logging
# print('app - 02')

# global objects; we could make them belong to the app object but it doesn't seem necessary
# unless two apps with a different endpint/config live along; TODO: move if necessary
# cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# orcid_cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# ads_cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# bibcode_cache = cachetools.TTLCache(maxsize=2048, ttl=3600, timer=time.time, missing=None, getsizeof=None)

# ALLOWED_CATEGORIES = set(['astronomy', 'planetary science', 'heliophysics', 'earth science', 'physics', 'other physics', 'other'])


proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
config = load_config(proj_home=proj_home)
logger = setup_logging('app.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))

# logger.info('Config file {}'.format(config))

# print('app - 03')
ALLOWED_CATEGORIES = set(config['ALLOWED_CATEGORIES'])

# print('app - 04')
class SciXClassifierCelery(ADSCelery):
        

    def is_blank(s):
        """ Check if a string is not None, not empty, and not only whitespace
        Based on https://stackoverflow.com/questions/9573244/how-to-check-if-the-string-is-empty-in-python
        """
        # return bool(not s and s.isspace())
        return not (s and not s.isspace())

    def handle_message_payload(self, message=None, payload=None):
        """
        Handles the message payload
        """

        parsed_message = json.loads(message)
        request_list = parsed_message['classifyRequests']


    def classify_record_from_scores(self, record):
        """
        Classify a record after it has been scored. 

        Parameters
        ----------
        record : dictionary (required) (default=None) Dictionary with the following
            keys: bibcode, text, validate, categories, scores, and model information

        Returns
        -------
        record : dictionary with the following keys: bibcode, text, validate, categories,
            scores, model information, and Collections
        """

        # Fetch thresholds from config file
        thresholds = config['CLASSIFICATION_THRESHOLDS']
        logger.info(f'Classification Thresholds: {thresholds}')


        scores = record['scores']
        categories = record['categories']

        meet_threshold = [score > threshold for score, threshold in zip(scores, thresholds)]

        # Extra step to check for "Earth Science" articles miscategorized as "Other"
        # This is expected to be less neccessary with improved training data
        if config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'] == 'active':
            logger.info('Additional Earth Science Processing')
            if meet_threshold[categories.index('Other')] is True:
                # If Earth Science score above additional threshold
                if scores[categories.index('Earth Science')] \
                        > config['ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD']:
                    meet_threshold[categories.index('Other')] = False
                    meet_threshold[categories.index('Earth Science')] = True

        # Append collections to record
        record['collections'] = [category for category, threshold in zip(categories, meet_threshold) if threshold is True]
        record['collection_scores'] = [score for score, threshold in zip(scores, meet_threshold) if threshold is True]
        record['collection_scores'] = [round(score, 2) for score in record['collection_scores']]


        return record


    def prepare_output_file(self, output_path, output_format=None):
        """
        Prepares an output file
        """

        header = ['bibcode','scix_id','title','abstract','run_id','categories','scores','collections','collection_scores','earth_science_adjustment','override']

        with open(output_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(header)

        logger.info(f'Prepared {output_path} for writing.')


    def add_record_to_output_file(self, record):
        """
        Adds a record to the output file
        """
        row = [record['bibcode'], record['scix_id'],record['title'], record['abstract'],record['run_id'], ', '.join(config['ALLOWED_CATEGORIES']), ', '.join(map(str,record['scores'])), ', '.join(record['collections']), ', '.join(map(str, record['collection_scores'])), config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'], '']

        with open(record['output_path'], 'a', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(row)


    def index_run(self):
        """
        Indexes a run into a database

        :param: none 
        :return: str Run table row ID
        """
        logger.info('Indexing run')
        with self.session_scope() as session:

            run_row = models.RunTable()

            session.add(run_row)
            session.commit()

            return run_row.id


    def index_record(self, record):
        """
        Saves a record into a database

        :param: record- dictionar
        :return: boolean - whether record successfuly added
                to the database
        """
        logger.info('Indexing record')
        logger.info(f'Record: {record}')

        if 'operation_step' not in record:
            record['operation_step'] = 'classify'
        if 'scix_id' not in record:
            record['scix_id'] = None

        with self.session_scope() as session:

            # Initial indexing of automatic classification results
            if record['operation_step'] == 'classify':
                logger.info('Indexing new record')

                # Model Table
                model = {'model' : config['CLASSIFICATION_PRETRAINED_MODEL'],
                         'revision' : config['CLASSIFICATION_PRETRAINED_MODEL_REVISION'],
                         'tokenizer' : config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER']
                         }
                postprocessing = {'ADDITIONAL_EARTH_SCIENCE_PROCESSING' : config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'],
                                  'ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD' : config['ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD'],
                                  'CLASSIFICATION_THRESHOLDS' : config['CLASSIFICATION_THRESHOLDS']
                                  }
                model_row = models.ModelTable(model=json.dumps(model),
                                              postprocessing=json.dumps(postprocessing)
                                              )

                logger.info('Checking model query')
                check_model_query = session.query(models.ModelTable).filter(and_(models.ModelTable.model == json.dumps(model), models.ModelTable.postprocessing == json.dumps(postprocessing))).order_by(models.ModelTable.created.desc()).first()

                logger.info(f'Check Model Query: {check_model_query}')
                if check_model_query is None:
                    session.add(model_row)
                    session.commit()
                    model_id = model_row.id
                else:
                    model_id = check_model_query.id

                # Run Table
                check_run_query = session.query(models.RunTable).filter(models.RunTable.id == record['run_id']).order_by(models.RunTable.created.desc()).first()

                logger.info(f'Check Run Query: {check_run_query}')
                if check_run_query is not None:

                    if check_run_query.model_id != model_id:
                        check_run_query.model_id = model_id 
                        session.commit()

                # Override Table
                check_overrides_query = session.query(models.OverrideTable).filter(or_(models.OverrideTable.scix_id == record['scix_id'], models.OverrideTable.bibcode == record['bibcode'])).order_by(models.OverrideTable.created.desc()).first()


                logger.info(f'Check Overrides Query: {check_overrides_query}')
                if check_overrides_query is not None:
                    final_collections = check_overrides_query.override
                    overrides_id = check_overrides_query.id
                else:
                    final_collections = record['collections']
                    overrides_id = None

                # Scores Table
                scores_dict = {cat:score for cat, score in zip(config['ALLOWED_CATEGORIES'], record['scores'])}
                scores = {'scores': scores_dict,
                          'earth_science_adjustment': config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'],
                          'collections': record['collections']}

                score_row = models.ScoreTable(bibcode=record['bibcode'], 
                                            scix_id=record['scix_id'],
                                            scores=json.dumps(scores),
                                            overrides_id = overrides_id,
                                            run_id = record['run_id']
                                            ) 

                # Check if EXACT record is already in the database
                check_scores_query = session.query(models.ScoreTable).filter(and_(or_(models.ScoreTable.scix_id == record['scix_id'], models.ScoreTable.bibcode == record['bibcode']), models.ScoreTable.scores == json.dumps(scores), models.ScoreTable.overrides_id == overrides_id, models.ScoreTable.run_id == record['run_id'])).order_by(models.ScoreTable.created.desc()).first()


                logger.info(f'Check Scores Query: {check_scores_query}')
                if check_scores_query is None:
                    session.add(score_row)
                    session.commit()
                    score_id = score_row.id

                final_collections_row = models.FinalCollectionTable(bibcode = record['bibcode'], 
                                                                        collection = final_collections,
                                                                        score_id = score_row.id
                                                                        )


                # Final Collection Table
                check_final_collection_query = session.query(models.FinalCollectionTable).filter(or_(models.FinalCollectionTable.scix_id == record['scix_id'], models.FinalCollectionTable.bibcode == record['bibcode'])).order_by(models.FinalCollectionTable.created.desc()).first()

                if check_final_collection_query is None:
                    session.add(final_collections_row)
                    session.commit()
                if check_final_collection_query is not None:
                    check_final_collection_query.final_collection = final_collections
                    session.commit()

                return record, True
                
            else:
                logger.info('Updating validated record')

                # Check for existing override
                check_overrides_query = session.query(models.OverrideTable).filter(and_(or_(models.OverrideTable.scix_id == record['scix_id'], models.OverrideTable.bibcode == record['bibcode']), models.OverrideTable.override == record['override'])).order_by(models.OverrideTable.created.desc()).first()

                if check_overrides_query is None:
                    override_row = models.OverrideTable(bibcode=record['bibcode'],
                                                        scix_id=record['scix_id'],
                                                        override=record['override'])
                    session.add(override_row)
                    session.commit()
                    overrides_id = override_row.id

                    update_scores_query = session.query(models.ScoreTable).filter(models.ScoreTable.bibcode == record['bibcode']).order_by(models.ScoreTable.created.desc()).all()
                    for element in update_scores_query:
                        element.overrides_id = overrides_id
                        session.commit()

                    update_final_collection_query = session.query(models.FinalCollectionTable).filter(models.FinalCollectionTable.bibcode == record['bibcode']).order_by(models.FinalCollectionTable.created.desc()).first()

                    update_final_collection_query.collection = record['override']
                    update_final_collection_query.validated = True
                    session.commit()

                return record, False


    # def update_validated_records(self, run_name):
    def update_validated_records(self, run_id):
        """
        Updates validated records in the database

        :param: run_id- Boolean
        :return: boolean - whether update successful
        """
        print(f'Updating run_id: {run_id}')

        with self.session_scope() as session:

            run_query = session.query(models.RunTable).filter(models.RunTable.id == run_id).first()

            if run_query is not None:

                run_id_query = session.query(models.ScoreTable).filter(models.ScoreTable.run_id == run_query.id).all()

                for record in run_id_query:

                    update_final_collection_query = session.query(models.FinalCollectionTable).filter(and_(models.FinalCollectionTable.bibcode == record.bibcode, models.FinalCollectionTable.validated == False)).order_by(models.FinalCollectionTable.created.desc()).first()

                    if update_final_collection_query is not None:
                        update_final_collection_query.validated = True
                        session.commit()

 



