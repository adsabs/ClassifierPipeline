import os
import json
import pickle
import zlib
import csv

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
# from ClassifierPipeline import tasks, classifier
from ClassifierPipeline import tasks
from ClassifierPipeline.classifier import Classifier
# import cachetools
# import datetime
# import os
# import random
# import time
# import traceback
# from adsputils import load_config
from adsputils import load_config, setup_logging
# from adsmg import ClassifyRequestRecord, ClassifyRequestRecordList
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from google.protobuf.json_format import MessageToDict
import classifyrecord_pb2
from google.protobuf.json_format import Parse

proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
# global objects; we could make them belong to the app object but it doesn't seem necessary
# unless two apps with a different endpint/config live along; TODO: move if necessary
# cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# orcid_cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# ads_cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# bibcode_cache = cachetools.TTLCache(maxsize=2048, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# config = load_config(proj_home=proj_home)
# ALLOWED_STATUS = set(['claimed', 'updated', 'removed', 'unchanged', 'forced', '#full-import'])

ALLOWED_CATEGORIES = set(['astronomy', 'planetary science', 'heliophysics', 'earth science', 'physics', 'other physics', 'other'])


# proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
config = load_config(proj_home=proj_home)
logger = setup_logging('app.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))

logger.info('Config file {}'.format(config))

classifier = Classifier()

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

def load_model_and_tokenizer(pretrained_model_name_or_path=None, revision=None, tokenizer_model_name_or_path=None):
    """
    Load the model and tokenizer for the classification task, as well as the
    label mappings. Returns the model, tokenizer, and label mappings as a
    dictionary.

    Parameters
    ----------
    pretrained_model_name_or_path : str (optional) (default=None) Specifies the
        model name or path to the model to load. If None, then reads from the 
        config file.
    revision : str (optional) (default=None) Specifies the revision of the model
    tokenizer_model_name_or_path : str (optional) (default=None) Specifies the
        model name or path to the tokenizer to load. If None, then defaults to
        the pretrained_model_name_or_path.
    """
    # Define labels and ID mappings
    labels = ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']
    id2label = {i:c for i,c in enumerate(labels) }
    label2id = {v:k for k,v in id2label.items()}

    # import pdb;pdb.set_trace()
    # Define model and tokenizer
    # if pretrained_model_name_or_path is None:
    #     pretrained_model_name_or_path = config['CLASSIFICATION_PRETRAINED_MODEL']
    # The above line leads to (Pdb) pretrained_model_name_or_path
    # <SciXClassifierCelery __main__ at 0xffff60657190>
    pretrained_model_name_or_path = "/app/ClassifierPipeline/tests/models/checkpoint-32100/"
    if revision is None:
        revision = config['CLASSIFICATION_PRETRAINED_MODEL_REVISION']
    if tokenizer_model_name_or_path is None:
        tokenizer_model_name_or_path = config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_model_name_or_path,
                                              revision=revision,
                                              do_lower_case=False)

    # load model
    # import pdb;pdb.set_trace()
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                    revision=revision,
                                    num_labels=len(labels),
                                    problem_type='multi_label_classification',
                                    id2label=id2label,
                                    label2id=label2id
                                        )
    # Output as dictionary
    model_dict = {'model': model,
                  'tokenizer': tokenizer,
                  'labels': labels,
                  'id2label': id2label,
                  'label2id': label2id}

    logger.info('Model loading function - top level of app.py')
    return model_dict

# logger.info('Loading model and tokenizer')
# MODEL_DICT = load_model_and_tokenizer()

# import pdb;pdb.set_trace()

# if config['LOAD_MODEL_SOURCE'] == "app_direct":
#     pass
    # MODEL_DICT = load_model_and_tokenizer()
    # MODEL_DICT2 = load_model_and_tokenizer()
    # MODEL_DICT3 = load_model_and_tokenizer()
    # MODEL_DICT4 = load_model_and_tokenizer()
# elif config['LOAD_MODEL_SOURCE'] == "test":
#     MODEL_DICT = None


class SciXClassifierCelery(ADSCelery):
    # def __init__(self, app_name, *args, **kwargs):
    #     ADSCelery.__init__(self, app_name, *args, **kwargs)
    #     self.model_dict = self.load_model_and_tokenizer()

        # pass
    # def __init__(self, *args, **kwargs):
    #     pass
        
    @staticmethod
    def load_model_and_tokenizer(self,pretrained_model_name_or_path=None, revision=None, tokenizer_model_name_or_path=None):
        """
        Load the model and tokenizer for the classification task, as well as the
        label mappings. Returns the model, tokenizer, and label mappings as a
        dictionary.

        Parameters
        ----------
        pretrained_model_name_or_path : str (optional) (default=None) Specifies the
            model name or path to the model to load. If None, then reads from the 
            config file.
        revision : str (optional) (default=None) Specifies the revision of the model
        tokenizer_model_name_or_path : str (optional) (default=None) Specifies the
            model name or path to the tokenizer to load. If None, then defaults to
            the pretrained_model_name_or_path.
        """
        # Define labels and ID mappings
        labels = ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']
        id2label = {i:c for i,c in enumerate(labels) }
        label2id = {v:k for k,v in id2label.items()}

        # Define model and tokenizer
        # if pretrained_model_name_or_path is None:
        #     pretrained_model_name_or_path = config['CLASSIFICATION_PRETRAINED_MODEL']
        # The above line leads to (Pdb) pretrained_model_name_or_path
        # <SciXClassifierCelery __main__ at 0xffff60657190>
        pretrained_model_name_or_path = "/app/ClassifierPipeline/tests/models/checkpoint-32100/"
        if revision is None:
            revision = config['CLASSIFICATION_PRETRAINED_MODEL_REVISION']
        if tokenizer_model_name_or_path is None:
            tokenizer_model_name_or_path = config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER']

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_model_name_or_path,
                                                  revision=revision,
                                                  do_lower_case=False)

        # load model
        # import pdb;pdb.set_trace()
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                        revision=revision,
                                        num_labels=len(labels),
                                        problem_type='multi_label_classification',
                                        id2label=id2label,
                                        label2id=label2id
                                            )
        # Output as dictionary
        # model_dict = {
        #               'model_config': model.config.to_dict(),
        #               'state_dict': model.state_dict(),
        model_dict = {'model': model,
                      'tokenizer': tokenizer,
                      'labels': labels,
                      'id2label': id2label,
                      'label2id': label2id}
        # model_dict = { 'model_config': model.config.to_dict(),
        #               'state_dict': model.state_dict(),
        #               'tokenizer': tokenizer
                      # }
        # for key, value in model_dict['state_dict'].items():
        #     # model_dict['state_dict'][key] = value.cpu().numpy().tolist()
        #     model_dict['state_dict'][key] = value.tolist()

        logger.info('Model loading function - instance of SciXClassifierCelery')
        return model_dict

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


    def prepare_batch_from_master(self, batch,tsv_output=True):
        """
        Takes a list of dictionaries of records and converts each record into a
        dictionary with the following keys: bibcode and text (a combination of 
        title and abstract). Sends each record to the classification queue.

        Parameters
        ----------
        records_path : str (required) (default=None) Path to a .csv file of records

        Returns
        -------
        no return
        """

        run_id = app.index_run()
        validate = False
        if tsv_output:
            output_path = os.path.join(proj_home, 'output', f'{run_id}_classified.tsv')
        else:
            output_path = os.path.join(proj_home, 'output', f'{run_id}_classified.csv')

        print('Preparing output file: {}'.format(output_path))
        print()
        prepare_output_file(output_path,tsv_output=tsv_output)

        for record in batch:
            record['text'] = record['title'] + ' ' + record['abstract']
            record['validate'] = validate
            record['run_id'] = run_id
            record['tsv_output'] = tsv_output

            record['override'] = None
            record['output_path'] = output_path
            # For Testing
            tasks.task_send_input_record_to_classifier(record)
            # For Production
            # tasks.task_send_input_record_to_classifier.delay(record)

    def prepare_validation_records(records_path,validate=True, tsv_output=True):
        """
        Takes a path to a .csv file of records and converts each record into a
        dictionary with the following keys: bibcode and text (a combination of 
        title and abstract). Sends each record to the classification queue.

        Parameters
        ----------
        records_path : str (required) (default=None) Path to a .csv file of records

        Returns
        -------
        no return
        """
        # Open .csv file and read in records
        # Note: that this method requres the input file to have the following
        # Initial input columns: bibcode, title, abstract
        # Corrected input columns:
        # bibcode,title,abstract,categories,scores,collections,collection_scores,earth_science_adjustment,override
        print('Processing records from: {}'.format(records_path))
        print()


        with open(records_path, 'r') as f: 
            csv_reader = csv.reader(f, delimiter='\t')
            headers = next(csv_reader)

            # Add run ID to record data
            # run_id = time.time() 
            # validate all records with same run_id
            # import pdb;pdb.set_trace()

            for row in csv_reader:
                record = {}
                record['bibcode'] = row[0]
                record['title'] = row[1]
                record['abstract'] = row[2]
                record['text'] = row[1] + ' ' + row[2]
                record['validate'] = validate
                record['run_id'] = run_id
                record['tsv_output'] = tsv_output

                record['override'] = row[9].split(',')
                run_id = row[3]
                # For Testing
                # Instead make a check of proper collections
                # if is_allowed(record['override']):
                #     tasks.task_index_classified_record(record)
                if not is_blank(record['override'][0]):
                    tasks.task_index_classified_record(record)
                # For Production
                # if not is_blank(record['override'][0]):
                #     tasks.task_index_classified_record.delay(record)

                # print('testing message')
                # import pdb;pdb.set_trace()
                # Now send record to classification queue
            tasks.task_update_validated_records(run_id)



    def score_record(self, record, model_dict=None,fake_data=False):
    # def score_record(self, record,fake_data=False):
        """
        Provide classification scores for a record using the following
            categories:
                0 - Astronomy
                1 - HelioPhysics
                2 - Planetary Science
                3 - Earth Science
                4 - Biological and Physical Sciences
                5 - Other Physics
                6 - Other
                7 - Garbage

        Parameters
        ----------
        records_path : str (required) (default=None) Path to a .csv file of records

        Returns
        -------
        records : dictionary with the following keys: bibcode, text,
                    categories, scores, and model information
        """

        
        # import pdb;pdb.set_trace()
        # Load model and tokenizer
        # if model_dict is None:
        #     model_dict = self.load_model_and_tokenizer()
        # model_dict = self.model_dict
        # model_dict = record['model_dict']
        # model_dict = MODEL_DICT

        # global MODEL_DICT
        # model_dict = MODEL_DICT
        # self.motto = "I am the SciXClassifierCelery"
        logger.info('Scoring record: {}'.format(record))
        # if MODEL_DICT:
        #     logger.info('MODEL DICT: {}'.format(MODEL_DICT))
        # if model_dict is None:
        #     model_dict = MODEL_DICT
        #     logger.info('local MODEL_DICT')
        # logger.info('MODEL DICT labels: {}'.format(model_dict['labels']))
        # logger.info('MODEL DICT model: {}'.format(model_dict['model']))
        # logger.info('MODEL DICT model type: {}'.format(type(model_dict['model'])))
        # import pdb;pdb.set_trace()

        # Classify record
        # record['categories'], record['scores'] = classifier.batch_assign_SciX_categories(
        #                             [record['text']],model_dict['tokenizer'],
        #                             model_dict['model'],model_dict['labels'],
        #                             model_dict['id2label'],model_dict['label2id'])
        # if fake_data is False:
        # logger.info('Config file {}'.format(config))
        model_implementation = config.get('MODEL_IMPLEMENTATION', 'object_load')
        # model_implementation = config.get('MODEL_IMPLEMENTATION', 'fake_data')
        if model_implementation == 'direct_load':
            logger.info('Trying to Scoring record - old method')
            try:
                record['categories'], record['scores'] = classifier.batch_assign_SciX_categories(
                                            [record['text']],model_dict['tokenizer'],
                                            model_dict['model'],model_dict['labels'],
                                            model_dict['id2label'],model_dict['label2id'])
            except:
                logger.exception('Error in scoring record')
            # record['categories'], record['scores'] = classifier.batch_assign_SciX_categories(
            #                             [record['text']],MODEL_DICT['tokenizer'],
            #                             MODEL_DICT['model'],MODEL_DICT['labels'],
            #                             MODEL_DICT['id2label'],MODEL_DICT['label2id'])

            # Because the classifier returns a list of lists so it can batch process
            # Take only the first element of each list
            # import pdb;pdb.set_trace()
            record['categories'] = record['categories'][0]
            record['scores'] = record['scores'][0]
        elif model_implementation == 'object_load':
        # elif config['MODEL_IMPLEMENTATION'] == 'object_load':
            logger.info('Trying to Scoring record - new method')
            try:
                record['categories'], record['scores'] = classifier.batch_assign_SciX_categories(
                                            [record['text']])
            except:
                logger.exception('Error in scoring record')
            # record['categories'], record['scores'] = classifier.batch_assign_SciX_categories(
            #                             [record['text']],MODEL_DICT['tokenizer'],
            #                             MODEL_DICT['model'],MODEL_DICT['labels'],
            #                             MODEL_DICT['id2label'],MODEL_DICT['label2id'])

            # Because the classifier returns a list of lists so it can batch process
            # Take only the first element of each list
            # import pdb;pdb.set_trace()
            record['categories'] = record['categories'][0]
            record['scores'] = record['scores'][0]
        # elif config['MODEL_IMPLEMENTATION'] == 'fake_data':
        else:
            logger.info('Using fake data')

            record['categories'] = ["Astronomy", "Heliophysics", "Planetary Science", "Earth Science", "NASA-funded Biophysics", "Other Physics", "Other", "Text Garbage"]
            record['scores'] = [0.0035331270191818476, 0.002528271870687604, 0.0018561003962531686, 0.017948558554053307, 0.0025463267229497433, 0.0006167808314785361, 0.983734130859375, 0.00024269080313388258]

        # Append model information to record
        # record['model'] = model_dict['model']
        # record['model'] = model_dict
        record['model'] = {'model' : config['CLASSIFICATION_PRETRAINED_MODEL'],
                           'revision' : config['CLASSIFICATION_PRETRAINED_MODEL_REVISION'],
                           'tokenizer' : config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER']}


        record['postprocessing'] = {'ADDITIONAL_EARTH_SCIENCE_PROCESSING' : True,
                                    'ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD' : 0.015,
                                    'CLASSIFICATION_THRESHOLDS' : [0.06, 0.03, 0.04, 0.02, 0.99, 0.02, 0.02, 0.99]}


        # print("checkpoint000")
        # import pdb;pdb.set_trace()

        # print("Record: {}".format(record['bibcode']))
        # print("Text: {}".format(record['text']))
        # print("Categories: {}".format(record['categories']))
        # print("Scores: {}".format(record['scores']))

        # import pdb;pdb.set_trace()
        return record

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
        # print('Thresholds: {}'.format(thresholds))


        # import pdb;pdb.set_trace()
        scores = record['scores']
        categories = record['categories']
        # max_score_index = scores.index(max(scores))
        # max_category = categories[max_score_index]
        # max_score = scores[max_score_index]

        meet_threshold = [score > threshold for score, threshold in zip(scores, thresholds)]

        # Extra step to check for "Earth Science" articles miscategorized as "Other"
        # This is expected to be less neccessary with improved training data
        if config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'] is True:
            # print('Additional Earth Science Processing')
            # import pdb;pdb.set_trace()
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
        record['earth_science_adjustment'] = config['ADDITIONAL_EARTH_SCIENCE_PROCESSING']


        return record


    def prepare_output_file(self, output_path,tsv_output=True):
        """
        Prepares an output file
        """

        # header = 'bibcode,title,abstract,run_id,categories,scores,collections,collection_scores,earth_science_adjustment,override\n'
        header = ['bibcode','title','abstract','run_id','categories','scores','collections','collection_scores','earth_science_adjustment','override']

        with open(output_path, 'w', newline='') as file:
            if tsv_output:
                writer = csv.writer(file, delimiter='\t')
            else:
                writer = csv.writer(file)
            writer.writerow(header)
            # file.write('\t'.join(header) + '\n')

    def add_record_to_output_file(self, record):
        """
        Adds a record to the output
        """
        #bibcode    title   abstract    categories  scores  collections collection_scores   earth_science_adjustment    run_id  override
        row = [record['bibcode'], record['title'], record['abstract'],record['run_id'], ', '.join(record['categories']), ', '.join(map(str,record['scores'])), ', '.join(record['collections']), ', '.join(map(str, record['collection_scores'])), record['earth_science_adjustment'], '']

        with open(record['output_path'], 'a', newline='') as file:
            if record['tsv_output']:
                writer = csv.writer(file, delimiter='\t')
            else:
                writer = csv.writer(file)
            writer.writerow(row)


    def handle_input_from_master(self, message, tsv_output=True): 
        """
        Handles input for the task
        This handles input from master checks if input is single bibcode or batch of bibcodes
        if Single just passes on to classfier tasks
        if batch will create a batch ID and send to classifier tasks and setup output file 

        :param: message - dictionary
        """

        # Check if input is single bibcode or path to file of bibcodes, title and abstracts
        self.logger.info('Handling input from master')
        # import pdb;pdb.set_trace()
        # tasks.task_send_input_record_to_classifier(message)

        # If batch of bibcodes
        # if message is type('classifyrecord_pb2.ClassifyRequestRecordList'):
        # if message.classify_requests:
        # if isinstance(message, classifyrecord_pb2.ClassifyRequestRecordList):
        # if message is type(list):

        self.logger.info('Batch of bibcodes')

        # import pdb;pdb.set_trace()
        run_id = self.index_run()
        # run_id = '001'
        validate = False
        if tsv_output:
            output_path = os.path.join(proj_home, 'output', f'{run_id}_classified.tsv')
        else:
            output_path = os.path.join(proj_home, 'output', f'{run_id}_classified.csv')

        self.logger.info('Preparing output file: {}'.format(output_path))
        # print()
        self.prepare_output_file(output_path,tsv_output=tsv_output)
        # prepare_output_file(output_path,tsv_output=tsv_output)

        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        for request in message.classify_requests:
            self.logger.info('Request: {}'.format(request))
            record = {'bibcode': request.bibcode,
                      'title': request.title,
                      'abstract': request.abstract,
                      'text': request.title + ' ' + request.abstract,
                      'validate': validate,
                      'run_id': run_id,
                      'tsv_output': tsv_output,
                      'override': None,
                      'output_path': output_path
                      }

            # import pdb;pdb.set_trace()
            tasks.task_send_input_record_to_classifier(record)
            # import pdb;pdb.set_trace()
                
                # message = classifyrecord_pb2.ClassifyRequestRecord(**record)
                # message = classifyrecord_pb2.ClassifyRequestRecord(record)
                # record_out = classifyrecord_pb2.ClassifyRequestRecord()
                # import pdb;pdb.set_trace()
                
                # Parse(json.dumps(record), record_out)
                # import pdb;pdb.set_trace()

                # Wrap message in a protobuf message


                # tasks.task_send_input_record_to_classifier.delay(json.dumps(record))
                # tasks.task_send_input_record_to_classifier.delay(record)
                # tasks.task_send_input_record_to_classifier.delay([record])
                # from adsmg import ClassifyRequestRecord, ClassifyRequestRecordList
                # record = ClassifyRequestRecord(bibcode=request.bibcode,
                #                                title=request.title,
                #                                abstract=request.abstract)
                # record = ClassifyRequestRecord(record)
                # tasks.task_send_input_record_to_classifier.apply_async([record])
                # tasks.task_send_input_record_to_classifier.apply_async(json.dumps(record))
                # tasks.task_send_input_record_to_classifier.apply_async(record)

            # import pdb;pdb.set_trace()

        # if message is type(dict):
        #     # print('Single bibcode')
        #     # import pdb;pdb.set_trace()
        #     record = {'bibcode': message['bibcode'],
        #               'title': message['title'],
        #               'abstract': message['abstract'],
        #               'text': message['title'] + ' ' + message['abstract']
        #               }
        #     tasks.task_send_input_record_to_classifier(record)

        # If batch of bibcodes
        # if 'filename' in message.keys():
        # if message is type('classifyrecord_pb2.ClassifyRequestRecordList'):
        # # if message is type(list):
        #     # print('Batch of bibcodes')
        #     import pdb;pdb.set_trace()
        #     for request in message.classify_requests:
        #         print('Request: {}'.format(request))
        #     import pdb;pdb.set_trace()
        #     prepare_batch_from_master(message)

    def index_run(self):
        """
        Indexes a run into a database

        :param: none 
        :return: run_id - id of the run
        """
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
        # print('Indexing record in index_record')
        # logging.info('**************************************')
        # logging.info('Indexing record in index_record')

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

        # import pdb; pdb.set_trace()
        logger.info('Indexing record in index_record')
        logger.info('Record: {}'.format(record))
        logger.info('Record bibcode: {}'.format(record['bibcode']))

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
                    model_id = model_row.id
                else:
                    model_id = check_model_query.id

                # Run Table
                # run_row = models.RunTable(model_id=models_id#,
                                          # run=record['run_name']
                                          # )

                # Check if run is already in the database
                # check_run_query = session.query(models.RunTable).filter(models.RunTable.run == record['run_name'] and models.RunTable.model_id == models_id).order_by(models.RunTable.created.desc()).first()
                check_run_query = session.query(models.RunTable).filter(models.RunTable.id == record['run_id']).order_by(models.RunTable.created.desc()).first()

                # import pdb; pdb.set_trace()
                if check_run_query is not None:

                    # run_id = check_run_query.id
                    if check_run_query.model_id != model_id:
                        check_run_query.model_id = model_id 
                        session.commit()
                # import pdb; pdb.set_trace()
                # else:
                #     session.add(run_row)
                #     session.commit()
                #     run_id = run_row.id

                # import pdb; pdb.set_trace()
                # if record['bibcode'] == record['run_bibcode']:
                    # session.add(run_row)
                    # session.commit()
                    # run_id = run_row.id
                # else:
                    # run_query = session.query(models.ScoreTable).filter(models.ScoreTable.bibcode == record['run_bibcode']).order_by(models.ScoreTable.created.desc()).first()
                    # run_id = run_query.run_id

                # record['run_id'] = run_id

                
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
                                            run_id = record['run_id']
                                            ) 

                # Check if EXACT record is already in the database
                # check_scores_query = session.query(models.ScoreTable).filter(models.ScoreTable.bibcode == record['bibcode'] and models.ScoreTable.scores == json.dumps(scores) and models.ScoreTable.overrides_id == overrides_id and models.ScoreTable.models_id == models_id).order_by(models.ScoreTable.created.desc()).first()
                check_scores_query = session.query(models.ScoreTable).filter(and_(models.ScoreTable.bibcode == record['bibcode'], models.ScoreTable.scores == json.dumps(scores), models.ScoreTable.overrides_id == overrides_id, models.ScoreTable.run_id == record['run_id'])).order_by(models.ScoreTable.created.desc()).first()
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



