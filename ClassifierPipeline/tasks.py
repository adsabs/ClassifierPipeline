import sys
import os
import json
import pickle
# from __future__ import absolute_import, unicode_literals
import adsputils
from adsputils import ADSCelery
# from adsmsg import OrcidClaims
# from SciXClassifier import app as app_module
# from SciXClassifier import updater
# from SciXClassifier.exceptions import ProcessingException, IgnorableException
# from SciXClassifier.models import KeyValue
# from .app import SciXClassifierCelery
import ClassifierPipeline.app as app_module
import ClassifierPipeline.utilities as utils
from ClassifierPipeline.classifier import Classifier
from adsputils import load_config, setup_logging
from kombu import Queue
# import datetime
# from .classifier import score_record
# sys.path.append(os.path.abspath('../..'))
# from run import score_record, classify_record_from_scores, add_record_to_output_file
import classifyrecord_pb2
from google.protobuf.json_format import Parse

from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
# import pdb;pdb.set_trace()
# ============================= INITIALIZATION ==================================== #

proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
app = app_module.SciXClassifierCelery(
        # app = SciXClassifierCelery(
    # "scixclassifier-pipeline",
    "classifier-pipeline",
    proj_home=proj_home,
    local_config=globals().get("local_config", {}),
)
# import pdb; pdb.set_trace()
# from adsputils import setup_logging, load_config
config = load_config(proj_home=proj_home)
logger = setup_logging('tasks.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))

app.conf.CELERY_QUEUES = (
    Queue("classify-record", app.exchange, routing_key="classify-record"),
    Queue("update-record", app.exchange, routing_key="update-record")
)
# logger = app.logger

classifier = Classifier()


# ============================= TASKS ============================================= #

# From Curators Daily Operations 

# Send data to the Classifier

# Populate database wit new data

# Return sorted classifications to Curators

# Query SOLR
#   - Finding records with given set of parameters (e.g. classification, model, etc.)

@app.task(queue="update-record")
# def task_handle_input_from_master(message):
def task_update_record(message, tsv_output=True):
    """
    Handle the input from the master

    :param message: contains the message inside the packet
        {
         'bibcode': String (19 chars),
         'title': String,
         'abstract':String
        }
    :return: no return
    Handles input for the task
    This handles input from master checks if input is single bibcode or batch of bibcodes
    if Single just passes on to classfier tasks
    if batch will create a batch ID and send to classifier tasks and setup output file 

    :param: message - dictionary
    """

    # Check if input is single bibcode or path to file of bibcodes, title and abstracts
    # print('Handling input from master')
    # import pdb;pdb.set_trace()
    # tasks.task_send_input_record_to_classifier(message)

    # If batch of bibcodes
    # if message is type('classifyrecord_pb2.ClassifyRequestRecordList'):
    # if message.classify_requests:
    # Always pass a list of records, even just a list of one record
    # if isinstance(message, classifyrecord_pb2.ClassifyRequestRecordList):
    # if message is type(list):
    logger.info('********************************************')
    logger.info('*** task_update_record ***')
    # logger.info('Batch of bibcodes')
    output_path = ''

    # import pdb;pdb.set_trace()
    run_id = app.index_run()
    logger.info('Run ID: {}'.format(run_id))
    # run_id = '001'
    validate = False
    if tsv_output:
        logger.info('output path A: {}'.format(output_path))
        # output_path = os.path.join(proj_home, 'output', f'{run_id}_classified.tsv')
        output_path = os.path.join(proj_home, 'logs', f'{run_id}_classified.tsv')
        # output_path = proj_home+f'/logs/{run_id}_classified.tsv'
        logger.info('output path B: {}'.format(output_path))
    else:
        output_path = os.path.join(proj_home, 'output', f'{run_id}_classified.csv')

    logger.info('Preparing output file: {}'.format(output_path))
    # print()
    utils.prepare_output_file(output_path,tsv_output=tsv_output)
    # prepare_output_file(output_path,tsv_output=tsv_output)
    logger.info('Output file prepared')


    # Delay setting
    
    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.info("Delay set for queue messages: {}".format(delay_message))

    # logger.info('Message to be parsed: {}'.format(message))
    parsed_message = json.loads(message)

    request_list = parsed_message['classifyRequests']


    # for request in message.classify_requests:
    # logger.info('Request list: {}'.format(request_list))
    for request in request_list:
        logger.info('Request: {}'.format(request))
        record = {'bibcode': request['bibcode'],
                  'title': request['title'],
                  'abstract': request['abstract'],
                  'text': request['title'] + ' ' + request['abstract'],
                  'validate': validate,
                  'run_id': run_id,
                  'tsv_output': tsv_output,
                  'override': None,
                  'output_path': output_path#,
                  # 'model_dict' : model_dict
                  }

        # import pdb;pdb.set_trace()
        out_message = parsed_message.copy()
        out_message['classifyRequests'] = [record] # protobuf is for list of dictionaries
        # import pdb;pdb.set_trace()
        # out_message = json.dumps(out_message)
        out_message = json.dumps(out_message)
        # import pdb;pdb.set_trace()

        # if not delay_message:
        #     import pdb;pdb.set_trace()

        logger.info('Output Record type: {}'.format(type(out_message)))
        logger.info('Output Record: {}'.format(out_message))
        if delay_message:
            logger.info('Using delay')
            task_send_input_record_to_classifier.delay(out_message)
            # task_send_input_record_to_classifier.apply_async(out_message)
        else:
            # import pdb;pdb.set_trace()
            task_send_input_record_to_classifier(out_message)  
            # import pdb;pdb.set_trace()
            

# @app.task(queue="unclassified-queue")
# @app.task(queue="update-record")
@app.task(queue="classify-record")
def task_send_input_record_to_classifier(message):
    """
    Send a new record to the classifier


    :param message: contains the message inside the packet
        {
         'bibcode': String (19 chars),
         'title': String,
         'abstract':String
        }
    :return: no return
    """

    logger.info('********************************************')
    logger.info('*** task_send_input_record_to_classifier ***')
    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.info("Delay set for queue messages: {}".format(delay_message))

    fake_data = config.get('FAKE_DATA', False) 
    # if config.get('MODEL_DICT_SOURCE') == "test":
    #     fake_data = True

    logger.info("Fake data set for queue messages: {}".format(fake_data))

    if not delay_message:
        # import pdb; pdb.set_trace()
        pass

    parsed_message = json.loads(message)

    record = parsed_message['classifyRequests'][0]

    # import pdb; pdb.set_trace()
    logger.info("Parsed message")
    logger.info(parsed_message)
    # logger.info('model to use: {}'.format(model))
    # logger.info("Record before classification and thresholding")
    # logger.info(record)
    # logger.info('Record type: {}'.format(type(record)))
    # logger.info("Record before classification and thresholding: {}".format(message))
    # First obtain the raw scores from the classifier model
    # import pdb; pdb.set_trace()
    # record = app.score_record(record, fake_data=fake_data)

    tasks_sources = ['tasks_celery_object', 'tasks_app_direct']
    app_sources = ['app_direct']
    # if MODEL_DICT is loaded in tasks then pass it
    # if config.get('LOAD_MODEL_SOURCE') == "test":
    if fake_data is False:
        logger.info('Performing Inference')
        # categories, scores = Classifier().batch_assign_SciX_categories(record['text'])
        categories, scores = classifier.batch_assign_SciX_categories([record['text']])
        record['categories'] = categories[0]
        record['scores'] = scores[0]
        logger.info('Categories: {}'.format(categories))
        logger.info('Scores: {}'.format(scores))
        # import pdb; pdb.set_trace()
    # elif config.get('LOAD_MODEL_SOURCE') in app_sources:
    else:
        logger.info('Skipping inference - generating fake data')
        record = utils.return_fake_data(record)

    record['model'] = {'model' : config['CLASSIFICATION_PRETRAINED_MODEL'],
                       'revision' : config['CLASSIFICATION_PRETRAINED_MODEL_REVISION'],
                       'tokenizer' : config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER']}

    record['postprocessing'] = {'ADDITIONAL_EARTH_SCIENCE_PROCESSING' : True,
                                'ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD' : 0.015,
                                'CLASSIFICATION_THRESHOLDS' : [0.06, 0.03, 0.04, 0.02, 0.99, 0.02, 0.02, 0.99]}


   # record = app.score_record(record, fake_data=fake_data, model_dict=MODEL_DICT)

    # Then classify the record based on the raw scores
    # import pdb; pdb.set_trace()
    logger.info('RECORD: {}'.format(record))
    # logger.info(record)
    record = app.classify_record_from_scores(record)
    # print('Collections: ')
    # print(message['collections'])
    logger.info("*****     /////     //////     /////     //////     *****")
    logger.info("Record after classification and thresholding: {}".format(record))
    logger.info("Record Type: {}".format(type(record)))

    out_message = parsed_message.copy()
    out_message['classifyRequests'] = [record]
    # logger.info("Out Message")
    # logger.info(out_message)
    out_message = json.dumps(out_message)

    # if not delay_message:
    #     import pdb;pdb.set_trace()
    # Write the classifications to output file
    # add_record_to_output_file(message)
    # may have add .async 
    # task_output_results(message)

    # import pdb; pdb.set_trace()
    # Write the new classification to the database

    if delay_message:
        task_index_classified_record.delay(out_message)
    else:
        task_index_classified_record(out_message) 


    # import pdb; pdb.set_trace()



@app.task(queue="classify-record")
# @app.task(queue="update-record")
def task_index_classified_record(message):
    """
    Update the database with the new classification

    :param message: contains the message inside the packet
        {
         'bibcode': String (19 chars),
         'collections': [String],
         'abstract':String,
         'validate': Boolean,
         'override': String
        }
    :return: no return
    """

    logger.info('********************************************')
    logger.info('*** task_index_classified_record ***')
    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.info("Delay set for queue messages: {}".format(delay_message))

    logger.info("Unpacking message for indexing")
    message = json.loads(message)
    record = message['classifyRequests'][0]
    logger.info(message)
    # test_var = app.return_text('test_text')
    # logger.info(f'Task preamble test var {test_var}')
    logger.info('Record type: {}'.format(type(message)))

    # if not delay_message:
    #     import pdb; pdb.set_trace()
        # pass

    # print('Indexing Classified Record')
    # import pdb; pdb.set_trace()

    record, success = app.index_record(record)
    if success is True:
        task_output_results(record)
    else:
        logger.info("Record failed to be indexed")
    # import pdb; pdb.set_trace()
    # pass

# @app.task(queue="classify-record")
def task_update_validated_records(message):
    """
    Update all records that have been validated that have same run_id

    :param message: contains the message inside the packet
        {
         'run_id': Boolean,
        }
    """

    # print('Updating Validated Records')
    app.update_validated_records(message)
    # import pdb; pdb.set_trace()
    # pass


# @app.task(queue="output-results")
# @app.task(queue="classify-record")
def task_output_results(message):
    """
    This worker will forward results to the outside
    exchange (typically an ADSImportPipeline) to be
    incorporated into the storage

    :param msg: contains the bibcode and the collections:

            {'bibcode': '....',
             'collections': [....]
            }
    :type: adsmsg.OrcidClaims
    :return: no return
    """
    app.add_record_to_output_file(message)



if __name__ == "__main__":
    app.start()
