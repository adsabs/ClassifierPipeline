import sys
import os
import json
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
from adsputils import load_config, setup_logging
from kombu import Queue
# import datetime
# from .classifier import score_record
sys.path.append(os.path.abspath('../..'))
# from run import score_record, classify_record_from_scores, add_record_to_output_file
import classifyrecord_pb2
from google.protobuf.json_format import Parse

from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
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
    logger.info('Batch of bibcodes')

    # import pdb;pdb.set_trace()
    run_id = app.index_run()
    logger.info('Run ID: {}'.format(run_id))
    # run_id = '001'
    validate = False
    if tsv_output:
        output_path = os.path.join(proj_home, 'output', f'{run_id}_classified.tsv')
    else:
        output_path = os.path.join(proj_home, 'output', f'{run_id}_classified.csv')

    logger.info('Preparing output file: {}'.format(output_path))
    # print()
    # app.prepare_output_file(output_path,tsv_output=tsv_output)
    # prepare_output_file(output_path,tsv_output=tsv_output)
    logger.info('Output file prepared')

    # import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()
    # logger.info('Message being sent')
    # message = classifyrecord_pb2.ClassifyRequestRecordList(**message)
    # logger.info(message)
    # logger.info('Message requests')
    # logger.info(dir(message))
    logger.info('message type: {}'.format(type(message)))
    # logger.info('message contents: {}'.format(message))

    # Delay setting
    
    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.info("Delay set for queue messages: {}".format(delay_message))

    parsed_message = json.loads(message)

    request_list = parsed_message['classifyRequests']

    # if not delay_message:
    #     import pdb;pdb.set_trace()

    # for request in message.classify_requests:
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
                  'output_path': output_path
                  }

        # import pdb;pdb.set_trace()
        # if not delay_message:
        #     import pdb;pdb.set_trace()
        # tasks.task_send_input_record_to_classifier(record)
        if delay_message:
            task_send_input_record_to_classifier.delay(record)
        else:
            # import pdb;pdb.set_trace()
            task_send_input_record_to_classifier(record)  
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
        # print('Single bibcode')
        # import pdb;pdb.set_trace()
        # record = {'bibcode': message['bibcode'],
        #           'title': message['title'],
        #           'abstract': message['abstract'],
        #           'text': message['title'] + ' ' + message['abstract']
        #           }
        # tasks.task_send_input_record_to_classifier(record)

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


# @app.task(queue="unclassified-queue")
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
    # print("task_send_input_record_to_classifier")
    # import pdb; pdb.set_trace()
    # print(message)
    if not delay_message:
        # import pdb; pdb.set_trace()
        pass

    # import pdb; pdb.set_trace()
    logger.info("Record before classification and thresholding")
    # logger.info("Record before classification and thresholding: {}".format(message))
    # First obtain the raw scores from the classifier model
    # import pdb; pdb.set_trace()
    message = app.score_record(message)

    # Then classify the record based on the raw scores
    # import pdb; pdb.set_trace()
    message = app.classify_record_from_scores(message)
    # print('Collections: ')
    # print(message['collections'])
    logger.info("Record after classification and thresholding: {}".format(message))

    # Write the classifications to output file
    # add_record_to_output_file(message)
    # may have add .async 
    # task_output_results(message)

    # import pdb; pdb.set_trace()
    # Write the new classification to the database
    if delay_message:
        task_index_classified_record.delay(message)
    else:
        task_index_classified_record(message) 
    # task_index_classified_record.delay(message)
    # task_index_classified_record.apply_async(json.dumps(message))
    # task_index_classified_record.apply_async(message)
    # task_index_classified_record.apply_async(message)

    # import pdb; pdb.set_trace()


# @app.task(queue="classify-record")
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

    if not delay_message:
        # import pdb; pdb.set_trace()
        pass
    # print('Indexing Classified Record')
    # import pdb; pdb.set_trace()
    app.index_record(message)
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
