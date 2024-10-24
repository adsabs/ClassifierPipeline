import sys
import os
import json
import adsputils
from adsputils import ADSCelery
import ClassifierPipeline.app as app_module
import ClassifierPipeline.utilities as utils
from ClassifierPipeline.classifier import Classifier
from adsputils import load_config, setup_logging
from kombu import Queue
import classifyrecord_pb2
from google.protobuf.json_format import Parse, MessageToDict

from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
# ============================= INITIALIZATION ==================================== #

proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
app = app_module.SciXClassifierCelery(
    "classifier-pipeline",
    proj_home=proj_home,
    local_config=globals().get("local_config", {}),
)
config = load_config(proj_home=proj_home)
logger = setup_logging('tasks.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))

app.conf.CELERY_QUEUES = (
    Queue("update-record", app.exchange, routing_key="update-record"),
    # Queue("classify-record", app.exchange, routing_key="classify-record"),
    # Queue("classify-record", app.exchange, routing_key="index-record")
)

classifier = Classifier()

# ============================= TASKS ============================================= #

@app.task(queue="update-record")
def task_update_record(message,pipeline='classifier', output_format='tsv'):
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

    # Always pass a list of records, even just a list of one record
    logger.info(f'Message type: {type(message)}')
    logger.info(f'Message: {message}')
    output_path = ''

    run_id = app.index_run()
    logger.info('Run ID: {}'.format(run_id))
    operation_step = 'classify'
    if output_format == 'tsv':
        output_path = os.path.join(proj_home, 'logs', f'{run_id}_classified.tsv')
    else:
        output_path = os.path.join(proj_home, 'logs', f'{run_id}_classified.csv')

    logger.info('Preparing output file: {}'.format(output_path))
    utils.prepare_output_file(output_path,output_format=output_format)
    logger.info('Output file prepared')

    # Delay setting
    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.info("Delay set for queue messages: {}".format(delay_message))

    # Needed for the test message - test aspect should be removed
    if pipeline =='classifier':
        request_list = message.classify_requests
        logger.info(f"Request List: {request_list}")
        request_list = [MessageToDict(request) for request in request_list]
        logger.info(f"Request List (dictionaries): {request_list}")
        
        # Needed until protobuff defines all processing data
        with open(config.get('TEST_INPUT_DATA'), 'r') as f:
            message_json = f.read()
            parsed_message = json.loads(message_json)
    else:
        logger.info("Test message being used")
        parsed_message = json.loads(message)
        request_list = parsed_message['classifyRequests']


    # for request in message.classify_requests:
    logger.info('Request list: {}'.format(request_list))
    for request in request_list:
        logger.info('Request: {}'.format(request))
        record = {'bibcode': request['bibcode'],
                  'title': request['title'],
                  'abstract': request['abstract'],
                  'text': request['title'] + ' ' + request['abstract'],
                  'operation_step': operation_step,
                  'run_id': run_id,
                  'output_format': output_format,
                  'override': None,
                  'output_path': output_path
                  }

        # only needed for test protobuf - remove after updating protobuf
        logger.info("creating output message")
        out_message = parsed_message.copy()
        out_message['classifyRequests'] = [record] # protobuf is for list of dictionaries
        out_message = json.dumps(out_message)

        logger.info('Output Record type: {}'.format(type(out_message)))
        logger.info('Output Record: {}'.format(out_message))
        if delay_message:
            logger.info('Using delay')
            task_send_input_record_to_classifier.delay(out_message)
        else:
            task_send_input_record_to_classifier(out_message)  
            

# @app.task(queue="unclassified-queue")
@app.task(queue="update-record")
# @app.task(queue="classify-record")
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

    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.info("Delay set for queue messages: {}".format(delay_message))

    fake_data = config.get('FAKE_DATA', False) 

    logger.info("Fake data set for queue messages: {}".format(fake_data))


    # Needed until protobuf updated
    parsed_message = json.loads(message)

    record = parsed_message['classifyRequests'][0]

    logger.info("Parsed message")
    logger.info(parsed_message)

    if fake_data is False:
        logger.info('Performing Inference')
        categories, scores = classifier.batch_assign_SciX_categories([record['text']])
        record['categories'] = categories[0]
        record['scores'] = scores[0]
        logger.info('Categories: {}'.format(categories))
        logger.info('Scores: {}'.format(scores))
    else:
        logger.info('Skipping inference - generating fake data')
        record = utils.return_fake_data(record)

    # Add model and classification information to record
    record['model'] = config['CLASSIFICATION_PRETRAINED_MODEL'],
    record['revision'] =  config['CLASSIFICATION_PRETRAINED_MODEL_REVISION'],
    record['tokenizer'] = config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER']

    record['ADDITIONAL_EARTH_SCIENCE_PROCESSING'] = config['ADDITIONAL_EARTH_SCIENCE_PROCESSING']
    record['ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD'] = config['ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD']
    record['CLASSIFICATION_THRESHOLDS'] = config['CLASSIFICATION_THRESHOLDS']



    logger.info('RECORD: {}'.format(record))

    # Decision making based on model scores
    record = app.classify_record_from_scores(record)

    logger.info("Record after classification and thresholding: {}".format(record))
    logger.info("Record Type: {}".format(type(record)))

    out_message = parsed_message.copy()
    out_message['classifyRequests'] = [record]
    out_message = json.dumps(out_message)

    # Write the new classification to the database
    if delay_message:
        task_index_classified_record.delay(out_message)
    else:
        task_index_classified_record(out_message) 




# @app.task(queue="classify-record")
@app.task(queue="update-record")
def task_index_classified_record(message):
    """
    Update the database with the new classification

    :param message: contains the message inside the packet
        {
         'bibcode': String (19 chars),
         'collections': [String],
         'abstract':String,
         'operation_step': String,
         'override': String
        }
    :return: no return
    """

    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.info("Delay set for queue messages: {}".format(delay_message))

    logger.info("Unpacking message for indexing")
    message = json.loads(message)
    record = message['classifyRequests'][0]
    logger.info(message)
    logger.info('Record type: {}'.format(type(message)))


    record, success = app.index_record(record)
    if success is True:
        logger.info("Record indexed, outputting results")
        task_output_results(record)
    else:
        logger.info("Record failed to be indexed")


# @app.task(queue="classify-record")
# @app.task(queue="update-record")
def task_update_validated_records(message):
    """
    Update all records that have been validated that have same run_id

    :param message: contains the message inside the packet
        {
         'run_id': Boolean,
        }
    """

    logger.info("Updating Validated Record")
    app.update_validated_records(message)


# @app.task(queue="output-results")
# @app.task(queue="classify-record")
@app.task(queue="update-record")
def task_output_results(message):
    """
    This worker will forward results to the outside
    exchange (typically an ADSMasterPipeline) to be
    incorporated into the storage
    Also, updates output file with classified record

    :param msg: contains the bibcode and the collections:

            {'bibcode': '....',
             'collections': [....]
            }
    :type: adsmsg.OrcidClaims
    :return: no return
    """
    logger.info('Output results ')
    logger.info(message)
    app.add_record_to_output_file(message)
    logger.info(f'Message: {message}')
    # app.forward_message(output_message)




if __name__ == "__main__":
    app.start()
