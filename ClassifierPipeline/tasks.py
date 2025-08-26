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
from google.protobuf.json_format import Parse, MessageToDict
from adsmsg import ClassifyRequestRecord, ClassifyRequestRecordList, ClassifyResponseRecord, ClassifyResponseRecordList

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
    All input is a classifyRequestRecordList even for a single record

    :param message: contains the message inside the packet
        {
         'bibcode': String (19 chars),
         'scix_id': String (19 chars),
         'title': String,
         'abstract':String
        }
    :return: no return

    :passes to queue: message:
        {
          'bibcode': String (19 chars),
          'scix_id': String (19 chars),
          'title': string,
          'abstract': string,
          'text': string,
          'operation_step': string,
          'run_id': int,
          'output_format': string,
          'override': 
          'output_path': string
        }
    """

    logger.debug(f'Message type: {type(message)}')
    logger.debug(f'Message: {message}')

    run_id = app.index_run()
    logger.info('Run ID: {}'.format(run_id))

    request_list = utils.classifyRequestRecordList_to_list(message)

    if 'operation_step' in request_list[0]:
        operation_step = request_list[0]['operation_step']
    else:
        operation_step = config.get('OPERATION_STEP', 'classify_verify')

    if 'output_path' in request_list[0]:
        try:
            filename = request_list[0]['output_path']
            filename = filename.split('/')[-1]
        except:
            filename = request_list[0]['output_path']
    else:
        filename = ''

    output_path = os.path.join(proj_home, 'logs', f'{filename}_{run_id}_classified.tsv')

    utils.prepare_output_file(output_path)
    logger.info('Prepared output file: {}'.format(output_path))

    # Delay setting for testing
    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.debug("Delay set for queue messages: {}".format(delay_message))


    logger.debug('Request list: {}'.format(request_list))
    for request in request_list:
        logger.debug('Request: {}'.format(request))
        record_bibcode = None
        record_scix_id = None
        if 'bibcode' in request:
            record_bibcode = request['bibcode']
        if 'scix_id' in request:
            record_scix_id = request['scix_id']
        if 'title' in request:
            record_title = request['title']
        else:
            record_title = "None"
        if 'abstract' in request:
            record_abstract = request['abstract']
        else:
            record_abstract = "None"
        record = {'bibcode': record_bibcode,
                  'scix_id': record_scix_id,
                  'title': record_title,
                  'abstract': record_abstract,
                  'text': record_title + ' ' + record_abstract,
                  'operation_step': operation_step,
                  'run_id': run_id,
                  'output_format': output_format,
                  'override': None,
                  'output_path': output_path
                  }

        # Protobuf takes a list of records
        logger.debug("creating output message")
        logger.debug(f"Record {record}")
        out_message = utils.list_to_ClassifyRequestRecordList([record])

        logger.debug('Output Record type: {}'.format(type(out_message)))
        logger.debug('Output Record: {}'.format(out_message))
        if delay_message:
            logger.debug('Using delay')
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
          'scix_id': String (19 chars),
          'title': string,
          'abstract': string,
          'text': string,
          'operation_step': string,
          'run_id': int,
          'output_format': string,
          'override': 
          'output_path': string
        }
    :return: no return: passes message packet to queue
    """

    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.debug("Delay set for queue messages: {}".format(delay_message))

    fake_data = config.get('FAKE_DATA', False) 

    logger.debug("Fake data set for queue messages: {}".format(fake_data))

    record = utils.classifyRequestRecordList_to_list(message)[0]

    if fake_data is False:
        logger.debug('Performing Inference')
        input_text = record['title'] + ' ' + record['abstract']
        categories, scores = classifier.batch_score_SciX_categories([input_text])
        record['categories'] = categories[0]
        record['scores'] = scores[0]
        logger.debug('Categories: {}'.format(categories))
        logger.debug('Allowed Categories: {}'.format(config['ALLOWED_CATEGORIES']))
        logger.debug('Scores: {}'.format(scores))
    else:
        logger.info('Skipping inference - generating fake data')
        record = utils.return_fake_data(record)


    logger.debug('RECORD: {}'.format(record))

    # Decision making based on model scores
    record = utils.classify_record_from_scores(record)

    logger.debug("Record after classification and thresholding: {}".format(record))
    logger.debug("Record Type: {}".format(type(record)))

    out_message = utils.list_to_ClassifyRequestRecordList([record])

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
         'scix_id': String (19 chars),
         'collections': [String],
         'abstract':String,
         'operation_step': String,
         'override': String
          'title': string,
          'abstract': string,
          'text': string,
          'operation_step': string,
          'run_id': int,
          'output_format': string,
          'override': 
          'output_path': string
        }
    :return: no return
    """

    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.debug("Delay set for queue messages: {}".format(delay_message))

    record = utils.classifyRequestRecordList_to_list(message)[0]
    logger.debug(f"Record: {record}")
    logger.debug(f'Record type: {type(message)}')

    record_id = None
    if 'scix_id' in record:
        record_id = record['scix_id']
    if 'bibcode' in record:
        record_id = record['bibcode']

    record, success = app.index_record(record)
    logger.debug(f'Record: {record}, Success: {success}')
    if success == "record_indexed":
        if record['operation_step'] == 'classify_verify':
            logger.info(f"Record {record_id} indexed")
            utils.add_record_to_output_file(record)
        if record['operation_step'] == 'classify':
            logger.info(f"Record {record_id} indexed")
            utils.add_record_to_output_file(record)
            message = utils.list_to_ClassifyRequestRecordList([record])
            task_resend_to_master(message)
            logger.info(f"Record {record_id} sent to master")

    elif success == "record_validated":
        message = utils.list_to_ClassifyRequestRecordList([record])
        task_message_to_master(message)
        logger.info(f"Record {record_id} sent to master")
    else:
        logger.info(f"Record {record_id} failed to be indexed")

@app.task(queue="update-record")
def task_message_to_master(message):
    """
    Return classified record to Master Pipeline.
    
    :param message: contains the message inside the packet
        {
         'bibcode': String (19 chars),
         'scix_id': String (19 chars),
         'collections': [String],
         'status': int
        }
    :return: no return
    """
    if isinstance(message, dict):
        out_message = utils.dict_to_ClassifyResponseRecord(message)
        logger.debug(f"Forwarding message to Master - Message: {out_message}")
        app.forward_message(out_message)
    if isinstance(message, list):
        for msg in message:
            out_message = utils.dict_to_ClassifyResponseRecord(msg)
            logger.debug(f"Forwarding message to Master - Message: {out_message}")
            app.forward_message(out_message)

# @app.task(queue="classify-record")
@app.task(queue="update-record")
def task_resend_to_master(message):
    """
    Resend records to master based on bibcode, scix_id or run_id
    """
    logger.info(f"Resending records to master")

    request_list = utils.classifyRequestRecordList_to_list(message)

    logger.debug('Request list: {}'.format(request_list))
    for request in request_list:
        logger.info('Request: {}'.format(request))

        if 'bibcode' in request:
            record_id = request['bibcode']
            record_list = app.query_final_collection_table(bibcode=request['bibcode'])
        elif 'scix_id' in request:
            record_list = app.query_final_collection_table(scix_id=request['scix_id'])
        elif 'run_id' in request:
            record_list = app.query_final_collection_table(run_id=request['run_id'])

        for record in record_list:
            record_id = None
            if 'scix_id' in record:
                record_id = record['scix_id']
            if 'bibcode' in record:
                record_id = record['bibcode']
            logger.info(f"Sending record {record_id} to master")
            task_message_to_master(record)


# @app.task(queue="classify-record")
@app.task(queue="update-record")
def task_update_validated_records(message):
    """
    Update all records that have been validated that have same run_id

    :param message: contains the message inside the packet
        {
         'run_id': Boolean,
        }
    """

    logger.info(f"Updating validated records")
    record = utils.classifyRequestRecordList_to_list(message)[0]
    record_list, success_list = app.update_validated_records(record['run_id'])
    for record, success in zip(record_list, success_list):
        logger.debug(f"Record: {record}")
        logger.debug(f"Success: {success}")
        if success == "success":
            record_id = None
            if 'scix_id' in record:
                record_id = record['scix_id']
            if 'bibcode' in record:
                record_id = record['bibcode']
            logger.info(f"Sending record {record_id} to master")
            task_message_to_master(record)


# @app.task(queue="output-results")
# @app.task(queue="classify-record")
@app.task(queue="update-record")
def task_output_results(message):
    """
    Updates output file with classified record

    :param msg: contains the bibcode and the collections:

            {'bibcode': '....',
             'scix_id': '....',
             'collections': [....]
            }
    :return: no return
    """

    record = utils.message_to_list(message)[0]
    logger.debug('Output results ')
    logger.debug(f'Record being output {message}')
    app.add_record_to_output_file(record)




if __name__ == "__main__":
    app.start()
