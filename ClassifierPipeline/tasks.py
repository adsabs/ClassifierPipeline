"""
Classifier Pipeline Celery Tasks

This module defines Celery tasks for managing and orchestrating the classification
pipeline for scientific records. It receives messages from ADS Master Pipeline,
performs classification using a model, updates the database, and sends responses 
back to the Master Pipeline.

The pipeline includes steps to:
    - Receive records
    - Perform classification
    - Index classification results into a database
    - Update output files
    - Validate and forward results to Master Pipeline

Celery Queue:
    - All tasks are currently assigned to the "update-record" queue. Other queues available include: "classify-record", "index-record", "send-record-to-classifier".

Dependencies:
    - adsputils
    - SQLAlchemy for database access
    - Google protobuf for message formatting
    - ClassifierPipeline (Classifier, utilities, models)

Tasks:
    - task_update_record
    - task_send_input_record_to_classifier
    - task_index_classified_record
    - task_message_to_master
    - task_resend_to_master
    - task_update_validated_records
    - task_output_results
"""
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
    Entry point task to receive classification requests from the master.

    Parses the message, generates initial record metadata, and forwards each 
    request to the classifier task.

    Parameters:
        message (ClassifyRequestRecordList): List of classification requests
        pipeline (str): Processing pipeline name (default "classifier")
        output_format (str): Output format for results (default "tsv")
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
    Task to perform classification inference on a record.

    If FAKE_DATA is set in config file, generates fake classifications instead of inference.

    Parameters:
        message (ClassifyRequestRecordList): A single-item list with record data
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
    Task to store classified records into the database.

    Also logs the result and forwards it to the master service if appropriate.

    Parameters:
        message (ClassifyRequestRecordList): Classified record to store
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
        task_resend_to_master(message)
        logger.info(f"Record {record_id} sent to master")
    else:
        logger.info(f"Record {record_id} failed to be indexed")

def out_message(message):
    """
    Helper function to convert and forward a message to the master pipeline.

    Parameters:
        message (dict): Dictionary containing classification result
    """

    out_message = utils.dict_to_ClassifyResponseRecord(message)
    logger.debug(f"Forwarding message to Master - Message: {out_message}")
    app.forward_message(out_message)

@app.task(queue="update-record")
def task_message_to_master(message):
    """
    Task to send the classified record(s) back to the master service.

    Parameters:
        message (dict or list): A single record or list of classified records
    """

    if isinstance(message, dict):
        out_message(message)
    if isinstance(message, list):
        for msg in message:
            out_message(message)

# @app.task(queue="classify-record")
@app.task(queue="update-record")
def task_resend_to_master(message):
    """
    Task to re-send a classified record to Master Pipeline based on bibcode, scix_id, or run_id.

    Parameters:
        message (ClassifyRequestRecordList): Message containing one record
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
    Task to mark a batch of records as validated in the database by run_id.

    Parameters:
        message (ClassifyRequestRecordList): Message with run_id field
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
    Task to append classified record results to the output file.

    Parameters:
        message (dict): Record containing bibcode, scix_id, and collections
    """

    record = utils.message_to_list(message)[0]
    logger.debug('Output results ')
    logger.debug(f'Record being output {message}')
    app.add_record_to_output_file(record)




if __name__ == "__main__":
    app.start()
