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
         'scix_id': String (19 chars),
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
    logger.debug(f'Message type: {type(message)}')
    logger.debug(f'Message: {message}')

    run_id = app.index_run()
    logger.info('Run ID: {}'.format(run_id))
    operation_step = 'classify'
    output_path = os.path.join(proj_home, 'logs', f'{run_id}_classified.tsv')

    logger.info('Preparing output file: {}'.format(output_path))
    utils.prepare_output_file(output_path,output_format=output_format)
    logger.info('Output file prepared')

    # Delay setting
    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.debug("Delay set for queue messages: {}".format(delay_message))

    request_list = message_to_list(message)

    logger.debug('Request list: {}'.format(request_list))
    for request in request_list:
        logger.info('Request: {}'.format(request))
        record = {'bibcode': request['bibcode'],
                  'scix_id': request['scix_id'],
                  'title': request['title'],
                  'abstract': request['abstract'],
                  'text': request['title'] + ' ' + request['abstract'],
                  'operation_step': operation_step,
                  'run_id': run_id,
                  'output_format': output_format,
                  'override': None,
                  'output_path': output_path
                  }

        # Protobuf takes a list of records
        logger.info("creating output message")
        out_message = list_to_message([record])

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
         'scix_id': String (19 chars),
         'title': String,
         'abstract':String
        }
    :return: no return
    """

    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.info("Delay set for queue messages: {}".format(delay_message))

    fake_data = config.get('FAKE_DATA', False) 

    logger.info("Fake data set for queue messages: {}".format(fake_data))

    record = message_to_list(message)[0]

    if fake_data is False:
        logger.info('Performing Inference')
        input_text = record['title'] + ' ' + record['abstract']
        categories, scores = classifier.batch_assign_SciX_categories([input_text])
        record['categories'] = categories[0]
        record['scores'] = scores[0]
        logger.info('Categories: {}'.format(categories))
        logger.info('Scores: {}'.format(scores))
    else:
        logger.info('Skipping inference - generating fake data')
        record = utils.return_fake_data(record)


    logger.debug('RECORD: {}'.format(record))

    # Decision making based on model scores
    record = app.classify_record_from_scores(record)

    logger.debug("Record after classification and thresholding: {}".format(record))
    logger.debug("Record Type: {}".format(type(record)))

    out_message = list_to_message([record])

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
         'scix_id': String (19 chars),
         'collections': [String],
         'abstract':String,
         'operation_step': String,
         'override': String
        }
    :return: no return
    """

    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.debug("Delay set for queue messages: {}".format(delay_message))

    record = message_to_list(message)[0]
    logger.delay(f"Message: {message}")
    logger.delay(f'Record type: {type(message)}')


    record, success = app.index_record(record)
    if success is True:
        logger.info("Record indexed, outputting results")
        app.add_record_to_output_file(record)
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

    record = message_to_list(message)[0]
    logger.info("Updating Validated Record")
    app.update_validated_records(record)


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
             'scix_id': '....',
             'collections': [....]
            }
    :type: adsmsg.OrcidClaims
    :return: no return
    """

    record = message_to_list(message)[0]
    logger.info('Output results ')
    logger.info(f'Record being output {message}')
    app.add_record_to_output_file(record)
    logger.info(f'Message: {message}')
    # app.forward_message(output_message)




if __name__ == "__main__":
    app.start()
