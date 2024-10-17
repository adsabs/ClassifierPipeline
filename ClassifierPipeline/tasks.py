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
# def task_handle_input_from_master(message):
# def task_update_record(message,pipeline, tsv_output=True):
def task_update_record(message,test_message=False, tsv_output=True):
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
    # if isinstance(message, classifyrecord_pb2.ClassifyRequestRecordList):
    # if message is type(list):
    logger.info('********************************************')
    logger.info('*** task_update_record ***')
    logger.info(f'Message type: {type(message)}')
    logger.info(f'Message: {message}')
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
    if test_message is True:
        logger.info("Test message being used")
        parsed_message = json.loads(message)
        request_list = parsed_message['classifyRequests']
    else:
        logger.info(f'Message contents: {message.classify_requests}')
        # open_message = classifyrecord_pb2.ClassifyRequestRecordList()
        # open_message.ParseFromString(message)
        # request_list = open_message.classify_requests
        request_list = message.classify_requests
        logger.info(f"Request List: {request_list}")
        request_list = [MessageToDict(request) for request in request_list]
        logger.info(f"Request List (dictionaries): {request_list}")
        
        # Needed until protobuff defines all processing data
        with open(config.get('TEST_INPUT_DATA'), 'r') as f:
            message_json = f.read()
            parsed_message = json.loads(message_json)

    # request_list, out_message = utils.extract_records_from_message(message)


    # for request in message.classify_requests:
    logger.info('Request list: {}'.format(request_list))
    for request in request_list:
        logger.info('Request: {}'.format(request))
        # request = MessageToDict(request)
        # logger.info('Request as Dictionary: {}'.format(request))
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
        logger.info("creating output message")
        out_message = parsed_message.copy()
        out_message['classifyRequests'] = [record] # protobuf is for list of dictionaries
        # import pdb;pdb.set_trace()
        out_message = json.dumps(out_message)
        # out_message = utils.package_records_to_message(record_list, out_message=out_message)
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
        logger.info("Record indexed, outputting results")
        task_output_results(record)
    else:
        logger.info("Record failed to be indexed")
    # import pdb; pdb.set_trace()
    # pass

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

    # print('Updating Validated Records')
    logger.info("Updating Validated Record")
    app.update_validated_records(message)
    # import pdb; pdb.set_trace()
    # pass


# @app.task(queue="output-results")
# @app.task(queue="classify-record")
@app.task(queue="update-record")
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
    logger.info('Output results ')
    logger.info(message)
    app.add_record_to_output_file(message)



if __name__ == "__main__":
    app.start()
