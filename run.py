#!/usr/bin/env python
"""
"""

# __author__ = 'rca'
# __maintainer__ = 'rca'
# __copyright__ = 'Copyright 2015'
# __version__ = '1.0'
# __email__ = 'ads@cfa.harvard.edu'
# __status__ = 'Production'
# __credit__ = ['J. Elliott']
# __license__ = 'MIT'

print('Run.py')
import os
import csv
# import sys
import time
import json
import argparse
import copy
# import logging
# import traceback
# import warnings
# from urllib3 import exceptions
# warnings.simplefilter('ignore', exceptions.InsecurePlatformWarning)

# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from adsputils import get_date
# from adsmsg import OrcidClaims
# from ClassifierPipeline import classifier, tasks
# from ClassifierPipeline import tasks
# from ClassifierPipeline import app as app_module
from ClassifierPipeline.tasks import task_update_record, task_update_validated_records
from ClassifierPipeline import tasks
# from ClassifierPipeline import classifier, tasks
# from ADSOrcid import updater, tasks
# from ADSOrcid.models import ClaimsLog, KeyValue, Records, AuthorInfo
from ClassifierPipeline.utilities import check_is_allowed_category

import classifyrecord_pb2
from google.protobuf.json_format import Parse

# # ============================= INITIALIZATION ==================================== #

from adsputils import setup_logging, load_config, get_date
proj_home = os.path.realpath(os.path.dirname(__file__))
# global config
config = load_config(proj_home=proj_home)
logger = setup_logging('run.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', False))

# app = app_module.SciXClassifierCelery(
# app = app_module.SciXClassifierCelery(
#         # app = SciXClassifierCelery(
#     "scixclassifier-pipeline",
#     proj_home=proj_home,
#     local_config=globals().get("local_config", {}),
#     )
# app = tasks.app
# app = SciXClassifierCelery()
# logger.info('Loading model and tokenizer')
# MODEL_DICT = app.load_model_and_tokenizer()

# =============================== FUNCTIONS ======================================= #

def record2protobuf(record):
    """
    Take an input dictionary and return a protobuf containing the dictionary

    Parameters
    ----------
    record : dict (required) dictionary with record information

    Returns
    -------
    protobuf
    """

    # message = classifyrecord_pb2.ClassifyRequestRecordList()
    # message['classifyRequests'] = input_list
    # message = json.dumps(message)
    # import pdb;pdb.set_trace()
    with open(config.get('TEST_INPUT_DATA'), 'r') as f:
        message_json = f.read()

    parsed_message = json.loads(message_json)
    request_list = parsed_message['classifyRequests']
    
    out_message = parsed_message.copy()
    out_message['classifyRequests'] = [record] # protobuf is for list of dictionaries
    # import pdb;pdb.set_trace()
    out_message = json.dumps(out_message)

    return out_message

def prepare_records(records_path, validate=True, tsv_output=True):
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
    # Initial input columns: bibcode, title, abstract
    # Corrected input columns:
    # bibcode,title,abstract,categories,scores,collections,collection_scores,earth_science_adjustment,override
    print('Processing records from: {}'.format(records_path))
    print()


    # Get run_id from filename
    run_id = records_path.split('/')[-1]
    run_id = int(run_id.split('_')[0])

    # import pdb;pdb.set_trace()

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
            # make a check of proper collections
            allowed = check_is_allowed_category(record['override'])
            if allowed:
                record = record2protobuf(record)
                tasks.task_index_classified_record(record)

            # Records that do not need an override
            # are marked as validated
            tasks.task_update_validated_records(run_id)



# =============================== MAIN ======================================= #

# To test the classifier
# python run.py -n -r ClassifierPipeline/tests/stub_data/stub_new_records.csv
# import pdb;pdb.set_trace()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process user input.')

    parser.add_argument('-n',
                        '--new_records',
                        dest='new_records',
                        action='store_true',
                        help='Process new records')

    parser.add_argument('-v',
                        '--validate',
                        dest='validate',
                        action='store_true',
                        help='Return list to manually validate classifications')

    parser.add_argument('-r',
                        '--records',
                        dest='records',
                        action='store',
                        help='Path to comma delimited list of new records' +
                             'to process: columns: bibcode, title, abstract')

    parser.add_argument('-t',
                        '--test',
                        dest='test',
                        action='store_true',
                        help='Run tests')



    args = parser.parse_args()

    # import pdb;pdb.set_trace()

    if args.records:
        records_path = args.records
        print(records_path)
        # Open .csv file and read in records
        # Convert records to send to classifier

    # import pdb;pdb.set_trace()
    if args.validate:
        # For testing : 
        # python3 run.py -v -r /app/logs/157_classified_corrected.tsv
        print("Validating records")
        # import pdb;pdb.set_trace()
        prepare_records(records_path,validate=True)

    # import pdb;pdb.set_trace()
    if args.new_records:
        print("Processing new records")
        prepare_records(records_path)

    if args.test:
        logger.debug("Running tests")
        logger.debug("Dev Env")

        # Remove delay for testing
        delay_message = config.get('DELAY_MESSAGE', True) 

        logger.debug("Delay set for queue messages: {}".format(delay_message))
        logger.debug("Config: TEST_INPUT_DATA: {}".format(config.get('TEST_INPUT_DATA')))

        # Read a protobuf from file
        with open(config.get('TEST_INPUT_DATA'), 'r') as f:
            message_json = f.read()
        
        
        logger.debug('Message for testing: {}'.format(message_json))
        if delay_message:
            message = task_update_record.delay(message_json,pipeline='test')
        else: 
            message = task_update_record(message_json)


    logger.info("Done - run.py")
