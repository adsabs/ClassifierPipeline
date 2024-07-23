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

import os
import csv
# import sys
import time
# import json
import argparse
# import logging
# import traceback
# import warnings
# from urllib3 import exceptions
# warnings.simplefilter('ignore', exceptions.InsecurePlatformWarning)

# import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from adsputils import get_date
# from adsmsg import OrcidClaims
from ClassifierPipeline import classifier, tasks, app
# from ClassifierPipeline import classifier, tasks
# from ADSOrcid import updater, tasks
# from ADSOrcid.models import ClaimsLog, KeyValue, Records, AuthorInfo

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
app = app.SciXClassifierCelery(
        # app = SciXClassifierCelery(
    "scixclassifier-pipeline",
    proj_home=proj_home,
    local_config=globals().get("local_config", {}),
    )
# app = tasks.app
# app = SciXClassifierCelery()

# =============================== FUNCTIONS ======================================= #


# =============================== MAIN ======================================= #

# To test the classifier
# python run.py -n -r ClassifierPipeline/tests/stub_data/stub_new_records.csv

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
        print("Validating records")
        prepare_records(records_path,validate=True)

    # import pdb;pdb.set_trace()
    if args.new_records:
        print("Processing new records")
        prepare_records(records_path)
        # records = score_records(records_path)

        # for record in records:
            # print("Record: {}".format(record['bibcode']))
            # print("Text: {}".format(record['text']))
            # print("Categories: {}".format(record['categories']))
            # print("Scores: {}".format(record['scores']))
        # records = classify_records_from_scores(records)

    # import pdb;pdb.set_trace
    if args.test:
        # print("Running tests")
        # print('more tests')
        # print('even more')
        # import pdb;pdb.set_trace
        # import pdb;pdb.set_trace
        # logger.info("Running tests")
        logger.debug("Running tests")

        # Remove delay for testing
        delay_message = config.get('DELAY_MESSAGE', True) 

        logger.info("Delay set for queue messages: {}".format(delay_message))

        # Read a protobuf from a
        with open('ClassifierPipeline/tests/stub_data/classifier_request_shorter.json', 'r') as f:
        # with open('ClassifierPipeline/tests/stub_data/classifier_request_short.json', 'r') as f:
            message_json = f.read()
        # with open('ClassifierPipeline/tests/stub_data/classifier_request.json', 'r') as f:
        #     message_json = f.read()
        
        # message = classifyrecord_pb2.ClassifyRequestRecordList()
        # Parse(message_json, message)
        
        # import pdb;pdb.set_trace
        logger.info('Message for testing: {}'.format(message_json))
        # message = app.handle_input_from_master(message)
        if delay_message:
            message = tasks.task_update_record.delay(message_json)
        # message = tasks.task_update_record.delay(message_json)
        else:
            message = tasks.task_update_record(message_json)

        # import pdb;pdb.set_trace

    # print("Done")
    logger.info("Done - run.py")
    # import pdb;pdb.set_trace()
