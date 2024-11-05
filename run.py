#!/usr/bin/env python
"""
"""

# __author__ = 'tsa'
# __maintainer__ = 'tsa'
# __copyright__ = 'Copyright 2024'
# __version__ = '1.0'
# __email__ = 'ads@cfa.harvard.edu'
# __status__ = 'Production'
# __credit__ = ['T. Allen']
# __license__ = 'MIT'

print('Run.py')
import os
import csv
import time
import json
import argparse
import copy
from ClassifierPipeline.tasks import task_update_record, task_update_validated_records
from ClassifierPipeline import tasks
# from ClassifierPipeline.utilities import check_is_allowed_category
import ClassifierPipeline.utilities as utils

import classifyrecord_pb2
from google.protobuf.json_format import Parse, MessageToDict
from adsmsg import ClassifyRequestRecordList

# # ============================= INITIALIZATION ==================================== #

from adsputils import setup_logging, load_config, get_date
proj_home = os.path.realpath(os.path.dirname(__file__))
# global config
config = load_config(proj_home=proj_home)
logger = setup_logging('run.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', False))


# =============================== FUNCTIONS ======================================= #

def row_to_dictionary(row):
    """
    Convert row of input file to dictionary.
    """

    record = {}
    if utils.check_identifier(row[0]) == 'bibcode':
        record['bibcode'] = row[0]
    elif utils.check_identifier(row[0]) == 'scix_id':
        record['scix_id'] = row[0]
    record['title'] = row[1]
    record['abstract'] = row[2]

    return record


def batch_new_records(records_path, batch_size=500):
    """
    Batch contents of input file and pass to the input_records queue
    """

    batch = []                                                     

    possible_headers = ['bibcode','scixid','scix_id']

    with open(records_path, 'r') as file:
        reader = csv.reader(file)
        
        # Peek at the first row to determine if it's a header
        first_row = next(reader)
        if str(first_row[0]).lower() in possible_headers:
            print("Header detected:", first_row)
        else:
            print("No header found, processing first row as data.")
            batch.append(row_to_dictionary(first_row))  # Add first row to the batch

        for i, row in enumerate(reader, 1):
            batch.append(row_to_dictionary(row))

            if i % batch_size == 0:
                print(f"Processing batch {i // batch_size}")
                message = utils.list_to_message(batch)
                task_update_record.delay(message)
                batch = []  # Clear the batch for the next one

        if batch:
            print("Processing final batch")
            message = utils.list_to_message(batch)
            # new_batch = message_to_list(message)
            # import pdb;pdb.set_trace()
            task_update_record.delay(message)


def records2_fake_protobuf(record):
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

def prepare_records(records_path, operation_step='validate'):
    """
    Takes a path to a .csv file of records and converts each record into a
    dictionary to be sent to the classification queu to be sent to the
    classifier.

    Parameters
    ----------
    records_path : str (required) (default=None) Path to a .csv file of records

    Returns
    -------
    no return
    """
    print('Processing records from: {}'.format(records_path))
    print()


    # Get run_id from filename
    run_id = records_path.split('/')[-1]
    run_id = int(run_id.split('_')[0])

    with open(records_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        headers = next(csv_reader)

        for row in csv_reader:
            record = {}
            if utils.check_identifier(row[0]) == 'bibcode':
                record['bibcode'] = row[0]
            elif utils.check_identifier(row[0]) == 'scix_id':
                record['scix_id'] = row[0]
            record['title'] = row[1]
            record['abstract'] = row[2]
            record['text'] = row[1] + ' ' + row[2]
            record['operation_step'] = operation_step
            record['run_id'] = run_id

            record['override'] = row[9].split(',')
            run_id = row[3]
            # make a check of proper collections
            allowed = utils.check_is_allowed_category(record['override'])
            if allowed:
                record = record2_fake_protobuf(record)
                tasks.task_index_classified_record(record)

            # Records that do not need an override
            # are marked as validated
            tasks.task_update_validated_records(run_id)



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


    if args.records:
        records_path = args.records
        print(records_path)

    if args.validate:
        print("Validating records")
        prepare_records(records_path,operation_step='validate')

    if args.new_records:
        print("Processing new records")
        batch_new_records(records_path)

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
            message = task_update_record(message_json,pipeline='test')


    logger.info("Done - run.py")
