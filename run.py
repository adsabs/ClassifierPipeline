#!/usr/bin/env python
"""
CLI utilities for batching, validating, and (re)sending classification records.

This script provides three primary operations:

1) Ingest "new records" from a CSV and enqueue them in batches to the
   classification pipeline.

2) Prepare a validation pass from a tab-delimited file produced by the pipeline,
   sending each record to the indexer with `operation_step='validate'`.

3) Resend previously processed records (by bibcode, SciX ID, or run_id) to the
   master pipeline.

Typical usage
-------------
# Process new records from a CSV (comma-delimited: bibcode|title|abstract)
python run.py -n -r path/to/new_records.csv

# Build "validate" requests from a TSV output file
python run.py -v -r path/to/classifier_output.tsv

# Resend a single record by bibcode
python run.py -s -b 2012ApJ...760...70A

# Resend a single record by SciX ID
python run.py -s -x scix:abcd-1234-ef56

# Resend all records by run_id
python run.py -s -i 2024-08-01T12:00:00Z

Input formats
-------------
New records CSV (-n):
  Columns: [bibcode_or_scix_id, title, abstract]
  Optionally with a header row (first column equal to one of: bibcode, scixid, scix_id).

Validation TSV (-v):
  Tab-delimited with at least the following columns by index:
    0: bibcode or blank
    1: scix_id or blank
    2: run_id
    3: title
   14: override (comma-separated list of categories)

Notes
-----
- All queue-submitting functions delegate to Celery tasks in `ClassifierPipeline.tasks`.
- Protobuf conversions are handled by helper functions in `ClassifierPipeline.utilities`.
"""

# __author__ = 'tsa'
# __maintainer__ = 'tsa'
# __copyright__ = 'Copyright 2024'
# __version__ = '1.0'
# __email__ = 'ads@cfa.harvard.edu'
# __status__ = 'Production'
# __credit__ = ['T. Allen']
# __license__ = 'MIT'

import os
import csv
import time
import json
import argparse
import copy
from ClassifierPipeline.tasks import task_update_record, task_update_validated_records, task_index_classified_record, task_resend_to_master
# import ClassifierPipeline.tasks as tasks
# from ClassifierPipeline.utilities import check_is_allowed_category
import ClassifierPipeline.utilities as utils

# import classifyrecord_pb2
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
    Convert a single CSV row into a minimal request dictionary.

    Parameters
    ----------
    row : list[str]
        A row with at least three elements:
        [identifier, title, abstract], where `identifier` is either a bibcode
        or a SciX ID (`scix:xxxx-xxxx-xxxx`).

    Returns
    -------
    dict
        Dictionary with keys:
          - 'bibcode' or 'scix_id' (detected)
          - 'title'
          - 'abstract'
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
    Read a CSV of new records and enqueue messages in batches.

    Parameters
    ----------
    records_path : str
        Path to the comma-delimited CSV containing new records. The file may
        optionally include a header row (first column equal to 'bibcode',
        'scixid', or 'scix_id').
    batch_size : int, default 500
        Number of rows per message batch.

    Side Effects
    ------------
    Submits each batch to Celery via `task_update_record.delay(message)`, where
    `message` is a `ClassifyRequestRecordList` protobuf built by
    `utils.list_to_ClassifyRequestRecordList`.
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
                # message = utils.list_to_message(batch)
                message = utils.list_to_ClassifyRequestRecordList(batch)
                task_update_record.delay(message)
                batch = []  # Clear the batch for the next one

        if batch:
            print("Processing final batch")
            # message = utils.list_to_message(batch)
            message = utils.list_to_ClassifyRequestRecordList(batch)
            # new_batch = message_to_list(message)
            # import pdb;pdb.set_trace()
            task_update_record.delay(message)


def records2_fake_protobuf(record):
    """
    Wrap a single record dict into a fake serialized request list message (JSON).

    This uses the JSON fixture at `config['TEST_INPUT_DATA']` as a template and
    replaces its 'classifyRequests' with a single-element list containing `record`.

    Parameters
    ----------
    record : dict
        Minimal record fields to embed (e.g., bibcode/scix_id, title, etc.).

    Returns
    -------
    str
        JSON string that mimics the serialized request list protobuf payload.
    """

    with open(config.get('TEST_INPUT_DATA'), 'r') as f:
        message_json = f.read()

    parsed_message = json.loads(message_json)
    request_list = parsed_message['classifyRequests']

    out_message = parsed_message.copy()
    out_message['classifyRequests'] = [record] # protobuf is for list of dictionaries
    out_message = json.dumps(out_message)

    return out_message

def prepare_records(records_path, operation_step='validate'):
    """
    Produce validation/index messages from a TSV produced by the classifier.

    Reads a tab-delimited file and for each row builds a minimal record dict,
    setting `operation_step` (default 'validate'), and then sends it to the
    indexer via `task_index_classified_record`.

    Parameters
    ----------
    records_path : str
        Path to the TSV file.
    operation_step : str, default 'validate'
        Operation step to attach to each record.

    Expected Columns (0-indexed)
    ----------------------------
    0 : bibcode (or blank)
    1 : scix_id (or blank)
    2 : run_id
    3 : title
    14: override (comma-separated categories)
    """
    print(f'Processing records from: {records_path}')
    print()

    with open(records_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        headers = next(csv_reader)

        for row in csv_reader:
            print(f'Processing row: {row}')
            record = {}
            if utils.check_identifier(row[0]) == 'bibcode':
                record['bibcode'] = row[0]
            elif utils.check_identifier(row[1]) == 'scix_id':
                record['scix_id'] = row[1]
            record['run_id'] = row[2]
            record['title'] = row[3]
            record['operation_step'] = operation_step

            record['override'] = row[14].split(',')
            print(f'validating record: {record}')
            logger.info(f'validating record: {record}')
            message = utils.list_to_ClassifyRequestRecordList([record])
            task_index_classified_record(message)



# =============================== MAIN ======================================= #

# To test the classifier
# python run.py -n -r ClassifierPipeline/tests/stub_data/stub_new_records.csv

if __name__ == '__main__':

    print('Run.py - parsing input')
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

    parser.add_argument('-s',
                        '--resend',
                        dest='resend',
                        action='store_true',
                        help='Resend results to Master Pipeline')

    parser.add_argument('-b',
                        '--bibcode',
                        dest='bibcode',
                        action='store',
                        help='Bibcode of record to resend')

    parser.add_argument('-x',
                        '--scix_id',
                        dest='scix_id',
                        action='store',
                        help='Scix_id of record to resend')

    parser.add_argument('-i',
                        '--run_id',
                        dest='run_id',
                        action='store',
                        help='Run_id of record batch to resend')


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

    if args.bibcode:
        print('Resending bibcode')
        bibcode = args.bibcode

    if args.resend:

        if args.bibcode:
            print(f'Resending bibcode {args.bibcode}')
            record = {'bibcode' : args.bibcode}
        if args.scix_id:
            print(f'Resending scix_id {args.scix_id}')
            record = {'scix_id' : args.scix_id}
        if args.run_id:
            print(f'Resending run_id {args.run_id}')
            record = {'run_id' : args.run_id}

        message = utils.list_to_ClassifyRequestRecordList([record])
        task_resend_to_master(message)

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


        parsed_message = json.loads(message_json)
        request_list = parsed_message['classifyRequests']
        # Add some fields
        print()
        print('Requests')
        for index, request in enumerate(request_list):
            print()
            # request.update({'operation_step' : 'classify'})
            request['scix_id'] = f'scix:0000-0000-000{index}'
            request['operation_step'] = 'classify'
            request['override'] = ['Astronomy', 'Earth Science']
            print(request)

        print()
        print('Input Dict')
        print(request_list[0])
        logger.debug('Message for testing: {}'.format(message_json))
        # message = utils.dict_to_ClassifyRequestRecord(request_list[0])
        message = utils.list_to_ClassifyRequestRecordList(request_list)
        print(message)
        import pdb;pdb.set_trace()
        test_list = utils.classifyRequestRecordList_to_list(message)
        print('Test list - Return')
        print(test_list[0])

        import pdb;pdb.set_trace()
        if delay_message:
            message = task_update_record.delay(message)#,pipeline='test')
        else:
            message = task_update_record(message,pipeline='test')


    logger.info("Done - run.py")
