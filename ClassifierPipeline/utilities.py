import os
import json
import pickle
import zlib
import csv
import re

from google.protobuf.json_format import Parse, MessageToDict
from adsmsg import ClassifyRequestRecordList, ClassifyResponseRecordList

from adsputils import get_date, ADSCelery, u2asc
from adsputils import load_config, setup_logging


proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))

config = load_config(proj_home=proj_home)
logger = setup_logging('utilities.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))


def prepare_output_file(output_path):
    """
    Prepares an output file
    """
    logger.info('Preparing output file - utilities.py')

    header = ['bibcode','title','abstract','run_id','categories','scores','collections','collection_scores','earth_science_adjustment','override']

    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(header)
    logger.info(f'Prepared {output_path} for writing.')


def check_is_allowed_category(categories_list):
    """
    Check if provided categories are in list of approved categories

    Parameters
    ----------
    categories_list : list (required) list of categories to check if allowed

    Returns
    ----------
    True if all categories in approved
    """
    allowed = config.get('ALLOWED_CATEGORIES')

    categories_list = [s.lower() for s in categories_list]

    result = [element in allowed for element in categories_list]

    # Only return True if all True
    if sum(result) == len(result):
        return True
    else:
        return False

def return_fake_data(record):
    """
    Return fake data as a stand in for classifier results.  Use for debugging.

    Parameters
    ----------
    record : dict (required) Dictionary of record information
    records_path : str (required) (default=None) Path to a .csv file of records

    Returns
    -------
    records : dictionary with the following keys: bibcode, text,
                categories, scores, and model information
    """

    
    logger.info('Retruning Fake data')

    record['categories'] = ["Astronomy", "Heliophysics", "Planetary Science", "Earth Science", "NASA-funded Biophysics", "Other Physics", "Other", "Text Garbage"]
    record['scores'] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    record['model'] = {'model' : config['CLASSIFICATION_PRETRAINED_MODEL'],
                       'revision' : config['CLASSIFICATION_PRETRAINED_MODEL_REVISION'],
                       'tokenizer' : config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER']}


    record['postprocessing'] = {'ADDITIONAL_EARTH_SCIENCE_PROCESSING' : config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'],
                                'ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD' : config['ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD'],
                                'CLASSIFICATION_THRESHOLDS' : config['CLASSIFICATION_THRESHOLDS']}


    return record

def list_to_message(input_list):
    """
    Convert a list of dictionaries to a protobuf message'
    """

    message = ClassifyRequestRecordList()

    for item in input_list:
        entry = message.classify_requests.add()
        try:
            entry.bibcode = item.get('bibcode')
        except:
            entry.bibcode = None
        try:
            entry.scix_id = item.get('scix_id')
        except:
            entry.scix_id = None
        try:
            entry.status = item.get('status')
        except:
            entry.status = None
        try:
            entry.title = item.get('title')
        except:
            entry.title = None
        try:
            entry.abstract = item.get('abstract')
        except:
            entry.abstract = None
        try:
            entry.operation_step = item.get('operation_step')
        except:
            entry.operation_step = None
        try:
            entry.run_id = item.get('run_id')
        except:
            entry.run_id = None
        try:
            entry.override = item.get('override')
        except:
            entry.override = None
        try:
            entry.output_path = item.get('output_path')
        except:
            entry.output_path = None
        try:
            entry.scores = item.get('scores')
        except:
            entry.scores = None
        try:
            entry.collections = item.get('collections')
        except:
            entry.collections = None
        try:
            entry.collection_scores = item.get('collection_scores')
        except:
            entry.collection_scores = None

    return message

def list_to_output_message(input_list):
    """
    Convert a list of dictionaries to a protobuf message to return to 
    the Master Pipeline

    """

    message = ClassifyResponseRecordList()

    for item in input_list:
        entry = message.classify_requests.add()
        try:
            entry.bibcode = item.get('bibcode')
        except:
            entry.bibcode = None
        try:
            entry.scix_id = item.get('scix_id')
        except:
            entry.scix_id = None
        try:
            entry.status = item.get('status')
        except:
            entry.status = None
        try:
            entry.status = item.get('collections')
        except:
            entry.status = None

    return message
     

def message_to_list(message):
    """
    Convert a protobuf ClassifyRequestRecordList to a list of dictionaries.
    """

    output_list = []
    request_list = message.classify_requests
    for request in request_list:
        output_list.append(MessageToDict(request))

    # import pdb;pdb.set_trace()

    return output_list


def check_identifier(identifier):
    """
    Determine form of identifier, bibcode or ScixID

    Parameters
    ----------
    identifier - str : either a bibcode or SciX ID - eventually SciX ID will 
    be primary identifier

    Returns
    ----------
    string or None: either 'bibcode' or 'scix_id' or None if fails
    """

    identifier = str(identifier)

    if len(identifier) != 19:
        return None
    scix_match_pattern = r'^scix:[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}$' 
    if re.match(scix_match_pattern, identifier) is not None:
        return 'scix'
    else:
        return 'bibcode'


def extract_records_from_message(message):
    """
    Extract records from a message. Use with json fake messages.

    Parameters
    ----------
    message - protobuff defined serialized message

    Returns
    -------
    record or list of records
    """
    parsed_message = json.loads(message)

    record = parsed_message['classifyRequests'][0]

    return record, parsed_message.copy()

def package_records_to_message(record_list, out_message=None):
    """
    Package records in a message. Use with json fake messages.

    Parameters
    ----------
    list of records - can be single element list

    Returns
    -------
    message - protobuff defined serialized message
    """
    if not_out_message:
        # handle here
        pass
  
    out_message['classifyRequests'] = [record]
    return json.dumps(out_message)

    # if not delay_message:

