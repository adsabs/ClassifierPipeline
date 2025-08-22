"""
Utility Functions for SciX Classification Pipeline

This module provides helper functions used throughout the classification pipeline,
including data formatting, file preparation, result scoring, threshold application,
and protobuf message conversions.

Key Functions:
    - Record classification from model scores
    - Output file creation and logging
    - Category validation
    - Fake data generation for debugging
    - Conversion to/from protobuf messages
    - Identifier validation (bibcode vs scix_id)

Dependencies:
    - adsputils for config and logging
    - adsmsg for protobuf message structures
    - protobuf.json_format for conversions
    - csv, json, re for formatting and regex matching
"""
import os
import json
import pickle
import zlib
import csv
import re

from google.protobuf.json_format import Parse, MessageToDict, ParseDict
from adsmsg import ClassifyRequestRecord, ClassifyRequestRecordList, ClassifyResponseRecord, ClassifyResponseRecordList

from adsputils import get_date, ADSCelery, u2asc
from adsputils import load_config, setup_logging


proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))

config = load_config(proj_home=proj_home)
logger = setup_logging('utilities.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))


def classify_record_from_scores(record):
    """
    Post-process a classified record based on configured thresholds.

    Applies classification thresholds from the config file and optionally 
    performs additional Earth Science reclassification logic.

    Parameters
    ----------
    record : dict
        Dictionary containing:
        - bibcode
        - text
        - validate
        - categories (list of str)
        - scores (list of float)
        - model (dict)

    Returns
    -------
    dict
        Updated record with additional fields:
        - collections (list of str): Categories meeting threshold.
        - collection_scores (list of float): Rounded scores for included collections.
    """
    logger.info('Classify Record From Scores')
    logger.info('RECORD: {}'.format(record))
    # Fetch thresholds from config file
    thresholds = config['CLASSIFICATION_THRESHOLDS']
    logger.info(f'Classification Thresholds: {thresholds}')


    scores = record['scores']
    categories = record['categories']

    meet_threshold = [score > threshold for score, threshold in zip(scores, thresholds)]

    # Extra step to check for "Earth Science" articles miscategorized as "Other"
    # This is expected to be less neccessary with improved training data
    if config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'] == 'active':
        logger.info('Additional Earth Science Processing')
        if meet_threshold[categories.index('Other')] is True:
            # If Earth Science score above additional threshold
            if scores[categories.index('Earth Science')] \
                    > config['ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD']:
                meet_threshold[categories.index('Other')] = False
                meet_threshold[categories.index('Earth Science')] = True

    record['collections'] = [category for category, threshold in zip(categories, meet_threshold) if threshold is True]
    record['collection_scores'] = [score for score, threshold in zip(scores, meet_threshold) if threshold is True]
    record['collection_scores'] = [round(score, 2) for score in record['collection_scores']]


    return record



def prepare_output_file(output_path):
    """
    Prepare a tab-delimited output file with predefined classification headers.

    Parameters
    ----------
    output_path : str
        Path where the output file will be created.

    Notes
    -----
    Overwrites existing files with the same name.
    """
    logger.info('Preparing output file - utilities.py')

    header = ['bibcode','scix_id','run_id','title','collections','collection_scores','astronomy_score','heliophysics_score','planetary_science_score','earth_science_score','biology_score','physics_score','other_score','garbage_score','override']

    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(header)
    logger.info(f'Prepared {output_path} for writing.')

def add_record_to_output_file(record):
    """
    Append a classified record to an existing output file.

    Parameters
    ----------
    record : dict
        Dictionary containing classification results. Must include:
        - bibcode
        - scix_id
        - run_id
        - title
        - collections
        - collection_scores
        - scores
        - output_path (str): file path to append data.

    Notes
    -----
    The row is written in tab-delimited format.
    """
    row = [record['bibcode'], record['scix_id'],record['run_id'],record['title'],', '.join(record['collections']), ', '.join(map(str, record['collection_scores'])), round(record['scores'][0],2), round(record['scores'][1],2), round(record['scores'][2],2), round(record['scores'][3],2), round(record['scores'][4],2), round(record['scores'][5],2), round(record['scores'][6],2), round(record['scores'][7],2), '']

    logger.debug(f'Writing {row}')
    with open(record['output_path'], 'a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(row)


def check_is_allowed_category(categories_list):
    """
    Validate that all provided categories are in the allowed categories list.

    Parameters
    ----------
    categories_list : list of str
        Categories to validate.

    Returns
    -------
    bool
        True if all categories are allowed, False otherwise.
    """
    logger.info(f"Cheking allowed categories for {categories_list}")
    allowed = config.get('ALLOWED_CATEGORIES')
    allowed = [s.lower() for s in allowed]

    categories_list = [s.lower() for s in categories_list]

    result = [element in allowed for element in categories_list]

    logger.info(f"Checking allowed categories for (after lowercase) {categories_list}")
    # Only return True if all True
    if sum(result) == len(result):
        return True
    else:
        return False


def check_if_list_single_empty_string(input_list):
    """
    Check whether the input is a list containing only a single empty string.

    Parameters
    ----------
    input_list : list
        List to evaluate.

    Returns
    -------
    bool
        True if input is exactly [''], otherwise False.
    """
    return isinstance(input_list, list) and len(input_list) == 1 and input_list[0] == ''

def return_fake_data(record):
    """
    Populate a record with fake classification data for debugging purposes.

    Parameters
    ----------
    record : dict
        Dictionary to be updated with fake data.

    Returns
    -------
    dict
        Updated record with mock categories, scores, model, and postprocessing info.
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

def filter_allowed_fields(input_dict, allowed_fields=None,response=False):
    """
    Filter a dictionary to retain only fields allowed by protobuf message definitions.

    Parameters
    ----------
    input_dict : dict
        Original dictionary of data.
    allowed_fields : set of str, optional
        Explicitly allowed fields. If None, defaults to allowed fields for
        ClassifyRequestRecord or ClassifyResponseRecord based on `response`.
    response : bool, default False
        Whether to filter for response message fields.

    Returns
    -------
    dict
        Dictionary with only allowed fields.
    """
    if allowed_fields is None:
        if response is False:
            allowed_fields = {'bibcode', 'scix_id', 'status', 'title', 'abstract', 
                            'operation_step', 'run_id', 'override', 'output_path', 
                            'scores', 'collections', 'collection_scores'}
        else:
            allowed_fields = {'bibcode', 'scix_id', 'status', 'collections'}

    return {key: value for key, value in input_dict.items() if key in allowed_fields}


def dict_to_ClassifyRequestRecord(input_dict):
    """
    Convert a dictionary to a ClassifyRequestRecord protobuf message.

    Parameters
    ----------
    input_dict : dict
        Dictionary containing request fields.

    Returns
    -------
    ClassifyRequestRecord
        Parsed protobuf message.
    """
    input_dict = filter_allowed_fields(input_dict)

    request_message = ClassifyRequestRecord()
    message = ParseDict(input_dict, request_message)
    return message

def list_to_ClassifyRequestRecordList(input_list):
    """
    Convert a list of dictionaries to a ClassifyRequestRecordList protobuf message.

    Parameters
    ----------
    input_list : list of dict
        List of request dictionaries.

    Returns
    -------
    ClassifyRequestRecordList
        Parsed protobuf message containing the request list.
    """

    input_list = list(map(lambda d: filter_allowed_fields(d), input_list))

    logger.info(f'Created ClassifyResponseRecord message from list: {input_list}')

    request_list_dict = {
            'classify_requests' : input_list,
            # 'status' : 99
            }

    request_message = ClassifyRequestRecordList()
    message = ParseDict(request_list_dict, request_message)
    return message


def dict_to_ClassifyResponseRecord(input_dict):
    """
    Convert a dictionary to a ClassifyResponseRecord protobuf message.

    Parameters
    ----------
    input_dict : dict
        Dictionary containing response fields.

    Returns
    -------
    ClassifyResponseRecord
        Parsed protobuf message.
    """
    input_dict = filter_allowed_fields(input_dict, response=True)

    request_message = ClassifyResponseRecord()
    message = ParseDict(input_dict, request_message)
    return message

def list_to_ClassifyResponseRecordList(input_list):
    """
    Convert a list of dictionaries to a ClassifyResponseRecordList protobuf message.

    Parameters
    ----------
    input_list : list of dict
        List of response dictionaries.

    Returns
    -------
    ClassifyResponseRecordList
        Parsed protobuf message containing the response list.
    """

    input_list = list(map(lambda d: filter_allowed_fields(d, response=True), input_list))

    response_list_dict = {
            'classifyResponses' : input_list,
            # 'status' : 99
            }

    logger.info(f"Dictionary for Response Message {response_list_dict}")
    response_message = ClassifyResponseRecordList()
    message = ParseDict(response_list_dict, response_message)
    return message


def classifyRequestRecordList_to_list(message):
    """
    Convert a ClassifyRequestRecordList protobuf message into a list of dictionaries.

    Parameters
    ----------
    message : ClassifyRequestRecordList
        Protobuf message containing multiple classify requests.

    Returns
    -------
    list of dict
        List of request dictionaries.
    """

    logger.info(f'Converting message to list: {message}')
    output_list = []
    request_list = message.classify_requests
    for request in request_list:
        logger.info(f'Unpacking request: {request}')
        output_list.append(MessageToDict(request,preserving_proto_field_name=True))

    logger.info(f'Output list from message: {output_list}')

    return output_list


def check_identifier(identifier):
    """
    Determine whether an identifier is a bibcode or SciX ID.

    Parameters
    ----------
    identifier : str
        Candidate identifier string.

    Returns
    -------
    str or None
        'bibcode' if identifier is a bibcode,
        'scix_id' if identifier is a SciX ID,
        None if format is invalid.
    """

    identifier = str(identifier)

    if len(identifier) != 19:
        return None
    scix_match_pattern = r'^scix:[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}$' 
    if re.match(scix_match_pattern, identifier) is not None:
        return 'scix_id'
    else:
        return 'bibcode'



