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
import atexit

from google.protobuf.json_format import Parse, MessageToDict, ParseDict
from adsmsg import ClassifyRequestRecord, ClassifyRequestRecordList, ClassifyResponseRecord, ClassifyResponseRecordList

from adsputils import get_date, ADSCelery, u2asc
from adsputils import load_config, setup_logging


proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))

config = load_config(proj_home=proj_home)
logger = setup_logging('utilities.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))

_OUTPUT_ROW_BUFFERS = {}
_OUTPUT_BUFFER_FLUSH_EVERY = 25
_OUTPUT_FLUSH_REGISTERED = False

OUTPUT_HEADER = [
    'bibcode',
    'scix_id',
    'run_id',
    'title',
    'collections',
    'collection_scores',
    'astrophysics_score',
    'heliophysics_score',
    'planetary_science_score',
    'astronomy_score',
    'earth_science_score',
    'biology_score',
    'physics_score',
    'other_score',
    'general_score',
    'garbage_score',
    'gross_collection',
    'override',
]


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
    logger.debug('Classify Record From Scores')
    logger.debug('RECORD: {}'.format(record))
    # Fetch thresholds from config file
    thresholds = config['CLASSIFICATION_THRESHOLDS']
    logger.debug(f'Classification Thresholds: {thresholds}')


    scores = record['scores']
    categories = record['categories']

    meet_threshold = [score > threshold for score, threshold in zip(scores, thresholds)]

    # Extra step to check for "Earth Science" articles miscategorized as "Other"
    # This is expected to be less neccessary with improved training data
    if config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'] == 'active':
        logger.debug('Additional Earth Science Processing')
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
    global _OUTPUT_FLUSH_REGISTERED

    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(OUTPUT_HEADER)
    _OUTPUT_ROW_BUFFERS.pop(output_path, None)
    if not _OUTPUT_FLUSH_REGISTERED:
        atexit.register(flush_output_file)
        _OUTPUT_FLUSH_REGISTERED = True
    logger.info(f'Prepared {output_path} for writing.')


def flush_output_file(output_path=None):
    if output_path is not None:
        buffered_rows = _OUTPUT_ROW_BUFFERS.get(output_path, [])
        if not buffered_rows:
            return
        with open(output_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerows(buffered_rows)
        _OUTPUT_ROW_BUFFERS[output_path] = []
        return

    for path in list(_OUTPUT_ROW_BUFFERS.keys()):
        flush_output_file(path)


def reset_output_buffers_for_tests():
    _OUTPUT_ROW_BUFFERS.clear()

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
    row = build_output_row(record)

    logger.debug(f'Writing {row}')
    output_path = record['output_path']
    _OUTPUT_ROW_BUFFERS.setdefault(output_path, []).append(row)
    if len(_OUTPUT_ROW_BUFFERS[output_path]) >= _OUTPUT_BUFFER_FLUSH_EVERY:
        flush_output_file(output_path)


def _safe_score(scores, index):
    try:
        return round(float(scores[index]), 2)
    except (IndexError, TypeError, ValueError):
        logger.debug(f"Unable to parse score at index {index} from payload: {scores!r}")
        return 0.0


def _derive_summary_scores(scores):
    astrophysics_score = _safe_score(scores, 0)
    heliophysics_score = _safe_score(scores, 1)
    planetary_science_score = _safe_score(scores, 2)
    earth_science_score = _safe_score(scores, 3)
    biology_score = _safe_score(scores, 4)
    physics_score = _safe_score(scores, 5)
    other_score = _safe_score(scores, 6)
    garbage_score = _safe_score(scores, 7)

    astronomy_score = round(max(astrophysics_score, heliophysics_score, planetary_science_score), 2)
    general_score = round(max(earth_science_score, biology_score, other_score), 2)
    gross_collection = max(
        [
            ("astronomy", astronomy_score),
            ("physics", physics_score),
            ("general", general_score),
        ],
        key=lambda item: item[1],
    )[0]

    return {
        "astrophysics_score": astrophysics_score,
        "heliophysics_score": heliophysics_score,
        "planetary_science_score": planetary_science_score,
        "astronomy_score": astronomy_score,
        "earth_science_score": earth_science_score,
        "biology_score": biology_score,
        "physics_score": physics_score,
        "other_score": other_score,
        "general_score": general_score,
        "garbage_score": garbage_score,
        "gross_collection": gross_collection,
    }


def build_output_row(record):
    scores = _derive_summary_scores(record.get('scores', []))
    return [
        record.get('bibcode') or '',
        record.get('scix_id') or '',
        record.get('run_id') or '',
        record.get('title') or '',
        ', '.join(record.get('collections') or []),
        ', '.join(map(str, record.get('collection_scores') or [])),
        scores['astrophysics_score'],
        scores['heliophysics_score'],
        scores['planetary_science_score'],
        scores['astronomy_score'],
        scores['earth_science_score'],
        scores['biology_score'],
        scores['physics_score'],
        scores['other_score'],
        scores['general_score'],
        scores['garbage_score'],
        scores['gross_collection'],
        '',
    ]


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
    logger.debug(f"Cheking allowed categories for {categories_list}")
    allowed = config.get('ALLOWED_CATEGORIES')
    allowed = [s.lower() for s in allowed]

    categories_list = [s.lower() for s in categories_list]

    result = [element in allowed for element in categories_list]

    logger.debug(f"Checking allowed categories for (after lowercase) {categories_list}")
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

    
    logger.debug('Retruning Fake data')

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
                            'output_prepared',
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

    logger.debug(f'Created ClassifyResponseRecord message from list: {input_list}')

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

    logger.debug(f"Dictionary for Response Message {response_list_dict}")
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

    logger.debug(f'Converting message to list: {message}')
    output_list = []
    request_list = message.classify_requests
    for request in request_list:
        logger.debug(f'Unpacking request: {request}')
        output_list.append(MessageToDict(request,preserving_proto_field_name=True))

    logger.debug(f'Output list from message: {output_list}')

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
