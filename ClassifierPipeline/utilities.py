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
    Classify a record after it has been scored. 

    Parameters
    ----------
    record : dictionary (required) (default=None) Dictionary with the following
        keys: bibcode, text, validate, categories, scores, and model information

    Returns
    -------
    record : dictionary with the following keys: bibcode, text, validate, categories,
        scores, model information, and Collections
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

    # Append collections to record
    record['collections'] = [category for category, threshold in zip(categories, meet_threshold) if threshold is True]
    record['collection_scores'] = [score for score, threshold in zip(scores, meet_threshold) if threshold is True]
    record['collection_scores'] = [round(score, 2) for score in record['collection_scores']]


    return record



def prepare_output_file(output_path):
    """
    Prepares an output file
    """
    logger.info('Preparing output file - utilities.py')

    header = ['bibcode','scix_id','title','abstract','run_id','categories','scores','collections','collection_scores','earth_science_adjustment','override']

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
    logger.info(f"Cheking allowed categories for {categories_list}")
    allowed = config.get('ALLOWED_CATEGORIES')
    allowed = [s.lower() for s in allowed]

    categories_list = [s.lower() for s in categories_list]

    result = [element in allowed for element in categories_list]

    logger.info(f"Cheking allowed categories for (after lowercase) {categories_list}")
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

def filter_allowed_fields(input_dict, allowed_fields=None,response=False):
    """
    Return a new dictionary containing only the keys from input_dict 
    that are part of the ClassifyRequestRecord protobuf definition.
    
    :param input_dict: The original dictionary to filter.
    :param allowed_fields: A set of field names that are allowed.
    :param output: Boolean (False): True if ClassifyResponseRecord is desired output
    :return: A filtered dictionary with only allowed keys.
    """
    if allowed_fields is None:
	# allowed_fields = {'bibcode', 'scixId', 'status', 'title', 'abstract', 
	#	   'operationStep', 'runid', 'override', 'outputPath', 
	#	   'scores', 'collections', 'collectionScores'}
        if response is False:
            allowed_fields = {'bibcode', 'scix_id', 'status', 'title', 'abstract', 
                            'operation_step', 'run_id', 'override', 'output_path', 
                            'scores', 'collections', 'collection_scores'}
        else:
            allowed_fields = {'bibcode', 'scix_id', 'status', 'collections'}

    return {key: value for key, value in input_dict.items() if key in allowed_fields}

# Example usage:

def dict_to_ClassifyRequestRecord(input_dict):
    """
    Convert a dictionary to a protobuf message'
    """
    input_dict = filter_allowed_fields(input_dict)

    request_message = ClassifyRequestRecord()
    message = ParseDict(input_dict, request_message)


    # logger.info(f'Created ClassifyREquestRecord message from dictionary: {message}')
    return message

def list_to_ClassifyRequestRecordList(input_list):
    """
    Convert a list of dictionaries to a protobuf message'
    """

    input_list = list(map(lambda d: filter_allowed_fields(d), input_list))

    logger.info(f'Created ClassifyResponseRecord message from list: {input_list}')

    request_list_dict = {
            'classify_requests' : input_list,
            # 'status' : 99
            }

    request_message = ClassifyRequestRecordList()
    message = ParseDict(request_list_dict, request_message)


    # logger.info(f'Created ClassifyResponseRecord message from dictionary: {message}')
    return message


def dict_to_ClassifyResponseRecord(input_dict):
    """
    Convert a list of dictionaries to a protobuf message'
    """
    input_dict = filter_allowed_fields(input_dict, response=True)

    request_message = ClassifyResponseRecord()
    message = ParseDict(input_dict, request_message)


    # logger.info(f'Created ClassifyREquestRecord message from dictionary: {message}')
    return message

def list_to_ClassifyResponseRecordList(input_list):
    """
    Convert a list of dictionaries to a protobuf message'
    """

    input_list = list(map(lambda d: filter_allowed_fields(d, response=True), input_list))

    response_list_dict = {
            # 'classify_responses' : input_list,
            'classifyResponses' : input_list,
            # 'status' : 99
            }

    logger.info(f"Dictionary for Response Message {response_list_dict}")
    response_message = ClassifyResponseRecordList()
    message = ParseDict(response_list_dict, response_message)


    # logger.info(f'Created ClassifyResponseRecord message from dictionary: {message}')
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
        # try:
        #     entry.scix_id = item.get('scix_id')
        # except:
        #     entry.scix_id = None
        try:
            entry.status = item.get('status')
        except:
            entry.status = None
        try:
            entry.status = item.get('collections')
        except:
            entry.status = None

    return message
     

def classifyRequestRecordList_to_list(message):
    """
    Convert a protobuf ClassifyRequestRecordList to a list of dictionaries.
    """

    logger.info(f'Converting message to list: {message}')
    output_list = []
    request_list = message.classify_requests
    for request in request_list:
        logger.info(f'Unpacking request: {request}')
        output_list.append(MessageToDict(request,preserving_proto_field_name=True))

    # import pdb;pdb.set_trace()
    logger.info(f'Output list from message: {output_list}')

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
        return 'scix_id'
    else:
        return 'bibcode'



