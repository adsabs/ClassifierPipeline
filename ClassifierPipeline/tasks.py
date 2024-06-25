import sys
import os
# from __future__ import absolute_import, unicode_literals
import adsputils
from adsputils import ADSCelery
# from adsmsg import OrcidClaims
# from SciXClassifier import app as app_module
# from SciXClassifier import updater
# from SciXClassifier.exceptions import ProcessingException, IgnorableException
# from SciXClassifier.models import KeyValue
# from .app import SciXClassifierCelery
import ClassifierPipeline.app as app_module
# from kombu import Queue
# import datetime
# from .classifier import score_record
sys.path.append(os.path.abspath('../..'))
from run import score_record, classify_record_from_scores, add_record_to_output_file

from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
# ============================= INITIALIZATION ==================================== #

proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
app = app_module.SciXClassifierCelery(
        # app = SciXClassifierCelery(
    "scixclassifier-pipeline",
    proj_home=proj_home,
    local_config=globals().get("local_config", {}),
)
# import pdb; pdb.set_trace()
# from adsputils import setup_logging, load_config
# config = load_config(proj_home=proj_home)
# logger = setup_logging('run.py', proj_home=proj_home,
#                         level=config.get('LOGGING_LEVEL', 'INFO'),
#                         attach_stdout=config.get('LOG_STDOUT', False))

# app.conf.CELERY_QUEUES = (
#     Queue("unclassified-queue", app.exchange, routing_key="unclassified-queue"),
# )
# logger = app.logger


# ============================= TASKS ============================================= #

# From Curators Daily Operations 

# Send data to the Classifier

# Populate database wit new data

# Return sorted classifications to Curators

# Query SOLR
#   - Finding records with given set of parameters (e.g. classification, model, etc.)

@app.task(queue="update-record")
# def task_handle_input_from_master(message):
def task_update_record(message):
    """
    Handle the input from the master

    :param message: contains the message inside the packet
        {
         'bibcode': String (19 chars),
         'title': String,
         'abstract':String
        }
    :return: no return
    """
    message = app.handle_input_from_master(message)


# @app.task(queue="unclassified-queue")
@app.task(queue="classify-record")
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

    print("task_send_input_record_to_classifier")
    print(message)

    # import pdb; pdb.set_trace()
    # First obtain the raw scores from the classifier model
    message = score_record(message)

    # Then classify the record based on the raw scores
    # import pdb; pdb.set_trace()
    message = classify_record_from_scores(message)
    print('Collections: ')
    print(message['collections'])

    # Write the classifications to output file
    # add_record_to_output_file(message)
    # may have add .async 
    # task_output_results(message)

    # import pdb; pdb.set_trace()
    # Write the new classification to the database
    task_index_classified_record(message)

    # import pdb; pdb.set_trace()


@app.task(queue="classify-record")
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

    print('Indexing Classified Record')
    app.index_record(message)
    # import pdb; pdb.set_trace()
    # pass

@app.task(queue="classify-record")
def task_update_validated_records(message):
    """
    Update all records that have been validated that have same run_id

    :param message: contains the message inside the packet
        {
         'run_id': Boolean,
        }
    """

    print('Updating Validated Records')
    app.update_validated_records(message)
    # import pdb; pdb.set_trace()
    # pass


# @app.task(queue="output-results")
@app.task(queue="classify-record")
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
    add_record_to_output_file(message)



if __name__ == "__main__":
    app.start()
