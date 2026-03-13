"""
Classifier Pipeline Celery Tasks

This module defines Celery tasks for managing and orchestrating the classification
pipeline for scientific records. It receives messages from ADS Master Pipeline,
performs classification using a model, updates the database, and sends responses 
back to the Master Pipeline.

The pipeline includes steps to:
    - Receive records
    - Perform classification
    - Index classification results into a database
    - Update output files
    - Validate and forward results to Master Pipeline

Celery Queue:
    - All tasks are currently assigned to the "update-record" queue. Other queues available include: "classify-record", "index-record", "send-record-to-classifier".

Dependencies:
    - adsputils
    - SQLAlchemy for database access
    - Google protobuf for message formatting
    - ClassifierPipeline (Classifier, utilities, models)

Tasks:
    - task_update_record
    - task_send_input_record_to_classifier
    - task_index_classified_record
    - task_message_to_master
    - task_resend_to_master
    - task_update_validated_records
    - task_output_results
"""
import sys
import os
import json
import time
import adsputils
from adsputils import ADSCelery
import ClassifierPipeline.app as app_module
import ClassifierPipeline.utilities as utils
import ClassifierPipeline.perf_metrics as perf_metrics
from ClassifierPipeline.classifier import Classifier
from adsputils import load_config, setup_logging
from kombu import Queue
from google.protobuf.json_format import Parse, MessageToDict
from adsmsg import ClassifyRequestRecord, ClassifyRequestRecordList, ClassifyResponseRecord, ClassifyResponseRecordList

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
    Queue("classify-record", app.exchange, routing_key="classify-record"),
    Queue("classify-record", app.exchange, routing_key="index-record"),
    Queue("send-record-to-master", app.exchange, routing_key="send-record-to-master")
)

classifier = Classifier()


def _record_identifier(record):
    if record.get("scix_id"):
        return record.get("scix_id")
    return record.get("bibcode")


def _chunk_records(records, chunk_size):
    for index in range(0, len(records), chunk_size):
        yield records[index:index + chunk_size]

# ============================= TASKS ============================================= #

@app.task(queue="update-record")
def task_update_record(message,pipeline='classifier', output_format='tsv'):
    """
    Entry point task to receive classification requests from the master.

    Parses the message, generates initial record metadata, and forwards each 
    request to the classifier task.

    Parameters:
        message (ClassifyRequestRecordList): List of classification requests
        pipeline (str): Processing pipeline name (default "classifier")
        output_format (str): Output format for results (default "tsv")
    """

    logger.debug(f'Message type: {type(message)}')
    logger.debug(f'Message: {message}')

    request_list = utils.classifyRequestRecordList_to_list(message)

    if request_list and request_list[0].get("run_id"):
        run_id = request_list[0].get("run_id")
    else:
        run_id = app.index_run()
    logger.info('Run ID: {}'.format(run_id))

    if 'operation_step' in request_list[0]:
        operation_step = request_list[0]['operation_step']
    else:
        operation_step = config.get('OPERATION_STEP', 'classify_verify')

    if 'output_path' in request_list[0]:
        try:
            filename = request_list[0]['output_path']
            filename = filename.split('/')[-1]
        except:
            filename = request_list[0]['output_path']
    else:
        filename = ''

    output_path = os.path.join(proj_home, 'logs', f'{filename}_{run_id}_classified.tsv')

    utils.prepare_output_file(output_path)
    logger.info('Prepared output file: {}'.format(output_path))

    # Delay setting for testing
    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.debug("Delay set for queue messages: {}".format(delay_message))


    classify_batch_size = int(config.get("CLASSIFY_STAGE_BATCH_SIZE", 100) or 100)
    if classify_batch_size <= 0:
        classify_batch_size = 100

    normalized_records = []
    logger.debug('Request list: {}'.format(request_list))
    for request in request_list:
        logger.debug('Request: {}'.format(request))
        record_bibcode = None
        record_scix_id = None
        if 'bibcode' in request:
            record_bibcode = request['bibcode']
        if 'scix_id' in request:
            record_scix_id = request['scix_id']
        if 'title' in request:
            record_title = request['title']
        else:
            record_title = "None"
        if 'abstract' in request:
            record_abstract = request['abstract']
        else:
            record_abstract = "None"
        record = {'bibcode': record_bibcode,
                  'scix_id': record_scix_id,
                  'title': record_title,
                  'abstract': record_abstract,
                  'text': record_title + ' ' + record_abstract,
                  'operation_step': operation_step,
                  'run_id': run_id,
                  'output_format': output_format,
                  'override': None,
                  'output_path': output_path
                  }
        normalized_records.append(record)

    for sub_batch in _chunk_records(normalized_records, classify_batch_size):
        logger.debug("creating output message")
        logger.debug(f"Records {sub_batch}")
        out_message = utils.list_to_ClassifyRequestRecordList(sub_batch)

        logger.debug('Output Record type: {}'.format(type(out_message)))
        logger.debug('Output Record: {}'.format(out_message))
        enqueue_start = time.perf_counter()
        if delay_message:
            logger.debug('Using delay')
            task_send_input_record_to_classifier.delay(out_message)
        else:
            task_send_input_record_to_classifier(out_message)  
        enqueue_ms = (time.perf_counter() - enqueue_start) * 1000.0
        perf_metrics.emit_event(
            stage="ingest_enqueue",
            run_id=run_id,
            record_id=None,
            duration_ms=enqueue_ms,
            extra={
                "record_count": len(sub_batch),
                "delayed": bool(delay_message),
                "pipeline": pipeline,
                "batch_mode": True,
                "batch_size": len(sub_batch),
            },
            config=config,
        )

    return {"run_id": run_id, "records_submitted": len(request_list)}
            

# @app.task(queue="unclassified-queue")
# @app.task(queue="update-record")
@app.task(queue="classify-record")
def task_send_input_record_to_classifier(message):
    """
    Task to perform classification inference on a record.

    If FAKE_DATA is set in config file, generates fake classifications instead of inference.

    Parameters:
        message (ClassifyRequestRecordList): A single-item list with record data
    """

    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.debug("Delay set for queue messages: {}".format(delay_message))

    fake_data = config.get('FAKE_DATA', False)
    forced_fake_data = os.getenv("PERF_FORCE_FAKE_DATA")
    if forced_fake_data is not None:
        fake_data = forced_fake_data.strip().lower() in {"1", "true", "yes", "on", "active"}

    logger.debug("Fake data set for queue messages: {}".format(fake_data))

    records = utils.classifyRequestRecordList_to_list(message)
    if not records:
        return
    run_id = records[0].get("run_id")

    stage_start = time.perf_counter()
    try:
        processed_records = [None for _ in records]
        real_positions = []
        real_texts = []

        for index, record in enumerate(records):
            effective_fake_data = fake_data
            if record.get("fake_data") is not None:
                effective_fake_data = bool(record.get("fake_data"))

            if effective_fake_data is False:
                real_positions.append(index)
                real_texts.append(record['title'] + ' ' + record['abstract'])
            else:
                processed_records[index] = utils.return_fake_data(record)

        if real_texts:
            logger.debug('Performing Inference')
            categories, scores = classifier.batch_score_SciX_categories(real_texts)
            logger.debug('Categories: {}'.format(categories))
            logger.debug('Allowed Categories: {}'.format(config['ALLOWED_CATEGORIES']))
            logger.debug('Scores: {}'.format(scores))
            for output_index, record_index in enumerate(real_positions):
                record = records[record_index]
                record['categories'] = categories[output_index]
                record['scores'] = scores[output_index]
                processed_records[record_index] = record
        elif len(real_positions) == 0:
            logger.info('Skipping inference - generating fake data')

        for index, record in enumerate(processed_records):
            logger.debug('RECORD: {}'.format(record))
            processed_records[index] = utils.classify_record_from_scores(record)

        classify_status = "ok"
    except Exception:
        classify_status = "error"
        raise
    finally:
        fake_record_count = sum(1 for record in records if (bool(record.get("fake_data")) if record.get("fake_data") is not None else fake_data))
        perf_metrics.emit_event(
            stage="classify",
            run_id=run_id,
            record_id=None,
            duration_ms=(time.perf_counter() - stage_start) * 1000.0,
            status=classify_status,
            extra={
                "fake_data": bool(fake_data),
                "record_count": len(records),
                "batch_size": len(records),
                "real_record_count": len(records) - fake_record_count,
                "fake_record_count": fake_record_count,
                "batch_mode": True,
            },
            config=config,
        )

    logger.debug("Records after classification and thresholding: {}".format(processed_records))
    logger.debug("Record Type: {}".format(type(processed_records)))

    out_message = utils.list_to_ClassifyRequestRecordList(processed_records)

    if delay_message:
        task_index_classified_record.delay(out_message)
    else:
        task_index_classified_record(out_message) 




# @app.task(queue="classify-record")
@app.task(queue="index-record")
def task_index_classified_record(message):
    """
    Task to store classified records into the database.

    Also logs the result and forwards it to the master service if appropriate.

    Parameters:
        message (ClassifyRequestRecordList): Classified record to store
    """

    delay_message = config.get('DELAY_MESSAGE', False) 

    logger.debug("Delay set for queue messages: {}".format(delay_message))

    records = utils.classifyRequestRecordList_to_list(message)
    logger.debug(f"Record batch: {records}")
    logger.debug(f'Record type: {type(message)}')
    for record in records:
        record_id = _record_identifier(record)

        stage_start = time.perf_counter()
        try:
            record, success = app.index_record(record)
            index_status = "ok"
        except Exception:
            index_status = "error"
            raise
        finally:
            perf_metrics.emit_event(
                stage="index",
                run_id=record.get("run_id"),
                record_id=record_id,
                duration_ms=(time.perf_counter() - stage_start) * 1000.0,
                status=index_status,
                extra={"operation_step": record.get("operation_step")},
                config=config,
            )
        logger.debug(f'Record: {record}, Success: {success}')
        if success == "record_indexed":
            if record['operation_step'] == 'classify_verify':
                logger.info(f"Record {record_id} indexed")
                utils.add_record_to_output_file(record)
            if record['operation_step'] == 'classify':
                logger.info(f"Record {record_id} indexed")
                app.add_record_to_output_file(record)
                resend_message = utils.list_to_ClassifyRequestRecordList([record])
                task_resend_to_master(resend_message)
                logger.info(f"Record {record_id} sent to master")

        elif success == "record_validated":
            resend_message = utils.list_to_ClassifyRequestRecordList([record])
            task_resend_to_master(resend_message)
            logger.info(f"Record {record_id} sent to master")
        else:
            logger.info(f"Record {record_id} failed to be indexed")

def out_message(message):
    """
    Helper function to convert and forward a message to the master pipeline.

    Parameters:
        message (dict): Dictionary containing classification result
    """

    out_message = utils.dict_to_ClassifyResponseRecord(message)
    logger.debug(f"Forwarding message to Master - Message: {out_message}")
    app.forward_message(out_message)

@app.task(queue="send-record-to-master")
def task_message_to_master(message):
    """
    Task to send the classified record(s) back to the master service.

    Parameters:
        message (dict or list): A single record or list of classified records
    """

    if isinstance(message, dict):
        forward_start = time.perf_counter()
        status = "ok"
        try:
            out_message(message)
        except Exception:
            status = "error"
            raise
        finally:
            perf_metrics.emit_event(
                stage="forward",
                run_id=message.get("run_id"),
                record_id=_record_identifier(message),
                duration_ms=(time.perf_counter() - forward_start) * 1000.0,
                status=status,
                config=config,
            )
    if isinstance(message, list):
        for msg in message:
            forward_start = time.perf_counter()
            status = "ok"
            try:
                out_message(message)
            except Exception:
                status = "error"
                raise
            finally:
                perf_metrics.emit_event(
                    stage="forward",
                    run_id=msg.get("run_id"),
                    record_id=_record_identifier(msg),
                    duration_ms=(time.perf_counter() - forward_start) * 1000.0,
                    status=status,
                    config=config,
                )

# @app.task(queue="classify-record")
@app.task(queue="send-record-to-master")
def task_resend_to_master(message):
    """
    Task to re-send a classified record to Master Pipeline based on bibcode, scix_id, or run_id.

    Parameters:
        message (ClassifyRequestRecordList): Message containing one record
    """

    logger.info(f"Resending records to master")

    request_list = utils.classifyRequestRecordList_to_list(message)

    logger.debug('Request list: {}'.format(request_list))
    for request in request_list:
        logger.info('Request: {}'.format(request))

        if 'bibcode' in request:
            record_id = request['bibcode']
            record_list = app.query_final_collection_table(bibcode=request['bibcode'])
        elif 'scix_id' in request:
            record_list = app.query_final_collection_table(scix_id=request['scix_id'])
        elif 'run_id' in request:
            record_list = app.query_final_collection_table(run_id=request['run_id'])

        for record in record_list:
            record_id = None
            if 'scix_id' in record:
                record_id = record['scix_id']
            if 'bibcode' in record:
                record_id = record['bibcode']
            logger.info(f"Sending record {record_id} to master")
            task_message_to_master(record)


# @app.task(queue="classify-record")
@app.task(queue="index-record")
def task_update_validated_records(message):
    """
    Task to mark a batch of records as validated in the database by run_id.

    Parameters:
        message (ClassifyRequestRecordList): Message with run_id field
    """

    logger.info(f"Updating validated records")
    record = utils.classifyRequestRecordList_to_list(message)[0]
    record_list, success_list = app.update_validated_records(record['run_id'])
    for record, success in zip(record_list, success_list):
        logger.debug(f"Record: {record}")
        logger.debug(f"Success: {success}")
        if success == "success":
            record_id = None
            if 'scix_id' in record:
                record_id = record['scix_id']
            if 'bibcode' in record:
                record_id = record['bibcode']
            logger.info(f"Sending record {record_id} to master")
            task_message_to_master(record)


# @app.task(queue="output-results")
# @app.task(queue="classify-record")
@app.task(queue="update-record")
def task_output_results(message):
    """
    Task to append classified record results to the output file.

    Parameters:
        message (dict): Record containing bibcode, scix_id, and collections
    """

    record = utils.message_to_list(message)[0]
    logger.debug('Output results ')
    logger.debug(f'Record being output {message}')
    app.add_record_to_output_file(record)




if __name__ == "__main__":
    app.start()
