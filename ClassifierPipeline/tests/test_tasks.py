import os
import unittest
from unittest.mock import patch, MagicMock, call
from ClassifierPipeline.tasks import (
    task_update_record,
    task_send_input_record_to_classifier,
    task_index_classified_record,
    task_message_to_master,
    task_resend_to_master,
    task_update_validated_records,
    task_output_results,
)

# You'll need to import your module's objects
import ClassifierPipeline.tasks
from adsmsg import ClassifyRequestRecordList, ClassifyRequestRecord, ClassifyResponseRecord

class TestClassifierPipelineTasks(unittest.TestCase):

    def setUp(self):
        # Patch logger to suppress output and track calls
        self.patcher_log = patch.object(ClassifierPipeline.tasks, 'logger', autospec=True)
        self.mock_logger = self.patcher_log.start()

        # Patch app and utils globally for all tests
        # self.patcher_app = patch.object(ClassifierPipeline.tasks, 'app')
        self.patcher_app = patch('ClassifierPipeline.tasks.app') 
        self.mock_app = self.patcher_app.start()

        self.patcher_utils = patch.object(ClassifierPipeline.tasks, 'utils')
        self.mock_utils = self.patcher_utils.start()

        # Default config values
        ClassifierPipeline.tasks.config = {
            'DELAY_MESSAGE': False,
            'FAKE_DATA': False,
            'ALLOWED_CATEGORIES': ['astrophysics', 'heliophysics', 'planetary', 'earthscience', 'NASA-funded Biophysics', 'physics', 'general', 'Text Garbage'],
            'OPERATION_STEP': 'classify_verify'
        }

        # Prepare a dummy record, request list conversion, and protobuf wrapping
        self.record_dict = {
            'bibcode': '2017ApJ...845..161A',
            # 'scix_id': 'ID123',
            'title': 'Properties of the Closest Young Binaries. I. DF Tau’s Unequal Circumstellar Disk Evolution',
            'abstract': 'We present high-resolution, spatially resolved, near-infrared spectroscopy and imaging of the two components of DF Tau, a young, low-mass, visual binary in the Taurus star-forming region.'
        }
        self.request_list = [self.record_dict.copy()]
        self.record_list_protobuf = ClassifyRequestRecordList()
        # Configure utils to behave
        self.mock_utils.classifyRequestRecordList_to_list.return_value = self.request_list
        self.mock_utils.list_to_ClassifyRequestRecordList.return_value = self.record_list_protobuf

        # Make app.index_run generate a run ID
        self.mock_app.index_run.return_value = 'RUNID'

        # Dummy classifier scoring
        self.mock_classifier = patch.object(ClassifierPipeline.tasks, 'classifier', autospec=True).start()
        self.mock_classifier.batch_score_SciX_categories.return_value = ([['Cat']], [[0.9]])

        # Patches for file operations
        # self.patcher_prepare = patch.object(self.mock_utils, 'prepare_output_file', autospec=True)
        # self.mock_utils.prepare_output_file = self.patcher_prepare.start()
        self.mock_utils.prepare_output_file = MagicMock()
        # self.patcher_add_output = patch.object(self.mock_utils, 'add_record_to_output_file', autospec=True)
        # self.mock_utils.add_record_to_output_file = self.patcher_add_output.start()
        self.mock_utils.add_record_to_output_file = MagicMock()
        self.mock_utils.message_to_list = MagicMock()
        self.mock_app.index_record.return_value = (MagicMock(), 'record_indexed')

        # yield 

    def tearDown(self):
        patch.stopall()

    def test_task_update_record_with_minimal_message(self):
        # No operation_step or output_path in request -> defaults are used
        msg = self.record_list_protobuf
        task_update_record(msg)

        # index_run called
        self.mock_app.index_run.assert_called_once()

        # prepare_output_file called with derived path
        args, _ = self.mock_utils.prepare_output_file.call_args
        output_path_arg = args[0]
        self.assertTrue(output_path_arg.endswith('classified.tsv'))

        # utils.list_to_ClassifyRequestRecordList called once per record
        self.assertTrue(self.mock_utils.list_to_ClassifyRequestRecordList.called)

        # task_send_input_record_to_classifier called directly (DELAY_MESSAGE False)
        # task_send_input_record_to_classifier is in same module, so we patch to detect it.
        # For simplicity, check utils.list_to_ClassifyRequestRecordList was used to generate out_message
        self.mock_utils.list_to_ClassifyRequestRecordList.assert_called()

    def test_task_send_input_record_to_classifier_with_real_inference(self):
        msg = self.record_list_protobuf
        # Ensure FAKE_DATA is False
        ClassifierPipeline.tasks.config['FAKE_DATA'] = False
        task_send_input_record_to_classifier(msg)

        # classifier.batch_score_SciX_categories was called
        self.mock_classifier.batch_score_SciX_categories.assert_called()

        # classify_record_from_scores is invoked
        self.mock_utils.classify_record_from_scores.assert_called()

        # list_to_ClassifyRequestRecordList to wrap output
        self.mock_utils.list_to_ClassifyRequestRecordList.assert_called()

    def test_task_send_input_record_to_classifier_with_fake_data(self):
        ClassifierPipeline.tasks.config['FAKE_DATA'] = True
        msg = self.record_list_protobuf
        task_send_input_record_to_classifier(msg)

        # return_fake_data invoked
        self.mock_utils.return_fake_data.assert_called()

        # list_to_ClassifyRequestRecordList called
        self.mock_utils.list_to_ClassifyRequestRecordList.assert_called()

    def test_task_index_classified_record_success_indexed_verify_step(self):
        # Simulate record after classification
        rec = self.record_dict.copy()
        rec['operation_step'] = 'classify_verify'
        rec['scix_id'] = 'ID123'

        self.mock_utils.classifyRequestRecordList_to_list.return_value = [rec]
        self.mock_app.index_record.return_value = (rec, 'record_indexed')

        msg = self.record_list_protobuf
        task_index_classified_record(msg)

        # app.index_record called
        self.mock_app.index_record.assert_called_once_with(rec)

        # add_record_to_output_file invoked from utils (not app)
        self.mock_utils.add_record_to_output_file.assert_called_once_with(rec)

    def test_task_index_classified_record_success_indexed_classify_step(self):
        rec = self.record_dict.copy()
        rec['operation_step'] = 'classify'
        rec['bibcode'] = '2017ApJ...845..161A'
        self.mock_utils.classifyRequestRecordList_to_list.return_value = [rec]
        self.mock_app.index_record.return_value = (rec, 'record_indexed')

        task_index_classified_record(self.record_list_protobuf)
        # For classify step, uses app.add_record_to_output_file and then task_resend_to_master
        self.mock_utils.add_record_to_output_file.assert_called_once_with(rec)

    def test_task_index_classified_record_validated(self):
        rec = self.record_dict.copy()
        rec['operation_step'] = 'anything'
        self.mock_utils.classifyRequestRecordList_to_list.return_value = [rec]
        self.mock_app.index_record.return_value = (rec, 'record_validated')

        task_index_classified_record(self.record_list_protobuf)
        # task_resend_to_master should be called with appropriate message
        # self.mock_app.query_final_collection_table.assert_not_called()
        self.mock_app.query_final_collection_table.assert_called_once_with(bibcode=rec['bibcode'])


    def test_task_message_to_master_dict_and_list(self):
        # If passed a dict, out_message should be called once
        with patch.object(ClassifierPipeline.tasks, 'out_message') as om:
            task_message_to_master(self.record_dict)
            om.assert_called_once_with(self.record_dict)

        # If passed a list
        self.mock_app.reset_mock()
        with patch.object(ClassifierPipeline.tasks, 'out_message') as om2:
            task_message_to_master([self.record_dict, self.record_dict])
            # Each item should result in out_message called with the entire list
            self.assertEqual(om2.call_count, 2)

    def test_task_resend_to_master_bibcode_path(self):
        rec = self.record_dict.copy()
        self.mock_utils.classifyRequestRecordList_to_list.return_value = [rec]
        # Simulate query_final_collection_table returning some records
        self.mock_app.query_final_collection_table.return_value = [rec]
        with patch.object(ClassifierPipeline.tasks, 'task_message_to_master') as tmtm:
            task_resend_to_master(self.record_list_protobuf)
            # task_message_to_master called for each record
            tmtm.assert_called_once()

    def test_task_update_validated_records(self):
        rec = {'run_id': 'RUNID', 'bibcode': '2025Test'}
        self.mock_utils.classifyRequestRecordList_to_list.return_value = [rec]
        self.mock_app.update_validated_records.return_value = ([rec], ['success'])
        with patch.object(ClassifierPipeline.tasks, 'task_message_to_master') as tmtm:
            task_update_validated_records(self.record_list_protobuf)
            tmtm.assert_called_once_with(rec)

    def test_task_output_results(self):
        rec = {'bibcode': '2025Test'}
        # utils.message_to_list returns a list containing our rec
        self.mock_utils.message_to_list.return_value = [rec]
        task_output_results(rec)
        # app.add_record_to_output_file should be called with rec
        self.mock_app.add_record_to_output_file.assert_called_once_with(rec)


if __name__ == '__main__':
    unittest.main()

