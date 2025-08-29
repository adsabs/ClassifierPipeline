import unittest
from unittest.mock import MagicMock, patch
import ClassifierPipeline.app as app
import ClassifierPipeline.models as models
from ClassifierPipeline.app import SciXClassifierCelery, models

# app = app_module.SciXClassifierCelery(
#     "classifier-pipeline",
#     proj_home=proj_home,
#     local_config=globals().get("local_config", {}),
# )
# config = load_config(proj_home=proj_home)
# logger = setup_logging('tasks.py', proj_home=proj_home,
#                         level=config.get('LOGGING_LEVEL', 'INFO'),
#                         attach_stdout=config.get('LOG_STDOUT', True))

class TestIndexRun(unittest.TestCase):
    @patch('ClassifierPipeline.app.SciXClassifierCelery.session_scope')
    def test_index_run_creates_and_returns_id(self, mock_session_scope):
        mock_session = MagicMock()
        run_instance = MagicMock()
        run_instance.id = 42
        mock_session_scope.return_value.__enter__.return_value = mock_session

        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        with patch('ClassifierPipeline.models.RunTable', return_value=run_instance):
            classifier = SciXClassifierCelery("test-app")
            run_id = classifier.index_run()

        self.assertEqual(run_id, 42)
        mock_session.add.assert_called_with(run_instance)
        mock_session.commit.assert_called_once()


class TestIndexRecordClassify(unittest.TestCase):
    @patch('ClassifierPipeline.app.SciXClassifierCelery.session_scope')
    @patch('ClassifierPipeline.app.json')
    @patch('ClassifierPipeline.app.config', {
        'CLASSIFICATION_PRETRAINED_MODEL': 'modelA',
        'CLASSIFICATION_PRETRAINED_MODEL_REVISION': 'rev1',
        'CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER': 'tokA',
        'ADDITIONAL_EARTH_SCIENCE_PROCESSING': True,
        'ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD': 0.5,
        'CLASSIFICATION_THRESHOLDS': {'A': 0.5, 'B': 0.5},
        'ALLOWED_CATEGORIES': ['A', 'B']
    })
    def test_index_record_new_model_and_score(self, mock_json, mock_session_scope):
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        record = {
            'run_id': 1,
            'collections': ['A', 'B'],
            'scores': [0.1, 0.9],
            'bibcode': 'bib',
            'scix_id': 'scixid',
            'operation_step': 'classify'
        }

        mock_json.dumps.side_effect = lambda x: f"dump({x})"

        classifier = SciXClassifierCelery("test-app")
        out_record, status = classifier.index_record(record.copy())

        self.assertEqual(status, "record_indexed")
        self.assertEqual(out_record['collections'], record['collections'])
        mock_session.add.assert_called()
        mock_session.commit.assert_called()


class TestIndexRecordValidate(unittest.TestCase):
    @patch('ClassifierPipeline.app.SciXClassifierCelery.session_scope')
    @patch('ClassifierPipeline.utilities.check_is_allowed_category', return_value=True)
    @patch('ClassifierPipeline.utilities.check_if_list_single_empty_string', return_value=False)
    @patch('ClassifierPipeline.app.config', {
        'CLASSIFICATION_PRETRAINED_MODEL': 'modelA',
        'CLASSIFICATION_PRETRAINED_MODEL_REVISION': 'rev1',
        'CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER': 'tokA',
        'ADDITIONAL_EARTH_SCIENCE_PROCESSING': True,
        'ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD': 0.5,
        'CLASSIFICATION_THRESHOLDS': {'A': 0.5, 'B': 0.5},
        'ALLOWED_CATEGORIES': ['A', 'B']
    })
    def test_index_record_validate_override(self, mock_empty, mock_allowed, mock_session_scope):
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        # Setup mock for update_final_collection_query (used in validation logic)
        update_final_collection_mock = MagicMock()
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.side_effect = [
            None,  # for check_overrides_query
            update_final_collection_mock  # for update_final_collection_query
        ]


        record = {'bibcode': 'b', 'scix_id': 's', 'override': ['X'], 'run_id': 1, 'collections': ['A'], 'scores': [0.6], 'operation_step': 'validate'}

        classifier = SciXClassifierCelery("test-app")
        out_record, status = classifier.index_record(record.copy())

        self.assertEqual(status, "record_validated")
        mock_session.add.assert_called()
        mock_session.commit.assert_called()


class TestQueryFinalCollectionTable(unittest.TestCase):
    @patch('ClassifierPipeline.app.SciXClassifierCelery.session_scope')
    def test_query_by_bibcode(self, mock_session_scope):
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        fake_fc = MagicMock()
        fake_fc.scix_id = 's'
        fake_fc.collection = ['C']
        mock_session.query.return_value.filter.return_value.first.return_value = fake_fc

        classifier = SciXClassifierCelery("test-app")
        result = classifier.query_final_collection_table(bibcode='bcode')

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['bibcode'], 'bcode')
        self.assertEqual(result[0]['collections'], ['C'])


class TestUpdateValidatedRecords(unittest.TestCase):
    @patch('ClassifierPipeline.app.SciXClassifierCelery.session_scope')
    @patch.object(SciXClassifierCelery, 'query_final_collection_table')
    def test_update_validated_records(self, mock_query, mock_session_scope):
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        mock_query.return_value = [{'bibcode': 'b', 'scix_id': 's', 'collections': ['col']}]
        fake_fc = MagicMock()
        fake_fc.validated = False
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = fake_fc

        classifier = SciXClassifierCelery("test-app")
        records, successes = classifier.update_validated_records(run_id=999)

        self.assertEqual(len(records), 1)
        self.assertEqual(successes, ['success'])
        self.assertTrue(fake_fc.validated)
        mock_session.commit.assert_called()


if __name__ == '__main__':
    unittest.main()

