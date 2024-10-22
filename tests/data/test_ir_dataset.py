import unittest
from unittest.mock import Mock, patch
import pandas as pd
from mechir.data import MechIRDataset

class TestMechIRDataset(unittest.TestCase):

    def setUp(self):
        # Example data for mocking the dataset
        self.mock_ir_dataset = Mock()
        
        # Mock qrels, queries, and docs
        self.qrels_data = pd.DataFrame({
            'query_id': ['q1', 'q2', 'q3'],
            'doc_id': ['d1', 'd2', 'd3']
        })
        self.docs_data = {
            'd1': 'Document text 1',
            'd2': 'Document text 2',
            'd3': 'Document text 3'
        }
        self.queries_data = {
            'q1': 'Query text 1',
            'q2': 'Query text 2',
            'q3': 'Query text 3'
        }
        
        # Mock iterators
        self.mock_ir_dataset.qrels_iter.return_value = self.qrels_data.to_dict('records')
        self.mock_ir_dataset.docs_iter.return_value = [{'doc_id': k, 'text': v} for k, v in self.docs_data.items()]
        self.mock_ir_dataset.queries_iter.return_value = [{'query_id': k, 'text': v} for k, v in self.queries_data.items()]
        
        # Mock ir_datasets load function
        patcher = patch('ir_datasets.load', return_value=self.mock_ir_dataset)
        self.addCleanup(patcher.stop)
        self.mock_load = patcher.start()

    def test_dataset_initialization(self):
        # Initialize dataset with mocked ir_dataset
        dataset = MechIRDataset(ir_dataset="test_dataset")
        
        # Test that the dataset initialized correctly
        self.mock_load.assert_called_once_with("test_dataset")
        self.assertEqual(len(dataset), len(self.qrels_data))
        
    def test_get_item(self):
        # Initialize dataset with mocked ir_dataset
        dataset = MechIRDataset(ir_dataset="test_dataset")
        
        # Test __getitem__ method
        for idx, (query_id, doc_id) in enumerate(zip(self.qrels_data['query_id'], self.qrels_data['doc_id'])):
            query, doc = dataset[idx]
            self.assertEqual(query, self.queries_data[query_id])
            self.assertEqual(doc, self.docs_data[doc_id])

    def test_missing_column_error(self):
        # Test that a ValueError is raised if the required columns are missing
        invalid_pairs_df = pd.DataFrame({
            'invalid_query_col': ['q1', 'q2'],
            'invalid_doc_col': ['d1', 'd2']
        })
        
        with self.assertRaises(ValueError):
            MechIRDataset(ir_dataset="test_dataset", pairs=invalid_pairs_df)

    def test_default_pairs(self):
        # Test dataset initialization without passing the `pairs` dataframe (should use qrels from the dataset)
        dataset = MechIRDataset(ir_dataset="test_dataset")
        
        # Ensure pairs is initialized with the dataset's qrels
        self.assertTrue((dataset.pairs['query_id'] == self.qrels_data['query_id']).all())
        self.assertTrue((dataset.pairs['doc_id'] == self.qrels_data['doc_id']).all())

if __name__ == '__main__':
    unittest.main()
