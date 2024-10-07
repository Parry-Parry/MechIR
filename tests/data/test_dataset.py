import unittest
import pandas as pd
from . import MechDataset

class TestMechDataset(unittest.TestCase):

    def setUp(self):
        # Example data for testing
        self.data = {
            'query': ['query1', 'query2', 'query3'],
            'doc': ['doc1', 'doc2', 'doc3']
        }
        self.pairs_df = pd.DataFrame(self.data)
    
    def test_dataset_length(self):
        # Initialize the dataset
        dataset = MechDataset(pairs=self.pairs_df)
        
        # Test that the length of the dataset matches the dataframe length
        self.assertEqual(len(dataset), len(self.pairs_df))
    
    def test_get_item(self):
        # Initialize the dataset
        dataset = MechDataset(pairs=self.pairs_df)
        
        # Test __getitem__ method
        for idx, (expected_query, expected_doc) in enumerate(zip(self.data['query'], self.data['doc'])):
            query, doc = dataset[idx]
            self.assertEqual(query, expected_query)
            self.assertEqual(doc, expected_doc)
    
    def test_missing_column_error(self):
        # Test that a ValueError is raised if the expected columns are missing
        invalid_df = pd.DataFrame({
            'invalid_query_col': ['query1', 'query2'],
            'invalid_doc_col': ['doc1', 'doc2']
        })
        
        with self.assertRaises(ValueError):
            MechDataset(pairs=invalid_df, query_col='query', doc_col='doc')
    
    def test_default_column_names(self):
        # Test default column names ('query' and 'doc')
        dataset = MechDataset(pairs=self.pairs_df)
        query, doc = dataset[0]
        self.assertEqual(query, 'query1')
        self.assertEqual(doc, 'doc1')
    
    def test_custom_column_names(self):
        # Test using custom column names
        custom_pairs_df = pd.DataFrame({
            'custom_query': ['queryA', 'queryB'],
            'custom_doc': ['docA', 'docB']
        })
        
        dataset = MechDataset(pairs=custom_pairs_df, query_col='custom_query', doc_col='custom_doc')
        query, doc = dataset[0]
        self.assertEqual(query, 'queryA')
        self.assertEqual(doc, 'docA')

    def test_empty_dataframe(self):
        # Test behavior with an empty dataframe
        empty_df = pd.DataFrame(columns=['query', 'doc'])
        dataset = MechDataset(pairs=empty_df)
        
        # Length should be 0
        self.assertEqual(len(dataset), 0)

if __name__ == '__main__':
    unittest.main()
