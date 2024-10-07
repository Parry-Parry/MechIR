import unittest
from unittest.mock import Mock
from . import CatDataCollator, PairDataCollator, _make_pos_pairs

class TestDataCollators(unittest.TestCase):

    def setUp(self):
        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.convert_tokens_to_ids = Mock(return_value=999)  # Mock special token id
        self.tokenizer.return_tensors = "pt"
        self.tokenizer.side_effect = lambda queries, docs, *args, **kwargs: {
            'input_ids': [[1, 2, 3]] * len(queries),  # Mock input_ids
            'attention_mask': [[1, 1, 1]] * len(queries),  # Mock attention mask
        }

        # Simple transformation function that adds "_perturbed" to each document
        self.transformation_func = lambda doc, query: (doc, f"{doc}_perturbed")
        
    def test_cat_data_collator(self):
        # Instantiate the CatDataCollator
        collator = CatDataCollator(
            tokenizer=self.tokenizer,
            transformation_func=self.transformation_func,
            q_max_length=10,
            d_max_length=100,
            special_token="X"
        )
        
        # Example input batch
        batch = [
            ("query1", ["doc1", "doc2"], [1.0, 0.5]),  # query, documents, scores
            ("query2", ["doc3"], [0.9]),  # query, documents, scores
        ]
        
        # Call the collator
        output = collator(batch)
        
        # Check that output contains sequences and perturbed sequences
        self.assertIn("sequences", output)
        self.assertIn("perturbed_sequences", output)
        
        # Check the structure of tokenized outputs
        self.assertEqual(len(output["sequences"]['input_ids']), 3)  # 3 total query-doc pairs
        self.assertEqual(len(output["perturbed_sequences"]['input_ids']), 3)  # 3 total perturbed pairs
        
    def test_pair_data_collator(self):
        # Instantiate the PairDataCollator
        collator = PairDataCollator(
            tokenizer=self.tokenizer,
            transformation_func=self.transformation_func,
            max_length=512
        )
        
        # Example input batch
        batch = [
            ("query1", ["doc1", "doc2", "doc3"], [1.0, 0.5, 0.7]),  # query, documents, scores
        ]
        
        # Call the collator
        output = collator(batch)
        
        # Check that output contains sequences
        self.assertIn("sequences", output)
        
        # Check the structure of tokenized outputs
        # There should be 2 pairs generated from 3 documents
        self.assertEqual(len(output["sequences"]['input_ids']), 2)  # 2 query-document pairs
        
    def test_make_pos_pairs(self):
        # Test _make_pos_pairs function
        texts = ["doc1", "doc2", "doc3"]
        pairs = _make_pos_pairs(texts)
        
        # Expected output: [['doc1', 'doc2'], ['doc1', 'doc3']]
        self.assertEqual(pairs, [["doc1", "doc2"], ["doc1", "doc3"]])

if __name__ == '__main__':
    unittest.main()