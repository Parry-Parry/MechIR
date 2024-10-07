import unittest
from unittest.mock import Mock
from . import DotDataCollator, pad

class TestDotDataCollator(unittest.TestCase):

    def setUp(self):
        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.return_tensors = "pt"
        self.tokenizer.side_effect = lambda queries, *args, **kwargs: {
            'input_ids': [[1, 2, 3]] * len(queries),  # Mock input_ids
            'attention_mask': [[1, 1, 1]] * len(queries),  # Mock attention mask
            'special_tokens_mask': [[0, 1, 1]] * len(queries)  # Mock special tokens mask if special_mask=True
        }

        # Simple transformation function that adds "_perturbed" to each document
        self.transformation_func = lambda doc, query: (doc, f"{doc}_perturbed")
        
        # Mock pad function (since it comes from an external source)
        global pad
        pad = Mock(side_effect=lambda a, b, special_token=None: (a, b))  # Simply return a and b unmodified
        
    def test_dot_data_collator(self):
        # Instantiate DotDataCollator
        collator = DotDataCollator(
            tokenizer=self.tokenizer,
            transformation_func=self.transformation_func,
            special_mask=True,  # Set special mask to true for testing
            q_max_length=30,
            d_max_length=200
        )
        
        # Example input batch
        batch = [
            ("query1", ["doc1", "doc2"]),  # query and corresponding documents
            ("query2", ["doc3"]),          # another query and its document
        ]
        
        # Call the collator
        output = collator(batch)
        
        # Verify that the output contains expected keys
        self.assertIn("queries", output)
        self.assertIn("documents", output)
        self.assertIn("perturbed_documents", output)
        
        # Check the structure of tokenized outputs
        self.assertEqual(len(output["queries"]['input_ids']), 2)  # 2 queries
        self.assertEqual(len(output["documents"]['input_ids']), 3)  # 3 total documents
        self.assertEqual(len(output["perturbed_documents"]['input_ids']), 3)  # 3 perturbed documents
        
        # Verify special_tokens_mask when special_mask is True
        self.assertIn("special_tokens_mask", output["queries"])
        self.assertIn("special_tokens_mask", output["documents"])
        self.assertIn("special_tokens_mask", output["perturbed_documents"])
        
    def test_no_special_tokens_mask(self):
        # Instantiate DotDataCollator without special_mask
        collator = DotDataCollator(
            tokenizer=self.tokenizer,
            transformation_func=self.transformation_func,
            special_mask=False,  # No special tokens mask
            q_max_length=30,
            d_max_length=200
        )
        
        # Example input batch
        batch = [
            ("query1", ["doc1", "doc2"]),
        ]
        
        # Call the collator
        output = collator(batch)
        
        # Special tokens mask should not be returned in this case
        self.assertNotIn("special_tokens_mask", output["queries"])
        self.assertNotIn("special_tokens_mask", output["documents"])
        self.assertNotIn("special_tokens_mask", output["perturbed_documents"])
        
    def test_transformation_function(self):
        # Ensure transformation function is called correctly
        collator = DotDataCollator(
            tokenizer=self.tokenizer,
            transformation_func=self.transformation_func,
            q_max_length=30,
            d_max_length=200
        )
        
        # Input batch
        batch = [
            ("query1", ["doc1", "doc2"]),
        ]
        
        # Call the collator
        output = collator(batch)
        
        # Verify that the transformation function was called for each document
        expected_perturbed_docs = ["doc1_perturbed", "doc2_perturbed"]
        actual_perturbed_docs = output["perturbed_documents"]['input_ids']  # Mocked ids, so just check the length
        
        self.assertEqual(len(actual_perturbed_docs), 2)  # Should have perturbed both documents
    
if __name__ == '__main__':
    unittest.main()
