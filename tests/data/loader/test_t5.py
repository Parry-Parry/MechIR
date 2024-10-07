import unittest
from unittest.mock import Mock
from mechir.data import MonoT5DataCollator, pad

class TestMonoT5DataCollator(unittest.TestCase):

    def setUp(self):
        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.return_tensors = "pt"
        self.tokenizer.side_effect = lambda texts, *args, **kwargs: {
            'input_ids': [[1, 2, 3]] * len(texts),  # Mock input_ids
            'attention_mask': [[1, 1, 1]] * len(texts),  # Mock attention mask
        }

        # Simple transformation function that adds "_perturbed" to each document
        self.transformation_func = lambda doc, query: (doc, f"{doc}_perturbed")
        
        # Mock pad function (since it comes from an external source)
        global pad
        pad = Mock(side_effect=lambda a, b, special_token=None: (a, b))  # Simply return a and b unmodified
        
    def test_monoT5_data_collator(self):
        # Instantiate MonoT5DataCollator
        collator = MonoT5DataCollator(
            tokenizer=self.tokenizer,
            transformation_func=self.transformation_func,
            q_max_length=30,
            d_max_length=200
        )
        
        # Example input batch
        batch = [
            ("query1", ["doc1", "doc2"]),  # query, documents
            ("query2", ["doc3"]),          # another query, single document
        ]
        
        # Call the collator
        output = collator(batch)
        
        # Verify that the output contains expected keys
        self.assertIn("sequences", output)
        self.assertIn("perturbed_sequences", output)
        
        # Check the structure of tokenized outputs
        self.assertEqual(len(output["sequences"]['input_ids']), 3)  # 3 total query-doc pairs
        self.assertEqual(len(output["perturbed_sequences"]['input_ids']), 3)  # 3 total perturbed pairs
        
    def test_prompt_generation(self):
        # Test the prompt generation function
        query = "example_query"
        document = "example_document"
        expected_prompt = "query: example_query document: example_document relevant:"
        
        # Check that the prompt method produces the expected prompt string
        self.assertEqual(MonoT5DataCollator.prompt(query, document), expected_prompt)

    def test_transformation_function(self):
        # Ensure transformation function is called correctly
        collator = MonoT5DataCollator(
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
        actual_perturbed_docs = output["perturbed_sequences"]['input_ids']  # Mocked ids, so just check the length
        
        self.assertEqual(len(actual_perturbed_docs), 2)  # Should have perturbed both documents
    
if __name__ == '__main__':
    unittest.main()