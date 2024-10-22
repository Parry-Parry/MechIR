import unittest
from unittest.mock import MagicMock, patch
import pyterrier as pt 
if not pt.started(): pt.init()
from mechir.perturb.axiom.frequency import FrequencyPerturbation  


class TestFrequencyPerturbation(unittest.TestCase):

    def setUp(self):
        # Mocking IndexPerturbation and its methods
        self.mock_index_perturbation = MagicMock()
        
        patcher = patch('my_module.IndexPerturbation', return_value=self.mock_index_perturbation)
        self.addCleanup(patcher.stop)
        patcher.start()

        # Initialize FrequencyPerturbation with mock values
        self.frequency_perturbation = FrequencyPerturbation(index_location="mock_index")

        # Mock get_tf_text method to provide consistent results
        self.frequency_perturbation.get_tf_text = MagicMock(return_value={
            "term1": 3,
            "term2": 5,
            "term3": 1
        })

    def test_init_with_max_mode(self):
        perturbation = FrequencyPerturbation("mock_index", mode="max")
        self.assertEqual(perturbation.get_freq_terms, perturbation._get_max_freq_terms)

    def test_init_with_top_k_mode(self):
        perturbation = FrequencyPerturbation("mock_index", mode="top_k")
        self.assertEqual(perturbation.get_freq_terms, perturbation._get_top_k_freq_terms)

    def test_init_with_min_mode(self):
        perturbation = FrequencyPerturbation("mock_index", mode="min")
        self.assertEqual(perturbation.get_freq_terms, perturbation._get_min_freq_terms)

    def test_get_top_k_freq_terms(self):
        terms = self.frequency_perturbation._get_top_k_freq_terms("sample text")
        self.frequency_perturbation.get_tf_text.assert_called_once_with("sample text")
        self.assertEqual(terms, [("term2", 5), ("term1", 3)])

    def test_get_max_freq_terms(self):
        terms = self.frequency_perturbation._get_max_freq_terms("sample text")
        self.frequency_perturbation.get_tf_text.assert_called_once_with("sample text")
        self.assertEqual(terms, [5, 5])

    def test_get_min_freq_terms(self):
        terms = self.frequency_perturbation._get_min_freq_terms("sample text")
        self.frequency_perturbation.get_tf_text.assert_called_once_with("sample text")
        self.assertEqual(terms, [1, 1])

    def test_get_random_terms(self):
        random_terms = self.frequency_perturbation._get_random_terms("sample text")
        self.frequency_perturbation.get_tf_text.assert_called_once_with("sample text")
        self.assertEqual(len(random_terms), 1)  # Default num_additions is 1

    def test_apply_with_query_target(self):
        document = "This is a sample document."
        query = "sample query"
        self.frequency_perturbation.get_freq_terms = MagicMock(return_value=["term1", "term2"])
        
        result = self.frequency_perturbation.apply(document, query)
        self.assertEqual(result, f"{document} term1 term2")  # Assuming default loc is 'end'

    def test_apply_with_document_target(self):
        document = "This is a sample document."
        query = "sample query"
        self.frequency_perturbation.target = "document"
        self.frequency_perturbation.get_freq_terms = MagicMock(return_value=["term1", "term2"])

        result = self.frequency_perturbation.apply(document, query)
        self.assertEqual(result, f"term1 term2 {document}")  # Assuming default loc is 'start'

if __name__ == '__main__':
    unittest.main()