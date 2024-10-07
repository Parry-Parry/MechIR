import unittest
from unittest.mock import MagicMock, patch
from collections import Counter
from my_module import IndexPerturbation  # Adjust the import to your actual module

class TestIndexPerturbation(unittest.TestCase):

    def setUp(self):
        # Mock TerrierIndexContext and its methods
        self.mock_context = MagicMock()
        self.mock_context.terms.return_value = ["term1", "term2", "term1"]
        self.mock_context.term_frequency.return_value = 2
        self.mock_context.inverse_document_frequency.return_value = 0.5

        # Patch the TerrierIndexContext used within IndexPerturbation
        patcher = patch('my_module.TerrierIndexContext', return_value=self.mock_context)
        self.addCleanup(patcher.stop)
        self.mock_context_cls = patcher.start()

        # Initialize IndexPerturbation with mock values
        self.index_perturbation = IndexPerturbation(index_location="mock_index")

    def test_get_terms(self):
        text = "sample document"
        terms = self.index_perturbation.get_terms(text)
        self.mock_context.terms.assert_called_once_with(text)
        self.assertEqual(terms, ["term1", "term2", "term1"])

    def test_get_counts(self):
        text = "sample document"
        counts = self.index_perturbation.get_counts(text)
        self.mock_context.terms.assert_called_once_with(text)
        self.assertEqual(counts, Counter(["term1", "term2", "term1"]))

    def test_get_tf(self):
        term = "term1"
        text = "sample document"
        tf = self.index_perturbation.get_tf(term, text)
        self.mock_context.term_frequency.assert_called_once_with(text, term)
        self.assertEqual(tf, 2)

    def test_get_tf_text(self):
        text = "sample document"
        tf_dict = self.index_perturbation.get_tf_text(text)
        self.mock_context.terms.assert_called_once_with(text)
        self.mock_context.term_frequency.assert_any_call(text, "term1")
        self.mock_context.term_frequency.assert_any_call(text, "term2")
        self.assertEqual(tf_dict, {"term1": 2, "term2": 2})

    def test_get_idf(self):
        term = "term1"
        text = "sample document"
        idf = self.index_perturbation.get_idf(term, text)
        self.mock_context.inverse_document_frequency.assert_called_once_with(term)
        self.assertEqual(idf, 0.5)

    def test_get_idf_text(self):
        text = "sample document"
        idf_dict = self.index_perturbation.get_idf_text(text)
        self.mock_context.terms.assert_called_once_with(text)
        self.mock_context.inverse_document_frequency.assert_any_call("term1")
        self.mock_context.inverse_document_frequency.assert_any_call("term2")
        self.assertEqual(idf_dict, {"term1": 0.5, "term2": 0.5})

    def test_get_tfidf(self):
        term = "term1"
        text = "sample document"
        tfidf = self.index_perturbation.get_tfidf(term, text)
        self.mock_context.term_frequency.assert_called_once_with(text, term)
        self.mock_context.inverse_document_frequency.assert_called_once_with(term)
        self.assertEqual(tfidf, 2 * 0.5)

    def test_get_tfidf_text(self):
        text = "sample document"
        tfidf_dict = self.index_perturbation.get_tfidf_text(text)
        self.mock_context.terms.assert_called_once_with(text)
        self.mock_context.term_frequency.assert_any_call(text, "term1")
        self.mock_context.inverse_document_frequency.assert_any_call("term2")
        self.assertEqual(tfidf_dict, {"term1": 2 * 0.5, "term2": 2 * 0.5})

if __name__ == '__main__':
    unittest.main()
