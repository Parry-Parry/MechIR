import unittest
import pyterrier as pt 
if not pt.started(): pt.init()
from mechir.perturb import AbstractPerturbation, IdentityPerturbation, perturbation

class TestAbstractPerturbation(unittest.TestCase):
    
    def test_abstract_perturbation_instantiation(self):
        # Should raise error when trying to instantiate the abstract class
        with self.assertRaises(TypeError):
            AbstractPerturbation()

    def test_abstract_perturbation_apply_not_implemented(self):
        # Create a mock subclass and ensure the apply method raises the correct error
        class MockPerturbation(AbstractPerturbation):
            pass
        
        with self.assertRaises(NotImplementedError):
            mock_perturbation = MockPerturbation()
            mock_perturbation.apply("doc", "query")
    
    def test_abstract_perturbation_call_method(self):
        # Subclass with implementation
        class MockPerturbation(AbstractPerturbation):
            def apply(self, document, query=None):
                return "mock result"
        
        mock_perturbation = MockPerturbation()
        result = mock_perturbation("doc", "query")
        self.assertEqual(result, "mock result")

class TestIdentityPerturbation(unittest.TestCase):

    def test_identity_perturbation(self):
        perturbation = IdentityPerturbation()
        document = "This is a test document"
        result = perturbation(document)
        self.assertEqual(result, document)
    
    def test_identity_perturbation_with_query(self):
        perturbation = IdentityPerturbation()
        document = "This is a test document"
        query = "test query"
        result = perturbation(document, query)
        self.assertEqual(result, document)

class TestCustomPerturbationDecorator(unittest.TestCase):

    def test_custom_perturbation_no_query(self):
        # Function that doesn't need a query
        @perturbation
        def mock_perturbation(document):
            return document.upper()

        document = "test document"
        result = mock_perturbation(document)
        self.assertEqual(result, "TEST DOCUMENT")

    def test_custom_perturbation_with_query(self):
        # Function that takes both document and query
        @perturbation
        def mock_perturbation(document, query):
            return f"{document} {query}"

        document = "test document"
        query = "test query"
        result = mock_perturbation(document, query)
        self.assertEqual(result, "test document test query")

if __name__ == "__main__":
    unittest.main()
