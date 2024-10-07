import unittest
import torch
from torch import Tensor
from my_module import PatchingOutput, batched_dot_product, linear_rank_function, seed_everything, is_pyterrier_availible, is_ir_axioms_availible, is_ir_datasets_availible

class TestPatchingOutput(unittest.TestCase):
    def test_patching_output_initialization(self):
        # Create sample tensors
        result = torch.tensor([1, 2, 3])
        scores = torch.tensor([0.5, 0.2, 0.8])
        scores_p = torch.tensor([0.7, 0.4, 0.9])

        # Initialize PatchingOutput object
        patching_output = PatchingOutput(result=result, scores=scores, scores_p=scores_p)

        # Assert values are correctly assigned
        self.assertTrue(torch.equal(patching_output.result, result))
        self.assertTrue(torch.equal(patching_output.scores, scores))
        self.assertTrue(torch.equal(patching_output.scores_p, scores_p))


class TestBatchedDotProduct(unittest.TestCase):
    def test_batched_dot_product_3d(self):
        # 3D tensors
        a = torch.randn(2, 1, 3)
        b = torch.randn(2, 4, 3)

        result = batched_dot_product(a, b)
        
        # Manually compute dot product
        expected_result = torch.bmm(a, b.permute(0, 2, 1)).squeeze(1)
        self.assertTrue(torch.allclose(result, expected_result))

    def test_batched_dot_product_2d(self):
        # 2D tensors
        a = torch.randn(3, 1)
        b = torch.randn(3, 3)

        result = batched_dot_product(a, b)
        
        # Manually compute dot product
        expected_result = torch.matmul(a, b.T)
        self.assertTrue(torch.allclose(result, expected_result))

class TestLinearRankFunction(unittest.TestCase):
    def test_linear_rank_function(self):
        patch_score = torch.tensor([0.5, 0.6, 0.7])
        score = torch.tensor([0.3, 0.4, 0.5])
        score_p = torch.tensor([1.0, 1.0, 1.0])

        result = linear_rank_function(patch_score, score, score_p)
        
        expected_result = (patch_score - score) / (score_p - score)
        self.assertTrue(torch.allclose(result, expected_result))

    def test_linear_rank_function_edge_case(self):
        # Test for division by zero edge case
        patch_score = torch.tensor([0.5, 0.6])
        score = torch.tensor([0.5, 0.6])  # score equals score_p, should handle division by zero
        score_p = torch.tensor([0.5, 0.6])

        with self.assertRaises(ZeroDivisionError):
            linear_rank_function(patch_score, score, score_p)

class TestSeedEverything(unittest.TestCase):
    def test_seed_everything(self):
        seed_everything(42)
        # Test that random and torch generate the same numbers after setting seed
        random_val_1 = torch.rand(1)
        seed_everything(42)
        random_val_2 = torch.rand(1)
        self.assertTrue(torch.equal(random_val_1, random_val_2))

class TestDependencyChecks(unittest.TestCase):
    def test_is_pyterrier_availible(self):
        # Mock ImportError for pyterrier
        with unittest.mock.patch("builtins.__import__", side_effect=ImportError):
            self.assertFalse(is_pyterrier_availible())

        # Simulate pyterrier being available
        with unittest.mock.patch("builtins.__import__", return_value=True):
            self.assertTrue(is_pyterrier_availible())

    def test_is_ir_axioms_availible(self):
        # Mock ImportError for ir_axioms
        with unittest.mock.patch("builtins.__import__", side_effect=ImportError):
            self.assertFalse(is_ir_axioms_availible())

        # Simulate ir_axioms being available
        with unittest.mock.patch("builtins.__import__", return_value=True):
            self.assertTrue(is_ir_axioms_availible())

    def test_is_ir_datasets_availible(self):
        # Mock ImportError for ir_datasets
        with unittest.mock.patch("builtins.__import__", side_effect=ImportError):
            self.assertFalse(is_ir_datasets_availible())

        # Simulate ir_datasets being available
        with unittest.mock.patch("builtins.__import__", return_value=True):
            self.assertTrue(is_ir_datasets_availible())

if __name__ == "__main__":
    unittest.main()
