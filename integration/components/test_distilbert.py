import unittest
import torch
from transformer_lens import HookedTransformerConfig, ActivationCache
from mechir.modelling.hooked.HookedDistilBert import HookedDistilBert, HookedDistilBertForSequenceClassification

class TestHookedDistilBert(unittest.TestCase):
    
    def setUp(self):
        self.cfg = HookedTransformerConfig(
            n_layers=4,
            n_heads=8,
            d_model=256,
            d_vocab=1000,
            n_devices=1,
            device='cpu'
        )
        self.model = HookedDistilBert(self.cfg)
        self.input_tensor = torch.randint(0, self.cfg.d_vocab, (2, 10))  # (batch_size, sequence_length)

    def test_initialization(self):
        self.assertEqual(self.model.cfg.n_layers, 4)
        self.assertEqual(self.model.embed.embed.W_E.shape, (self.cfg.d_vocab, self.cfg.d_model))
        self.assertEqual(len(self.model.blocks), self.cfg.n_layers)

    def test_forward_logits(self):
        output = self.model(self.input_tensor, return_type="logits")
        self.assertEqual(output.shape, (2, self.cfg.d_vocab))  # Check output shape for logits

    def test_forward_embeddings(self):
        output = self.model(self.input_tensor, return_type="embeddings")
        self.assertEqual(output.shape, (2, 10, self.cfg.d_model))  # Check output shape for embeddings

    def test_forward_no_return(self):
        output = self.model(self.input_tensor, return_type=None)
        self.assertIsNone(output)  # Check output is None when return_type is None

    def test_run_with_cache(self):
        output, cache = self.model.run_with_cache(self.input_tensor, return_cache_object=True)
        self.assertEqual(output.shape, (2, self.cfg.d_vocab))
        self.assertIsInstance(cache, ActivationCache)

    def test_from_pretrained(self):
        model_name = "distilbert-base-uncased"
        model = HookedDistilBert.from_pretrained(model_name)
        self.assertIsNotNone(model)  # Check if model is loaded
        self.assertEqual(model.cfg.d_vocab, 30522)  # Check vocab size of the pretrained model
    
class TestHookedDistilBertForSequenceClassification(unittest.TestCase):

    def setUp(self):
        self.cfg = HookedTransformerConfig(
            n_layers=4,
            n_heads=8,
            d_model=256,
            d_vocab=1000,
            n_labels=2,
            n_devices=1,
            device='cpu'
        )
        self.model = HookedDistilBertForSequenceClassification(self.cfg)
        self.input_tensor = torch.randint(0, self.cfg.d_vocab, (2, 10))  # (batch_size, sequence_length)

    def test_initialization(self):
        self.assertIsInstance(self.model.classifier, nn.Linear)  # Check classifier initialization
        self.assertEqual(self.model.classifier.in_features, self.cfg.d_model)
        self.assertEqual(self.model.classifier.out_features, self.cfg.n_labels)

    def test_forward_logits(self):
        output = self.model(self.input_tensor, return_type="logits")
        self.assertEqual(output.shape, (2, self.cfg.n_labels))  # Check output shape for logits

    def test_forward_no_return(self):
        output = self.model(self.input_tensor, return_type=None)
        self.assertIsNone(output)  # Check output is None when return_type is None

    def test_run_with_cache(self):
        output, cache = self.model.run_with_cache(self.input_tensor, return_cache_object=True)
        self.assertEqual(output.shape, (2, self.cfg.n_labels))
        self.assertIsInstance(cache, ActivationCache)

if __name__ == "__main__":
    unittest.main()
