import pytest


@pytest.mark.integration
def test_dot_model_end_to_end():
    # TODO: unimplemented. The body was left as a bare `tokenizer` expression,
    # which raises NameError, so this failed rather than passing vacuously
    # whenever it was collected. Skipped explicitly until a real end-to-end
    # case is written: build a Dot model, apply a perturbation, patch, and
    # assert on the resulting scores.
    pytest.skip("end-to-end Dot coverage not yet written")
