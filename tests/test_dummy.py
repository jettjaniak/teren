import pytest
import torch
from beartype.roar import BeartypeException

from teren.dummy import dummy_function


def test_dummy_function():

    float_input = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [1.1, 1.2, 1.3],
        ]
    )

    assert torch.all(dummy_function(float_input) == torch.tensor([1, 2]))
    int_input = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    with pytest.raises(BeartypeException):
        dummy_function(int_input)
    flat_input = torch.tensor([0.1, 0.2, 0.3])
    with pytest.raises(BeartypeException):
        dummy_function(flat_input)
