import random

import numpy as np

from image_tagger.util import Credentials, TemporarySeed, human_join, total_size
from tests._workflow_split import export_tests

export_tests(globals(), "core")


def test_human_join_handles_common_lengths() -> None:
    """Format one, two, and many items with a conjunction."""
    assert human_join([]) == ""
    assert human_join(["one"]) == "one"
    assert human_join(["one", "two"]) == "one and two"
    assert human_join(["one", "two", "three"], conjunction="or") == "one, two, or three"


def test_temporary_seed_restores_random_state() -> None:
    """Restore generator state after exiting the context manager."""
    random.seed(12345)
    before = random.random()
    with TemporarySeed(99):
        seeded_value = random.random()
    after = random.random()

    random.seed(12345)
    _ = random.random()
    expected_after = random.random()
    with TemporarySeed(99):
        assert random.random() == seeded_value

    assert before == 0.41661987254534116
    assert after == expected_after


def test_credentials_repr_redacts_secrets() -> None:
    """Hide sensitive values while preserving non-secret keys."""
    creds = Credentials(
        {
            "api_key": "abc",
            "access_token": "def",
            "password": "ghi",
            "endpoint": "https://example.com",
        }
    )
    rendered = repr(creds)

    assert "endpoint" in rendered
    assert "https://example.com" in rendered
    assert "abc" not in rendered
    assert "def" not in rendered
    assert "ghi" not in rendered
    assert rendered.count("********") == 3


def test_total_size_handles_cycles_and_ndarray() -> None:
    """Avoid infinite recursion and account for ndarray nbytes."""
    cyclical: list[object] = []
    cyclical.append(cyclical)
    assert total_size(cyclical) > 0

    array = np.ones((4, 5), dtype=np.float64)
    assert total_size(array) == array.nbytes
