"""Test the abstract dataset."""
from pathlib import Path
from typing import Any, TypedDict

from datasets import IterableDataset, load_dataset
import pytest

from sparse_autoencoder.source_data.abstract_dataset import (
    PreprocessTokenizedPrompts,
    SourceDataset,
)


TEST_CONTEXT_SIZE: int = 4


class MockHuggingFaceDatasetItem(TypedDict):
    """Mock Hugging Face dataset item typed dict."""

    text: str
    meta: dict


class MockSourceDataset(SourceDataset[MockHuggingFaceDatasetItem]):
    """Mock source dataset for testing the inherited abstract dataset."""

    def preprocess(
        self,
        source_batch: MockHuggingFaceDatasetItem,  # noqa: ARG002
        *,
        context_size: int,  # noqa: ARG002
    ) -> PreprocessTokenizedPrompts:
        """Preprocess a batch of prompts."""
        # Assuming a very simple preprocess that just tokenizes the 'text' field
        tokenized_texts = [[1, 5, 23, 2], [1, 2, 61, 12]]
        return {"input_ids": tokenized_texts}

    def __init__(
        self,
        dataset_path: str = "mock_dataset_path",
        dataset_split: str = "test",
        context_size: int = TEST_CONTEXT_SIZE,
        buffer_size: int = 1000,
        preprocess_batch_size: int = 1000,
    ):
        """Initialise the dataset."""
        super().__init__(
            dataset_path,
            dataset_split,
            context_size,
            buffer_size,
            preprocess_batch_size,
        )


@pytest.fixture()
def mock_hugging_face_load_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the `load_dataset` function from Hugging Face.

    Instead load the text data from mocks/text_dataset.txt, using a restored `load_dataset` method.
    """

    def mock_load_dataset(*args: Any, **kwargs: Any) -> IterableDataset:  # noqa:   ANN401
        """Mock load dataset function."""
        mock_path = Path(__file__).parent / "mocks" / "text_dataset.txt"
        return load_dataset(
            "text", data_files={"train": [str(mock_path)]}, streaming=True, split="train"
        )  # type: ignore

    monkeypatch.setattr(
        "sparse_autoencoder.source_data.abstract_dataset.load_dataset", mock_load_dataset
    )


def test_extended_dataset_initialization(mock_hugging_face_load_dataset: pytest.Function) -> None:
    """Test the initialization of the extended dataset."""
    data = MockSourceDataset()
    assert data is not None
    assert isinstance(data, SourceDataset)


def test_extended_dataset_iterator(mock_hugging_face_load_dataset: pytest.Function) -> None:
    """Test the iterator of the extended dataset."""
    data = MockSourceDataset()
    iterator = iter(data)
    assert iterator is not None

    first_item = next(iterator)
    assert len(first_item["input_ids"]) == TEST_CONTEXT_SIZE
