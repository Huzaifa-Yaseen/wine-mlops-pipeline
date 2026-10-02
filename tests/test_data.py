"""Unit tests for Wine dataset pipeline."""
from src.data import load_and_validate_data, get_train_test_data, EXPECTED_FEATURE_COUNT


def test_load_and_validate_data():
    """Verify feature dimension and absence of missing values."""
    x_data, y_data = load_and_validate_data()
    assert x_data.shape[1] == EXPECTED_FEATURE_COUNT
    assert not x_data.isnull().values.any()
    assert not y_data.isnull().values.any()


def test_stratified_split_proportions():
    """Verify 80/20 train-test split and sample consistency."""
    X_train, X_test, y_train, y_test = get_train_test_data()
    total_samples = len(X_train) + len(X_test)
    assert total_samples == 178
    assert len(X_test) == 36
    assert len(X_train) == 142
    assert set(y_train.unique()) == {0, 1, 2}
