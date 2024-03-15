import pytest
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import generate_predictions


@pytest.fixture
def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[:1]
    result = generate_predictions(single_row)
    return result

# output is not none
def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None

# data type is string
def test_single_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get('prediction')[0],str)

# check the output is Y
def test_single_pred_validate(single_prediction):
    assert single_prediction.get('prediction')[0] == 'Y'