import pytest
from unittest.mock import patch, MagicMock
from src.tools.api import get_prices, prices_to_df, Price

@patch('src.tools.api.requests.get')
def test_get_prices_success(mock_get):
    """
    Tests the get_prices function for a successful API call.
    """
    # Arrange
    mock_response = MagicMock()
    mock_response.status_code = 200
    # The mock response must match the PriceResponse Pydantic model
    mock_response.json.return_value = {
        "ticker": "TEST",
        "prices": [
            {"time": "2023-01-01", "open": 99, "high": 101, "low": 98, "close": 100, "volume": 1000},
            {"time": "2023-01-02", "open": 100, "high": 103, "low": 99, "close": 102, "volume": 1200}
        ]
    }
    mock_get.return_value = mock_response

    # Act
    prices = get_prices("TEST", "2023-01-01", "2023-01-02")

    # Assert
    assert prices is not None
    assert len(prices) == 2
    assert prices[0].close == 100

@patch('src.tools.api.requests.get')
def test_get_prices_failure(mock_get):
    """
    Tests the get_prices function for a failed API call.
    """
    # Arrange
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found" # Add text for the exception message
    mock_get.return_value = mock_response

    # Act & Assert
    with pytest.raises(Exception, match="Error fetching data: FAIL - 404 - Not Found"):
        get_prices("FAIL", "2023-01-01", "2023-01-02")

def test_prices_to_df():
    """
    Tests the prices_to_df function to ensure it correctly converts
    the price list to a pandas DataFrame.
    """
    # Arrange
    # The function expects a list of Price objects, not dicts
    prices = [
        Price(time="2023-01-01", open=99, high=101, low=98, close=100, volume=1000),
        Price(time="2023-01-02", open=100, high=103, low=99, close=102, volume=1200)
    ]

    # Act
    df = prices_to_df(prices)

    # Assert
    assert not df.empty
    assert "close" in df.columns
    assert len(df) == 2
    assert df["close"].iloc[0] == 100 