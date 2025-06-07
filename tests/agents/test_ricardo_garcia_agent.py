import pytest
from unittest.mock import patch, MagicMock
from src.agents.ricardo_garcia import ricardo_garcia_agent
from src.graph.state import AgentState

@pytest.fixture
def initial_state():
    """Fixture for the initial state of the agent."""
    return AgentState(
        messages=[],
        data={
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "tickers": ["TEST"],
            "analyst_signals": {},
        },
        metadata={
            "model_name": "test_model",
            "model_provider": "test_provider",
            "show_reasoning": False,
        },
    )

@patch('src.agents.ricardo_garcia.generate_ricardo_garcia_signals')
def test_ricardo_garcia_agent_success(mock_generate_signals, initial_state):
    """
    Tests the ricardo_garcia_agent to ensure it processes data correctly
    and returns the expected state.
    """
    # Arrange: Mock the output of the signal generation function
    mock_analysis = {
        "TEST": {
            "signal": "bullish",
            "confidence": 80,
            "reasoning": "Strong fundamentals.",
            "llm_signal": "bullish",
            "llm_confidence": 0.8,
            "llm_reasoning": "AI predicts upward movement.",
            "ml_data": {"feature1": 1.0}
        }
    }
    mock_generate_signals.return_value = mock_analysis

    # Act: Run the agent
    result_state = ricardo_garcia_agent(initial_state)

    # Assert: Check that the state was updated correctly
    mock_generate_signals.assert_called_once_with(
        ["TEST"], "2023-01-01", "2023-01-31", "test_model", "test_provider"
    )

    assert "ricardo_garcia_agent" in result_state["data"]["analyst_signals"]
    assert result_state["data"]["analyst_signals"]["ricardo_garcia_agent"] == mock_analysis
    assert len(result_state["messages"]) == 1
    message = result_state["messages"][0]
    assert message.name == "ricardo_garcia_agent" 