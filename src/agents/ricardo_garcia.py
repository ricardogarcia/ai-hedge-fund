import math

from langchain_core.messages import HumanMessage
from datetime import datetime
from graph.state import AgentState, show_agent_reasoning

import json
import pandas as pd
import numpy as np

from tools.api import get_prices, prices_to_df, get_next_earnings_date
from utils.progress import progress
from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm



class RicardoGarciaSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


RICARDO_GARCIA_STRATEGY = """
    Ricardo Garcia is a financial analyst who uses the following principles in most important to least important order:

    1. He invests mostly when the overall S&P 500 is down 10% or more over the last month and when a particular stock has strong revenue and earnings growth.
    2. He sells when the stock has had strong gains and is overbought on the RSI.  
    3. He never buys stocks until a 2-4 weeks before earnings
    4. The most important principle of his strategy is he usually sells a week or two after the stock has had strong gains or earnings reporrts.
    5. The penultimate strategy is to invest in companies that are using disruptive technology to grow their business.
    especially in the AI, robotics, and biotechnology sectors.  We should always take into account the potential for a strong upcoming earnings report and not invest in companies right after earnings.  Buy companies in the weeks running up to earnings, usually 2-4 weeks.  
    6. Lastly, almost always invest in companies where the marjority shareowners have funded the policitians that are currently in power.  This would mean that those companies stand to beneft from the policies of the current administration.

    An example of this is Jeff Bezos, who is the majority shareholder of Amazon.  He has funded many politicians who are in power and has a lot of influence over the policies of the United States.
    Another example is Elon Musk, who is the majority shareholder of Tesla.  He has funded many politicians who are in power and has a lot of influence over the policies of the United States.
    Another example is Mark Zuckerberg, who is the majority shareholder of Facebook.  He has funded many politicians who are in power and has a lot of influence over the policies of the United States.
    Another example is Tim Cook, who is the majority shareholder of Apple.  He has funded many politicians who are in power and has a lot of influence over the policies of the United States.
    Another example of this is Peter Thiel, who is the majority shareholder of Palantir.  He has funded many politicians who are in power and has a lot of influence over the policies of the United States.
    Another example of this is Jack Ma, who is the majority shareholder of Alibaba.  He has funded many politicians who are in power and has a lot of influence over the policies of the United States.
    """

def ricardo_garcia_agent(state: AgentState):
   
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    data["ricardo_garcia_strategy"] = RICARDO_GARCIA_STRATEGY

    # Initialize analysis for each ticker
    ricardo_garcia_analysis = {}
    for ticker in tickers:
        progress.update_status("ricardo_garcia_agent", ticker, "Analyzing price data")

        # Get the historical price data
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        if not prices:
            progress.update_status("ricardo_garcia_agent", ticker, "Failed: No price data found")
            continue

        # Convert prices to a DataFrame
        prices_df = prices_to_df(prices)

        progress.update_status("ricardo_garcia_agent", ticker, "Calculating trend signals")
        trend_signals = calculate_trend_signals(prices_df)

        progress.update_status("ricardo_garcia_agent", ticker, "Calculating mean reversion")
        mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

        progress.update_status("ricardo_garcia_agent", ticker, "Calculating momentum")
        momentum_signals = calculate_momentum_signals(prices_df)

        progress.update_status("ricardo_garcia_agent", ticker, "Analyzing volatility")
        volatility_signals = calculate_volatility_signals(prices_df)

        progress.update_status("ricardo_garcia_agent", ticker, "Statistical analysis")
        market_analysis_signals = calculate_market_analysis(prices_df)

        # Combine all signals using a weighted ensemble approach
        strategy_weights = {
            "trend": 0.10,
            "mean_reversion": 0.10,
            "momentum": 0.10,
            "volatility": 0.10,
            "market_analysis": 0.60,
        }

        progress.update_status("ricardo_garcia_agent", ticker, "Fetching financial metrics")
        # You can adjust these parameters (period="annual"/"ttm", limit=5/10, etc.)
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)

        progress.update_status("ricardo_garcia_agent", ticker, "Gathering financial line items")
        # Request multiple periods of data (annual or TTM) for a more robust view.
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "gross_margin",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                "research_and_development",
                "capital_expenditure",
                "operating_expense",
            ],
            end_date,
            period="annual",
            limit=5
        )

        # Transform to dictionary
        line_items_dict = {
            item.report_period: {
                k: v for k, v in item.model_dump().items() 
                if k not in ['ticker', 'report_period', 'period', 'currency']
            }
            for item in financial_line_items
        }

        progress.update_status("ricardo_garcia_agent", ticker, "Combining signals")
        combined_signal = weighted_signal_combination(
            {
                "trend": trend_signals,
                "mean_reversion": mean_reversion_signals,
                "momentum": momentum_signals,
                "volatility": volatility_signals,
                "market_analysis": market_analysis_signals,
            },
            strategy_weights,
        )

        # Generate detailed analysis report for this ticker
        ricardo_garcia_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "strategy_signals": {
                "trend_following": {
                    "signal": trend_signals["signal"],
                    "confidence": round(trend_signals["confidence"] * 100),
                    "metrics": normalize_pandas(trend_signals["metrics"]),
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": round(mean_reversion_signals["confidence"] * 100),
                    "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": round(momentum_signals["confidence"] * 100),
                    "metrics": normalize_pandas(momentum_signals["metrics"]),
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": round(volatility_signals["confidence"] * 100),
                    "metrics": normalize_pandas(volatility_signals["metrics"]),
                },
                "market_analysis": {
                    "signal": market_analysis_signals["signal"],
                    "confidence": round(market_analysis_signals["confidence"] * 100),
                    "metrics": normalize_pandas(market_analysis_signals["metrics"]),
                },
            },
        }


        # Generate a Ricardo Garcia-style investment signal
        ricardo_garcia_signal = generate_ricardo_garcia_output(
            ticker=ticker,
            analysis_data=ricardo_garcia_analysis[ticker],
            financial_data=line_items_dict,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        ricardo_garcia_analysis[ticker] = {
            "signal": ricardo_garcia_signal.signal,
            "confidence": ricardo_garcia_signal.confidence,
            "reasoning": ricardo_garcia_signal.reasoning
        }
        progress.update_status("ricardo_garcia_agent", ticker, "Done")

    # Create the technical analyst message
    message = HumanMessage(
        content=json.dumps(ricardo_garcia_analysis),
        name="ricardo_garcia_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(ricardo_garcia_analysis, "Ricardo Garcia")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["ricardo_garcia_agent"] = ricardo_garcia_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def calculate_trend_signals(prices_df):
    """
    Advanced trend following strategy using multiple timeframes and indicators
    """
    # Calculate EMAs for multiple timeframes
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)

    # Calculate ADX for trend strength
    adx = calculate_adx(prices_df, 14)

    # Determine trend direction and strength
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55

    # Combine signals with confidence weighting
    trend_strength = adx["adx"].iloc[-1] / 100.0

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "bullish"
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": float(adx["adx"].iloc[-1]),
            "trend_strength": float(trend_strength),
        },
    }


def calculate_mean_reversion_signals(prices_df):
    """
    Mean reversion strategy using statistical measures and Bollinger Bands
    """
    # Calculate z-score of price relative to moving average
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50

    # Calculate Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)

    # Calculate RSI with multiple timeframes
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    # Mean reversion signals
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # Combine signals
    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = "bullish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = "bearish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": float(z_score.iloc[-1]),
            "price_vs_bb": float(price_vs_bb),
            "rsi_14": float(rsi_14.iloc[-1]),
            "rsi_28": float(rsi_28.iloc[-1]),
        },
    }


def calculate_momentum_signals(prices_df):
    """
    Multi-factor momentum strategy
    """
    # Price momentum
    returns = prices_df["close"].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()

    # Volume momentum
    volume_ma = prices_df["volume"].rolling(21).mean()
    volume_momentum = prices_df["volume"] / volume_ma

    # Relative strength
    # (would compare to market/sector in real implementation)

    # Calculate momentum score
    momentum_score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m).iloc[-1]

    # Volume confirmation
    volume_confirmation = volume_momentum.iloc[-1] > 1.0

    if momentum_score > 0.05 and volume_confirmation:
        signal = "bullish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal = "bearish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": float(mom_1m.iloc[-1]),
            "momentum_3m": float(mom_3m.iloc[-1]),
            "momentum_6m": float(mom_6m.iloc[-1]),
            "volume_momentum": float(volume_momentum.iloc[-1]),
        },
    }


def calculate_volatility_signals(prices_df):
    """
    Volatility-based trading strategy
    """
    # Calculate various volatility metrics
    returns = prices_df["close"].pct_change()

    # Historical volatility
    hist_vol = returns.rolling(21).std() * math.sqrt(252)

    # Volatility regime detection
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma

    # Volatility mean reversion
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    # ATR ratio
    atr = calculate_atr(prices_df)
    atr_ratio = atr / prices_df["close"]

    # Generate signal based on volatility regime
    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if current_vol_regime < 0.8 and vol_z < -1:
        signal = "bullish"  # Low vol regime, potential for expansion
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal = "bearish"  # High vol regime, potential for contraction
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": float(hist_vol.iloc[-1]),
            "volatility_regime": float(current_vol_regime),
            "volatility_z_score": float(vol_z),
            "atr_ratio": float(atr_ratio.iloc[-1]),
        },
    }


def calculate_market_analysis(prices_df):
    """
    Market analysis signals based on price action and market timing analysis
    """
    # Calculate price distribution statistics
    returns = prices_df["close"].pct_change()

    # Generate signal based on statistical properties
    if returns.iloc[-1] > -0.1:
        signal = "bullish"
        confidence = .8
    elif returns.iloc[-1] > 0.1:
        signal = "bearish"
        confidence = .8
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "return": float(returns.iloc[-1]),
        },
    }


def weighted_signal_combination(signals, weights):
    """
    Combines multiple trading signals using a weighted approach
    """
    # Convert signals to numeric values
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    weighted_sum = 0
    total_confidence = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        weight = weights[strategy]
        confidence = signal["confidence"]

        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence

    # Normalize the weighted sum
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    # Convert back to signal
    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {"signal": signal, "confidence": abs(final_score)}


def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        df: DataFrame with price data
        window: EMA period

    Returns:
        pd.Series: EMA values
    """
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with OHLC data
        period: Period for calculations

    Returns:
        DataFrame with ADX values
    """
    # Calculate True Range
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    # Calculate Directional Movement
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]

    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    # Calculate ADX
    df["+di"] = 100 * (df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["-di"] = 100 * (df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent to determine long-term memory of time series
    H < 0.5: Mean reverting series
    H = 0.5: Random walk
    H > 0.5: Trending series

    Args:
        price_series: Array-like price data
        max_lag: Maximum lag for R/S calculation

    Returns:
        float: Hurst exponent
    """
    lags = range(2, max_lag)
    # Add small epsilon to avoid log(0)
    tau = [max(1e-8, np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag])))) for lag in lags]

    # Return the Hurst exponent from linear fit
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]  # Hurst exponent is the slope
    except (ValueError, RuntimeWarning):
        # Return 0.5 (random walk) if calculation fails
        return 0.5
    
 
 
def generate_ricardo_garcia_output(
    ticker: str,
    analysis_data: dict[str, any],
    financial_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> RicardoGarciaSignal:
    
    # get todays date as a string
    today = datetime.now().strftime("%Y-%m-%d")

    
    """
    Generates investment decisions in the style of Ricardo Garcia.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Ricardo Garcia AI agent, making investment decisions using his principles:\n\n"
            " 1. He invests mostly when the overall S&P 500 is down 10% or more over the last month and when a particular stock has strong revenue and earnings growth.\n"
            " 2. He buys stocks when its 4 weeks or less days before earnings and earnings are projected to be very strong.\n"
            " 3. He sells when the stock has had strong gains and more than 2 weeks but less than 4weeks since the last earnings report.  \n"
            " 4. He invests in companies that are using disruptive technology to grow their business. especially in the AI, robotics, and biotechnology sectors.  We should always take into account the potential for a strong upcoming earnings report and not invest in companies right after earnings.  Buy companies in the weeks running up to earnings, usually 2-4 weeks.  \n"
            " 5. Almost always invest in companies where the marjority shareowners have funded the policitians that are currently in power.  This would mean that those companies stand to beneft from the policies of the current administration.  \n"
            " An example of this is Jeff Bezos, who is the majority shareholder of Amazon.  He has funded many politicians who are in power and has a lot of influence over the policies of the United States. \n"
            " Another example is Elon Musk, who is the majority shareholder of Tesla.  He has funded many politicians who are in power and has a lot of influence over the policies of the United States. \n"
            " Another example is Mark Zuckerberg, who is the majority shareholder of Facebook.  He has funded many politicians who are in power and has a lot of influence over the policies of the United States. \n"
            " Another example is Tim Cook, who is the majority shareholder of Apple.  He has funded many politicians who are in power and has a lot of influence over the policies of the United States. \n"
            " Another example of this is Peter Thiel, who is the majority shareholder of Palantir.  He has funded many politicians who are in power and has a lot of influence over the policies of the United States. \n"
            " Another example of this is Jack Ma, who is the majority shareholder of Alibaba.  He has funded many politicians who are in power and has a lot of influence over the policies of the United States. \n"
            
            "7. Seek companies leveraging disruptive innovation.\n"
            "8. Emphasize exponential growth potential, large TAM.\n"
            "9. Focus on technology, healthcare, or other future-facing sectors.\n"
            "10. Consider multi-year time horizons for potential breakthroughs.\n"
            "11. Accept higher volatility in pursuit of high returns.\n"
            "12. Evaluate management's vision and ability to invest in R&D.\n\n"
            "Rules:\n"
            "- Identify disruptive or breakthrough technology.\n"
            "- Evaluate strong potential for multi-year revenue growth.\n"
            "- Check if the company can scale effectively in a large market.\n"
            "- Use a growth-biased valuation approach.\n"
            "- Provide a data-driven recommendation (bullish, bearish, or neutral)."""
        ),
        (
            "human",
            """Based on the following analysis, create a Ricardo Garcia-style investment signal. Make sure to look up the last and next earnings date against the current date, {today}, in order to better facilitate a signal.\n\n"
            "Last Earnings Date: When was the last earnings date for {ticker}?\n\n"
            "Next Earnings Date: What is the next earnings date for {ticker}?\n\n"
            "Financial Data for {ticker}:\n"
            "{financial_data}\n\n"
            "Analysis Data for {ticker}:\n"
            "{analysis_data}\n\n"
            "Return the trading signal in this JSON format:\n"
            "{{\n  \"signal\": \"bullish/bearish/neutral\",\n  \"confidence\": float (0-100),\n  \"reasoning\": \"string\"\n}}"""
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker,
        "today": today,
        "financial_data": json.dumps(financial_data, indent=2),
    })

    def create_default_ricardo_garcia_signal():
        return RicardoGarciaSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=RicardoGarciaSignal,
        agent_name="ricardo_garcia_agent",
        default_factory=create_default_ricardo_garcia_signal,
    )
