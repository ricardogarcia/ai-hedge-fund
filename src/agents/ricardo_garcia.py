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


def prepare_ml_data(analysis: dict,llm_signal: str) -> dict:
    """
    Prepare data for machine learning model training.
    """
    ml_data = {}
    for ticker, analysis_data in analysis.items():
        print(analysis_data)
        strategy_signals = analysis_data["strategy_signals"]
        for signal_name, signal_data in strategy_signals.items():
            metrics = signal_data["metrics"]
            for metric_name, metric_value in metrics.items():
                ml_data[f"{signal_name}_{metric_name}"] = metric_value
        ml_data[f"llm_signal"] = llm_signal
    return ml_data


def generate_ricardo_garcia_signals(tickers: list[str], start_date: str, end_date: str,model_name: str, model_provider: str) -> dict:
    
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

        # Calculate the Hurst Exponent

        progress.update_status("ricardo_garcia_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

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

        progress.update_status("ricardo_garcia_agent", ticker, "Fetching financial metrics")
        # You can adjust these parameters (period="annual"/"ttm", limit=5/10, etc.)
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)

        # Transform to dictionary
        metrics_dict = {
            item.report_period: {
                k: v for k, v in item.model_dump().items() 
                if k not in ['ticker', 'report_period', 'period', 'currency']
            }
            for item in metrics
        }

        progress.update_status("ricardo_garcia_agent", ticker, "Calculating trend signals")
        trend_signals = calculate_trend_signals(prices_df)

        progress.update_status("ricardo_garcia_agent", ticker, "Calculating mean reversion")
        mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

        progress.update_status("ricardo_garcia_agent", ticker, "Analyzing disruptive potential")
        disruptive_analysis = analyze_disruptive_potential(metrics, financial_line_items)

        progress.update_status("ricardo_garcia_agent", ticker, "Analyzing valuation")
        valuation_analysis = analyze_valuation(financial_line_items, market_cap)

        # Combine all signals using a weighted ensemble approach
        strategy_weights = {
            "trend": 0.10,
            "mean_reversion": 0.20,
            "disruptive_analysis": 0.30,
            "valuation_analysis": 0.30,
        }

        progress.update_status("ricardo_garcia_agent", ticker, "Combining signals")
        combined_signal = weighted_signal_combination(
            {
                "trend": trend_signals,
                "mean_reversion": mean_reversion_signals,
                "disruptive_analysis": disruptive_analysis,
                "valuation_analysis": valuation_analysis,
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
                "disruptive_analysis": {
                    "signal": disruptive_analysis["signal"],
                    "score": disruptive_analysis["score"],
                    "metrics": normalize_pandas(disruptive_analysis["metrics"]),
                    "details": disruptive_analysis["details"],
                    "max_score": disruptive_analysis["max_score"],
                },
                "valuation_analysis": {
                    "signal": valuation_analysis["signal"],
                    "score": valuation_analysis["score"],
                    "max_score": valuation_analysis["max_score"],
                    "metrics": normalize_pandas(valuation_analysis["metrics"]),
                    "details": valuation_analysis["details"],
                },
            },
        }
        # Generate a Ricardo Garcia-style LLM investment signal
        ricardo_garcia_signal = generate_ricardo_garcia_output(
            ticker=ticker,
            analysis_data=ricardo_garcia_analysis[ticker],
            model_name=model_name,
            model_provider=model_provider,
        )

        ml_data = prepare_ml_data(ricardo_garcia_analysis, ricardo_garcia_signal.signal)

        ricardo_garcia_analysis[ticker] = {
            "signal": ricardo_garcia_signal.signal,
            "confidence": ricardo_garcia_signal.confidence,
            "reasoning": ricardo_garcia_signal.reasoning,
            "ml_data": ml_data
        }
        progress.update_status("ricardo_garcia_agent", ticker, "Done")
    return ricardo_garcia_analysis

def ricardo_garcia_agent(state: AgentState):
   
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    ricardo_garcia_analysis = generate_ricardo_garcia_signals(tickers, start_date, end_date, state["metadata"]["model_name"], state["metadata"]["model_provider"]) 
    
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
    
def analyze_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
    Focus on long-term exponential growth potential. We can do
    a simplified approach looking for a large total addressable market (TAM) and the
    company's ability to capture a sizable portion.
    """
    max_possible_score = 5
    if not financial_line_items or market_cap is None:
        return {
            "signal":"neutral",
            "score": 0,
            "max_score": max_possible_score,
            "details": "Insufficient data for valuation",
            "confidence": 0,
            "metrics": {}
        }

    latest = financial_line_items[-1]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0

    # Instead of a standard DCF, let's assume a higher growth rate for an innovative company.
    # Example values:
    growth_rate = 0.20  # 20% annual growth
    discount_rate = 0.15
    terminal_multiple = 25
    projection_years = 5

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    # Terminal Value
    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) \
                     / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value

    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    score = 0
    if margin_of_safety > 0.5:
        score += 5  
    elif margin_of_safety > 0.2:
        score += 3
    elif margin_of_safety > 0:
        score += 1

    details = [
        f"Calculated intrinsic value: ~{intrinsic_value:,.2f}",
        f"Market cap: ~{market_cap:,.2f}",
        f"Margin of safety: {margin_of_safety:.2%}"
    ]

    max_possible_score = 5
    signal = "bullish" if score >= 0.7 * max_possible_score else "bearish" if score <= 0.3 * max_possible_score else "neutral"
    return {
        "signal": signal,
        "score": score,
        "details": "; ".join(details),
        "confidence": score / max_possible_score,
        "raw_score": score,
        "max_score": max_possible_score,
        "metrics": {
            "market_cap": market_cap,
            "margin_of_safety": margin_of_safety,
        }
    } 

def analyze_disruptive_potential(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze whether the company has disruptive products, technology, or business model.
    Evaluates multiple dimensions of disruptive potential:
    1. Revenue Growth Acceleration - indicates market adoption
    2. R&D Intensity - shows innovation investment
    3. Gross Margin Trends - suggests pricing power and scalability
    4. Operating Leverage - demonstrates business model efficiency
    5. Market Share Dynamics - indicates competitive position
    """
    score = 0
    details = []
    max_possible_score = 12  # Sum of all possible points
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "max_score": max_possible_score,
            "details": "Insufficient data to analyze disruptive potential",
            "confidence": 0,
            "metrics": {}
        }

    # 1. Revenue Growth Analysis - Check for accelerating growth
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    growth_rate = 0
    if len(revenues) >= 3:  # Need at least 3 periods to check acceleration
        growth_rates = []
        for i in range(len(revenues)-1):
            if revenues[i] and revenues[i+1]:
                growth_rate = (revenues[i+1] - revenues[i]) / abs(revenues[i]) if revenues[i] != 0 else 0
                growth_rates.append(growth_rate)

        # Check if growth is accelerating
        if len(growth_rates) >= 2 and growth_rates[-1] > growth_rates[0]:
            score += 2
            details.append(f"Revenue growth is accelerating: {(growth_rates[-1]*100):.1f}% vs {(growth_rates[0]*100):.1f}%")

        # Check absolute growth rate
        latest_growth = growth_rates[-1] if growth_rates else 0
        if latest_growth > 1.0:
            score += 3
            details.append(f"Exceptional revenue growth: {(latest_growth*100):.1f}%")
        elif latest_growth > 0.5:
            score += 2
            details.append(f"Strong revenue growth: {(latest_growth*100):.1f}%")
        elif latest_growth > 0.2:
            score += 1
            details.append(f"Moderate revenue growth: {(latest_growth*100):.1f}%")
    else:
        details.append("Insufficient revenue data for growth analysis")

    # 2. Gross Margin Analysis - Check for expanding margins
    gross_margins = [item.gross_margin for item in financial_line_items if hasattr(item, 'gross_margin') and item.gross_margin is not None]
    margin_trend = 0
    if len(gross_margins) >= 2:
        margin_trend = gross_margins[-1] - gross_margins[0]
        if margin_trend > 0.05:  # 5% improvement
            score += 2
            details.append(f"Expanding gross margins: +{(margin_trend*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"Slightly improving gross margins: +{(margin_trend*100):.1f}%")

        # Check absolute margin level
        if gross_margins[-1] > 0.50:  # High margin business
            score += 2
            details.append(f"High gross margin: {(gross_margins[-1]*100):.1f}%")
    else:
        details.append("Insufficient gross margin data")

    # 3. Operating Leverage Analysis
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    operating_expenses = [
        item.operating_expense
        for item in financial_line_items
        if hasattr(item, "operating_expense") and item.operating_expense
    ]

    if len(revenues) >= 2 and len(operating_expenses) >= 2:
        rev_growth = (revenues[-1] - revenues[0]) / abs(revenues[0])
        opex_growth = (operating_expenses[-1] - operating_expenses[0]) / abs(operating_expenses[0])

        if rev_growth > opex_growth:
            score += 2
            details.append("Positive operating leverage: Revenue growing faster than expenses")
    else:
        details.append("Insufficient data for operating leverage analysis")

    # 4. R&D Investment Analysis
    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, 'research_and_development') and item.research_and_development is not None]
    rd_intensity = 0
    if rd_expenses and revenues:
        rd_intensity = rd_expenses[-1] / revenues[-1]
        if rd_intensity > 0.15:  # High R&D intensity
            score += 3
            details.append(f"High R&D investment: {(rd_intensity*100):.1f}% of revenue")
        elif rd_intensity > 0.08:
            score += 2
            details.append(f"Moderate R&D investment: {(rd_intensity*100):.1f}% of revenue")
        elif rd_intensity > 0.05:
            score += 1
            details.append(f"Some R&D investment: {(rd_intensity*100):.1f}% of revenue")
    else:
        details.append("No R&D data available")

    

    signal = "bullish" if score >= 0.65 * max_possible_score else "bearish" if score <= 0.4 * max_possible_score else "neutral"
    return {
        "signal": signal,
        "score": score,
        "details": "; ".join(details),
        "confidence": score / max_possible_score,
        "raw_score": score,
        "max_score": max_possible_score,
        "metrics": {
            "revenue": revenues[-1],
            "rd_intensity": rd_intensity,
            "margin_trend": margin_trend,
            "growth_rate": growth_rate,
        }
    }

def generate_ricardo_garcia_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> RicardoGarciaSignal:

    """
    Generates investment decisions in the style of Ricardo Garcia.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Ricardo Garcia AI agent, making investment decisions using his principles:\n\n"
            " 1. He invests mostly when a particular stock has strong revenue and earnings growth.\n"
            " 2. He invests in growth companies that are using AI technology to grow their business. especially in the military, robotics, and biotechnology sectors. \n"
            " 3. Almost always invest in companies where the majority shareowners have funded the policitians that are currently in power.  This would mean that those companies stand to beneft from the policies of the current administration.  \n"
            " 4. Emphasize exponential growth potential, large TAM.\n"
            " 5. Focus on technology, healthcare, or other future-facing sectors.\n"
            " 6. He invests in growth stocks that have a market cap of less than 1 trillion dollars.\n"
            " 7. He takes into account the macroeconomic factors for the sector that the company is in. If the sector is doing well, he will invest in the company. If it is not, he will not invest in the company.\n"
            " 8. He looks at the RSI to determine if the stock is overbought or oversold. If it is overbought, he will not invest in the company. If it is oversold, he will invest in the company.\n"
            " 9. He considers analyst consensus price targets to determine if the stock is undervalued or overvalued. If the price target is higher than the current price, he will invest in the company. If the price target is lower than the current price, he will not invest in the company.\n"
            "Rules:\n"
            "- Identify disruptive or breakthrough technology.\n"
            "- Evaluate strong potential for multi-year revenue growth.\n"
            "- Check if the company can scale effectively in a large market.\n"
            "- Use a growth-biased valuation approach.\n"
            "- Provide a data-driven recommendation (bullish, bearish, or neutral)."""
        ),
        (
            "human",
            """Based on the following analysis, create a Ricardo Garcia-style investment signal.\n\n"
            "Analysis Data for {ticker}:\n"
            "{analysis_data}\n\n"
            "Return the trading signal in this JSON format:\n"
            "{{\n  \"signal\": \"bullish/bearish/neutral\",\n  \"confidence\": float (0-100),\n  \"reasoning\": \"string\"\n}}"""
        )
    ])


    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker,
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
