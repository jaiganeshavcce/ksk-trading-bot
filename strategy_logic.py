from collections import deque


def build_strategy_state(cfg: dict) -> dict:
    """Builds strategy parameters from config for reuse in the main loop."""
    return {
        "window": int(cfg.get("sma_window", 5)),
        "threshold": float(cfg.get("entry_threshold_pct", 0.12)) / 100.0,
    }


def evaluate_signal(prices: deque, state: dict) -> dict:
    """Returns BUY/SELL/HOLD signal using simple SMA threshold crossover."""
    window = state["window"]
    threshold = state["threshold"]

    if len(prices) < window:
        return {"signal": "HOLD", "sma": 0.0}

    ltp = float(prices[-1])
    sma = sum(prices) / window

    if ltp > sma * (1.0 + threshold):
        return {"signal": "BUY", "sma": sma}
    if ltp < sma * (1.0 - threshold):
        return {"signal": "SELL", "sma": sma}
    return {"signal": "HOLD", "sma": sma}
