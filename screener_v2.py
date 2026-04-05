from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import glob
import inspect
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from neo_api_client import NeoAPI  # type: ignore
except ImportError:
    NeoAPI = None  # type: ignore  # not available on Streamlit Cloud; screener uses yfinance only

# Directories relative to this file
_HERE = Path(__file__).resolve().parent
NSE_DATA_DIR = _HERE.parent / "nse data"
NSE_STOCK_LIST_DIR = _HERE.parent / "stock list"

# Map UI label → CSV filename prefix in stock list dir
NSE_INDEX_CSV_MAP: dict[str, str] = {
    "Nifty 50":                   "MW-NIFTY-50",
    "Nifty Bank":                 "MW-NIFTY-BANK",
    "Nifty 100":                  "MW-NIFTY-100",
    "Nifty Next 50":              "MW-NIFTY-NEXT-50",
    "Nifty Midcap Select":        "MW-NIFTY-MIDCAP-SELECT",
    "Nifty Financial Services":   "MW-NIFTY-FINANCIAL-SERVICES",
}


# Nifty 50 constituents — verified from NSE live market data (April 2026)
NIFTY50_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC",
    "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "TATASTEEL", "TECHM",
    "TITAN", "TMPV", "TRENT", "ULTRACEMCO", "WIPRO",
]

INTERVAL_LABELS = {
    "1D": "day",
    "1H": "hour",
    "15M": "15minute",
    "5M": "5minute",
}

LOGIN_URL = "https://mis.kotaksecurities.com/login/1.0/tradeApiLogin"
VALIDATE_URL = "https://mis.kotaksecurities.com/login/1.0/tradeApiValidate"


class HistoryMethodUnavailableError(RuntimeError):
    pass


@dataclass
class V2AuthSession:
    token_id: str
    view_token: str
    view_sid: str
    trade_token: str
    trade_sid: str
    base_url: str


@dataclass
class ScreenerCredentials:
    token_id: str
    mobile_number: str
    mpin: str
    totp: str
    environment: str = "prod"
    ucc: str = ""


def create_client(credentials: ScreenerCredentials) -> NeoAPI:
    token_id = credentials.token_id.strip()
    environment = credentials.environment.strip() or "prod"

    return NeoAPI(
        access_token=token_id,
        environment=environment,
    )


def _login_with_totp(credentials: ScreenerCredentials) -> tuple[str, str]:
    headers = {
        "Authorization": credentials.token_id.strip(),
        "neo-fin-key": "neotradeapi",
        "Content-Type": "application/json",
    }
    payload = {
        "mobileNumber": credentials.mobile_number.strip(),
        "ucc": credentials.ucc.strip(),
        "totp": credentials.totp.strip(),
    }
    response = requests.post(LOGIN_URL, headers=headers, json=payload, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"tradeApiLogin failed ({response.status_code}): {response.text}")

    data = response.json().get("data") or {}
    view_token = data.get("token")
    view_sid = data.get("sid")
    if not view_token or not view_sid:
        raise RuntimeError(f"tradeApiLogin succeeded but token/sid missing: {response.text}")
    return str(view_token), str(view_sid)


def _validate_with_mpin(credentials: ScreenerCredentials, view_token: str, view_sid: str) -> tuple[str, str, str]:
    headers = {
        "Authorization": credentials.token_id.strip(),
        "neo-fin-key": "neotradeapi",
        "sid": view_sid,
        "Auth": view_token,
        "Content-Type": "application/json",
    }
    payload = {"mpin": credentials.mpin.strip()}
    response = requests.post(VALIDATE_URL, headers=headers, json=payload, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"tradeApiValidate failed ({response.status_code}): {response.text}")

    data = response.json().get("data") or {}
    trade_token = data.get("token")
    trade_sid = data.get("sid")
    base_url = data.get("baseUrl")
    if not trade_token or not trade_sid or not base_url:
        raise RuntimeError(f"tradeApiValidate succeeded but trade token/baseUrl missing: {response.text}")
    return str(trade_token), str(trade_sid), str(base_url)


def _check_orders(session: V2AuthSession) -> dict[str, Any]:
    headers = {
        "Authorization": session.token_id,
        "neo-fin-key": "neotradeapi",
        "Auth": session.trade_token,
        "Sid": session.trade_sid,
    }
    response = requests.get(f"{session.base_url}/quick/user/orders", headers=headers, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"orders check failed ({response.status_code}): {response.text}")
    return response.json()


def authenticate_client(client: NeoAPI, credentials: ScreenerCredentials) -> dict[str, Any]:
    mobile_number = credentials.mobile_number.strip()
    user_id = credentials.ucc.strip() or None
    mpin = credentials.mpin.strip()
    totp = credentials.totp.strip()

    if not credentials.token_id.strip():
        raise RuntimeError("Token ID is required.")
    if not mobile_number:
        raise RuntimeError("Mobile Number is required.")
    if not mpin:
        raise RuntimeError("MPIN is required.")
    if not totp:
        raise RuntimeError("TOTP is required.")

    try:
        view_token, view_sid = _login_with_totp(credentials)
        trade_token, trade_sid, base_url = _validate_with_mpin(
            credentials,
            view_token=view_token,
            view_sid=view_sid,
        )
        session = V2AuthSession(
            token_id=credentials.token_id.strip(),
            view_token=view_token,
            view_sid=view_sid,
            trade_token=trade_token,
            trade_sid=trade_sid,
            base_url=base_url,
        )
        orders_probe = _check_orders(session)
        return {
            "auth_session": session,
            "orders_probe_keys": list(orders_probe.keys()) if isinstance(orders_probe, dict) else [],
        }
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "Expecting value" in message:
            raise RuntimeError(
                "Neo v2 login returned a non-JSON response before TOTP verification. "
                "Check token_id, environment, and IP allowlist in Kotak Neo."
            ) from exc
        raise RuntimeError(f"Neo v2 authentication failed: {exc}") from exc


def history_support_details(client: NeoAPI) -> tuple[bool, str]:
    history_fn = getattr(client, "history", None)
    if callable(history_fn):
        try:
            signature = str(inspect.signature(history_fn))
        except (TypeError, ValueError):
            signature = "(signature unavailable)"
        return True, signature
    return False, "Installed neo_api_client does not expose history()."


def _normalize_history_response(payload: Any) -> pd.DataFrame:
    candidate = payload
    if isinstance(payload, dict):
        for key in ("data", "candles", "history", "result", "response"):
            value = payload.get(key)
            if value:
                candidate = value
                break

    if isinstance(candidate, list) and candidate and isinstance(candidate[0], (list, tuple)):
        frame = pd.DataFrame(candidate)
        if frame.shape[1] >= 6:
            frame = frame.iloc[:, :6]
            frame.columns = ["timestamp", "open", "high", "low", "close", "volume"]
            return frame

    if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
        frame = pd.DataFrame(candidate)
        rename_map = {
            "time": "timestamp",
            "datetime": "timestamp",
            "date": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
        frame = frame.rename(columns=rename_map)
        required = {"timestamp", "open", "high", "low", "close"}
        if required.issubset(frame.columns):
            if "volume" not in frame.columns:
                frame["volume"] = 0.0
            return frame[["timestamp", "open", "high", "low", "close", "volume"]]

    raise RuntimeError(f"Unsupported history payload format: {type(payload).__name__}")


YF_INTERVAL_MAP: dict[str, str] = {
    "1D": "1d",
    "1H": "1h",
    "15M": "15m",
    "5M": "5m",
}
# yfinance max lookback (days) per interval
YF_MAX_DAYS: dict[str, int] = {
    "1d": 36500,
    "1h": 730,
    "15m": 60,
    "5m": 60,
}

# Known NSE symbol → exact Yahoo Finance ticker overrides
YF_SYMBOL_OVERRIDES: dict[str, str] = {
    "ETERNAL": "ETERNAL.NS",      # Zomato rebranded to Eternal Ltd (2025)
    "BAJAJ-AUTO": "BAJAJ-AUTO.BO", # Yahoo Finance NSE feed broken; BSE works
    "M&M": "M&M.NS",
    "JIOFIN": "JIOFIN.NS",
    "TMPV": "TMPV.NS",             # Tata Motors Passenger Vehicles (demerged 2025)
}

# Major index tickers on Yahoo Finance
INDEX_TICKERS: dict[str, str] = {
    "Nifty 50": "^NSEI",
    "Nifty Bank": "^NSEBANK",
    "Sensex": "^BSESN",
    "Nifty IT": "^CNXIT",
    "Nifty Next 50": "^NSMIDCP",
    "Nifty 500": "^CRSLDX",
}


def _to_yf_symbol(symbol: str) -> str:
    """Strip exchange suffix, apply override map, then append .NS."""
    clean = symbol.upper().strip()
    for suffix in ("-EQ", "-BE", "-BL", "-SM", "-GR"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
            break
    return YF_SYMBOL_OVERRIDES.get(clean, clean + ".NS")


def fetch_ohlc_history(
    symbol: str,
    interval_label: str = "1D",
    lookback_bars: int = 250,
) -> pd.DataFrame:
    import yfinance as yf  # type: ignore

    yf_symbol = _to_yf_symbol(symbol)
    yf_interval = YF_INTERVAL_MAP.get(interval_label, "1d")
    max_days = YF_MAX_DAYS.get(yf_interval, 36500)

    end_dt = datetime.now()
    # Add a 50% buffer to account for weekends / holidays
    if yf_interval == "1d":
        start_dt = end_dt - timedelta(days=min(int(lookback_bars * 1.5), max_days))
    elif yf_interval == "1h":
        start_dt = end_dt - timedelta(hours=min(int(lookback_bars * 1.5), max_days * 24))
    elif yf_interval in ("15m", "5m"):
        minutes = 15 if yf_interval == "15m" else 5
        start_dt = end_dt - timedelta(minutes=min(int(lookback_bars * minutes * 1.5), max_days * 1440))
    else:
        start_dt = end_dt - timedelta(days=min(int(lookback_bars * 1.5), max_days))

    df = pd.DataFrame()
    # Try primary ticker, then BSE fallback (.NS → .BO) for symbols not in override map
    tickers_to_try = [yf_symbol]
    if yf_symbol.endswith(".NS") and yf_symbol not in YF_SYMBOL_OVERRIDES.values():
        tickers_to_try.append(yf_symbol.replace(".NS", ".BO"))

    for ticker_attempt in tickers_to_try:
        for attempt in range(2):
            df = yf.download(
                ticker_attempt,
                start=start_dt,
                end=end_dt,
                interval=yf_interval,
                progress=False,
                auto_adjust=True,
            )
            if not df.empty:
                break
            if attempt == 0:
                time.sleep(1.5)
        if not df.empty:
            break

    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {yf_symbol}")

    # Flatten MultiIndex columns (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]

    df = df.reset_index()
    # Rename the date/datetime index column
    date_col = next(
        (c for c in df.columns if c in ("date", "datetime", "index", "price")),
        df.columns[0],
    )
    df = df.rename(columns={date_col: "timestamp"})

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise RuntimeError(f"Missing column '{col}' in yfinance data for {yf_symbol}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").tail(lookback_bars).reset_index(drop=True)
    return df


def _calculate_indicators_fallback(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    close = result["close"]
    volume = result["volume"].fillna(0.0)

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    result["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    result["ema_20"] = close.ewm(span=20, adjust=False).mean()
    result["ema_50"] = close.ewm(span=50, adjust=False).mean()
    result["sma_200"] = close.rolling(window=200, min_periods=1).mean()

    typical_price = (result["high"] + result["low"] + result["close"]) / 3.0
    cumulative_volume = volume.replace(0, pd.NA).cumsum()
    result["vwap"] = ((typical_price * volume).cumsum() / cumulative_volume).fillna(close)
    result["avg_volume_20"] = volume.rolling(window=20, min_periods=1).mean()
    return result


def calculate_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        result = frame.copy()
        if "timestamp" in result.columns:
            result = result.set_index(pd.DatetimeIndex(pd.to_datetime(result["timestamp"])))

        close  = result["close"]
        high   = result["high"]
        low    = result["low"]
        volume = result["volume"].fillna(0.0)

        # RSI(14) — Wilder EMA method
        delta    = close.diff()
        avg_gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        result["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

        # EMA 20 / 50
        result["ema_20"] = close.ewm(span=20, adjust=False).mean()
        result["ema_50"] = close.ewm(span=50, adjust=False).mean()

        # SMA 200 (rolling so it works with < 200 bars)
        result["sma_200"] = close.rolling(window=200, min_periods=1).mean()

        # VWAP (intraday cumulative)
        typical  = (high + low + close) / 3.0
        cumvol   = volume.replace(0, pd.NA).cumsum()
        result["vwap"] = ((typical * volume).cumsum() / cumvol).fillna(close)

        # Avg volume 20
        result["avg_volume_20"] = volume.rolling(window=20, min_periods=1).mean()

        result = result.ffill().bfill()
        if "timestamp" not in result.columns:
            result = result.reset_index().rename(columns={"index": "timestamp"})
        return result
    except Exception:
        return _calculate_indicators_fallback(frame)


def _crossover_signal(frame: pd.DataFrame, fast_col: str, slow_col: str, lookback: int = 5) -> str:
    """Returns 'Bull ▲', 'Bear ▼', or '-' if fast crossed slow within recent lookback bars."""
    if len(frame) < lookback + 1:
        return "-"
    recent = frame[[fast_col, slow_col]].iloc[-(lookback + 1):]
    fast = recent[fast_col].values
    slow = recent[slow_col].values
    currently_above = float(fast[-1]) > float(slow[-1])
    was_below_any = any(float(fast[i]) <= float(slow[i]) for i in range(len(fast) - 1))
    was_above_any = any(float(fast[i]) >= float(slow[i]) for i in range(len(fast) - 1))
    if currently_above and was_below_any:
        return "Bull ▲"
    if not currently_above and was_above_any:
        return "Bear ▼"
    return "-"


def _screen_latest_row(symbol: str, frame: pd.DataFrame) -> dict[str, Any]:
    enriched = calculate_indicators(frame)
    current = enriched.iloc[-1]
    previous = enriched.iloc[-2] if len(enriched) > 1 else current

    close = round(float(current["close"]), 2)
    ema_20 = round(float(current["ema_20"]), 2)
    ema_50 = round(float(current["ema_50"]), 2)
    sma_200 = round(float(current["sma_200"]), 2)
    vwap = round(float(current["vwap"]), 2)
    rsi = round(float(current["rsi_14"]), 2)
    volume = int(float(current["volume"]))
    avg_vol = round(float(current["avg_volume_20"]), 2)

    # 52-week high/low from the full bar range passed in
    w52_high = round(float(enriched["high"].max()), 2)
    w52_low  = round(float(enriched["low"].min()), 2)

    bullish_cross = float(current["ema_50"]) > float(current["sma_200"]) and float(previous["ema_50"]) <= float(previous["sma_200"])
    bearish_cross = float(current["ema_50"]) < float(current["sma_200"]) and float(previous["ema_50"]) >= float(previous["sma_200"])

    return {
        "Symbol": symbol,
        "Close": close,
        "52W High": w52_high,
        "52W Low": w52_low,
        "RSI(14)": rsi,
        "RSI Zone": "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"),
        "vs MA20": "Above" if close >= ema_20 else "Below",
        "vs MA50": "Above" if close >= ema_50 else "Below",
        "vs MA200": "Above" if close >= sma_200 else "Below",
        "vs VWAP": "Above" if close >= vwap else "Below",
        "EMA20×50": _crossover_signal(enriched, "ema_20", "ema_50"),
        "GoldenDeath": _crossover_signal(enriched, "ema_50", "sma_200"),
        "Vol Spike": bool(volume > 2 * avg_vol),
        "Volume": volume,
        "EMA 20": ema_20,
        "EMA 50": ema_50,
        "SMA 200": sma_200,
        "VWAP": vwap,
        "Avg Vol 20": avg_vol,
        "Bullish MA Cross": bullish_cross,
        "Bearish MA Cross": bearish_cross,
        "Bias": "Bullish" if close >= ema_50 else "Bearish",
        "Bars": int(len(enriched)),
    }


def run_screener(
    symbols: list[str],
    interval_label: str = "1D",
    lookback_bars: int = 250,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for symbol in symbols:
        try:
            frame = fetch_ohlc_history(
                symbol=symbol,
                interval_label=interval_label,
                lookback_bars=lookback_bars,
            )
            if frame.empty:
                errors.append(f"{symbol}: no OHLC rows returned")
                continue
            rows.append(_screen_latest_row(symbol, frame))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{symbol}: {exc}")

    if not rows:
        return pd.DataFrame(), errors

    return pd.DataFrame(rows), errors


def apply_filters(
    frame: pd.DataFrame,
    *,
    oversold_only: bool = False,
    overbought_only: bool = False,
    volume_spikes_only: bool = False,
    ma_cross_only: bool = False,
    above_vwap_only: bool = False,
    below_vwap_only: bool = False,
    above_ma20: bool = False,
    below_ma20: bool = False,
    above_ma50: bool = False,
    below_ma50: bool = False,
    above_ma200: bool = False,
    below_ma200: bool = False,
    ema20_bull_cross: bool = False,
    golden_cross: bool = False,
) -> pd.DataFrame:
    filtered = frame.copy()
    if oversold_only:
        filtered = filtered[filtered["RSI Zone"] == "Oversold"]
    if overbought_only:
        filtered = filtered[filtered["RSI Zone"] == "Overbought"]
    if volume_spikes_only:
        filtered = filtered[filtered["Vol Spike"]]
    if ma_cross_only:
        filtered = filtered[filtered["Bullish MA Cross"] | filtered["Bearish MA Cross"]]
    if above_vwap_only:
        filtered = filtered[filtered["vs VWAP"] == "Above"]
    if below_vwap_only:
        filtered = filtered[filtered["vs VWAP"] == "Below"]
    if above_ma20:
        filtered = filtered[filtered["vs MA20"] == "Above"]
    if below_ma20:
        filtered = filtered[filtered["vs MA20"] == "Below"]
    if above_ma50:
        filtered = filtered[filtered["vs MA50"] == "Above"]
    if below_ma50:
        filtered = filtered[filtered["vs MA50"] == "Below"]
    if above_ma200:
        filtered = filtered[filtered["vs MA200"] == "Above"]
    if below_ma200:
        filtered = filtered[filtered["vs MA200"] == "Below"]
    if ema20_bull_cross:
        filtered = filtered[filtered["EMA20\u00d750"] == "Bull \u25b2"]
    if golden_cross:
        filtered = filtered[filtered["GoldenDeath"] == "Bull \u25b2"]
    return filtered.reset_index(drop=True)


# Columns shown in the UI by default (internal bool cols used for row styling are excluded)
SCREENER_DISPLAY_COLS = [
    "Symbol", "Close", "52W High", "52W Low", "RSI(14)", "RSI Zone",
    "vs MA20", "vs MA50", "vs MA200", "vs VWAP",
    "EMA20\u00d750", "GoldenDeath",
    "Vol Spike", "Volume",
    "EMA 20", "EMA 50", "SMA 200", "VWAP",
    "Bias",
]


def style_results(frame: pd.DataFrame):
    def _row_style(row: pd.Series) -> list[str]:
        if row.get("Bullish MA Cross"):
            return ["background-color: #153d1d; color: #d6ffd6;"] * len(row)
        if row.get("Bearish MA Cross"):
            return ["background-color: #4a1717; color: #ffd6d6;"] * len(row)
        return [""] * len(row)

    def _rsi_style(value: Any) -> str:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return ""
        if val < 30:
            return "background-color: #4a1717; color: #ffd6d6;"
        if val > 70:
            return "background-color: #3d3214; color: #fff0b3;"
        return ""

    def _pos_style(value: Any) -> str:
        v = str(value)
        if v == "Above":
            return "color: #7fff7f;"
        if v == "Below":
            return "color: #ff8080;"
        return ""

    def _cross_style(value: Any) -> str:
        v = str(value)
        if "\u25b2" in v:
            return "color: #7fff7f; font-weight: bold;"
        if "\u25bc" in v:
            return "color: #ff8080; font-weight: bold;"
        return ""

    styled = frame.style.apply(_row_style, axis=1)
    if "RSI(14)" in frame.columns:
        styled = styled.map(_rsi_style, subset=["RSI(14)"])
    for col in ("vs MA20", "vs MA50", "vs MA200", "vs VWAP"):
        if col in frame.columns:
            styled = styled.map(_pos_style, subset=[col])
    for col in ("EMA20\u00d750", "GoldenDeath"):
        if col in frame.columns:
            styled = styled.map(_cross_style, subset=[col])
    return styled


def fetch_dividend_pivot(symbols: list[str]) -> pd.DataFrame:
    """Return one row per symbol with dividend data pivoted by year (2024/2025/2026).

    Columns: Symbol | 2024 | 2025 | 2026 | Next Ex-Div
    Each year cell contains 'DD-Mon ₹X.XX' entries (newline-joined if multiple).
    'Next Ex-Div' shows the upcoming ex-dividend date from ticker.info.
    Only data from 2024-01-01 onwards is shown.
    """
    import yfinance as yf  # type: ignore
    from datetime import timezone

    YEARS = [2024, 2025, 2026]
    today = datetime.now().date()
    rows: list[dict] = []

    for symbol in symbols:
        yf_sym = _to_yf_symbol(symbol)
        row: dict = {"Symbol": symbol}
        for yr in YEARS:
            row[str(yr)] = "—"
        row["Next Ex-Div"] = "—"

        try:
            ticker = yf.Ticker(yf_sym)

            # ── Dividend history ──────────────────────────────────────────────
            try:
                divs = ticker.dividends
                if divs is not None and not divs.empty:
                    divs_df = divs.reset_index()
                    divs_df.columns = ["Date", "Amount"]
                    divs_df["Date"] = (
                        pd.to_datetime(divs_df["Date"])
                        .dt.tz_localize(None)
                        .dt.date
                    )
                    # Filter 2024 onwards
                    divs_df = divs_df[divs_df["Date"] >= datetime(2024, 1, 1).date()]
                    divs_df["Year"] = [d.year for d in divs_df["Date"]]
                    for yr in YEARS:
                        yr_rows = divs_df[divs_df["Year"] == yr].sort_values("Date")
                        if not yr_rows.empty:
                            entries = [
                                f"{r['Date'].strftime('%d-%b')}  ₹{r['Amount']:.2f}"
                                for _, r in yr_rows.iterrows()
                            ]
                            row[str(yr)] = "\n".join(entries)
            except Exception:
                pass

            # ── Next Ex-Div from info ─────────────────────────────────────────
            try:
                info = ticker.info or {}
                ex_div_raw = info.get("exDividendDate") or info.get("ex_dividend_date")
                if ex_div_raw:
                    if isinstance(ex_div_raw, (int, float)):
                        ex_date = datetime.fromtimestamp(int(ex_div_raw), tz=timezone.utc).date()
                    else:
                        ex_date = pd.to_datetime(ex_div_raw).date()
                    div_rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
                    label = ex_date.strftime("%d-%b-%Y")
                    if div_rate:
                        label += f"  ₹{float(div_rate):.2f}"
                    if ex_date >= today:
                        label = f"↑ {label}"
                    row["Next Ex-Div"] = label
            except Exception:
                pass

            time.sleep(0.1)
        except Exception:
            pass

        rows.append(row)

    return pd.DataFrame(rows, columns=["Symbol"] + [str(y) for y in YEARS] + ["Next Ex-Div"])


def fetch_index_snapshot(lookback_bars: int = 252) -> tuple[pd.DataFrame, list[str]]:
    """Fetch RSI/MA/VWAP indicator snapshot for major indices using yfinance."""
    import yfinance as yf  # type: ignore

    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for display_name, yf_ticker in INDEX_TICKERS.items():
        try:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=int(lookback_bars * 1.5))
            df = yf.download(
                yf_ticker,
                start=start_dt,
                end=end_dt,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                errors.append(f"{display_name}: no data returned")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [str(col[0]).lower() for col in df.columns]
            else:
                df.columns = [str(col).lower() for col in df.columns]

            df = df.reset_index()
            date_col = next(
                (c for c in df.columns if c in ("date", "datetime", "index", "price")),
                df.columns[0],
            )
            df = df.rename(columns={date_col: "timestamp"})
            for col in ("open", "high", "low", "close"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if "volume" not in df.columns:
                df["volume"] = 0.0
            else:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

            df = (
                df[["timestamp", "open", "high", "low", "close", "volume"]]
                .dropna(subset=["timestamp", "open", "high", "low", "close"])
                .sort_values("timestamp")
                .tail(lookback_bars)
                .reset_index(drop=True)
            )
            if df.empty:
                errors.append(f"{display_name}: empty after normalization")
                continue

            rows.append(_screen_latest_row(display_name, df))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{display_name}: {exc}")

    if not rows:
        return pd.DataFrame(), errors
    return pd.DataFrame(rows), errors


def load_nse_symbols(index_label: str) -> list[str]:
    """Load stock symbols from the latest NSE market watch CSV in the stock list directory."""
    prefix = NSE_INDEX_CSV_MAP.get(index_label)
    if not prefix:
        return []
    pattern = str(NSE_STOCK_LIST_DIR / f"{prefix}-*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return []
    latest = files[-1]
    try:
        df = pd.read_csv(latest, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        if "SYMBOL" not in df.columns:
            return []
        syms = [str(s).strip() for s in df["SYMBOL"].tolist()]
        # First row is often the index name itself (e.g. "NIFTY 50") — remove it
        syms = [s for s in syms if s and " " not in s and s.upper() == s]
        return syms
    except Exception:
        return []


# NSE futures symbol → CSV file slug
_FUTURES_CSV_SLUGS: dict[str, str] = {
    "NIFTY":     "MW-FO-nse50_fut",
    "BANKNIFTY": "MW-FO-nifty_bank_fut",
}

# Map index display name → NSE futures symbol
INDEX_TO_FUTURES_SYMBOL: dict[str, str] = {
    "Nifty 50":   "NIFTY",
    "Nifty Bank": "BANKNIFTY",
}


def load_nse_futures() -> pd.DataFrame:
    """
    Load current-month index futures from the latest NSE F&O CSV files.
    Returns a DataFrame with columns: Index, Symbol, Expiry, LTP, Chng%, OI.
    """
    rows: list[dict[str, Any]] = []
    for index_name, fut_symbol in INDEX_TO_FUTURES_SYMBOL.items():
        slug = _FUTURES_CSV_SLUGS.get(fut_symbol, "")
        if not slug:
            continue
        pattern = str(NSE_DATA_DIR / f"{slug}*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            continue
        latest = files[-1]
        try:
            df = pd.read_csv(latest, skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            for col in df.columns:
                df[col] = df[col].astype(str).str.strip()
            df = df[df["SYMBOL"].str.strip() == fut_symbol]
            if df.empty:
                continue
            # First entry = nearest (current month) expiry
            row = df.iloc[0]
            ltp_raw = row.get("LTP", "").replace(",", "")
            chng_raw = row.get("%CHNG", "").replace(",", "")
            oi_raw = row.get("OPEN INTEREST", row.get("OI", "")).replace(",", "")
            rows.append({
                "Index":   index_name,
                "Symbol":  fut_symbol + " FUT",
                "Expiry":  row.get("EXPIRY DATE", "-"),
                "LTP":     round(float(ltp_raw), 2) if ltp_raw else None,
                "Chng%":   round(float(chng_raw), 2) if chng_raw else None,
                "OI":      int(float(oi_raw)) if oi_raw else None,
            })
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)