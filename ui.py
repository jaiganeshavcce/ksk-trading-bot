import json
import os
import csv
import io
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as st_components

from v2_bot import (
    authenticate,
    build_order_payload,
    check_limits,
    check_orders,
    check_positions,
    fetch_ltp,
    get_token_id,
    place_order,
    resolve_quote_candidates,
)
from screener_v2 import (
    NIFTY50_SYMBOLS as SCREENER_NIFTY50,
    NSE_INDEX_CSV_MAP,
    SCREENER_DISPLAY_COLS,
    run_screener,
    apply_filters,
    style_results,
    fetch_dividend_pivot,
    fetch_index_snapshot as _fetch_index_snapshot_raw,
    load_nse_symbols,
    load_nse_futures as _load_nse_futures_raw,
)

# ── Cached wrappers (5-min TTL) ──────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_index_snapshot(lookback_bars: int = 252):
    return _fetch_index_snapshot_raw(lookback_bars)


@st.cache_data(ttl=300, show_spinner=False)
def load_nse_futures():
    return _load_nse_futures_raw()


BASE_DIR = Path(__file__).resolve().parent
NSE_DATA_DIR = BASE_DIR.parent / "nse data"
CONFIG_PATH = BASE_DIR / "config.json"
BOT_PATH = BASE_DIR / "v2_bot.py"
VALIDATOR_PATH = BASE_DIR / "v2_validate.py"
BOT_LOG_PATH = BASE_DIR / "bot_runtime.log"
NSE_BASE_URL = "https://www.nseindia.com"



NIFTY50_SYMBOLS = [
    "ADANIPORTS-EQ", "APOLLOHOSP-EQ", "ASIANPAINT-EQ", "AXISBANK-EQ",
    "BAJAJ-AUTO-EQ", "BAJFINANCE-EQ", "BAJAJFINSV-EQ", "BEL-EQ",
    "BPCL-EQ", "BHARTIARTL-EQ", "BRITANNIA-EQ", "CIPLA-EQ",
    "COALINDIA-EQ", "DRREDDY-EQ", "EICHERMOT-EQ", "ETERNAL-EQ",
    "GRASIM-EQ", "HCLTECH-EQ", "HDFCBANK-EQ", "HDFCLIFE-EQ",
    "HEROMOTOCO-EQ", "HINDALCO-EQ", "HINDUNILVR-EQ", "ICICIBANK-EQ",
    "ITC-EQ", "INDUSINDBK-EQ", "INFY-EQ", "JSWSTEEL-EQ",
    "JIOFIN-EQ", "KOTAKBANK-EQ", "LT-EQ", "LICI-EQ",
    "M&M-EQ", "MARUTI-EQ", "NTPC-EQ", "NESTLEIND-EQ",
    "ONGC-EQ", "POWERGRID-EQ", "RELIANCE-EQ", "SBILIFE-EQ",
    "SHRIRAMFIN-EQ", "SBIN-EQ", "SUNPHARMA-EQ", "TCS-EQ",
    "TATACONSUM-EQ", "TATAMOTORS-EQ", "TATASTEEL-EQ", "TECHM-EQ",
    "TITAN-EQ", "TRENT-EQ", "ULTRACEMCO-EQ", "WIPRO-EQ",
]

EXCHANGE_SEGMENTS: dict[str, str] = {
    "nse_cm": "nse_cm — NSE Cash Market (Equity)",
    "nse_fo": "nse_fo — NSE F&O (Futures & Options)",
    "bse_cm": "bse_cm — BSE Cash Market (Equity)",
    "bse_fo": "bse_fo — BSE F&O",
    "mcx_fo": "mcx_fo — MCX Futures (Commodities)",
    "cde_fo": "cde_fo — Currency Derivatives",
}

PRODUCT_TYPES: dict[str, str] = {
    "MIS": "MIS — Intraday",
    "CNC": "Cash (CNC) — Delivery equity",
    "NRML": "NRML — Overnight / carry-forward",
    "MTF": "Pay Later (MTF) — Margin Trading Facility",
    "CO": "CO — Cover Order",
    "BO": "BO — Bracket Order",
}

ORDER_TYPES: dict[str, str] = {
    "MKT": "Market (MKT)",
    "L": "Limit (L)",
    "SL": "SL-LMT (SL)",
    "SL-M": "SL-MKT (SL-M)",
}


def _looks_equity_symbol(symbol: str) -> bool:
    upper = str(symbol).upper().strip()
    return upper.endswith(("-EQ", "-BE", "-BL", "-SM", "-GR"))


def _ensure_valid_segment_symbol(cfg: dict) -> tuple[dict, str | None]:
    symbol_cfg = cfg.setdefault("symbol", {})
    seg = str(symbol_cfg.get("exchange_segment", "")).lower().strip()
    sym = str(symbol_cfg.get("trading_symbol", "")).strip()
    token = str(symbol_cfg.get("instrument_token", "")).strip()
    if seg in {"nse_fo", "bse_fo", "mcx_fo", "cde_fo"} and _looks_equity_symbol(sym) and not token:
        symbol_cfg["exchange_segment"] = "nse_cm"
        return cfg, f"Auto-corrected segment to nse_cm for equity symbol {sym}."
    return cfg, None


def _nse_session() -> requests.Session:
    session = st.session_state.get("nse_http_session")
    if isinstance(session, requests.Session):
        return session
    session = requests.Session()
    headers = {
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "accept": "application/json,text/plain,*/*",
        "referer": "https://www.nseindia.com/",
    }
    session.headers.update(headers)
    session.get("https://www.nseindia.com", timeout=20)
    st.session_state["nse_http_session"] = session
    return session


def _nse_get_json(endpoint: str, params: dict | None = None) -> dict:
    url = f"{NSE_BASE_URL}{endpoint}"
    session = _nse_session()
    response = session.get(url, params=params, timeout=30)
    if response.status_code in (401, 403):
        st.session_state.pop("nse_http_session", None)
        session = _nse_session()
        response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _fetch_nse_option_chain(symbol: str) -> dict:
    return _nse_get_json("/api/option-chain-indices", params={"symbol": symbol})


def _build_option_chain_frame(chain_payload: dict, expiry: str) -> pd.DataFrame:
    records = chain_payload.get("records") or {}
    data_rows = records.get("data") or []
    rows: list[dict] = []
    for item in data_rows:
        if str(item.get("expiryDate", "")).strip() != expiry:
            continue
        strike = item.get("strikePrice")
        ce = item.get("CE") or {}
        pe = item.get("PE") or {}
        rows.append(
            {
                "Strike": float(strike) if strike is not None else None,
                "CE OI": ce.get("openInterest"),
                "CE IV": ce.get("impliedVolatility"),
                "CE LTP": ce.get("lastPrice"),
                "CE Bid": ce.get("bidprice"),
                "CE Ask": ce.get("askPrice"),
                "PE Bid": pe.get("bidprice"),
                "PE Ask": pe.get("askPrice"),
                "PE LTP": pe.get("lastPrice"),
                "PE IV": pe.get("impliedVolatility"),
                "PE OI": pe.get("openInterest"),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).dropna(subset=["Strike"]).sort_values("Strike").reset_index(drop=True)
    return frame


def _format_expiry_tokens(expiry_text: str) -> list[str]:
    raw = str(expiry_text).strip()
    out = {raw.upper().replace("-", "")}
    for fmt in ("%d-%b-%Y", "%d %b %Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            out.add(dt.strftime("%d%b%y").upper())
            out.add(dt.strftime("%d%b%Y").upper())
            out.add(dt.strftime("%d-%b-%Y").upper())
        except ValueError:
            continue
    return [x for x in out if x]


def _extract_lot_size_from_row(row: dict) -> int | None:
    for key, value in row.items():
        k = str(key).lower()
        if "lot" not in k:
            continue
        try:
            num = int(float(str(value).replace(",", "").strip()))
            if num > 0:
                return num
        except (TypeError, ValueError):
            continue
    return None


def _find_fo_contract_token(
    session: object,
    token_id: str,
    underlying: str,
    expiry: str,
    strike: float,
    option_type: str,
) -> tuple[str | None, str | None, int | None]:
    symbols = _load_master_symbols(session, token_id, "nse_fo")
    expiry_tokens = _format_expiry_tokens(expiry)
    strike_text = f"{int(round(float(strike)))}"
    opt = option_type.upper()
    best: dict | None = None
    for row in symbols:
        norm = row.get("norm", "")
        if underlying.upper() not in norm:
            continue
        if opt not in norm:
            continue
        if strike_text not in norm:
            continue
        if not any(tok.replace("-", "") in norm for tok in expiry_tokens):
            continue
        best = row
        break
    if not best:
        return None, None, None
    return (
        best.get("token") or best.get("symbol"),
        best.get("symbol"),
        _extract_lot_size_from_row(best),
    )


# ---------------------------------------------------------------------------
# NSE data helpers
# ---------------------------------------------------------------------------

def _find_nse_files() -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {"indices": [], "futures": [], "options": []}
    if not NSE_DATA_DIR.exists():
        return result
    for p in sorted(NSE_DATA_DIR.glob("*.csv")):
        name = p.name.lower()
        if "all-indices" in name:
            result["indices"].append(p)
        elif "mw-fo" in name:
            result["futures"].append(p)
        elif "option-chain" in name:
            result["options"].append(p)
    return result


def _latest_local_option_chain_file(underlying: str) -> Path | None:
    if not NSE_DATA_DIR.exists():
        return None
    target = "BANKNIFTY" if underlying.upper() == "BANKNIFTY" else "NIFTY"
    candidates = [
        p for p in NSE_DATA_DIR.glob("option-chain-ED-*.csv")
        if target in p.name.upper()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _expiry_from_chain_filename(path: Path) -> str:
    # Example: option-chain-ED-NIFTY-07-Apr-2026.csv
    stem = path.stem
    parts = stem.split("-")
    if len(parts) >= 6:
        return "-".join(parts[-3:])
    return "LocalSnapshot"


def _clean_num(val: str) -> float | None:
    try:
        stripped = str(val).replace(",", "").strip()
        if stripped in ("-", "", "nan", "NaN"):
            return None
        return float(stripped)
    except (ValueError, TypeError):
        return None


def _load_indices_df(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=0, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.applymap(lambda x: str(x).strip() if pd.notna(x) else "")
        return df
    except Exception:
        return pd.DataFrame()


def _load_futures_df(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=0, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.applymap(lambda x: str(x).strip() if pd.notna(x) else "")
        return df
    except Exception:
        return pd.DataFrame()


def _parse_option_chain(path: Path) -> pd.DataFrame:
    """Parse NSE option chain CSV (row1=CALLS,,PUTS, row2=column headers)."""
    try:
        import csv as _csv
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            all_rows = list(_csv.reader(f))
        # Row 0: CALLS,,PUTS — skip
        # Row 1: column headers
        # Row 2+: data
        if len(all_rows) < 3:
            return pd.DataFrame()
        raw_headers = [h.strip() for h in all_rows[1]]
        data_rows = all_rows[2:]
        # Find STRIKE column index
        strike_idx = next(
            (i for i, h in enumerate(raw_headers) if "STRIKE" in h.upper()), None
        )
        if strike_idx is None:
            return pd.DataFrame()
        records = []
        for row in data_rows:
            if len(row) <= strike_idx:
                continue
            strike_val = _clean_num(row[strike_idx])
            if strike_val is None:
                continue
            call_ltp = _clean_num(row[5]) if len(row) > 5 else None
            call_oi = _clean_num(row[1]) if len(row) > 1 else None
            call_iv = _clean_num(row[4]) if len(row) > 4 else None
            call_bid = _clean_num(row[8]) if len(row) > 8 else None
            call_ask = _clean_num(row[9]) if len(row) > 9 else None
            # PUT side: after strike (index 11)
            put_ltp = _clean_num(row[17]) if len(row) > 17 else None
            put_oi = _clean_num(row[21]) if len(row) > 21 else None
            put_iv = _clean_num(row[18]) if len(row) > 18 else None
            put_bid = _clean_num(row[13]) if len(row) > 13 else None
            put_ask = _clean_num(row[14]) if len(row) > 14 else None
            records.append({
                "Strike": strike_val,
                "CE OI": call_oi,
                "CE IV": call_iv,
                "CE LTP": call_ltp,
                "CE Bid": call_bid,
                "CE Ask": call_ask,
                "PE Bid": put_bid,
                "PE Ask": put_ask,
                "PE LTP": put_ltp,
                "PE IV": put_iv,
                "PE OI": put_oi,
            })
        df = pd.DataFrame(records).sort_values("Strike").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def _fmt_option_val(val: float | None) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    return f"{val:,.2f}"


def _detect_atm(df: pd.DataFrame, spot: float | None) -> float | None:
    if spot is None or df.empty:
        return None
    strikes = df["Strike"].dropna()
    if strikes.empty:
        return None
    idx = (strikes - spot).abs().idxmin()
    return float(strikes.loc[idx])


def render_nse_data_panel() -> None:
    st.subheader("NSE Market Data")
    st.caption("Reference data from downloaded NSE files. Refresh by re-downloading from NSE and saving to the 'nse data' folder.")
    nse_files = _find_nse_files()

    if not any(nse_files.values()):
        st.warning(f"No NSE data files found in: {NSE_DATA_DIR}. Download MW / options chain CSVs from NSE and save them there.")
        return

    tab_labels = []
    if nse_files["indices"]:
        tab_labels.append("Indices")
    for p in nse_files["futures"]:
        label = p.stem.replace("MW-FO-", "").replace("-", " ").title()
        tab_labels.append(f"Fut: {label[:20]}")
    for p in nse_files["options"]:
        label = p.stem
        if "NIFTY" in label.upper() and "BANK" not in label.upper():
            tab_labels.append("NIFTY Options")
        elif "BANKNIFTY" in label.upper() or "BANK" in label.upper():
            tab_labels.append("BANKNIFTY Options")
        else:
            tab_labels.append(f"Options: {label[:18]}")

    if not tab_labels:
        st.info("No recognizable NSE data files found.")
        return

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # --- Indices tab ---
    if nse_files["indices"]:
        with tabs[tab_idx]:
            df = _load_indices_df(nse_files["indices"][0])
            if df.empty:
                st.info("Could not load indices file.")
            else:
                # Pick key columns present in file
                key_cols = [c for c in ["INDEX \n", "INDEX", "CURRENT \n", "CURRENT",
                                         "%CHNG \n", "%CHNG", "OPEN \n", "OPEN",
                                         "HIGH \n", "HIGH", "LOW \n", "LOW",
                                         "PREV. CLOSE \n", "PREV. CLOSE"] if c in df.columns]
                display_df = df[key_cols] if key_cols else df.iloc[:, :8]
                # Clean column names for display
                display_df = display_df.copy()
                display_df.columns = [c.replace(" \n", "").strip() for c in display_df.columns]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        tab_idx += 1

    # --- Futures tabs ---
    for path in nse_files["futures"]:
        with tabs[tab_idx]:
            df = _load_futures_df(path)
            if df.empty:
                st.info("Could not load futures file.")
            else:
                display_df = df.copy()
                display_df.columns = [c.replace(" \n", "").replace("\n", " ").strip() for c in display_df.columns]
                key_cols = [c for c in ["INSTRUMENT TYPE", "SYMBOL", "EXPIRY DATE",
                                         "LTP", "CHNG", "%CHNG", "OPEN", "HIGH", "LOW",
                                         "VOLUME (Contracts)", "OPEN INTEREST", "UNDERLYING VALUE"]
                            if c in display_df.columns]
                show_df = display_df[key_cols] if key_cols else display_df.iloc[:, :10]
                st.dataframe(show_df, use_container_width=True, hide_index=True)

                # Highlight contracts
                if "LTP" in show_df.columns:
                    st.caption(f"Contracts available: {len(show_df)} | File: {path.name}")
        tab_idx += 1

    # --- Options chain tabs ---
    for path in nse_files["options"]:
        with tabs[tab_idx]:
            is_bank = "BANK" in path.name.upper()
            underlying_label = "BANKNIFTY" if is_bank else "NIFTY 50"
            spot_key = "latest_nifty_bank_spot" if is_bank else "latest_nifty_index"
            spot: float | None = st.session_state.get(spot_key)

            df = _parse_option_chain(path)
            if df.empty:
                st.info("Could not parse option chain file.")
                tab_idx += 1
                continue

            st.markdown(f"**{underlying_label} Option Chain** — {path.stem}")
            if spot:
                st.caption(f"Spot (live Nifty RSI watch): {spot:.2f}")

            strikes_all = sorted(df["Strike"].dropna().unique().tolist())
            atm = _detect_atm(df, spot)

            # Strike range filter
            if atm:
                default_range = 20 if not is_bank else 10  # number of strikes each side
                atm_idx = next((i for i, s in enumerate(strikes_all) if s >= atm), len(strikes_all) // 2)
                lo_idx = max(0, atm_idx - default_range)
                hi_idx = min(len(strikes_all) - 1, atm_idx + default_range)
                lo_default = strikes_all[lo_idx]
                hi_default = strikes_all[hi_idx]
            else:
                lo_default = strikes_all[0]
                hi_default = strikes_all[-1]

            col_lo, col_hi = st.columns(2)
            with col_lo:
                lo_val = st.selectbox(
                    "From Strike",
                    options=strikes_all,
                    index=strikes_all.index(lo_default),
                    key=f"opt_lo_{path.stem}",
                    format_func=lambda x: f"{x:,.0f}",
                )
            with col_hi:
                hi_val = st.selectbox(
                    "To Strike",
                    options=strikes_all,
                    index=strikes_all.index(hi_default),
                    key=f"opt_hi_{path.stem}",
                    format_func=lambda x: f"{x:,.0f}",
                )

            filtered = df[(df["Strike"] >= lo_val) & (df["Strike"] <= hi_val)].copy()

            # Broker-style chain table: CE side | STRIKE | PE side
            def highlight_atm(row: pd.Series) -> list[str]:
                if atm and abs(row["Strike"] - atm) < 1:
                    return ["background-color:#1a3a1a;color:#7fff7f"] * len(row)
                return [""] * len(row)

            display = filtered[[
                "CE OI", "CE IV", "CE LTP", "CE Bid", "CE Ask",
                "Strike",
                "PE Bid", "PE Ask", "PE LTP", "PE IV", "PE OI",
            ]].copy()
            display["Strike"] = display["Strike"].apply(lambda x: f"{x:,.0f}")
            for col in ["CE OI", "CE IV", "CE LTP", "CE Bid", "CE Ask",
                        "PE Bid", "PE Ask", "PE LTP", "PE IV", "PE OI"]:
                display[col] = display[col].apply(_fmt_option_val)

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )
            if atm:
                st.caption(f"ATM strike ~{atm:,.0f} (highlighted in green)")

            # Strike selection for order
            st.markdown("**Place Options Order from Chain**")
            sel_strike = st.selectbox(
                "Select Strike",
                options=strikes_all,
                index=strikes_all.index(atm) if atm and atm in strikes_all else 0,
                key=f"opt_sel_strike_{path.stem}",
                format_func=lambda x: f"{x:,.0f}",
            )
            opt_type = st.radio(
                "Option Type",
                options=["CE (Call)", "PE (Put)"],
                horizontal=True,
                key=f"opt_type_{path.stem}",
            )
            opt_qty = st.number_input(
                "Lot Qty (1 lot = 65 for NIFTY, 30 for BANKNIFTY)",
                min_value=1,
                value=65 if not is_bank else 30,
                step=65 if not is_bank else 30,
                key=f"opt_qty_{path.stem}",
            )

            # Show selected option LTP from chain
            sel_row = df[df["Strike"] == sel_strike]
            if not sel_row.empty:
                ce_ltp = sel_row.iloc[0]["CE LTP"]
                pe_ltp = sel_row.iloc[0]["PE LTP"]
                if "CE" in opt_type:
                    ref_ltp = ce_ltp
                    ltp_label = "CE LTP (from chain)"
                else:
                    ref_ltp = pe_ltp
                    ltp_label = "PE LTP (from chain)"
                if ref_ltp:
                    st.caption(f"{ltp_label}: {ref_ltp:,.2f} | Est. value: {ref_ltp * opt_qty:,.2f}")

            # Build symbol hint for Kotak NEO F&O
            ce_pe_code = "CE" if "CE" in opt_type else "PE"
            expiry_hint = path.stem.split("-")[-1]  # e.g. 07-Apr-2026
            symbol_hint = f"{underlying_label.replace(' ', '')} {sel_strike:,.0f} {ce_pe_code} {expiry_hint}"
            st.info(
                f"To trade this option in the bot, set Exchange Segment = nse_fo, "
                f"and use the exact Kotak NeoSymbol for: **{symbol_hint}**. "
                f"Use Load/Refresh Symbol Master in the RSI Charts section to find the exact token."
            )
        tab_idx += 1


def _extract_master_file_urls(payload: object) -> list[str]:
    urls: list[str] = []

    def _walk(obj: object) -> None:
        if isinstance(obj, str):
            if obj.lower().endswith(".csv"):
                urls.append(obj)
            return
        if isinstance(obj, list):
            for item in obj:
                _walk(item)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_lower = str(key).lower()
                if key_lower in {"filespaths", "filepaths", "paths", "files"}:
                    _walk(value)
                elif isinstance(value, str) and value.lower().endswith(".csv"):
                    urls.append(value)
                elif isinstance(value, (list, dict)):
                    _walk(value)

    _walk(payload)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in urls:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _normalize_sym(value: str) -> str:
    return "".join(str(value).upper().split())


def _load_master_symbols(session, token_id: str, exchange_segment: str) -> list[dict]:
    cache_key = f"master_symbols_{exchange_segment}"
    cached = st.session_state.get(cache_key)
    if isinstance(cached, list) and cached:
        return cached

    paths_url = f"{session.base_url}/script-details/1.0/masterscrip/file-paths"
    r = requests.get(paths_url, headers={"Authorization": token_id}, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"Scrip master file-paths failed ({r.status_code}): {r.text}")

    entries = _extract_master_file_urls(r.json())
    seg_lower = exchange_segment.lower()
    file_url = next((url for url in entries if seg_lower in url.lower()), None)
    if not file_url:
        raise RuntimeError(f"No scrip master file found for segment '{exchange_segment}'.")

    r2 = requests.get(file_url, timeout=120)
    if r2.status_code >= 400:
        raise RuntimeError(f"Scrip master download failed ({r2.status_code}).")

    content = r2.content.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(content))
    symbols: list[dict] = []
    for row in reader:
        raw_sym = (
            row.get("pTrdSymbol")
            or row.get("TradingSymbol")
            or row.get("trdSym")
            or row.get("symbol")
            or ""
        ).strip()
        raw_token = (
            row.get("pSymbol")
            or row.get("pInstrumentToken")
            or row.get("Token")
            or row.get("token")
            or ""
        ).strip()
        if not raw_sym:
            continue
        token = raw_token[:-2] if raw_token.endswith(".0") else raw_token
        symbols.append(
            {
                "symbol": raw_sym,
                "token": token,
                "display": f"{raw_sym} ({token or '-'})",
                "norm": _normalize_sym(raw_sym),
                "lot_size": _extract_lot_size_from_row(row),
            }
        )

    if not symbols:
        raise RuntimeError(f"Scrip master parsed but no symbols found for '{exchange_segment}'.")

    st.session_state[cache_key] = symbols
    return symbols


def require_totp(otp: str) -> str:
    value = otp.strip()
    if not value:
        raise RuntimeError("Enter a fresh TOTP in the UI before using manual test actions.")
    return value


def authenticate_for_ui(config: dict, otp: str):
    totp = require_totp(otp)
    token_id = get_token_id(config)
    session = authenticate(config, token_id=token_id, otp=totp)
    return session, token_id


def clear_manual_error() -> None:
    st.session_state.pop("manual_action_error", None)


def reset_ui_session() -> None:
    st.session_state.pop("ui_session", None)
    st.session_state.pop("ui_token_id", None)
    st.session_state.pop("ui_session_symbol", None)


def connect_ui_session(config: dict, otp: str) -> None:
    session, token_id = authenticate_for_ui(config, otp)
    st.session_state["ui_session"] = session
    st.session_state["ui_token_id"] = token_id
    st.session_state["ui_session_symbol"] = str(config.get("symbol", {}).get("trading_symbol", ""))
    st.session_state["manual_action_output"] = "Neo v2 session connected successfully."


def get_connected_ui_session(config: dict) -> tuple[object, str]:
    session = st.session_state.get("ui_session")
    token_id = st.session_state.get("ui_token_id")
    if session and token_id:
        return session, token_id
    raise RuntimeError("No active API session. Enter a fresh TOTP and click Connect Session first.")


def extract_fund_metrics(limits: dict) -> dict[str, str]:
    return {
        "Net": str(limits.get("Net", "-")),
        "MarginUsed": str(limits.get("MarginUsed", "-")),
        "CollateralValue": str(limits.get("CollateralValue", "-")),
        "NotionalCash": str(limits.get("NotionalCash", "-")),
        "Category": str(limits.get("Category", "-")),
    }


def parse_money(value: object) -> float | None:
    try:
        text = str(value).replace(",", "").strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def run_manual_action(config: dict, otp: str, mode: str, action: str) -> None:
    try:
        session, token_id = get_connected_ui_session(config)
        symbol_cfg = config["symbol"]

        if action == "funds":
            limits = check_limits(session)
            st.session_state["manual_limits"] = limits
            st.session_state["manual_action_output"] = "Funds refreshed successfully."
            return

        if action == "orders":
            orders = check_orders(session)
            st.session_state["manual_orders"] = orders
            st.session_state["manual_action_output"] = "Order book refreshed successfully."
            return

        if action == "positions":
            positions = check_positions(session)
            st.session_state["manual_positions"] = positions
            st.session_state["manual_action_output"] = "Positions refreshed successfully."
            return

        instruments = resolve_quote_candidates(symbol_cfg, session=session, token_id=token_id)
        st.session_state["manual_quote_candidates"] = instruments
        ltp = fetch_ltp(
            base_url=session.base_url,
            token_id=token_id,
            exchange_segment=str(symbol_cfg["exchange_segment"]),
            instruments=instruments,
        )
        st.session_state["manual_ltp"] = ltp

        if action == "quote":
            st.session_state["manual_action_output"] = f"LTP fetched successfully: {ltp:.2f}"
            return

        side = "B" if action == "buy" else "S"
        payload = build_order_payload(config, side=side, ltp=ltp)
        st.session_state["manual_order_payload"] = payload
        limits = check_limits(session)
        st.session_state["manual_limits"] = limits
        if action == "buy" and mode == "live":
            available_net = parse_money(limits.get("Net"))
            full_value = ltp * int(symbol_cfg.get("quantity", 1))
            product = str(config.get("symbol", {}).get("product", "CNC")).upper()
            # MTF requires ~25% upfront margin; MIS/CO/BO are intraday with ~20% margin
            # CNC and NRML require full order value
            if product == "MTF":
                required_value = full_value * 0.25
                margin_note = f"MTF margin required (~25%): {required_value:.2f}"
            elif product in ("MIS", "CO", "BO"):
                required_value = full_value * 0.20
                margin_note = f"Intraday margin required (~20%): {required_value:.2f}"
            else:
                required_value = full_value
                margin_note = f"Full order value required: {required_value:.2f}"
            if available_net is not None and available_net < required_value:
                raise RuntimeError(
                    f"{margin_note} | Available Net: {available_net:.2f}. "
                    "Add funds or use a cheaper symbol/lower quantity."
                )
        if mode == "paper":
            st.session_state["manual_order_result"] = {"mode": "paper", "payload": payload}
            st.session_state["manual_action_output"] = f"Paper {action.upper()} prepared at LTP {ltp:.2f}."
        else:
            result = place_order(session, payload)
            st.session_state["manual_order_result"] = result
            st.session_state["manual_action_output"] = f"Live {action.upper()} order submitted at LTP {ltp:.2f}."

        st.session_state["manual_orders"] = check_orders(session)
        st.session_state["manual_positions"] = check_positions(session)
        st.session_state["manual_limits"] = check_limits(session)
    except Exception as exc:
        error_text = str(exc)
        if "tradeApiLogin failed" in error_text or "tradeApiValidate failed" in error_text or "Invalid Totp" in error_text or "Invalid TOTP" in error_text or "Invalid Token" in error_text or "Unauthorized" in error_text:
            reset_ui_session()
        st.session_state["manual_action_error"] = str(exc)


# Credential keys — stored only in st.session_state, never written to disk
_CRED_KEYS = {"token_id", "mobile_number", "ucc", "mpin"}


def load_config() -> dict:
    """Load config.json and overlay any credentials cached in session_state."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file_handle:
        cfg = json.load(file_handle)
    # Overlay credentials from session_state (entered via UI, never stored on disk)
    cached_creds = st.session_state.get("_cached_creds", {})
    for key in _CRED_KEYS:
        if cached_creds.get(key):
            cfg[key] = cached_creds[key]
    return cfg


def save_config(data: dict) -> None:
    """Write non-credential settings to config.json; cache credentials in session_state."""
    # Persist credentials to session_state only
    cached = st.session_state.setdefault("_cached_creds", {})
    for key in _CRED_KEYS:
        if data.get(key):
            cached[key] = data[key]
    # Strip credentials before writing to disk
    safe = {k: v for k, v in data.items() if k not in _CRED_KEYS}
    with CONFIG_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(safe, file_handle, indent=4)


def _fmt_money(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _order_type_label(code: str) -> str:
    labels = {
        "MKT": "Market",
        "L": "Limit",
        "SL": "SL-LMT",
        "SL-M": "SL-MKT",
    }
    return labels.get(code.upper(), code)


def _side_label(code: str) -> str:
    return "BUY" if str(code).upper() == "B" else "SELL"


def _status_badge(status: str) -> tuple[str, str]:
    up = status.upper()
    if any(token in up for token in ("COMPLETE", "SUCCESS", "EXECUTED")):
        return "#0a3", "SUCCESS"
    if any(token in up for token in ("REJECT", "FAILED", "ERROR")):
        return "#c33", "FAILED"
    if any(token in up for token in ("OPEN", "PENDING", "TRIGGER")):
        return "#d98b00", "PENDING"
    return "#4b8", status or "UNKNOWN"


def _extract_orders_list(orders_payload: object) -> list[dict]:
    if isinstance(orders_payload, list):
        return [item for item in orders_payload if isinstance(item, dict)]
    if isinstance(orders_payload, dict):
        data = orders_payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return []


def _extract_executed_orders(orders_payload: object) -> list[dict]:
    orders = _extract_orders_list(orders_payload)
    rows: list[dict] = []
    for order in orders:
        status = str(order.get("ordSt") or order.get("status") or order.get("stat") or "")
        status_upper = status.upper()
        if not any(token in status_upper for token in ("COMPLETE", "EXECUTED", "TRADED", "FILLED")):
            continue

        rows.append(
            {
                "Time": str(order.get("ordDtTm") or order.get("ordTm") or order.get("trdTm") or "-"),
                "Order ID": str(order.get("nOrdNo") or order.get("ordNo") or order.get("ordId") or "-"),
                "Symbol": str(order.get("trdSym") or order.get("sym") or order.get("ts") or "-"),
                "Side": "BUY" if str(order.get("trnsTp") or order.get("tt") or "").upper() == "B" else "SELL",
                "Qty": str(order.get("fldQty") or order.get("qty") or order.get("qt") or "-"),
                "Avg Price": str(order.get("avgPrc") or order.get("prc") or order.get("pr") or "-"),
                "Product": str(order.get("prdCode") or order.get("prd") or order.get("pc") or "-"),
                "Status": status or "-",
            }
        )
    return rows


def _read_log_incremental(path: Path) -> list[str]:
    if not path.exists():
        st.session_state["log_file_offset"] = 0
        st.session_state["log_lines"] = []
        return []

    prev_offset = int(st.session_state.get("log_file_offset", 0))
    buffer_lines = list(st.session_state.get("log_lines", []))
    file_size = path.stat().st_size

    if file_size < prev_offset:
        prev_offset = 0
        buffer_lines = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(prev_offset)
        chunk = f.read()
        st.session_state["log_file_offset"] = f.tell()

    if chunk:
        new_lines = chunk.replace("\r\n", "\n").split("\n")
        if new_lines and new_lines[-1] == "":
            new_lines = new_lines[:-1]
        buffer_lines.extend(new_lines)
        buffer_lines = buffer_lines[-350:]
        st.session_state["log_lines"] = buffer_lines

    return buffer_lines


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _build_ohlc_with_rsi(ticks: list[dict], timeframe: str) -> pd.DataFrame:
    if not ticks:
        return pd.DataFrame()

    df = pd.DataFrame(ticks)
    if df.empty or "ts" not in df.columns or "price" not in df.columns:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["ts", "price"])
    if df.empty:
        return pd.DataFrame()

    ohlc = (
        df.set_index("ts")["price"]
        .resample(timeframe)
        .ohlc()
        .dropna(how="any")
        .reset_index()
    )
    if ohlc.empty:
        return pd.DataFrame()

    ohlc["rsi_14"] = _compute_rsi(ohlc["close"], period=14)
    return ohlc


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_yf_rsi(yf_ticker: str, interval: str, bars: int) -> pd.DataFrame:
    """Download OHLC from yfinance and return a frame with rsi_14 column.
    Cached for 5 minutes so repeated widget interactions don't re-fetch.
    interval: '1d', '1h', '15m', '5m'
    """
    import yfinance as yf  # type: ignore
    from datetime import timedelta

    period_map = {"1d": 365, "1h": 60, "15m": 30, "5m": 10}
    days = period_map.get(interval, 60)
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(yf_ticker, start=start, end=end, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    df = df.reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "ts"})
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).tail(bars).reset_index(drop=True)
    df["rsi_14"] = _compute_rsi(df["close"], period=14)
    return df


def _render_yf_chart(title: str, yf_ticker: str, interval: str, bars: int) -> None:
    """Render close-price + RSI chart for a yfinance ticker."""
    df = _fetch_yf_rsi(yf_ticker, interval, bars)
    if df.empty:
        st.info(f"No data for {title} ({yf_ticker}) at {interval}.")
        return
    ts = df["ts"].dt.strftime("%d-%b %H:%M" if interval != "1d" else "%d-%b-%y")
    plot_df = pd.DataFrame({"Close": df["close"].values, "RSI": df["rsi_14"].values}, index=ts)
    col_price, col_rsi = st.columns(2)
    with col_price:
        st.markdown(f"**{title} — Price ({interval})**")
        st.line_chart(plot_df[["Close"]], height=220)
        last = df.iloc[-1]
        st.caption(f"Close: {last['close']:.2f}")
    with col_rsi:
        st.markdown(f"**{title} — RSI(14) ({interval})**")
        st.line_chart(plot_df[["RSI"]], height=220)
        rsi_val = last["rsi_14"]
        zone = "Overbought 🔴" if rsi_val > 70 else ("Oversold 🟢" if rsi_val < 30 else "Neutral ⚪")
        st.caption(f"RSI: {rsi_val:.2f} — {zone}")


def _append_tick(key: str, price: float) -> None:
    ticks = list(st.session_state.get(key, []))
    ticks.append({"ts": datetime.now().isoformat(timespec="seconds"), "price": float(price)})
    if len(ticks) > 4000:
        ticks = ticks[-4000:]
    st.session_state[key] = ticks


def _render_chart_block(title: str, ticks: list[dict], timeframe: str) -> None:
    st.markdown(f"**{title} ({timeframe})**")
    frame = _build_ohlc_with_rsi(ticks, timeframe=timeframe)
    if frame.empty:
        st.info("Waiting for enough ticks to form candles...")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(frame.set_index("ts")["close"], height=220)
    with c2:
        rsi_df = frame.set_index("ts")[["rsi_14"]]
        st.line_chart(rsi_df, height=220)
        st.caption("RSI(14): above 70 = overbought, below 30 = oversold")

    latest = frame.iloc[-1]
    st.caption(
        f"Latest candle O:{latest['open']:.2f} H:{latest['high']:.2f} "
        f"L:{latest['low']:.2f} C:{latest['close']:.2f} | RSI14:{latest['rsi_14']:.2f}"
    )


def _filter_log_lines(lines: list[str], compact: bool) -> list[str]:
    if not compact:
        return lines

    keep_tokens = (
        "=== START",
        "Step 1/2",
        "Step 2/2",
        "Running Neo v2 bot",
        "Symbol=",
        "Quote symbol candidates=",
        "Signal=",
        "Cooldown active",
        "[PAPER]",
        "[LIVE]",
        "Traceback",
        "RuntimeError",
        "Error",
        "failed",
        "rejected",
        "Validation successful",
    )
    filtered: list[str] = []
    last_ltp: str | None = None
    for line in lines:
        if "LTP=" in line:
            ltp_value = line.split("LTP=", 1)[1].strip()
            if ltp_value == last_ltp:
                continue
            last_ltp = ltp_value
            filtered.append(line)
            continue
        if any(token in line for token in keep_tokens):
            filtered.append(line)

    return filtered[-220:]


def render_intro() -> None:
    st.set_page_config(page_title="Kotak Intraday Bot", layout="wide")
    st.title("Kotak Neo v2 Intraday Bot")
    st.caption("Configure token, strategy and run Neo v2 bot locally.")
    st.info("Update your trading logic in strategy_logic.py by editing evaluate_signal().")


def render_config_form(config: dict) -> dict:
    # Pre-fill credential fields from session_state cache if config.json has blanks
    _cached = st.session_state.get("_cached_creds", {})
    for key in _CRED_KEYS:
        if not config.get(key) and _cached.get(key):
            config[key] = _cached[key]

    c1, c2 = st.columns(2)
    with c1:
        token_value = config.get("token_id") or config.get("access_token") or config.get("consumer_key", "")
        config["token_id"] = st.text_input("Token ID (Access Token)", value=token_value, key="cfg_token_id")
        config["mobile_number"] = st.text_input("Mobile Number", value=config.get("mobile_number", ""), key="cfg_mobile_number")
        config["ucc"] = st.text_input("UCC", value=config.get("ucc", ""), key="cfg_ucc")
        config["mpin"] = st.text_input("MPIN", value=config.get("mpin", ""), type="password", key="cfg_mpin")
        env_options = ["prod", "uat"]
        current_env = config.get("environment", "prod")
        env_index = env_options.index(current_env) if current_env in env_options else 0
        config["environment"] = st.selectbox("Environment", options=env_options, index=env_index, key="cfg_environment")

    symbol_cfg = config.setdefault("symbol", {})
    with c2:
        current_sym = symbol_cfg.get("trading_symbol", "TCS-EQ")
        nifty50_opts = NIFTY50_SYMBOLS + ["Other (type below)"]
        if current_sym in NIFTY50_SYMBOLS:
            sym_idx = NIFTY50_SYMBOLS.index(current_sym)
            default_custom = ""
        else:
            sym_idx = len(nifty50_opts) - 1
            default_custom = current_sym
        selected_sym = st.selectbox("Trading Symbol (Nifty 50)", options=nifty50_opts, index=sym_idx, key="cfg_symbol_dropdown")
        custom_sym = st.text_input(
            "Custom Symbol (overrides above)",
            value=default_custom,
            key="cfg_custom_symbol",
            help="For stocks outside Nifty 50, e.g. CENTRALBK-EQ, ITBEES-EQ. Leave empty to use dropdown.",
        )
        if custom_sym.strip():
            symbol_cfg["trading_symbol"] = custom_sym.strip()
        elif selected_sym != "Other (type below)":
            symbol_cfg["trading_symbol"] = selected_sym
        symbol_cfg["instrument_token"] = st.text_input(
            "Instrument Token",
            value=str(symbol_cfg.get("instrument_token", "")),
            key="cfg_instrument_token",
            help="Leave empty — bot auto-resolves from Kotak scrip master on startup.",
        )
        seg_keys = list(EXCHANGE_SEGMENTS.keys())
        current_seg = symbol_cfg.get("exchange_segment", "nse_cm")
        seg_idx = seg_keys.index(current_seg) if current_seg in seg_keys else 0
        symbol_cfg["exchange_segment"] = st.selectbox(
            "Exchange Segment",
            options=seg_keys,
            format_func=lambda k: EXCHANGE_SEGMENTS[k],
            index=seg_idx,
            key="cfg_exchange_segment",
        )
        prod_keys = list(PRODUCT_TYPES.keys())
        current_prod = symbol_cfg.get("product", "MIS")
        prod_idx = prod_keys.index(current_prod) if current_prod in prod_keys else 0
        symbol_cfg["product"] = st.selectbox(
            "Product",
            options=prod_keys,
            format_func=lambda k: PRODUCT_TYPES[k],
            index=prod_idx,
            key="cfg_product",
        )
        symbol_cfg["quantity"] = st.number_input(
            "Quantity",
            min_value=1,
            value=int(symbol_cfg.get("quantity", 1)),
            step=1,
            key="cfg_quantity",
        )

    st.subheader("Strategy")
    s1, s2, s3 = st.columns(3)
    with s1:
        config["sma_window"] = st.number_input(
            "SMA Window",
            min_value=2,
            value=int(config.get("sma_window", 5)),
            step=1,
            key="cfg_sma_window",
        )
    with s2:
        config["entry_threshold_pct"] = st.number_input(
            "Entry Threshold %",
            min_value=0.01,
            value=float(config.get("entry_threshold_pct", 0.12)),
            step=0.01,
            format="%.2f",
            key="cfg_entry_threshold_pct",
        )
    with s3:
        config["cooldown_seconds"] = st.number_input(
            "Cooldown Seconds",
            min_value=1,
            value=int(config.get("cooldown_seconds", 30)),
            step=1,
            key="cfg_cooldown_seconds",
        )

    t1, t2 = st.columns(2)
    with t1:
        config["poll_interval_seconds"] = st.number_input(
            "Price Poll Interval Seconds",
            min_value=1,
            value=int(config.get("poll_interval_seconds", 2)),
            step=1,
            key="cfg_poll_interval_seconds",
        )
    with t2:
        config["decision_interval_seconds"] = st.number_input(
            "Decision Interval Seconds",
            min_value=1,
            value=int(config.get("decision_interval_seconds", 5)),
            step=1,
            key="cfg_decision_interval_seconds",
        )

    ot1, ot2 = st.columns(2)
    with ot1:
        ot_options = list(ORDER_TYPES.keys())
        current_ot = str(config.get("order_type", "MKT")).upper()
        ot_idx = ot_options.index(current_ot) if current_ot in ot_options else 0
        config["order_type"] = st.selectbox(
            "Order Type",
            options=ot_options,
            index=ot_idx,
            format_func=lambda k: ORDER_TYPES[k],
            key="cfg_order_type",
        )
    with ot2:
        config["limit_price_offset_pct"] = st.number_input(
            "Limit Price Offset %",
            min_value=0.01,
            value=float(config.get("limit_price_offset_pct", 0.1)),
            step=0.01,
            format="%.2f",
            help="Used for Limit and SL-LMT orders. BUY price = LTP + offset%, SELL price = LTP - offset%.",
            key="cfg_limit_price_offset_pct",
        )

    st.caption(
        "Product labels mirror Kotak Neo app naming. Order types follow Neo v2 API codes: "
        "Limit=L, Market=MKT, SL-LMT=SL, SL-MKT=SL-M."
    )

    trigger_col, save_col = st.columns(2)
    with trigger_col:
        config["trigger_price_offset_pct"] = st.number_input(
            "Trigger Price Offset %",
            min_value=0.01,
            value=float(config.get("trigger_price_offset_pct", 0.1)),
            step=0.01,
            format="%.2f",
            help="Used only for SL-LMT and SL-MKT. BUY trigger = LTP + offset%, SELL trigger = LTP - offset%.",
            key="cfg_trigger_price_offset_pct",
        )
    with save_col:
        st.write("")
        st.write("")
        if st.button("Save Config", use_container_width=True):
            save_config(config)
            st.success("Settings saved. Credentials (Token, Mobile, UCC, MPIN) are cached in this session only — not written to disk.")

    with st.expander("How the Strategy Works", expanded=False):
        window = int(config.get("sma_window", 5))
        threshold = float(config.get("entry_threshold_pct", 0.12))
        poll = int(config.get("poll_interval_seconds", 2))
        decision = int(config.get("decision_interval_seconds", 5))
        cooldown = int(config.get("cooldown_seconds", 30))
        order_t = config.get("order_type", "MKT")
        trigger_offset = float(config.get("trigger_price_offset_pct", 0.1))
        st.markdown(f"""
**Current Strategy: SMA Momentum Crossover**

The bot collects live price ticks and generates BUY/SELL signals using a Simple Moving Average:

| Setting | Value | Meaning |
|---|---|---|
| SMA Window | {window} bars | Averages the last {window} price readings |
| Entry Threshold | {threshold}% | Minimum deviation from SMA to trigger a signal |
| Poll Interval | {poll}s | Price is fetched every {poll} seconds |
| Decision Interval | {decision}s | Signal is evaluated every {decision} seconds |
| Cooldown | {cooldown}s | Minimum gap between two consecutive orders |
| Order Type | {order_t} | {ORDER_TYPES.get(order_t, order_t)} |
| Trigger Offset | {trigger_offset}% | Used only for SL-LMT / SL-MKT |

**Signal logic:**
- **BUY** → LTP rises more than **{threshold}%** above SMA → price has momentum upward
- **SELL** → LTP falls more than **{threshold}%** below SMA → price has momentum downward
- **HOLD** → LTP is within ±{threshold}% of SMA → no clear edge, wait

**Order behavior:**
- **Market** places immediately at market price.
- **Limit** places at LTP adjusted by limit offset.
- **SL-LMT** uses a trigger plus a limit price.
- **SL-MKT** uses a trigger and converts to market order once hit.

**To change the logic**, edit `evaluate_signal()` in `strategy_logic.py`.  
Replace SMA with RSI, VWAP, breakout, or any custom condition — the rest of the bot (auth, polling, order placement) stays the same.
        """)

    return config


def run_bot(mode: str, otp: str) -> None:
    command = [sys.executable, str(BOT_PATH), "--mode", mode]
    if otp.strip():
        command.extend(["--otp", otp.strip()])

    BOT_LOG_PATH.write_text("", encoding="utf-8")
    log_handle = BOT_LOG_PATH.open("a", encoding="utf-8", buffering=1)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_handle.write(f"=== START v2_bot.py | mode={mode} | started_at={started_at} ===\n")
    log_handle.flush()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdout=log_handle,
        stderr=log_handle,
        text=True,
        env=env,
    )
    st.session_state["bot_process"] = process
    st.session_state["bot_started_at"] = started_at


def run_validation(otp: str) -> None:
    command = [sys.executable, str(VALIDATOR_PATH)]
    if otp.strip():
        command.extend(["--totp", otp.strip()])

    result = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    st.text_area("Validation Output", value=output[-8000:], height=200)
    if result.returncode == 0:
        st.success("Neo v2 validation passed.")
    else:
        st.error("Validation failed. Check output above.")


def render_derivatives_panel(current_config: dict, mode: str, otp: str) -> None:
    st.subheader("Derivatives Workbench")
    st.caption("Broker-style options/futures panel driven by live NSE API + Kotak master symbols.")

    session = st.session_state.get("ui_session")
    token_id = st.session_state.get("ui_token_id")

    top = st.columns([1, 1, 1, 1])
    with top[0]:
        underlying = st.selectbox("Select Options", options=["NIFTY", "BANKNIFTY"], key="dw_underlying")
    with top[1]:
        chain_refresh = st.button("🔄 Refresh Option Chain", use_container_width=True)
    with top[2]:
        if st.button("Load Kotak F&O Master", use_container_width=True):
            if not session or not token_id:
                st.warning("Connect Session first.")
            else:
                try:
                    _load_master_symbols(session, token_id, "nse_fo")
                    st.success("Kotak nse_fo master loaded.")
                except Exception as exc:
                    st.error(f"Master load failed: {exc}")
    with top[3]:
        st.caption("Connect Session (Home tab) for order placement.")

    cache_key = f"nse_chain_{underlying}"
    if chain_refresh or cache_key not in st.session_state:
        try:
            chain_payload = _fetch_nse_option_chain(underlying)
            st.session_state[cache_key] = chain_payload
            st.session_state["nse_chain_error"] = None
            # Force ATM update on next render by clearing stored strike keys
            st.session_state.pop(f"dw_call_strike_{underlying}", None)
            st.session_state.pop(f"dw_put_strike_{underlying}", None)
        except Exception as exc:
            st.session_state["nse_chain_error"] = str(exc)

    chain_err = st.session_state.get("nse_chain_error")
    if chain_err:
        st.error(f"NSE API option-chain fetch failed: {chain_err}")
        return

    chain_payload = st.session_state.get(cache_key)
    if not isinstance(chain_payload, dict):
        st.info("Click Refresh Option Chain to load live data.")
        return

    records = chain_payload.get("records") or {}
    expiry_dates = records.get("expiryDates") or []
    underlying_value = records.get("underlyingValue")
    fallback_df: pd.DataFrame | None = None
    fallback_expiry: str | None = None
    if not expiry_dates:
        local_path = _latest_local_option_chain_file(underlying)
        if local_path is not None:
            parsed = _parse_option_chain(local_path)
            if not parsed.empty:
                fallback_df = parsed
                fallback_expiry = _expiry_from_chain_filename(local_path)
                expiry_dates = [fallback_expiry]
                st.warning(
                    "NSE option-chain API returned empty payload. "
                    f"Using local snapshot: {local_path.name}"
                )
        if not expiry_dates:
            st.error(
                "No expiry dates found from NSE API, and no local option-chain CSV in nse data/ folder. "
                "Click Refresh Option Chain again."
            )
            return

    row1 = st.columns([1, 1, 1, 1, 1, 1])
    with row1[0]:
        selected_expiry = st.selectbox("Select Date", options=expiry_dates, key=f"dw_exp_{underlying}")

    if fallback_df is not None and fallback_expiry and selected_expiry == fallback_expiry:
        chain_df = fallback_df.copy()
    else:
        chain_df = _build_option_chain_frame(chain_payload, selected_expiry)
    if chain_df.empty:
        st.info("No option rows found for selected expiry.")
        return

    strikes = sorted(chain_df["Strike"].dropna().unique().tolist())
    if not strikes:
        st.info("No strikes available.")
        return

    # ── Live spot: prefer NSE API value, fall back to yfinance ───────────────
    spot = float(underlying_value) if underlying_value is not None else None
    if spot is None:
        try:
            import yfinance as yf  # type: ignore
            yf_idx = "^NSEBANK" if underlying == "BANKNIFTY" else "^NSEI"
            _sp = yf.download(yf_idx, period="1d", interval="1m", progress=False, auto_adjust=True)
            if not _sp.empty:
                spot = float(_sp["Close"].iloc[-1])
        except Exception:
            pass

    atm = _detect_atm(chain_df, spot) if spot is not None else strikes[len(strikes) // 2]
    call_default = atm if atm in strikes else strikes[len(strikes) // 2]
    put_default = call_default

    # Force selectbox to ATM when chain was just refreshed (keys were cleared above)
    call_sk_key = f"dw_call_strike_{underlying}"
    put_sk_key  = f"dw_put_strike_{underlying}"
    if call_sk_key not in st.session_state:
        st.session_state[call_sk_key] = call_default
    if put_sk_key not in st.session_state:
        st.session_state[put_sk_key] = put_default

    with row1[1]:
        call_strike = st.selectbox(
            "Call Strike (ATM auto)",
            options=strikes,
            index=strikes.index(st.session_state[call_sk_key])
                  if st.session_state[call_sk_key] in strikes else strikes.index(call_default),
            key=call_sk_key,
            format_func=lambda x: f"{x:,.0f}",
        )
    with row1[2]:
        put_strike = st.selectbox(
            "Put Strike (ATM auto)",
            options=strikes,
            index=strikes.index(st.session_state[put_sk_key])
                  if st.session_state[put_sk_key] in strikes else strikes.index(put_default),
            key=put_sk_key,
            format_func=lambda x: f"{x:,.0f}",
        )
    with row1[3]:
        lots = st.number_input("Qty (Lots)", min_value=1, value=1, step=1, key=f"dw_lots_{underlying}")
    with row1[4]:
        product = st.selectbox("Product", options=["NRML", "MIS", "MTF", "CNC"], key=f"dw_product_{underlying}")
    with row1[5]:
        order_type = st.selectbox("Order Type", options=["MKT", "L", "SL", "SL-M"], key=f"dw_ordertype_{underlying}")

    call_row = chain_df[chain_df["Strike"] == call_strike]
    put_row  = chain_df[chain_df["Strike"] == put_strike]
    call_ltp = float(call_row.iloc[0]["CE LTP"]) if not call_row.empty and pd.notna(call_row.iloc[0]["CE LTP"]) else 0.0
    put_ltp  = float(put_row.iloc[0]["PE LTP"])  if not put_row.empty  and pd.notna(put_row.iloc[0]["PE LTP"])  else 0.0
    call_iv  = float(call_row.iloc[0]["CE IV"])  if not call_row.empty and pd.notna(call_row.iloc[0]["CE IV"])  else None
    put_iv   = float(put_row.iloc[0]["PE IV"])   if not put_row.empty  and pd.notna(put_row.iloc[0]["PE IV"])   else None

    # Default lot size fallback (latest exchange revision).
    lot_size = 30 if underlying == "BANKNIFTY" else 65
    lot_source = "fallback"
    if session and token_id:
        try:
            _, _, detected_lot = _find_fo_contract_token(
                session=session, token_id=token_id, underlying=underlying,
                expiry=selected_expiry, strike=call_strike, option_type="CE",
            )
            if detected_lot and detected_lot > 0:
                lot_size = detected_lot
                lot_source = "kotak master"
        except Exception:
            pass
    qty = int(lots) * lot_size

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(f"{underlying} Spot", f"{spot:.2f}" if spot is not None else "-")
    with m2:
        st.metric(f"ATM Strike", f"{int(atm):,}" if atm is not None else "-")
    with m3:
        iv_str = f"{call_iv:.1f}%" if call_iv else "-"
        st.metric(f"Call LTP ({int(call_strike)})", f"{call_ltp:.2f}", delta=f"IV {iv_str}" if call_iv else None)
    with m4:
        iv_str2 = f"{put_iv:.1f}%" if put_iv else "-"
        st.metric(f"Put LTP ({int(put_strike)})", f"{put_ltp:.2f}", delta=f"IV {iv_str2}" if put_iv else None)
    with m5:
        # PCR — total put OI / total call OI
        total_ce_oi = pd.to_numeric(chain_df["CE OI"], errors="coerce").sum()
        total_pe_oi = pd.to_numeric(chain_df["PE OI"], errors="coerce").sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else None
        st.metric("PCR (Put/Call OI)", f"{pcr:.2f}" if pcr is not None else "-",
                  delta="Bullish" if pcr and pcr < 0.8 else ("Bearish" if pcr and pcr > 1.2 else "Neutral"))
    st.caption(f"Lot size: {lot_size} ({lot_source}) | Total qty: {qty}")

    # ── Option chain table with ATM highlighted ───────────────────────────────
    preview = chain_df[["CE OI", "CE IV", "CE LTP", "Strike", "PE LTP", "PE IV", "PE OI"]].copy()
    # Format columns
    for col in ("CE OI", "PE OI"):
        preview[col] = pd.to_numeric(preview[col], errors="coerce").apply(
            lambda x: f"{x/1e5:.1f}L" if pd.notna(x) and x >= 1e5 else (f"{x:,.0f}" if pd.notna(x) else "-")
        )
    for col in ("CE LTP", "PE LTP"):
        preview[col] = pd.to_numeric(preview[col], errors="coerce").apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "-"
        )
    for col in ("CE IV", "PE IV"):
        preview[col] = pd.to_numeric(preview[col], errors="coerce").apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "-"
        )

    def _chain_row_style(row: pd.Series) -> list[str]:
        strike_val = _clean_num(str(row["Strike"]).replace(",", ""))
        if atm is not None and strike_val is not None and abs(strike_val - atm) < 1:
            return ["font-weight:bold; background-color:#1a3a1a;"] * len(row)
        return [""] * len(row)

    preview["Strike"] = preview["Strike"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(
        preview.style.apply(_chain_row_style, axis=1),
        use_container_width=True, hide_index=True,
        column_config={
            "CE OI":  st.column_config.TextColumn("CE OI",  width="small"),
            "CE IV":  st.column_config.TextColumn("CE IV",  width="small"),
            "CE LTP": st.column_config.TextColumn("CE LTP", width="small"),
            "Strike": st.column_config.TextColumn("Strike", width="medium"),
            "PE LTP": st.column_config.TextColumn("PE LTP", width="small"),
            "PE IV":  st.column_config.TextColumn("PE IV",  width="small"),
            "PE OI":  st.column_config.TextColumn("PE OI",  width="small"),
        },
    )

    if not session or not token_id:
        st.warning("Connect Session to place manual option/future orders from this panel.")

    def _place_option_order(side: str, opt_type: str, strike_value: float, ltp_hint: float) -> None:
        if mode == "live" and not bool(st.session_state.get("confirm_live_orders", False)):
            st.session_state["manual_action_error"] = "Tick live confirmation checkbox in Run Bot section first."
            return
        if not session or not token_id:
            st.session_state["manual_action_error"] = "Connect Session first."
            return
        try:
            token, symbol_name, detected_lot = _find_fo_contract_token(
                session=session,
                token_id=token_id,
                underlying=underlying,
                expiry=selected_expiry,
                strike=strike_value,
                option_type=opt_type,
            )
            if not token:
                raise RuntimeError(
                    f"Could not map {underlying} {int(strike_value)} {opt_type} {selected_expiry} in Kotak F&O master. "
                    "Click Load Kotak F&O Master and try again."
                )

            order_qty = int(lots) * (detected_lot if detected_lot and detected_lot > 0 else lot_size)
            temp_cfg = json.loads(json.dumps(current_config))
            temp_cfg.setdefault("symbol", {})
            temp_cfg["symbol"]["exchange_segment"] = "nse_fo"
            temp_cfg["symbol"]["trading_symbol"] = str(token)
            temp_cfg["symbol"]["instrument_token"] = str(token)
            temp_cfg["symbol"]["quantity"] = order_qty
            temp_cfg["symbol"]["product"] = product
            temp_cfg["order_type"] = order_type

            side_key = "buy" if side == "B" else "sell"
            run_manual_action(temp_cfg, otp, mode, side_key)
            if "manual_action_error" not in st.session_state:
                st.session_state["manual_action_output"] = (
                    f"{mode.upper()} {side_key.upper()} {underlying} {int(strike_value)} {opt_type} | "
                    f"NeoSymbol={symbol_name or token} | Qty={order_qty}"
                )
        except Exception as exc:
            st.session_state["manual_action_error"] = str(exc)

    actions = st.columns(4)
    with actions[0]:
        if st.button("Buy Call", use_container_width=True):
            clear_manual_error()
            _place_option_order("B", "CE", call_strike, call_ltp)
    with actions[1]:
        if st.button("Sell Call", use_container_width=True):
            clear_manual_error()
            _place_option_order("S", "CE", call_strike, call_ltp)
    with actions[2]:
        if st.button("Buy Put", use_container_width=True):
            clear_manual_error()
            _place_option_order("B", "PE", put_strike, put_ltp)
    with actions[3]:
        if st.button("Sell Put", use_container_width=True):
            clear_manual_error()
            _place_option_order("S", "PE", put_strike, put_ltp)

    apply_row = st.columns([1, 1])
    with apply_row[0]:
        if st.button("Use Call Contract For Bot", use_container_width=True):
            if not session or not token_id:
                st.warning("Connect Session first.")
            else:
                tok, sym, detected_lot = _find_fo_contract_token(session, token_id, underlying, selected_expiry, call_strike, "CE")
                if not tok:
                    st.error("Could not resolve call contract token from Kotak master.")
                else:
                    current_config["symbol"]["exchange_segment"] = "nse_fo"
                    current_config["symbol"]["trading_symbol"] = str(tok)
                    current_config["symbol"]["instrument_token"] = str(tok)
                    current_config["symbol"]["quantity"] = int(lots) * (detected_lot if detected_lot and detected_lot > 0 else lot_size)
                    current_config["symbol"]["product"] = product
                    current_config["order_type"] = order_type
                    save_config(current_config)
                    st.success(f"Bot symbol set to CALL contract {sym or tok}.")
    with apply_row[1]:
        if st.button("Use Put Contract For Bot", use_container_width=True):
            if not session or not token_id:
                st.warning("Connect Session first.")
            else:
                tok, sym, detected_lot = _find_fo_contract_token(session, token_id, underlying, selected_expiry, put_strike, "PE")
                if not tok:
                    st.error("Could not resolve put contract token from Kotak master.")
                else:
                    current_config["symbol"]["exchange_segment"] = "nse_fo"
                    current_config["symbol"]["trading_symbol"] = str(tok)
                    current_config["symbol"]["instrument_token"] = str(tok)
                    current_config["symbol"]["quantity"] = int(lots) * (detected_lot if detected_lot and detected_lot > 0 else lot_size)
                    current_config["symbol"]["product"] = product
                    current_config["order_type"] = order_type
                    save_config(current_config)
                    st.success(f"Bot symbol set to PUT contract {sym or tok}.")


def render_market_watch(current_config: dict) -> None:
    st.subheader("RSI Charts")

    # ── Section 1: yfinance historical charts (no login required) ────────────
    st.markdown("#### Historical RSI — Yahoo Finance")
    st.caption("Live data from Yahoo Finance. No login required. Auto-cached 5 min.")

    _YF_SYMBOLS = {
        "Nifty 50":    "^NSEI",
        "Bank Nifty":  "^NSEBANK",
        "Sensex":      "^BSESN",
        "Nifty IT":    "^CNXIT",
        "Nifty Midcap":"^NSMIDCP",
    }
    _INTERVALS = {"Daily (1D)": "1d", "Hourly (1H)": "1h", "15 Min": "15m", "5 Min": "5m"}
    _BARS_MAP  = {"1d": 120, "1h": 96, "15m": 80, "5m": 60}

    hc1, hc2, hc3 = st.columns([2, 2, 1])
    with hc1:
        h_symbols = st.multiselect(
            "Index / Symbol",
            options=list(_YF_SYMBOLS.keys()),
            default=["Nifty 50", "Bank Nifty"],
            key="rsi_chart_symbols",
        )
    with hc2:
        h_interval_label = st.selectbox("Interval", options=list(_INTERVALS.keys()),
                                        index=0, key="rsi_chart_interval")
    with hc3:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Charts", key="rsi_chart_refresh"):
            # Clear cache so next render re-fetches
            _fetch_yf_rsi.clear()
            st.rerun()

    h_interval = _INTERVALS[h_interval_label]
    h_bars = _BARS_MAP[h_interval]

    for sym_label in h_symbols:
        yf_ticker = _YF_SYMBOLS[sym_label]
        _render_yf_chart(sym_label, yf_ticker, h_interval, h_bars)

    st.divider()

    # ── Section 2: Live Kotak tick-based charts (requires session) ────────────
    st.markdown("#### Live Tick Charts — Kotak Neo (requires session)")
    st.caption("Builds real-time 3-min and 5-min OHLC candles from Kotak quote ticks. Connect Session on Home tab first.")

    watch_cfg = current_config.setdefault("market_watch", {})
    default_fut_symbol = str(watch_cfg.get("nifty_fut_symbol", ""))

    session = st.session_state.get("ui_session")
    token_id = st.session_state.get("ui_token_id")

    c_load1, c_load2 = st.columns([1, 2])
    with c_load1:
        refresh_master = st.button("Load/Refresh Symbol Master", use_container_width=True)
    with c_load2:
        st.caption("Populates futures dropdown from Kotak master scrip after Connect Session.")

    if refresh_master:
        if not session or not token_id:
            st.warning("Connect Session first to load symbol dropdowns.")
        else:
            try:
                _load_master_symbols(session, token_id, "nse_fo")
                _load_master_symbols(session, token_id, "nse_cm")
                st.success("Symbol master loaded for nse_fo and nse_cm.")
            except Exception as exc:
                st.error(f"Failed to load symbol master: {exc}")

    watch_cfg["nifty_index_segment"] = "nse_cm"

    index_options = ["Nifty 50", "NIFTY 50", "NIFTY"]
    curr_idx_symbol = str(watch_cfg.get("nifty_index_symbol", "Nifty 50"))
    if curr_idx_symbol not in index_options:
        index_options = [curr_idx_symbol] + index_options
    idx_sel = st.selectbox(
        "Nifty 50 Symbol",
        options=index_options,
        index=index_options.index(curr_idx_symbol),
        key="mw_nifty_symbol_dropdown",
    )
    watch_cfg["nifty_index_symbol"] = idx_sel

    watch_cfg["nifty_fut_segment"] = st.selectbox("Nifty Fut Segment", options=["nse_fo"],
                                                   index=0, key="mw_fut_seg")

    fut_entries = []
    try:
        if session and token_id:
            fut_entries = _load_master_symbols(session, token_id, watch_cfg["nifty_fut_segment"])
    except Exception as exc:
        st.warning(f"Could not fetch futures dropdown from master: {exc}")

    filtered_fut = [
        row for row in fut_entries
        if "NIFTY" in row.get("norm", "") and ("FUT" in row.get("norm", "") or row.get("norm", "").startswith("NIFTY"))
    ]
    if not filtered_fut:
        filtered_fut = fut_entries[:200]

    if filtered_fut:
        fut_display = [row["display"] for row in filtered_fut]
        default_display = None
        for row in filtered_fut:
            if default_fut_symbol and (row.get("symbol") == default_fut_symbol or row.get("token") == default_fut_symbol):
                default_display = row["display"]
                break
        if not default_display:
            default_display = fut_display[0]
        selected_display = st.selectbox("Nifty Fut Contract (from Kotak master)", options=fut_display,
                                        index=fut_display.index(default_display), key="mw_fut_symbol_dropdown")
        selected_row = next((row for row in filtered_fut if row["display"] == selected_display), filtered_fut[0])
        watch_cfg["nifty_fut_symbol"] = selected_row.get("token") or selected_row.get("symbol")
        watch_cfg["nifty_fut_symbol_label"] = selected_row.get("symbol")
        st.caption(
            f"Nifty Fut: {watch_cfg.get('nifty_fut_symbol_label', '-')} | "
            f"instrument: {watch_cfg.get('nifty_fut_symbol', '-')}"
        )
    else:
        watch_cfg["nifty_fut_symbol"] = default_fut_symbol
        st.info("No futures dropdown loaded yet. Connect Session and click Load/Refresh Symbol Master.")

    r1, r2, r3 = st.columns([1, 1, 2])
    with r1:
        if st.button("Start RSI Watch", use_container_width=True):
            st.session_state["market_watch_active"] = True
            st.success("RSI watch started.")
    with r2:
        if st.button("Stop RSI Watch", use_container_width=True):
            st.session_state["market_watch_active"] = False
            st.info("RSI watch stopped.")
    with r3:
        if st.button("Clear RSI Data", use_container_width=True):
            st.session_state["ticks_nifty_index"] = []
            st.session_state["ticks_nifty_fut"] = []
            st.success("RSI tick buffers cleared.")
    if st.button("Save RSI Watch Settings", use_container_width=True):
        save_config(current_config)
        st.success("RSI watch settings saved.")

    active = bool(st.session_state.get("market_watch_active", False))

    if active:
        if not session or not token_id:
            st.warning("Connect Session first to fetch live quotes for tick charts.")
        else:
            idx_symbol = str(watch_cfg.get("nifty_index_symbol", "Nifty 50")).strip()
            idx_seg    = str(watch_cfg.get("nifty_index_segment", "nse_cm")).strip()
            fut_symbol = str(watch_cfg.get("nifty_fut_symbol", default_fut_symbol)).strip()
            fut_seg    = str(watch_cfg.get("nifty_fut_segment", "nse_fo")).strip()

            try:
                idx_price = fetch_ltp(base_url=session.base_url, token_id=token_id,
                                      exchange_segment=idx_seg, instruments=[idx_symbol])
                _append_tick("ticks_nifty_index", idx_price)
                st.session_state["latest_nifty_index"] = idx_price
                st.session_state.pop("market_watch_error_index", None)
            except Exception as exc:
                st.session_state["market_watch_error_index"] = str(exc)

            try:
                fut_price = fetch_ltp(base_url=session.base_url, token_id=token_id,
                                      exchange_segment=fut_seg, instruments=[fut_symbol])
                _append_tick("ticks_nifty_fut", fut_price)
                st.session_state["latest_nifty_fut"] = fut_price
                st.session_state.pop("market_watch_error_fut", None)
            except Exception as exc:
                st.session_state["market_watch_error_fut"] = str(exc)

    i_err = st.session_state.get("market_watch_error_index")
    f_err = st.session_state.get("market_watch_error_fut")
    if i_err:
        st.error(f"Nifty 50 quote error: {i_err}")
    if f_err:
        st.error(f"Nifty Fut quote error: {f_err}")

    k1, k2 = st.columns(2)
    with k1:
        if st.session_state.get("latest_nifty_index") is not None:
            st.metric("Nifty 50 LTP", f"{float(st.session_state['latest_nifty_index']):.2f}")
    with k2:
        if st.session_state.get("latest_nifty_fut") is not None:
            st.metric("Nifty Fut LTP", f"{float(st.session_state['latest_nifty_fut']):.2f}")

    idx_ticks = list(st.session_state.get("ticks_nifty_index", []))
    fut_ticks = list(st.session_state.get("ticks_nifty_fut", []))

    st.markdown("**Nifty 50 Live Tick Charts**")
    cidx1, cidx2 = st.columns(2)
    with cidx1:
        _render_chart_block("Nifty 50", idx_ticks, "3min")
    with cidx2:
        _render_chart_block("Nifty 50", idx_ticks, "5min")

    st.markdown("**Nifty Fut Live Tick Charts**")
    cfut1, cfut2 = st.columns(2)
    with cfut1:
        _render_chart_block("Nifty Fut", fut_ticks, "3min")
    with cfut2:
        _render_chart_block("Nifty Fut", fut_ticks, "5min")


def render_runner(current_config: dict) -> None:
    st.subheader("Run Bot")
    mode = st.radio("Execution Mode", options=["paper", "live"], horizontal=True, key="execution_mode")
    otp = st.text_input("TOTP (optional)", value=st.session_state.get("totp_input", ""), key="totp_input")

    process = st.session_state.get("bot_process")
    if process and process.poll() is None:
        started_at = st.session_state.get("bot_started_at", "unknown")
        st.info(f"Active runner: v2_bot.py | started at {started_at}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Bot"):
            if st.session_state.get("bot_process") and st.session_state["bot_process"].poll() is None:
                st.warning("Bot is already running.")
            else:
                fixed_cfg, fix_note = _ensure_valid_segment_symbol(current_config)
                current_config = fixed_cfg
                if fix_note:
                    st.warning(fix_note)
                save_config(current_config)
                run_bot(mode, otp)
                st.success("Bot started. Logs appear below.")

    with c2:
        if st.button("Stop Bot"):
            process = st.session_state.get("bot_process")
            if process and process.poll() is None:
                process.terminate()
                st.success("Stop signal sent.")
            else:
                st.info("No active bot process.")

    if st.button("Validate API Connection"):
        fixed_cfg, fix_note = _ensure_valid_segment_symbol(current_config)
        current_config = fixed_cfg
        if fix_note:
            st.warning(fix_note)
        save_config(current_config)
        run_validation(otp)

    st.markdown("**Manual Test Controls**")
    st.caption(
        "Use these controls for one-by-one testing. Connect once with a fresh TOTP, then reuse the same Neo v2 session for funds, quote, buy, sell, orders, and positions."
    )
    current_session = st.session_state.get("ui_session")
    if current_session:
        connected_symbol = st.session_state.get("ui_session_symbol", "")
        st.info(f"Connected Neo v2 session ready for manual actions. Symbol context: {connected_symbol or 'current config'}")
    else:
        st.warning("No active session for manual actions. Enter a fresh TOTP and click Connect Session.")

    if mode == "live":
        st.warning("Live mode will place a real exchange order when you click Buy or Sell.")
    confirm_live = st.checkbox(
        "I understand Buy/Sell in live mode will place a real order",
        value=bool(st.session_state.get("confirm_live_orders", False)),
        key="confirm_live_orders",
        help="Required before live Buy or Sell. You can tick this before or after switching to live mode.",
    )

    active_symbol = current_config.get("symbol", {}).get("trading_symbol", "")
    active_product = current_config.get("symbol", {}).get("product", "")
    active_order_type = current_config.get("order_type", "")
    active_quantity = current_config.get("symbol", {}).get("quantity", 1)
    st.caption(
        f"Active manual config: symbol={active_symbol} | product={active_product} | "
        f"order_type={active_order_type} | quantity={active_quantity}"
    )

    session_row = st.columns(3)
    with session_row[0]:
        if st.button("Connect Session", use_container_width=True):
            clear_manual_error()
            connect_ui_session(current_config, otp)
    with session_row[1]:
        if st.button("Disconnect Session", use_container_width=True):
            reset_ui_session()
            st.session_state["manual_action_output"] = "Manual API session cleared."
    with session_row[2]:
        st.caption("Reconnect with a new TOTP if the session expires.")

    action_row_1 = st.columns(3)
    with action_row_1[0]:
        if st.button("Refresh Funds", use_container_width=True):
            clear_manual_error()
            run_manual_action(current_config, otp, mode, "funds")
    with action_row_1[1]:
        if st.button("Fetch LTP", use_container_width=True):
            clear_manual_error()
            run_manual_action(current_config, otp, mode, "quote")
    with action_row_1[2]:
        if st.button("Refresh Orders", use_container_width=True):
            clear_manual_error()
            run_manual_action(current_config, otp, mode, "orders")

    action_row_2 = st.columns(3)
    with action_row_2[0]:
        if st.button("Refresh Positions", use_container_width=True):
            clear_manual_error()
            run_manual_action(current_config, otp, mode, "positions")
    with action_row_2[1]:
        if st.button("Buy 1x Config Qty", use_container_width=True):
            clear_manual_error()
            if mode == "live" and not confirm_live:
                st.session_state["manual_action_error"] = "Tick the live confirmation checkbox before placing a live Buy order."
            else:
                run_manual_action(current_config, otp, mode, "buy")
    with action_row_2[2]:
        if st.button("Sell 1x Config Qty", use_container_width=True):
            clear_manual_error()
            if mode == "live" and not confirm_live:
                st.session_state["manual_action_error"] = "Tick the live confirmation checkbox before placing a live Sell order."
            else:
                run_manual_action(current_config, otp, mode, "sell")

    limits = st.session_state.get("manual_limits")
    if isinstance(limits, dict):
        metrics = extract_fund_metrics(limits)
        metric_cols = st.columns(len(metrics))
        for index, (label, value) in enumerate(metrics.items()):
            metric_cols[index].metric(label, value)

    current_ltp = st.session_state.get("manual_ltp")
    if current_ltp is not None:
        quantity = int(current_config.get("symbol", {}).get("quantity", 1))
        estimated = float(current_ltp) * quantity
        st.caption(f"Current LTP: {current_ltp:.2f} | Config quantity: {quantity} | Estimated order value: {estimated:.2f}")

    action_output = st.session_state.get("manual_action_output")
    if action_output:
        st.success(action_output)

    action_error = st.session_state.get("manual_action_error")
    if action_error:
        st.error(action_error)

    quote_candidates = st.session_state.get("manual_quote_candidates")
    if quote_candidates:
        st.caption(f"Quote symbol candidates used: {quote_candidates}")

    manual_payload = st.session_state.get("manual_order_payload")
    manual_result = st.session_state.get("manual_order_result")
    if manual_payload or manual_result:
        st.markdown("**Order Ticket**")
        ticket = manual_payload or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Side", _side_label(str(ticket.get("tt", "B"))))
        c2.metric("Symbol", str(ticket.get("ts", active_symbol)))
        c3.metric("Quantity", str(ticket.get("qt", active_quantity)))
        c4.metric("Product", str(ticket.get("pc", active_product)))
        c5, c6, c7 = st.columns(3)
        c5.metric("Order Type", _order_type_label(str(ticket.get("pt", active_order_type))))
        c6.metric("Price", _fmt_money(ticket.get("pr", "0")))
        c7.metric("Trigger", _fmt_money(ticket.get("tp", "0")))

        if manual_result:
            status_text = ""
            if isinstance(manual_result, dict):
                status_text = str(
                    manual_result.get("stat")
                    or manual_result.get("stCode")
                    or manual_result.get("errMsg")
                    or ""
                )
            result_data = manual_result.get("data") if isinstance(manual_result, dict) else None
            if isinstance(result_data, list) and result_data:
                status_text = str(result_data[0].get("ordSt") or status_text)
            color, badge = _status_badge(status_text)
            st.markdown(
                f"<div style='padding:8px 12px;border:1px solid {color};border-radius:8px;'>"
                f"<strong>Order Status:</strong> <span style='color:{color};'>{badge}</span>"
                f" &nbsp; <span style='opacity:.85'>{status_text or '-'}</span></div>",
                unsafe_allow_html=True,
            )

            if st.toggle("Show raw order response", value=False, key="show_raw_order_result"):
                st.json(manual_result)

    positions = st.session_state.get("manual_positions")
    if positions:
        st.markdown("**Positions Snapshot**")
        st.json(positions)

    orders = st.session_state.get("manual_orders")
    if orders:
        executed_rows = _extract_executed_orders(orders)
        st.markdown("**Today's Executed Orders**")
        if executed_rows:
            st.dataframe(executed_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No executed orders found in today's order book response.")

        if st.toggle("Show raw order book response", value=False, key="show_raw_orders"):
            st.json(orders)

    lc1, lc2 = st.columns([1, 2])
    with lc1:
        if st.button("Clear Bot Logs"):
            BOT_LOG_PATH.write_text("", encoding="utf-8")
            st.session_state["log_file_offset"] = 0
            st.session_state["log_lines"] = []
            st.success("Bot logs cleared.")
    with lc2:
        compact_logs = st.toggle(
            "Compact logs (hide repetitive LTP ticks)",
            value=True,
            key="compact_logs_toggle",
        )

    log_lines = _read_log_incremental(BOT_LOG_PATH)
    visible_lines = _filter_log_lines(log_lines, compact=compact_logs)
    if visible_lines:
        if any("auth_test.py" in line for line in visible_lines):
            st.warning(
                "Legacy auth_test.py traceback detected in logs. "
                "Use 'Clear Bot Logs' and start again; current runner is v2_bot.py."
            )
        st.markdown("**Bot Logs**")
        escaped = "\n".join(visible_lines).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        log_html = f"""
<div id="logbox" style="background:#0e1117;color:#e0e0e0;font-family:monospace;
font-size:13px;padding:12px;border-radius:6px;border:1px solid #333;
height:320px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;">{escaped}</div>
<script>
  var el=document.getElementById('logbox');
  el.scrollTop=el.scrollHeight;
</script>
"""
        st_components.html(log_html, height=340, scrolling=False)

    # Auto-refresh while bot is running or market watch is active.
    active = st.session_state.get("bot_process")
    watch_active = bool(st.session_state.get("market_watch_active", False))
    if (active and active.poll() is None) or watch_active:
        time.sleep(2)
        st.rerun()


def render_screener_tab(current_config: dict) -> None:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")

    # ── Column config for 2-decimal display ──────────────────────────────────
    _NUM2 = {"format": "%.2f"}
    _IDX_COL_CFG = {
        "Close":     st.column_config.NumberColumn(**_NUM2),
        "52W High":  st.column_config.NumberColumn(**_NUM2),
        "52W Low":   st.column_config.NumberColumn(**_NUM2),
        "RSI(14)":   st.column_config.NumberColumn(**_NUM2),
        "EMA 20":    st.column_config.NumberColumn(**_NUM2),
        "EMA 50":    st.column_config.NumberColumn(**_NUM2),
        "SMA 200":   st.column_config.NumberColumn(**_NUM2),
        "VWAP":      st.column_config.NumberColumn(**_NUM2),
        "Avg Vol 20":st.column_config.NumberColumn(**_NUM2),
    }

    # ── Auto-refresh bot ─────────────────────────────────────────────────────
    now_ist = datetime.now(tz=IST)
    mkt_open  = now_ist.replace(hour=9,  minute=10, second=0, microsecond=0)
    mkt_close = now_ist.replace(hour=15, minute=35, second=0, microsecond=0)
    in_market = (mkt_open <= now_ist <= mkt_close) and (now_ist.weekday() < 5)

    bot_running = st.session_state.get("screener_bot_running", False)
    b1, b2, b3 = st.columns([2, 1, 2])
    with b1:
        status_icon = "🟢" if bot_running else "🔴"
        mkt_label   = "🔔 Market Open" if in_market else "💤 Market Closed"
        st.caption(f"Auto-refresh Bot: {status_icon} {'Running' if bot_running else 'Stopped'}  |  {mkt_label}  |  {now_ist.strftime('%H:%M IST')}")
    with b2:
        if bot_running:
            if st.button("⏹ Stop Bot", key="bot_stop"):
                st.session_state["screener_bot_running"] = False
                st.rerun()
        else:
            if st.button("▶ Start Bot", key="bot_start", type="primary"):
                st.session_state["screener_bot_running"] = True
                st.rerun()

    if bot_running:
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore
            st_autorefresh(interval=5 * 60 * 1000, key="screener_autorefresh")
        except ImportError:
            st.warning("streamlit-autorefresh not installed. Run: pip install streamlit-autorefresh")

    st.divider()

    # ── Index Overview ────────────────────────────────────────────────────────
    st.subheader("Index Overview")
    st.caption("Live RSI/MA/VWAP snapshot for major indices via Yahoo Finance.")

    # Only fetch on first load or when bot autorefresh fires (detected by
    # the autorefresh counter incrementing, not on every widget interaction)
    _autorefresh_count = st.session_state.get("screener_autorefresh", 0)
    _last_fetched_at   = st.session_state.get("_idx_fetched_at_count", -1)
    _first_load        = "index_snapshot_df" not in st.session_state
    _autorefresh_fired = bot_running and (_autorefresh_count != _last_fetched_at)

    if _first_load or _autorefresh_fired:
        with st.spinner("Loading index snapshot…"):
            try:
                idx_df, idx_errors = fetch_index_snapshot(lookback_bars=252)
                st.session_state["index_snapshot_df"] = idx_df
                st.session_state["index_snapshot_errors"] = idx_errors
                st.session_state["_idx_fetched_at_count"] = _autorefresh_count
            except Exception as exc:
                st.session_state["index_snapshot_df"] = None
                st.session_state["index_snapshot_errors"] = [str(exc)]

    if st.button("🔄 Refresh Indices", key="idx_refresh_btn"):
        with st.spinner("Refreshing index snapshot…"):
            try:
                idx_df, idx_errors = fetch_index_snapshot(lookback_bars=252)
                st.session_state["index_snapshot_df"] = idx_df
                st.session_state["index_snapshot_errors"] = idx_errors
                st.session_state["_idx_fetched_at_count"] = _autorefresh_count
            except Exception as exc:
                st.session_state["index_snapshot_df"] = None
                st.session_state["index_snapshot_errors"] = [str(exc)]

    idx_df = st.session_state.get("index_snapshot_df")
    if idx_df is not None and not idx_df.empty:
        display_cols = [c for c in SCREENER_DISPLAY_COLS if c in idx_df.columns]
        st.dataframe(
            style_results(idx_df),
            column_order=display_cols,
            column_config=_IDX_COL_CFG,
            use_container_width=True,
            hide_index=True,
        )
    elif idx_df is not None:
        st.info("No index data available.")
    for err in st.session_state.get("index_snapshot_errors", []):
        st.caption(f"⚠ {err}")

    # ── Index Futures (from latest NSE CSV) ──────────────────────────────────
    st.markdown("**Current Month Futures**")
    if st.button("Load Futures", key="futures_load_btn"):
        with st.spinner("Loading futures from NSE CSV…"):
            st.session_state["index_futures_df"] = load_nse_futures()
    if "index_futures_df" not in st.session_state:
        # Load once silently — it's a fast local CSV read
        st.session_state["index_futures_df"] = load_nse_futures()

    fut_df = st.session_state.get("index_futures_df")
    if fut_df is not None and not fut_df.empty:
        st.dataframe(
            fut_df.style.format({"LTP": "{:.2f}", "Chng%": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No futures CSV found in nse data/ folder.")

    st.divider()

    # ── Stock Screener ────────────────────────────────────────────────────────
    st.subheader("Stock Screener")
    st.caption("Screens stocks using Yahoo Finance data (no login required). Indicators: RSI(14), EMA(20/50), SMA(200), VWAP.")

    sc1, sc2 = st.columns([2, 2])
    with sc1:
        index_choices = list(NSE_INDEX_CSV_MAP.keys())
        selected_index = st.selectbox(
            "Load symbols from index",
            options=["Custom"] + index_choices,
            index=1,
            key="screener_index_select",
        )
    with sc2:
        interval = st.selectbox("Interval", options=["1D", "1H", "15M", "5M"], index=0, key="screener_interval")
        lookback_bars = st.number_input("Lookback Bars", min_value=50, max_value=500, value=250, step=10, key="screener_lookback")

    # Auto-populate symbols from selected index CSV
    if selected_index != "Custom":
        csv_syms = load_nse_symbols(selected_index)
        default_syms = csv_syms if csv_syms else SCREENER_NIFTY50[:10]
        options_pool = csv_syms if csv_syms else SCREENER_NIFTY50
    else:
        options_pool = SCREENER_NIFTY50
        default_syms = SCREENER_NIFTY50[:10]

    selected_symbols = st.multiselect(
        "Symbols to Screen",
        options=options_pool,
        default=default_syms,
        key="screener_symbols",
    )

    with st.expander("Filters", expanded=False):
        st.markdown("**RSI**")
        r1, r2 = st.columns(2)
        with r1:
            oversold_only    = st.checkbox("RSI < 30 (Oversold)",    key="screener_oversold")
        with r2:
            overbought_only  = st.checkbox("RSI > 70 (Overbought)",  key="screener_overbought")

        st.markdown("**Price vs Moving Averages**")
        m1, m2, m3 = st.columns(3)
        with m1:
            above_ma20 = st.checkbox("Above MA20", key="screener_above_ma20")
            below_ma20 = st.checkbox("Below MA20", key="screener_below_ma20")
        with m2:
            above_ma50 = st.checkbox("Above MA50", key="screener_above_ma50")
            below_ma50 = st.checkbox("Below MA50", key="screener_below_ma50")
        with m3:
            above_ma200 = st.checkbox("Above MA200", key="screener_above_ma200")
            below_ma200 = st.checkbox("Below MA200", key="screener_below_ma200")

        st.markdown("**VWAP & Volume**")
        v1, v2, v3 = st.columns(3)
        with v1:
            above_vwap_only = st.checkbox("Above VWAP", key="screener_vwap_above")
        with v2:
            below_vwap_only = st.checkbox("Below VWAP", key="screener_vwap_below")
        with v3:
            volume_spikes   = st.checkbox("Volume Spike (2× avg)", key="screener_volspike")

        st.markdown("**Crossovers (within last 5 bars)**")
        c1, c2, c3 = st.columns(3)
        with c1:
            ema20_bull   = st.checkbox("EMA20 Bull ▲ EMA50",          key="screener_ema20cross")
        with c2:
            golden_cross = st.checkbox("Golden Cross EMA50▲SMA200",   key="screener_golden")
        with c3:
            ma_cross_any = st.checkbox("Any MA Crossover",             key="screener_macross")

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_clicked = st.button("▶ Run Screener", type="primary", key="screener_run")

    if run_clicked or (bot_running and in_market and "screener_result_df" not in st.session_state):
        if not selected_symbols:
            st.warning("Select at least one symbol.")
        else:
            with st.spinner(f"Screening {len(selected_symbols)} symbols…"):
                try:
                    result_df, errors = run_screener(
                        symbols=selected_symbols,
                        interval_label=str(interval),
                        lookback_bars=int(lookback_bars),
                    )
                    st.session_state["screener_result_df"] = result_df
                    st.session_state["screener_errors"] = errors
                except Exception as exc:
                    st.error(f"Screener failed: {exc}")
                    st.session_state.pop("screener_result_df", None)

    result_df = st.session_state.get("screener_result_df")
    errors    = st.session_state.get("screener_errors", [])

    if result_df is not None and not result_df.empty:
        filtered = apply_filters(
            result_df,
            oversold_only=oversold_only,
            overbought_only=overbought_only,
            volume_spikes_only=volume_spikes,
            ma_cross_only=ma_cross_any,
            above_vwap_only=above_vwap_only,
            below_vwap_only=below_vwap_only,
            above_ma20=above_ma20,    below_ma20=below_ma20,
            above_ma50=above_ma50,    below_ma50=below_ma50,
            above_ma200=above_ma200,  below_ma200=below_ma200,
            ema20_bull_cross=ema20_bull,
            golden_cross=golden_cross,
        )
        st.caption(f"Showing {len(filtered)} of {len(result_df)} screened symbols")
        if filtered.empty:
            st.info("No symbols match the active filters.")
        else:
            display_cols = [c for c in SCREENER_DISPLAY_COLS if c in filtered.columns]
            st.dataframe(
                style_results(filtered),
                column_order=display_cols,
                column_config=_IDX_COL_CFG,
                use_container_width=True,
                hide_index=True,
            )
    elif result_df is not None:
        st.info("Screener ran but returned no results.")

    if errors:
        with st.expander(f"Fetch errors ({len(errors)})", expanded=False):
            for err in errors:
                st.text(err)

    # ── Events & Dividends ────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Dividends by Year")
    st.caption("Dividend history per symbol — 2024 · 2025 · 2026. Each cell shows DD-Mon ₹Amount. Upcoming ex-div dates are marked ↑.")

    events_symbols = list(selected_symbols) if selected_symbols else []
    if st.button("Fetch Dividends", key="screener_events_btn"):
        if not events_symbols:
            st.warning("Select symbols above first.")
        else:
            with st.spinner(f"Fetching dividends for {len(events_symbols)} symbols…"):
                div_df = fetch_dividend_pivot(events_symbols)
                st.session_state["screener_events"] = div_df

    div_data = st.session_state.get("screener_events")
    if div_data is not None and not div_data.empty:
        # Highlight the Next Ex-Div column if upcoming
        def _exdiv_style(val: str) -> str:
            return "color: #7fff7f; font-weight: bold;" if str(val).startswith("↑") else ""

        year_cols = [c for c in div_data.columns if c in ("2024", "2025", "2026")]
        styled_div = div_data.style
        if "Next Ex-Div" in div_data.columns:
            styled_div = styled_div.map(_exdiv_style, subset=["Next Ex-Div"])
        st.dataframe(
            styled_div,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol":      st.column_config.TextColumn("Symbol", width="small"),
                "2024":        st.column_config.TextColumn("2024", width="medium"),
                "2025":        st.column_config.TextColumn("2025", width="medium"),
                "2026":        st.column_config.TextColumn("2026", width="medium"),
                "Next Ex-Div": st.column_config.TextColumn("Next Ex-Div", width="medium"),
            },
        )
    elif div_data is not None:
        st.info("No dividend data found for selected symbols.")



if __name__ == "__main__":
    render_intro()
    current_config = load_config()
    tab_home, tab_charts, tab_derivatives, tab_screener = st.tabs(
        ["Home", "RSI Charts", "Derivatives", "Screener"]
    )

    with tab_home:
        current_config = render_config_form(current_config)
        render_runner(current_config)

    with tab_charts:
        render_market_watch(current_config)

    with tab_derivatives:
        mode = str(st.session_state.get("execution_mode", "paper"))
        otp = str(st.session_state.get("totp_input", ""))
        render_derivatives_panel(current_config, mode, otp)

    with tab_screener:
        render_screener_tab(current_config)

