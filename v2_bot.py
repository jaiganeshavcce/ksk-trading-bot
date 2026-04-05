import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import csv
import io
import requests

from strategy_logic import build_strategy_state, evaluate_signal


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
LOGIN_URL = "https://mis.kotaksecurities.com/login/1.0/tradeApiLogin"
VALIDATE_URL = "https://mis.kotaksecurities.com/login/1.0/tradeApiValidate"


@dataclass
class V2Session:
    trade_token: str
    trade_sid: str
    base_url: str


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def get_token_id(cfg: dict) -> str:
    token = cfg.get("token_id") or cfg.get("access_token") or cfg.get("consumer_key")
    if not token:
        raise RuntimeError("Missing token_id in config.json")
    return str(token)


def _headers_for_login(token_id: str) -> dict:
    return {
        "Authorization": token_id,
        "neo-fin-key": "neotradeapi",
        "Content-Type": "application/json",
    }


def _trade_headers(session: V2Session, *, content_type: str | None = None, accept_json: bool = True) -> dict:
    headers = {
        "Auth": session.trade_token,
        "Sid": session.trade_sid,
        "neo-fin-key": "neotradeapi",
    }
    if accept_json:
        headers["accept"] = "application/json"
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def login_with_totp(cfg: dict, token_id: str, totp: str) -> tuple[str, str]:
    payload = {
        "mobileNumber": cfg["mobile_number"],
        "ucc": cfg["ucc"],
        "totp": totp,
    }
    response = requests.post(LOGIN_URL, headers=_headers_for_login(token_id), json=payload, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"tradeApiLogin failed ({response.status_code}): {response.text}")

    data = response.json().get("data") or {}
    view_token = data.get("token")
    view_sid = data.get("sid")
    if not view_token or not view_sid:
        raise RuntimeError(f"Login failed: {response.text}")
    return view_token, view_sid


def validate_with_mpin(cfg: dict, token_id: str, view_token: str, view_sid: str) -> V2Session:
    headers = {
        "Authorization": token_id,
        "neo-fin-key": "neotradeapi",
        "sid": view_sid,
        "Auth": view_token,
        "Content-Type": "application/json",
    }
    payload = {"mpin": cfg["mpin"]}

    response = requests.post(VALIDATE_URL, headers=headers, json=payload, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"tradeApiValidate failed ({response.status_code}): {response.text}")

    data = response.json().get("data") or {}
    trade_token = data.get("token")
    trade_sid = data.get("sid")
    base_url = data.get("baseUrl")
    if not trade_token or not trade_sid or not base_url:
        raise RuntimeError(f"MPIN validate failed: {response.text}")
    return V2Session(trade_token=trade_token, trade_sid=trade_sid, base_url=base_url)


def authenticate(cfg: dict, token_id: str, otp: str | None) -> V2Session:
    totp = (otp or input("Enter current 6-digit TOTP: ")).strip()
    if not totp:
        raise RuntimeError("TOTP is required")

    print("Step 1/2: Login with TOTP...", flush=True)
    view_token, view_sid = login_with_totp(cfg, token_id, totp)
    print("Step 2/2: Validate with MPIN...", flush=True)
    return validate_with_mpin(cfg, token_id, view_token, view_sid)


def _parse_ltp_from_quote_response(response_json: object) -> float | None:
    if isinstance(response_json, list) and response_json:
        ltp_raw = response_json[0].get("ltp")
        if ltp_raw is not None:
            return float(ltp_raw)

    if isinstance(response_json, dict):
        data = response_json.get("data")
        if isinstance(data, list) and data:
            ltp_raw = data[0].get("ltp")
            if ltp_raw is not None:
                return float(ltp_raw)
    return None


def _extract_scrip_file_urls(payload: object) -> list[str]:
    """Extract CSV URLs from multiple possible scrip-master response shapes."""
    urls: list[str] = []

    def _collect_from_obj(obj: object) -> None:
        if isinstance(obj, str):
            if obj.lower().endswith(".csv"):
                urls.append(obj)
            return
        if isinstance(obj, list):
            for item in obj:
                _collect_from_obj(item)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in {"filespaths", "filepaths", "paths", "files"}:
                    _collect_from_obj(value)
                elif isinstance(value, str) and value.lower().endswith(".csv"):
                    urls.append(value)
                elif isinstance(value, (list, dict)):
                    _collect_from_obj(value)

    _collect_from_obj(payload)

    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def _normalize_symbol(value: str) -> str:
    return "".join(value.upper().split())


def _normalize_token(value: str) -> str:
    token = str(value).strip()
    if token.endswith(".0"):
        token = token[:-2]
    return token


def fetch_ltp(base_url: str, token_id: str, exchange_segment: str, instruments: list[str]) -> float:
    headers = {
        "Authorization": token_id,
        "Content-Type": "application/json",
    }

    errors: list[str] = []
    for instrument in instruments:
        endpoint = f"{base_url}/script-details/1.0/quotes/neosymbol/{exchange_segment}|{instrument}/ltp"
        response = requests.get(endpoint, headers=headers, timeout=20)
        if response.status_code >= 400:
            errors.append(f"{instrument}: HTTP {response.status_code} {response.text}")
            continue

        response_json = response.json()
        ltp = _parse_ltp_from_quote_response(response_json)
        if ltp is not None:
            return ltp

        errors.append(f"{instrument}: Unexpected payload {response.text}")

    raise RuntimeError("Quote lookup failed for all symbol candidates. " + " | ".join(errors))


def build_order_payload(cfg: dict, side: str, ltp: float) -> dict:
    symbol_cfg = cfg["symbol"]
    order_type = str(cfg.get("order_type", "L")).upper()
    offset_pct = float(cfg.get("limit_price_offset_pct", 0.1)) / 100.0
    trigger_offset_pct = float(cfg.get("trigger_price_offset_pct", 0.1)) / 100.0

    if side == "B":
        trigger_price = ltp * (1.0 + trigger_offset_pct)
        limit_price = ltp * (1.0 + offset_pct)
    else:
        trigger_price = ltp * (1.0 - trigger_offset_pct)
        limit_price = ltp * (1.0 - offset_pct)

    if order_type == "L":
        price_str = f"{limit_price:.2f}"
        trigger_price_str = "0"
    elif order_type == "MKT":
        price_str = "0"
        trigger_price_str = "0"
    elif order_type == "SL":
        price_str = f"{limit_price:.2f}"
        trigger_price_str = f"{trigger_price:.2f}"
    elif order_type == "SL-M":
        price_str = "0"
        trigger_price_str = f"{trigger_price:.2f}"
    else:
        raise RuntimeError("order_type must be one of 'L', 'MKT', 'SL', or 'SL-M'")

    return {
        "am": "NO",
        "dq": "0",
        "es": str(symbol_cfg["exchange_segment"]),
        "mp": "0",
        "pc": str(symbol_cfg.get("product", "MIS")),
        "pf": "N",
        "pr": price_str,
        "pt": order_type,
        "qt": str(symbol_cfg["quantity"]),
        "rt": "DAY",
        "tp": trigger_price_str,
        "ts": str(symbol_cfg["trading_symbol"]),
        "tt": side,
    }


def place_order(session: V2Session, payload: dict) -> dict:
    endpoint = f"{session.base_url}/quick/order/rule/ms/place"
    headers = _trade_headers(session, content_type="application/x-www-form-urlencoded")
    response = requests.post(
        endpoint,
        headers=headers,
        data={"jData": json.dumps(payload, separators=(",", ":"))},
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Place order failed ({response.status_code}): {response.text}")
    return response.json()


def check_orders(session: V2Session) -> dict:
    headers = _trade_headers(session)
    endpoint = f"{session.base_url}/quick/user/orders"
    response = requests.get(endpoint, headers=headers, timeout=20)
    if response.status_code >= 400:
        raise RuntimeError(f"Orders check failed ({response.status_code}): {response.text}")
    return response.json()


def check_positions(session: V2Session) -> dict:
    headers = _trade_headers(session, content_type="application/x-www-form-urlencoded")
    endpoint = f"{session.base_url}/quick/user/positions"
    response = requests.get(endpoint, headers=headers, timeout=20)
    if response.status_code >= 400:
        raise RuntimeError(f"Positions check failed ({response.status_code}): {response.text}")
    return response.json()


def check_limits(session: V2Session, seg: str = "ALL", exch: str = "ALL", prod: str = "ALL") -> dict:
    headers = _trade_headers(session, content_type="application/x-www-form-urlencoded")
    endpoint = f"{session.base_url}/quick/user/limits"
    payload = {"seg": seg, "exch": exch, "prod": prod}
    response = requests.post(
        endpoint,
        headers=headers,
        data={"jData": json.dumps(payload, separators=(",", ":"))},
        timeout=20,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Limits check failed ({response.status_code}): {response.text}")
    return response.json()


def resolve_quote_candidates(symbol_cfg: dict, session: V2Session, token_id: str) -> list[str]:
    quote_candidates: list[str] = []

    if symbol_cfg.get("quote_instrument"):
        quote_candidates.append(str(symbol_cfg["quote_instrument"]).strip())

    if symbol_cfg.get("instrument_token"):
        quote_candidates.append(str(symbol_cfg["instrument_token"]).strip())
    else:
        print(
            f"No instrument_token set. Resolving '{symbol_cfg['trading_symbol']}' "
            f"from scrip master (segment: {symbol_cfg['exchange_segment']})...",
            flush=True,
        )
        resolved = fetch_scrip_token_from_master(
            session=session,
            token_id=token_id,
            exchange_segment=str(symbol_cfg["exchange_segment"]),
            trading_symbol=str(symbol_cfg["trading_symbol"]),
        )
        if resolved:
            print(f"Scrip master resolved instrument token: {resolved}", flush=True)
            quote_candidates.append(resolved)
        else:
            print("Could not resolve from scrip master. Falling back to trading symbol.", flush=True)

    if symbol_cfg.get("trading_symbol"):
        quote_candidates.append(str(symbol_cfg["trading_symbol"]).strip())

    seen: set[str] = set()
    instruments_for_quotes: list[str] = []
    for candidate in quote_candidates:
        if candidate and candidate not in seen:
            instruments_for_quotes.append(candidate)
            seen.add(candidate)

    return instruments_for_quotes


_EQUITY_LIKE_SEGMENTS = {"nse_cm", "bse_cm"}
_FO_LIKE_SEGMENTS = {"nse_fo", "bse_fo", "mcx_fo", "cde_fo"}
_EQUITY_SUFFIXES = ("-EQ", "-BE", "-BL", "-SM", "-GR")


def _validate_symbol_segment(symbol_cfg: dict) -> None:
    """Warn early when segment/symbol combination looks obviously wrong."""
    seg = str(symbol_cfg.get("exchange_segment", "")).lower().strip()
    sym = str(symbol_cfg.get("trading_symbol", "")).upper().strip()
    if not seg or not sym:
        return
    is_eq_sym = any(sym.endswith(suf) for suf in _EQUITY_SUFFIXES)
    if seg in _FO_LIKE_SEGMENTS and is_eq_sym:
        raise RuntimeError(
            f"Symbol '{sym}' looks like an equity symbol but exchange_segment is '{seg}' (F&O). "
            "For equities use nse_cm or bse_cm. "
            "For F&O symbols set the segment to nse_fo and pick the exact contract token from "
            "'Load/Refresh Symbol Master' in the UI RSI Charts section."
        )


def run_loop(cfg: dict, session: V2Session, token_id: str, mode: str) -> None:
    symbol_cfg = cfg["symbol"]
    _validate_symbol_segment(symbol_cfg)
    instruments_for_quotes = resolve_quote_candidates(symbol_cfg, session=session, token_id=token_id)

    poll_interval = max(1, int(cfg.get("poll_interval_seconds", 2)))
    decision_interval = max(1, int(cfg.get("decision_interval_seconds", 5)))
    cooldown_seconds = max(1, int(cfg.get("cooldown_seconds", 30)))

    state = build_strategy_state(cfg)
    window = max(2, int(state["window"]))
    prices: deque[float] = deque(maxlen=window)

    last_decision_at = 0.0
    last_trade_at = 0.0

    print(f"Running Neo v2 bot in {mode} mode.", flush=True)
    print(
        f"Symbol={symbol_cfg['trading_symbol']} Poll={poll_interval}s Decision={decision_interval}s "
        f"Cooldown={cooldown_seconds}s OrderType={cfg.get('order_type', 'L')}",
        flush=True,
    )
    print(f"Quote symbol candidates={instruments_for_quotes}", flush=True)

    while True:
        ltp = fetch_ltp(
            base_url=session.base_url,
            token_id=token_id,
            exchange_segment=symbol_cfg["exchange_segment"],
            instruments=instruments_for_quotes,
        )
        prices.append(ltp)
        print(f"{datetime.now().strftime('%H:%M:%S')} | LTP={ltp:.2f}", flush=True)

        now = time.time()
        if now - last_decision_at >= decision_interval:
            signal_data = evaluate_signal(prices, state)
            signal = signal_data.get("signal", "HOLD")
            sma = float(signal_data.get("sma", 0.0))
            print(f"{datetime.now().strftime('%H:%M:%S')} | Signal={signal} SMA={sma:.2f}", flush=True)
            last_decision_at = now

            if signal in ("BUY", "SELL"):
                if now - last_trade_at < cooldown_seconds:
                    print("Cooldown active. Trade skipped.", flush=True)
                else:
                    side = "B" if signal == "BUY" else "S"
                    payload = build_order_payload(cfg, side=side, ltp=ltp)
                    if mode == "paper":
                        print(f"[PAPER] Order payload: {payload}", flush=True)
                    else:
                        response = place_order(session, payload)
                        print(f"[LIVE] Order response: {response}", flush=True)
                    last_trade_at = now

        time.sleep(poll_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kotak Neo v2 interval strategy runner")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to config json")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--otp", default=None, help="Current 6-digit TOTP")
    parser.add_argument("--validate-only", action="store_true", help="Auth + orderbook check, then exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config))

    required = ["mobile_number", "ucc", "mpin", "symbol"]
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise RuntimeError(f"Missing required config keys: {', '.join(missing)}")

    symbol_required = ["exchange_segment", "trading_symbol", "quantity", "product"]
    symbol_missing = [key for key in symbol_required if not cfg["symbol"].get(key)]
    if symbol_missing:
        raise RuntimeError(f"Missing required symbol keys: {', '.join(symbol_missing)}")

    token_id = get_token_id(cfg)
    session = authenticate(cfg, token_id=token_id, otp=args.otp)

    if args.validate_only:
        orders = check_orders(session)
        print("Validation successful.")
        print(f"BASE_URL: {session.base_url}")
        print(f"Orders API keys: {list(orders.keys())}")
        return 0

    run_loop(cfg, session=session, token_id=token_id, mode=args.mode)
    return 0


def fetch_scrip_token_from_master(
    session: V2Session, token_id: str, exchange_segment: str, trading_symbol: str
) -> str | None:
    """Download Kotak scrip master CSV and resolve numeric neosymbol token for a trading symbol."""
    try:
        paths_url = f"{session.base_url}/script-details/1.0/masterscrip/file-paths"
        r = requests.get(paths_url, headers={"Authorization": token_id}, timeout=30)
        if r.status_code >= 400:
            print(f"Scrip master file-paths failed ({r.status_code}): {r.text}", flush=True)
            return None

        data = r.json()
        entries = _extract_scrip_file_urls(data)
        seg_lower = exchange_segment.lower()
        file_url: str | None = None
        for entry in entries:
            if isinstance(entry, str) and seg_lower in entry.lower():
                file_url = entry
                break
            elif isinstance(entry, dict):
                seg_val = str(
                    entry.get("seg", "") or entry.get("exchangeSegment", "") or entry.get("segment", "")
                ).lower()
                url_val = entry.get("filePath") or entry.get("fileUrl") or entry.get("url")
                if seg_lower in seg_val and url_val:
                    file_url = url_val
                    break

        if not file_url:
            print(f"No scrip master file found for segment '{exchange_segment}'.", flush=True)
            return None

        print(f"Using scrip master file: {file_url}", flush=True)

        r2 = requests.get(file_url, timeout=120)
        if r2.status_code >= 400:
            print(f"Scrip master download failed ({r2.status_code}).", flush=True)
            return None

        content = r2.content.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(content))
        ts_norm = _normalize_symbol(trading_symbol)
        for row in reader:
            raw_sym = (
                row.get("pTrdSymbol")
                or row.get("TradingSymbol")
                or row.get("trdSym")
                or row.get("symbol")
                or ""
            )
            sym_norm = _normalize_symbol(raw_sym)
            if sym_norm == ts_norm:
                token = (
                    row.get("pSymbol")
                    or row.get("pInstrumentToken")
                    or row.get("Token")
                    or row.get("token")
                )
                if token:
                    return _normalize_token(str(token))

        print(f"Symbol '{trading_symbol}' not found in scrip master.", flush=True)
    except Exception as exc:
        print(f"Scrip master lookup error: {exc}", flush=True)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
