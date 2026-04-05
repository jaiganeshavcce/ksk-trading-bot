import argparse
import json
from pathlib import Path

import requests


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
LOGIN_URL = "https://mis.kotaksecurities.com/login/1.0/tradeApiLogin"
VALIDATE_URL = "https://mis.kotaksecurities.com/login/1.0/tradeApiValidate"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def get_token_id(cfg: dict) -> str:
    token = cfg.get("token_id") or cfg.get("access_token") or cfg.get("consumer_key")
    if not token:
        raise RuntimeError("Missing token_id in config.json")
    return str(token)


def login_with_totp(cfg: dict, token_id: str, totp: str) -> tuple[str, str]:
    headers = {
        "Authorization": token_id,
        "neo-fin-key": "neotradeapi",
        "Content-Type": "application/json",
    }
    payload = {
        "mobileNumber": cfg["mobile_number"],
        "ucc": cfg["ucc"],
        "totp": totp,
    }

    response = requests.post(LOGIN_URL, headers=headers, json=payload, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"tradeApiLogin failed ({response.status_code}): {response.text}")

    data = response.json().get("data") or {}
    view_token = data.get("token")
    view_sid = data.get("sid")
    if not view_token or not view_sid:
        raise RuntimeError(f"Login failed: {response.text}")
    return view_token, view_sid


def validate_with_mpin(cfg: dict, token_id: str, view_token: str, view_sid: str) -> tuple[str, str, str]:
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
    return trade_token, trade_sid, base_url


def check_orders(token_id: str, trade_token: str, trade_sid: str, base_url: str) -> dict:
    headers = {
        "Authorization": token_id,
        "neo-fin-key": "neotradeapi",
        "Auth": trade_token,
        "Sid": trade_sid,
    }
    url = f"{base_url}/quick/user/orders"
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"orders check failed ({response.status_code}): {response.text}")
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Kotak Neo v2 auth validator (HTTP endpoints)")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--totp", default=None, help="6-digit TOTP from authenticator")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    required = ["mobile_number", "ucc", "mpin"]
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise RuntimeError(f"Missing required config keys: {', '.join(missing)}")

    token_id = get_token_id(cfg)
    totp = (args.totp or input("Enter current 6-digit TOTP: ")).strip()
    if not totp:
        raise RuntimeError("TOTP is required")

    print("Step 1/3: Login with TOTP...")
    view_token, view_sid = login_with_totp(cfg, token_id, totp)
    print("Step 2/3: Validate with MPIN...")
    trade_token, trade_sid, base_url = validate_with_mpin(cfg, token_id, view_token, view_sid)
    print("Step 3/3: Read-only API check (orders)...")
    orders_response = check_orders(token_id, trade_token, trade_sid, base_url)

    print("\nValidation successful.")
    print(f"BASE_URL: {base_url}")
    print(f"Orders API status keys: {list(orders_response.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
