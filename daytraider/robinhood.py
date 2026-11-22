"""Robinhood Crypto Trading API client (Ed25519 key-pair authentication).

Implements the auth scheme from https://docs.robinhood.com/crypto/trading/:
every request is signed with your Ed25519 private key over the message
    api_key + timestamp + path + method + body
and sent with x-api-key / x-signature / x-timestamp headers.

!! REAL MONEY !! Robinhood has no paper-trading mode. Nothing in this
project places Robinhood orders automatically; this module gives you an
authenticated client and a small CLI to use deliberately.

Setup:
  1. python robinhood.py keygen
       -> prints a base64 PUBLIC key and writes the PRIVATE key to .env
  2. Go to robinhood.com -> Account -> Crypto -> API trading, create a
     credential, paste in the PUBLIC key, and copy the API key it gives you
     into .env as ROBINHOOD_API_KEY.
  3. Verify:  python robinhood.py account
              python robinhood.py price BTC-USD
              python robinhood.py holdings

CLI:
  python robinhood.py keygen
  python robinhood.py account
  python robinhood.py holdings
  python robinhood.py price <SYMBOL>              e.g. BTC-USD
  python robinhood.py buy <SYMBOL> <USD_AMOUNT>   (asks for confirmation)
  python robinhood.py sell <SYMBOL> <USD_AMOUNT>  (asks for confirmation)
  python robinhood.py orders
"""
from __future__ import annotations

import base64
import json
import sys
import time
import uuid

import requests
from nacl.signing import SigningKey

import config

BASE_URL = "https://trading.robinhood.com"


# ---------------------------------------------------------------- keygen --

def keygen() -> None:
    """Generate an Ed25519 key pair; store private key, show public key."""
    signing_key = SigningKey.generate()
    private_b64 = base64.b64encode(bytes(signing_key)).decode()
    public_b64 = base64.b64encode(bytes(signing_key.verify_key)).decode()

    env_path = config.PROJECT_ROOT / ".env"
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    lines = [l for l in lines if not l.startswith("ROBINHOOD_PRIVATE_KEY=")]
    if not any(l.startswith("ROBINHOOD_API_KEY=") for l in lines):
        lines.append("ROBINHOOD_API_KEY=")
    lines.append(f"ROBINHOOD_PRIVATE_KEY={private_b64}")
    env_path.write_text("\n".join(lines) + "\n")

    print("Private key saved to .env (keep it secret).\n")
    print("PUBLIC key -- paste this into Robinhood's API credentials page:")
    print(f"\n  {public_b64}\n")
    print("Then copy the API key Robinhood gives you into .env as "
          "ROBINHOOD_API_KEY=...")


# ---------------------------------------------------------------- client --

class RobinhoodCrypto:
    def __init__(self, api_key: str | None = None,
                 private_key_b64: str | None = None):
        import os
        self.api_key = api_key or os.getenv("ROBINHOOD_API_KEY", "")
        seed_b64 = private_key_b64 or os.getenv("ROBINHOOD_PRIVATE_KEY", "")
        if not self.api_key or not seed_b64:
            sys.exit("Missing ROBINHOOD_API_KEY / ROBINHOOD_PRIVATE_KEY in "
                     ".env -- run 'python robinhood.py keygen' and follow "
                     "the setup steps in this file's docstring.")
        self.signing_key = SigningKey(base64.b64decode(seed_b64))

    def _headers(self, method: str, path: str, body: str) -> dict:
        timestamp = str(int(time.time()))
        message = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.signing_key.sign(message.encode())
        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode(),
            "x-timestamp": timestamp,
            "Content-Type": "application/json; charset=utf-8",
        }

    def _request(self, method: str, path: str, body: dict | None = None):
        body_str = json.dumps(body) if body else ""
        headers = self._headers(method, path, body_str)
        resp = requests.request(method, BASE_URL + path, headers=headers,
                                data=body_str or None, timeout=15)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"{method} {path} -> {resp.status_code}: {resp.text}")
        return resp.json() if resp.text else {}

    # -- account / market data --

    def account(self) -> dict:
        return self._request("GET", "/api/v1/crypto/trading/accounts/")

    def holdings(self, asset_code: str | None = None) -> dict:
        path = "/api/v1/crypto/trading/holdings/"
        if asset_code:
            path += f"?asset_code={asset_code}"
        return self._request("GET", path)

    def trading_pairs(self, symbol: str | None = None) -> dict:
        path = "/api/v1/crypto/trading/trading_pairs/"
        if symbol:
            path += f"?symbol={symbol}"
        return self._request("GET", path)

    def best_bid_ask(self, symbol: str) -> dict:
        return self._request(
            "GET", f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={symbol}")

    def estimated_price(self, symbol: str, side: str, quantity: str) -> dict:
        return self._request(
            "GET", "/api/v1/crypto/marketdata/estimated_price/"
                   f"?symbol={symbol}&side={side}&quantity={quantity}")

    # -- orders --

    def orders(self) -> dict:
        return self._request("GET", "/api/v1/crypto/trading/orders/")

    def market_order(self, symbol: str, side: str,
                     usd_amount: float | None = None,
                     asset_quantity: float | None = None) -> dict:
        """Market order sized in USD (converted via best quote) or units."""
        if asset_quantity is None:
            quote = self.best_bid_ask(symbol)["results"][0]
            ref = float(quote["ask_inclusive_of_buy_spread"] if side == "buy"
                        else quote["bid_inclusive_of_sell_spread"])
            asset_quantity = round(usd_amount / ref, 8)
        body = {
            "client_order_id": str(uuid.uuid4()),
            "side": side,
            "type": "market",
            "symbol": symbol,
            "market_order_config": {"asset_quantity": f"{asset_quantity}"},
        }
        return self._request("POST", "/api/v1/crypto/trading/orders/", body)

    def cancel_order(self, order_id: str) -> dict:
        return self._request(
            "POST", f"/api/v1/crypto/trading/orders/{order_id}/cancel/")


# ------------------------------------------------------------------- cli --

def _confirm_real_money(action: str) -> bool:
    print(f"\n!! This places a REAL-MONEY {action} on Robinhood. !!")
    return input("Type YES to proceed: ").strip() == "YES"


def main():
    args = sys.argv[1:]
    if not args or args[0] == "keygen":
        if not args:
            print(__doc__)
            return
        keygen()
        return

    client = RobinhoodCrypto()
    cmd = args[0]
    if cmd == "account":
        print(json.dumps(client.account(), indent=2))
    elif cmd == "holdings":
        print(json.dumps(client.holdings(), indent=2))
    elif cmd == "orders":
        print(json.dumps(client.orders(), indent=2))
    elif cmd == "price":
        print(json.dumps(client.best_bid_ask(args[1]), indent=2))
    elif cmd in ("buy", "sell"):
        symbol, usd = args[1], float(args[2])
        if not _confirm_real_money(f"{cmd} of ~${usd:,.2f} {symbol}"):
            print("Aborted.")
            return
        print(json.dumps(
            client.market_order(symbol, cmd, usd_amount=usd), indent=2))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
