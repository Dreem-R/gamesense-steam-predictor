"""Fetch each game's review history from Steam.

Steam's review histogram gives the positive and negative reviews a game received in each month, or each
week for games with a short history. Results are appended to data/reviews/histograms.jsonl, one game per
line, so a run can be stopped and resumed.

Usage: python -m gamesense.pipeline.fetch_reviews [--ids FILE] [--limit N] [--delay SECONDS]
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gamesense import config

URL = "https://store.steampowered.com/appreviewhistogram/{app_id}?l=english"
USER_AGENT = "gamesense-research/0.1 (+https://github.com/Dreem-R/gamesense-steam-predictor)"
RATE_LIMITED = {403, 429}
# Waits after Steam refuses a request, and after any other failure.
RATE_LIMIT_WAITS = [60, 120, 300, 600, 900]
ERROR_WAITS = [5, 20]


class RateLimited(Exception):
    pass


def fetch_histogram(app_id: str, timeout: float = 20) -> dict[str, Any]:
    request = urllib.request.Request(URL.format(app_id=app_id), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        if error.code in RATE_LIMITED:
            raise RateLimited(f"HTTP {error.code}") from error
        raise
    # Steam answers some requests with an HTML page instead of an error status.
    return json.loads(body)


def parse_histogram(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the rollup size and one [start, positive, negative] entry per week or month."""
    results = payload.get("results") or {}
    rollups = results.get("rollups") or []
    if payload.get("success") != 1 or not rollups:
        return {"status": "empty"}
    return {
        "status": "ok",
        "rollup_type": results.get("rollup_type"),
        "rollups": [[r["date"], r["recommendations_up"], r["recommendations_down"]] for r in rollups],
    }


def finished_ids(path: Path) -> set[str]:
    """Games already fetched. Games that failed are tried again."""
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            record = json.loads(line)
            if record["status"] != "error":
                done.add(record["app_id"])
    return done


def fetch_with_retries(
    app_id: str, fetch: Callable[[str], dict[str, Any]], sleep: Callable[[float], None], stats: dict[str, int]
) -> dict[str, Any]:
    rate_limit_waits, error_waits = iter(RATE_LIMIT_WAITS), iter(ERROR_WAITS)
    while True:
        try:
            return parse_histogram(fetch(app_id))
        except RateLimited as limited:
            stats["rate_limited"] += 1
            wait = next(rate_limit_waits, None)
            detail = str(limited)
        except (OSError, ValueError) as error:
            stats["retries"] += 1
            wait = next(error_waits, None)
            detail = f"{type(error).__name__}: {error}"
        if wait is None:
            return {"status": "error", "detail": detail[:200]}
        sleep(wait)


def crawl(
    app_ids: Iterable[str],
    out: Path,
    delay: float = 1.5,
    fetch: Callable[[str], dict[str, Any]] = fetch_histogram,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Fetch every game not already in `out`, appending one line per game as it goes."""
    done = finished_ids(out)
    todo = [a for a in dict.fromkeys(app_ids) if a not in done]
    stats = {"fetched": 0, "ok": 0, "empty": 0, "error": 0, "rate_limited": 0, "retries": 0}
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with out.open("a", encoding="utf-8") as lines:
        for i, app_id in enumerate(todo, start=1):
            record = fetch_with_retries(app_id, fetch, sleep, stats)
            stats["fetched"] += 1
            stats[record["status"]] += 1
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            lines.write(json.dumps({"app_id": app_id, "fetched_at": stamp, **record}) + "\n")
            lines.flush()
            if i % 100 == 0 or i == len(todo):
                minutes = (time.monotonic() - started) / 60
                print(f"{i:,}/{len(todo):,} games in {minutes:.1f} min  {stats}", flush=True)
            if i < len(todo):
                sleep(delay)
    return stats


def all_app_ids() -> list[str]:
    import pandas as pd

    return pd.read_parquet(config.RAW_PATH, columns=["appID"])["appID"].astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ids", type=Path, help="text file with one app id per line (default: every game)")
    parser.add_argument("--out", type=Path, default=config.REVIEW_HISTORY_PATH)
    parser.add_argument("--limit", type=int, help="stop after this many games")
    parser.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    args = parser.parse_args()

    app_ids = args.ids.read_text().split() if args.ids else all_app_ids()
    crawl(app_ids[: args.limit], args.out, delay=args.delay)


if __name__ == "__main__":
    main()
