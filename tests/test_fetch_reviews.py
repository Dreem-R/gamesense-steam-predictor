import json

from gamesense.pipeline.fetch_reviews import (
    ERROR_WAITS,
    RATE_LIMIT_WAITS,
    RateLimited,
    crawl,
    finished_ids,
    parse_histogram,
)


def payload(*rollups: tuple[int, int, int], rollup_type: str = "month") -> dict:
    return {
        "success": 1,
        "results": {
            "rollup_type": rollup_type,
            "rollups": [{"date": d, "recommendations_up": up, "recommendations_down": down} for d, up, down in rollups],
        },
    }


def read(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_parse_keeps_rollup_size_and_counts():
    parsed = parse_histogram(payload((1700000000, 12, 3), (1702600000, 5, 0), rollup_type="week"))
    assert parsed == {"status": "ok", "rollup_type": "week", "rollups": [[1700000000, 12, 3], [1702600000, 5, 0]]}


def test_parse_marks_games_without_reviews_as_empty():
    assert parse_histogram(payload()) == {"status": "empty"}
    assert parse_histogram({"success": 2}) == {"status": "empty"}


def test_crawl_writes_one_line_per_game(tmp_path):
    out = tmp_path / "histograms.jsonl"
    pages = {"10": payload((1, 4, 1)), "20": payload()}
    stats = crawl(["10", "20", "10"], out, fetch=pages.__getitem__, sleep=lambda s: None)

    records = read(out)
    assert [r["app_id"] for r in records] == ["10", "20"]
    assert [r["status"] for r in records] == ["ok", "empty"]
    assert records[0]["rollups"] == [[1, 4, 1]]
    assert stats["fetched"] == 2


def test_crawl_resumes_where_it_stopped(tmp_path):
    out = tmp_path / "histograms.jsonl"
    asked = []

    def fetch(app_id: str) -> dict:
        asked.append(app_id)
        return payload((1, 1, 0))

    crawl(["10", "20"], out, fetch=fetch, sleep=lambda s: None)
    crawl(["10", "20", "30"], out, fetch=fetch, sleep=lambda s: None)
    assert asked == ["10", "20", "30"]
    assert finished_ids(out) == {"10", "20", "30"}


def test_crawl_waits_and_retries_when_rate_limited(tmp_path):
    out = tmp_path / "histograms.jsonl"
    answers: list[dict | Exception] = [RateLimited("HTTP 429"), RateLimited("HTTP 429"), payload((1, 2, 0))]
    waits: list[float] = []

    def fetch(app_id: str) -> dict:
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    stats = crawl(["10"], out, fetch=fetch, sleep=waits.append)
    assert waits == RATE_LIMIT_WAITS[:2]
    assert stats["rate_limited"] == 2
    assert read(out)[0]["status"] == "ok"


def test_failed_games_are_recorded_and_tried_again_next_run(tmp_path):
    out = tmp_path / "histograms.jsonl"

    def broken(app_id: str) -> dict:
        raise ValueError("not json")

    waits: list[float] = []
    crawl(["10"], out, fetch=broken, sleep=waits.append)
    assert waits == ERROR_WAITS
    assert read(out)[0]["status"] == "error"
    assert finished_ids(out) == set()

    crawl(["10"], out, fetch=lambda app_id: payload((1, 1, 1)), sleep=lambda s: None)
    assert finished_ids(out) == {"10"}
