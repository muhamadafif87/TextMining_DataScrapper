#!/usr/bin/env python3
"""Normalize the project text dumps into the tweet CSV schema."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

WIB = timezone(timedelta(hours=7))
FIELDS = [
    "tweet_id",
    "created_at",
    "username",
    "raw_text",
    "reply_count",
    "retweet_count",
    "like_count",
    "quote_count",
    "conversation_id",
    "search_query_used",
    "scrape_type",
    "root_tweet_id",
    "root_tweet_text",
    "is_reply",
    "in_reply_to_tweet_id",
    "lang",
    "hashtags",
    "is_spam_suspect",
    "sentiment_label",
    "scraped_at",
]
INPUT_FILES = [
    "replies-output.txt",
    "replies.txt",
    "tambahan-replies.txt",
    "tweet-batch.txt",
]
SPAM_WORDS = (
    "slot",
    "gacor",
    "judi",
    "togel",
    "maxwin",
    "bandar",
    "casino",
    "sbobet",
    "kripto",
    "crypto",
    "airdrop",
    "binary option",
    "forex",
    "pinjol",
    "pinjaman online",
)
HASH_TAG_RE = re.compile(r"(?<!\w)#([\w]+)", re.UNICODE)
SPAM_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in SPAM_WORDS) + r")\b",
    re.IGNORECASE,
)
REPLY_HEADER_RE = re.compile(
    r"^\s*@(?P<username>[\w]+)\s+\((?P<id>\d+)\)\s+"
    r"\[likes=(?P<likes>\d+),\s*reposts=(?P<retweets>\d+),\s*"
    r"replies=(?P<replies>\d+),\s*bookmarks=\d+,\s*(?P<date>[^]]+)\]\s*$"
)
TWEET_USER_RE = re.compile(r"\(@(?P<username>[\w]+)\)\s*$")
TWEET_ID_RE = re.compile(r"/status/(\d+)")


def hash_username(username: str, salt: str) -> str:
    """Match the scraper's salted SHA-256 username pseudonym."""
    return hashlib.sha256((salt + username.lower()).encode("utf-8")).hexdigest()[:16]


def parse_reply_date(value: str) -> str:
    try:
        date = datetime.strptime(
            f"{value.strip()} {datetime.now(WIB).year}",
            "%a %b %d %H:%M %Y",
        )
    except ValueError:
        return ""
    date = date.replace(year=datetime.now(WIB).year, tzinfo=WIB)
    return date.isoformat(timespec="minutes")


def parse_tweet_date(value: str) -> str:
    try:
        date = datetime.strptime(value.strip(), "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return ""
    return date.astimezone(WIB).isoformat(timespec="seconds")


def _hashtags(text: str) -> str:
    tags = dict.fromkeys(HASH_TAG_RE.findall(text))
    return "|".join(tags)


def _record(
    tweet_id: str,
    username: str,
    text: str,
    metrics: dict[str, int],
    created_at: str,
    scraped_at: str,
    *,
    root_id: str = "",
    is_reply: bool = False,
) -> dict[str, str]:
    spam = bool(SPAM_RE.search(text) or len(_hashtags(text).split("|")) > 4)
    return {
        "tweet_id": tweet_id,
        "created_at": created_at,
        "username": username,
        "raw_text": text.strip(),
        "reply_count": str(metrics.get("replies", 0)),
        "retweet_count": str(metrics.get("retweets", 0)),
        "like_count": str(metrics.get("likes", 0)),
        "quote_count": str(metrics.get("quotes", 0)),
        "conversation_id": root_id or tweet_id,
        "search_query_used": "",
        "scrape_type": "reply" if is_reply else "search",
        "root_tweet_id": root_id if is_reply else "",
        "root_tweet_text": "",
        "is_reply": str(is_reply),
        "in_reply_to_tweet_id": root_id if is_reply else "",
        "lang": "",
        "hashtags": _hashtags(text),
        "is_spam_suspect": str(spam),
        "sentiment_label": "",
        "scraped_at": scraped_at,
    }


def _parse_reply_dump(path: Path, salt: str, scraped_at: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    section = "search"
    root_id = ""
    current: dict[str, object] | None = None

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        text = "\n".join(current["text"]).strip()
        if text:
            rows.append(
                _record(
                    str(current["id"]),
                    hash_username(str(current["username"]), salt),
                    text,
                    current["metrics"],
                    str(current["created_at"]),
                    scraped_at,
                    root_id=str(current["root_id"]),
                    is_reply=bool(current["is_reply"]),
                )
            )
        current = None

    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("====="):
            finish()
            section, root_id = "search", ""
            continue
        if stripped.startswith("--- replies"):
            finish()
            section = "reply"
            continue
        if stripped.startswith("--- thread"):
            finish()
            section = "thread"
            continue
        if stripped.startswith("(more available:"):
            continue

        header_line = stripped
        is_root_header = header_line.startswith("### ROOT POST:")
        if is_root_header:
            header_line = header_line.removeprefix("### ROOT POST:").strip()
        match = REPLY_HEADER_RE.match(header_line)
        if match:
            finish()
            tweet_id = match.group("id")
            is_reply = section == "reply"
            if is_root_header:
                root_id = tweet_id
                is_reply = False
            current = {
                "id": tweet_id,
                "username": match.group("username"),
                "metrics": {
                    "likes": int(match.group("likes")),
                    "retweets": int(match.group("retweets")),
                    "replies": int(match.group("replies")),
                },
                "created_at": parse_reply_date(match.group("date")),
                "root_id": root_id if is_reply else "",
                "is_reply": is_reply,
                "text": [],
            }
            continue

        if current is not None and stripped:
            current["text"].append(stripped)
    finish()
    return rows


def _parse_tweet_batch(path: Path, salt: str, scraped_at: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current: dict[str, object] | None = None

    def finish() -> None:
        nonlocal current
        if current is None or not current["id"]:
            current = None
            return
        text = "\n".join(current["text"]).strip()
        rows.append(
            _record(
                str(current["id"]),
                hash_username(str(current["username"]), salt),
                text,
                current["metrics"],
                str(current["created_at"]),
                scraped_at,
            )
        )
        current = None

    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            finish()
            user_match = TWEET_USER_RE.search(stripped)
            current = {
                "id": "",
                "username": user_match.group("username") if user_match else "",
                "created_at": "",
                "metrics": {},
                "text": [],
                "body": False,
            }
            continue
        if current is None:
            continue
        if stripped.startswith("URL:"):
            id_match = TWEET_ID_RE.search(stripped)
            if id_match:
                current["id"] = id_match.group(1)
            continue
        if stripped.startswith("created:"):
            current["created_at"] = parse_tweet_date(stripped.removeprefix("created:"))
            continue
        metrics_match = re.search(
            r"likes=(\d+)\s+reposts=(\d+)\s+replies=(\d+)\s+quotes=(\d+)",
            stripped,
        )
        if metrics_match:
            current["metrics"] = {
                "likes": int(metrics_match.group(1)),
                "retweets": int(metrics_match.group(2)),
                "replies": int(metrics_match.group(3)),
                "quotes": int(metrics_match.group(4)),
            }
            current["body"] = True
            continue
        if stripped.startswith("---") or stripped.startswith("media[") or stripped.startswith("PS "):
            continue
        if current["body"] and stripped:
            current["text"].append(stripped)
    finish()
    return rows


def parse_file(path: Path, salt: str, scraped_at: str) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if re.search(r"(?m)^URL:\s+https?://", text):
        rows = _parse_reply_dump(path, salt, scraped_at)
        rows.extend(_parse_tweet_batch(path, salt, scraped_at))
        return rows
    return _parse_reply_dump(path, salt, scraped_at)


def merge_duplicate(existing: dict[str, str], incoming: dict[str, str]) -> None:
    for field in ("reply_count", "retweet_count", "like_count", "quote_count"):
        existing[field] = str(max(int(existing[field] or 0), int(incoming[field] or 0)))
    if len(incoming["raw_text"]) > len(existing["raw_text"]):
        existing["raw_text"] = incoming["raw_text"]
    for field in FIELDS:
        if not existing[field] and incoming[field]:
            existing[field] = incoming[field]
    existing["hashtags"] = "|".join(
        dict.fromkeys(filter(None, (existing["hashtags"] + "|" + incoming["hashtags"]).split("|")))
    )
    existing["is_spam_suspect"] = str(
        existing["is_spam_suspect"] == "True" or incoming["is_spam_suspect"] == "True"
    )
    if incoming["scrape_type"] == "search":
        existing["scrape_type"] = "search"
        existing["is_reply"] = "False"
        existing["root_tweet_id"] = ""
        existing["root_tweet_text"] = ""
        existing["in_reply_to_tweet_id"] = ""


def attach_root_tweet_text(rows_by_id: dict[str, dict[str, str]]) -> None:
    root_text_by_id = {
        tweet_id: row["raw_text"]
        for tweet_id, row in rows_by_id.items()
        if row["is_reply"] != "True" and row["raw_text"]
    }
    for row in rows_by_id.values():
        if row["is_reply"] == "True" and not row["root_tweet_text"]:
            row["root_tweet_text"] = root_text_by_id.get(row["root_tweet_id"], "")


def load_salt(path: Path) -> str:
    env_salt = os.environ.get("X_SCRAPER_SALT")
    if env_salt:
        return env_salt
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(16), encoding="utf-8")
    salt = path.read_text(encoding="utf-8").strip()
    if not salt:
        raise ValueError(f"File salt kosong: {path}")
    return salt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=INPUT_FILES, help="Berkas .txt yang dibersihkan")
    parser.add_argument("--out", default="cleaned_data.csv", help="Path CSV keluaran")
    parser.add_argument("--salt-file", default="output/.salt", help="Salt bersama dengan x_scraper.py")
    parser.add_argument("--drop-spam", action="store_true", help="Buang baris yang ditandai sebagai spam")
    args = parser.parse_args()

    salt = load_salt(Path(args.salt_file))
    scraped_at = datetime.now(WIB).isoformat(timespec="seconds")
    rows_by_id: dict[str, dict[str, str]] = {}
    source_counts: dict[str, int] = {}
    for input_name in args.inputs:
        path = Path(input_name)
        if not path.is_file():
            raise FileNotFoundError(f"Berkas input tidak ditemukan: {path}")
        source_rows = parse_file(path, salt, scraped_at)
        source_counts[path.name] = len(source_rows)
        for row in source_rows:
            if row["tweet_id"] in rows_by_id:
                merge_duplicate(rows_by_id[row["tweet_id"]], row)
            else:
                rows_by_id[row["tweet_id"]] = row

    attach_root_tweet_text(rows_by_id)
    rows = list(rows_by_id.values())
    if args.drop_spam:
        rows = [row for row in rows if row["is_spam_suspect"] == "False"]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV tersimpan: {output} ({len(rows)} baris unik)")
    for name, count in source_counts.items():
        print(f"  {name}: {count} baris terbaca")
    if args.drop_spam:
        print(f"  Baris spam dibuang: {len(rows_by_id) - len(rows)}")


if __name__ == "__main__":
    main()
