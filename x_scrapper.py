#!/usr/bin/env python3
"""
x_scraper.py - Base scraper X/Twitter (tweet + replies) -> CSV untuk text mining.

Berbasis `twscrape` (akun login/cookies, async, rotasi akun + auto-wait saat rate limit).

Subcommand:
    add-account   Tambah akun X ke pool (disarankan pakai cookies)
    accounts      Lihat status akun di pool
    tweet         Scrape satu tweet (URL/ID) + replies-nya
    search        Scrape berdasarkan keyword/hashtag (operator pencarian X didukung)
    user          Scrape timeline tweet dari satu akun
    batch         Jalankan kombinasi target dari file JSON

Contoh:
    python x_scraper.py add-account --username akun_scrape --cookies "auth_token=...; ct0=..."
    python x_scraper.py tweet https://x.com/NASA/status/1234567890 --limit 500
    python x_scraper.py search "#pemilu lang:id since:2026-01-01" --limit 300 --expand-replies 20
    python x_scraper.py user nasa --limit 100
    python x_scraper.py batch --config targets.json --out output/dataset.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from twscrape import API
from twscrape.logger import set_log_level

DB_PATH = "accounts.db"
DEFAULT_OUT = "output/dataset.csv"

COLUMNS = [
    "tweet_id",  # simpan sebagai string! (ID 19 digit rusak kalau dibaca sebagai float di Excel)
    "url",
    "created_at",
    "scrape_type",  # tweet | reply | search | user
    "source",  # tweet id / query / username yang memicu data ini
    "root_tweet_id",  # untuk reply: ID tweet utama yang di-scrape
    "is_reply",
    "in_reply_to_tweet_id",
    "conversation_id",
    "user_id",
    "username",
    "displayname",
    "user_followers",
    "user_verified",
    "lang",
    "text",  # teks asli
    "text_clean",  # URL & mention dibuang, whitespace dirapikan
    "reply_count",
    "retweet_count",
    "like_count",
    "quote_count",
    "view_count",
    "hashtags",  # dipisah "|"
    "mentions",  # dipisah "|"
    "links",  # dipisah "|"
    "scraped_at",
]

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")
TWEET_ID_RE = re.compile(r"(?:x|twitter)\.com/[^/\s]+/status/(\d+)")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_tweet_id(target: str) -> int:
    """Terima URL tweet atau ID mentah, kembalikan ID (int)."""
    target = target.strip()
    if target.isdigit():
        return int(target)
    m = TWEET_ID_RE.search(target)
    if not m:
        raise ValueError(f"Tidak bisa mengambil tweet ID dari: {target!r}")
    return int(m.group(1))


def clean_text(text: str) -> str:
    """Pembersihan ringan. Lowercase/stopword/stemming dilakukan di tahap preprocessing."""
    text = html.unescape(text or "")
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def tweet_to_row(t: Any, scrape_type: str, source: str, root_tweet_id: int | None = None) -> dict:
    user = getattr(t, "user", None)
    text = getattr(t, "rawContent", "") or ""
    created = getattr(t, "date", None)
    return {
        "tweet_id": str(t.id),
        "url": getattr(t, "url", ""),
        "created_at": created.isoformat() if created else "",
        "scrape_type": scrape_type,
        "source": source,
        "root_tweet_id": str(root_tweet_id) if root_tweet_id else "",
        "is_reply": getattr(t, "inReplyToTweetId", None) is not None,
        "in_reply_to_tweet_id": str(getattr(t, "inReplyToTweetId", "") or ""),
        "conversation_id": str(getattr(t, "conversationId", "") or ""),
        "user_id": str(getattr(user, "id", "")) if user else "",
        "username": getattr(user, "username", "") if user else "",
        "displayname": getattr(user, "displayname", "") if user else "",
        "user_followers": getattr(user, "followersCount", "") if user else "",
        "user_verified": getattr(user, "verified", "") if user else "",
        "lang": getattr(t, "lang", ""),
        "text": text,
        "text_clean": clean_text(text),
        "reply_count": getattr(t, "replyCount", 0),
        "retweet_count": getattr(t, "retweetCount", 0),
        "like_count": getattr(t, "likeCount", 0),
        "quote_count": getattr(t, "quoteCount", 0),
        "view_count": getattr(t, "viewCount", "") or "",
        "hashtags": "|".join(getattr(t, "hashtags", None) or []),
        "mentions": "|".join(u.username for u in (getattr(t, "mentionedUsers", None) or [])),
        "links": "|".join(getattr(l, "url", "") for l in (getattr(t, "links", None) or [])),
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class CsvSink:
    """Tulis CSV secara inkremental + dedup berdasarkan tweet_id.

    - Aman di-resume: kalau file sudah ada, ID yang sudah tersimpan dilewati.
    - Encoding utf-8-sig supaya emoji/aksara non-Latin tampil benar saat dibuka di Excel.
    """

    def __init__(self, path: str | Path, flush_every: int = 50):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.seen: set[str] = set()
        self._since_flush = 0
        self.written = 0

        has_data = self.path.exists() and self.path.stat().st_size > 0
        if has_data:
            with open(self.path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    self.seen.add(row.get("tweet_id", ""))

        self._fh = open(self.path, "a", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS, extrasaction="ignore")
        if not has_data:
            self._writer.writeheader()

    def write(self, row: dict) -> bool:
        tid = row["tweet_id"]
        if tid in self.seen:
            return False
        self.seen.add(tid)
        self._writer.writerow(row)
        self.written += 1
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self._fh.flush()
            self._since_flush = 0
        return True

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    def __enter__(self) -> "CsvSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Scraping logic
# --------------------------------------------------------------------------- #
async def expand_replies(api: API, root: Any, limit: int, sink: CsvSink) -> int:
    """Ambil replies dari sebuah tweet dan simpan ke sink."""
    if not limit or not getattr(root, "replyCount", 0):
        return 0
    added = 0
    async for r in api.tweet_replies(root.id, limit=limit):
        if sink.write(tweet_to_row(r, "reply", str(root.id), root_tweet_id=root.id)):
            added += 1
    return added


async def scrape_tweet(api: API, sink: CsvSink, target: str, limit: int, with_replies: bool = True) -> None:
    tid = parse_tweet_id(target)
    root = await api.tweet_details(tid)
    if root is None:
        print(f"[!] Tweet {tid} tidak ditemukan / tidak bisa diakses", file=sys.stderr)
        return
    sink.write(tweet_to_row(root, "tweet", str(tid), root_tweet_id=root.id))
    n = await expand_replies(api, root, limit, sink) if with_replies else 0
    print(f"[tweet {tid}] +{n} replies (reply_count di X: {root.replyCount})")


async def scrape_search(
    api: API, sink: CsvSink, query: str, limit: int, product: str = "Latest", expand: int = 0
) -> None:
    n = 0
    n_replies = 0
    async for t in api.search(query, limit=limit, kv={"product": product}):
        if sink.write(tweet_to_row(t, "search", query)):
            n += 1
        n_replies += await expand_replies(api, t, expand, sink)
    print(f"[search {query!r}] +{n} tweets, +{n_replies} replies")


async def scrape_user(api: API, sink: CsvSink, username: str, limit: int, expand: int = 0) -> None:
    username = username.lstrip("@")
    user = await api.user_by_login(username)
    if user is None:
        print(f"[!] User @{username} tidak ditemukan", file=sys.stderr)
        return
    n = 0
    n_replies = 0
    async for t in api.user_tweets(user.id, limit=limit):
        if sink.write(tweet_to_row(t, "user", username)):
            n += 1
        n_replies += await expand_replies(api, t, expand, sink)
    print(f"[user @{username}] +{n} tweets, +{n_replies} replies")


async def run_batch(api: API, sink: CsvSink, config_path: str) -> None:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

    for item in cfg.get("tweets", []):
        try:
            await scrape_tweet(api, sink, item["url"], item.get("limit", 200), item.get("with_replies", True))
        except Exception as e:  # satu target gagal tidak menghentikan batch
            print(f"[!] tweet {item} gagal: {e!r}", file=sys.stderr)

    for item in cfg.get("searches", []):
        try:
            await scrape_search(
                api, sink, item["query"], item.get("limit", 200),
                item.get("product", "Latest"), item.get("expand_replies", 0),
            )
        except Exception as e:
            print(f"[!] search {item} gagal: {e!r}", file=sys.stderr)

    for item in cfg.get("users", []):
        try:
            await scrape_user(api, sink, item["username"], item.get("limit", 100), item.get("expand_replies", 0))
        except Exception as e:
            print(f"[!] user {item} gagal: {e!r}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scraper tweet + replies X/Twitter -> CSV")
    p.add_argument("--db", default=DB_PATH, help="Path database akun twscrape (default: accounts.db)")
    p.add_argument("--log", default="INFO", help="Log level twscrape (DEBUG/INFO/WARNING)")
    sub = p.add_subparsers(dest="cmd", required=True)

    out = argparse.ArgumentParser(add_help=False)
    out.add_argument("--out", default=DEFAULT_OUT, help=f"File CSV output (default: {DEFAULT_OUT})")

    a = sub.add_parser("add-account", help="Tambah akun ke pool")
    a.add_argument("--username", required=True)
    a.add_argument("--password", default="x", help="Tidak dipakai kalau --cookies diberikan")
    a.add_argument("--email", default="x@example.com", help="Tidak dipakai kalau --cookies diberikan")
    a.add_argument("--email-password", default="x", help="Tidak dipakai kalau --cookies diberikan")
    a.add_argument("--cookies", help='String cookies, mis. "auth_token=...; ct0=..."')
    a.add_argument("--proxy", help="mis. http://user:pass@host:port")

    sub.add_parser("accounts", help="Lihat status akun")

    t = sub.add_parser("tweet", parents=[out], help="Satu tweet + replies")
    t.add_argument("target", help="URL tweet atau tweet ID")
    t.add_argument("--limit", type=int, default=200, help="Maks replies (-1 = semua)")
    t.add_argument("--no-replies", action="store_true")

    s = sub.add_parser("search", parents=[out], help="Keyword/hashtag")
    s.add_argument("query", help='Query X, mis. "#topik lang:id since:2026-01-01 until:2026-02-01"')
    s.add_argument("--limit", type=int, default=200)
    s.add_argument("--product", choices=["Latest", "Top"], default="Latest")
    s.add_argument("--expand-replies", type=int, default=0, help="Ambil N replies per tweet hasil pencarian")

    u = sub.add_parser("user", parents=[out], help="Timeline akun")
    u.add_argument("username")
    u.add_argument("--limit", type=int, default=100)
    u.add_argument("--expand-replies", type=int, default=0)

    b = sub.add_parser("batch", parents=[out], help="Kombinasi target dari JSON")
    b.add_argument("--config", required=True, help="Path targets.json")
    return p


async def amain(args: argparse.Namespace) -> None:
    set_log_level(args.log)
    api = API(args.db)

    if args.cmd == "add-account":
        await api.pool.add_account(
            args.username, args.password, args.email, args.email_password,
            cookies=args.cookies, proxy=args.proxy,
        )
        if not args.cookies:
            await api.pool.login_all()  # login via password (sering kena challenge; cookies lebih andal)
        print("Akun ditambahkan.")
        return

    if args.cmd == "accounts":
        for acc in await api.pool.accounts_info():
            print(acc)
        return

    with CsvSink(args.out) as sink:
        if args.cmd == "tweet":
            await scrape_tweet(api, sink, args.target, args.limit, not args.no_replies)
        elif args.cmd == "search":
            await scrape_search(api, sink, args.query, args.limit, args.product, args.expand_replies)
        elif args.cmd == "user":
            await scrape_user(api, sink, args.username, args.limit, args.expand_replies)
        elif args.cmd == "batch":
            await run_batch(api, sink, args.config)
        print(f"Selesai. {sink.written} baris baru -> {sink.path}")


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nDihentikan. Data yang sudah ditulis tetap tersimpan di CSV.", file=sys.stderr)


if __name__ == "__main__":
    main()