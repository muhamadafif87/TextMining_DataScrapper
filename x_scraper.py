#!/usr/bin/env python3
"""
scraper_x_perpres111.py
=======================
Scraper tweet publik X (Twitter) untuk riset text mining:
"Analisis Sentimen Publik terhadap Pencantuman Isu LGBTQ+ sebagai Ancaman Nonmiliter
 dalam Perpres Nomor 111 Tahun 2025 Menggunakan Algoritma Multinomial Naive Bayes".

Engine : twscrape (async, akun login via cookies, rotasi akun, auto-wait saat rate limit)
Output : CSV UTF-8-SIG (+ opsional JSONL), append inkremental, resume otomatis.

Struktur (separation of concerns):
    Config           -> semua parameter yang bisa diubah
    AuthManager      -> membaca cookies.json & mendaftarkan akun ke pool twscrape
    QueryBuilder     -> membangun query Boolean per jendela tanggal
    TweetParser      -> tweet -> baris dataset (+ filter retweet / flag spam / hash username)
    StorageManager   -> CSV/JSONL inkremental, dedup tweet_id, catatan query yang sudah selesai
    ScraperEngine    -> orkestrasi: loop query x jendela tanggal, jitter, backoff, ekspansi replies

Contoh:
    python scraper_x_perpres111.py --dry-run                       # lihat query tanpa scraping
    python scraper_x_perpres111.py --since 2026-07-05 --until 2026-07-08 --limit-per-window 200
    python scraper_x_perpres111.py                                 # jalankan penuh (resume otomatis)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import random
import re
import secrets
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from twscrape import API
from twscrape.logger import set_log_level

log = logging.getLogger("scraper_x")
WIB = ZoneInfo("Asia/Jakarta")


# =============================================================================
# 1. CONFIG
# =============================================================================
@dataclass
class QueryGroup:
    """Satu strategi query = (salah satu istilah kebijakan) AND (salah satu istilah isu)."""

    name: str
    policy_terms: list[str]
    issue_terms: list[str]


@dataclass
class Config:
    # --- Auth -----------------------------------------------------------------
    cookies_file: str = "cookies.json"
    db_path: str = "accounts.db"  # database akun twscrape (JANGAN di-commit)

    # --- Rentang tanggal ------------------------------------------------------
    # Perpres ditandatangani 24 Okt 2025, tetapi polemik publik baru ramai sekitar awal Juli 2026.
    # `until` bersifat EKSKLUSIF (aturan operator X). Ubah sesuai kebutuhan.
    since: str = "2026-07-01"
    until: str = "2026-11-27"
    window_days: int = 3  # pecah rentang jadi jendela kecil agar tidak mentok batas kedalaman search X

    # --- Query ----------------------------------------------------------------
    lang: str = "id"
    exclude_retweets: bool = True
    product: str = "Latest"  # "Latest" | "Top"
    query_groups: list[QueryGroup] = field(
        default_factory=lambda: [
            QueryGroup(
                name="core",  # presisi tinggi: wajib menyebut kebijakan DAN isu
                policy_terms=[
                    "Perpres 111",
                    "Perpres Nomor 111",
                    "Perpres 111 Tahun 2025",
                    "Perpres 111/2025",
                    "#Perpres111",
                    "ancaman nonmiliter",
                    "ancaman non-militer",
                    "kebijakan umum pertahanan negara",
                    "jakumhaneg",
                    "boti hunter",  # ejaan sesuai prompt; ejaan resmi Kemhan: "Jakumhaneg"
                ],
                issue_terms=["LGBT", "LGBTQ", "LGBTQ+", "perilaku menyimpang"],
            ),
            QueryGroup(
                name="loose",  # recall lebih tinggi: bahasa informal tanpa nomor Perpres
                policy_terms=["Perpres", "ancaman negara", "pertahanan negara"],
                issue_terms=["LGBT", "LGBTQ"],
            ),
        ]
    )

    # --- Volume & replies -----------------------------------------------------
    per_window_limit: int = 500  # maks tweet per (query x jendela); -1 = tanpa batas
    expand_replies_per_tweet: int = 50  # 0 = nonaktif; >0 = ambil N replies per tweet hasil pencarian
    min_replies_to_expand: int = 1  # hanya ekspansi kalau reply_count >= nilai ini

    # --- Anti-ban / resilience -----------------------------------------------
    page_size: int = 20  # kira-kira jumlah tweet per halaman twscrape
    delay_min: float = 3.0  # jitter antar "halaman" (detik)
    delay_max: float = 8.0
    reply_delay_min: float = 1.0  # jeda setelah tiap ekspansi replies (lebih pendek agar tidak terlalu lambat)
    reply_delay_max: float = 3.0
    max_retries: int = 5
    backoff_base: float = 10.0  # 10s, 20s, 40s, 80s, 160s (+ jitter), dibatasi backoff_cap
    backoff_cap: float = 300.0
    max_empty_streak: int = 5  # hentikan run jika N query pertama berturut-turut 0 hasil

    # --- Pre-filtering --------------------------------------------------------
    max_hashtags: int = 4
    spam_keywords: list[str] = field(
        default_factory=lambda: [
            "slot", "gacor", "judi", "togel", "maxwin", "bandar", "casino", "sbobet",
            "kripto", "crypto", "airdrop", "binary option", "forex", "pinjol", "pinjaman online",
        ]
    )
    drop_spam: bool = False  # False = hanya diberi flag `is_spam_suspect` (disarankan, agar bisa diaudit)

    # --- Privasi --------------------------------------------------------------
    hash_username: bool = True  # SHA-256 + salt lokal; displayname/profil tidak disimpan
    salt_file: str = "output/.salt"

    # --- Output ---------------------------------------------------------------
    output_csv: str = "output/perpres111_raw.csv"
    output_jsonl: str | None = "output/perpres111_raw.jsonl"  # None untuk menonaktifkan
    done_file: str = "output/.done_queries.txt"
    checkpoint_every: int = 50  # flush ke disk tiap N tweet baru


FIELDS = [
    "tweet_id",
    "created_at",  # ISO 8601, WIB (UTC+7)
    "username",  # hash (default) atau handle mentah
    "raw_text",
    "reply_count",
    "retweet_count",
    "like_count",
    "quote_count",
    "conversation_id",
    "search_query_used",
    # --- kolom tambahan untuk kebutuhan riset ---
    "scrape_type",  # search | reply
    "root_tweet_id",  # untuk reply: tweet yang memicu ekspansi
    "is_reply",
    "in_reply_to_tweet_id",
    "lang",
    "hashtags",  # dipisah "|"
    "is_spam_suspect",
    "sentiment_label",  # kosong; diisi saat anotasi manual (pro/kontra/netral)
    "scraped_at",
]


# =============================================================================
# 2. AUTH
# =============================================================================
class AuthManager:
    """Membaca cookies.json lalu mendaftarkan akun ke pool twscrape.

    Format yang didukung:
      a) {"username": "akun1", "auth_token": "...", "ct0": "..."}
      b) [ {akun a}, {akun b}, ... ]                 -> beberapa akun (rotasi otomatis)
      c) Ekspor ekstensi browser (mis. Cookie-Editor): [{"name": "auth_token", "value": "..."}, ...]
    """

    def __init__(self, cfg: Config, api: API):
        self.cfg = cfg
        self.api = api

    @staticmethod
    def _normalize(data: Any) -> list[dict]:
        if isinstance(data, list) and data and isinstance(data[0], dict) and "name" in data[0]:
            c = {x["name"]: x["value"] for x in data if "name" in x and "value" in x}
            return [{"username": "scraper_1", **{k: c[k] for k in ("auth_token", "ct0") if k in c}}]
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError("Format cookies.json tidak dikenali")

    async def setup(self, reset: bool = False) -> int:
        path = Path(self.cfg.cookies_file)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} tidak ditemukan. Buat file ini dulu (lihat README.md, bagian 'Cookies')."
            )
        accounts = self._normalize(json.loads(path.read_text(encoding="utf-8")))
        added = 0
        for i, acc in enumerate(accounts, start=1):
            token, ct0 = acc.get("auth_token"), acc.get("ct0")
            if not token or not ct0:
                raise ValueError(f"Akun #{i} di {path} harus punya 'auth_token' dan 'ct0'")
            username = acc.get("username") or f"scraper_{i}"
            if reset:
                await self.api.pool.delete_accounts(username)
            # Dengan cookies, password/email tidak dipakai (placeholder).
            await self.api.pool.add_account(
                username, "x", "x@example.com", "x",
                cookies=f"auth_token={token}; ct0={ct0}", proxy=acc.get("proxy"),
            )
            added += 1
        log.info("Akun terdaftar di pool: %d", added)
        return added


# =============================================================================
# 3. QUERY BUILDER
# =============================================================================
class QueryBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _term(t: str) -> str:
        # hashtag / kata tunggal dibiarkan; frasa & karakter khusus di-quote
        return t if re.fullmatch(r"#?\w+", t) else f'"{t}"'

    def _or_group(self, terms: list[str]) -> str:
        return "(" + " OR ".join(self._term(t) for t in terms) + ")"

    def build(self, group: QueryGroup, since: str, until: str) -> str:
        parts = [self._or_group(group.policy_terms), self._or_group(group.issue_terms)]  # spasi = AND
        if self.cfg.lang:
            parts.append(f"lang:{self.cfg.lang}")
        if self.cfg.exclude_retweets:
            parts.append("-filter:retweets")
        parts += [f"since:{since}", f"until:{until}"]
        return " ".join(parts)

    @staticmethod
    def date_windows(since: str, until: str, step_days: int) -> Iterator[tuple[str, str]]:
        cur, end = date.fromisoformat(since), date.fromisoformat(until)
        while cur < end:
            nxt = min(cur + timedelta(days=step_days), end)
            yield cur.isoformat(), nxt.isoformat()
            cur = nxt


# =============================================================================
# 4. PARSER (+ light pre-filtering)
# =============================================================================
class TweetParser:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._salt = self._load_salt() if cfg.hash_username else ""
        kw = "|".join(re.escape(k) for k in cfg.spam_keywords)
        self._spam_re = re.compile(rf"\b(?:{kw})\b", re.IGNORECASE) if kw else None

    def _load_salt(self) -> str:
        env = os.environ.get("X_SCRAPER_SALT")
        if env:
            return env
        p = Path(self.cfg.salt_file)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(secrets.token_hex(16), encoding="utf-8")
            log.info("Salt hash baru dibuat di %s (simpan & jangan di-commit)", p)
        return p.read_text(encoding="utf-8").strip()

    def _user(self, username: str) -> str:
        if not self.cfg.hash_username:
            return username
        return hashlib.sha256((self._salt + username.lower()).encode()).hexdigest()[:16]

    @staticmethod
    def is_retweet(t: Any) -> bool:
        text = getattr(t, "rawContent", "") or ""
        return getattr(t, "retweetedTweet", None) is not None or text.startswith("RT @")

    def is_spam(self, text: str, hashtags: list[str]) -> bool:
        if len(hashtags) > self.cfg.max_hashtags:
            return True
        return bool(self._spam_re and self._spam_re.search(text))

    def parse(self, t: Any, query: str, scrape_type: str = "search", root_id: int | None = None) -> dict | None:
        """Kembalikan dict baris dataset, atau None kalau harus dibuang."""
        if self.is_retweet(t):
            return None
        text = getattr(t, "rawContent", "") or ""
        hashtags = list(getattr(t, "hashtags", None) or [])
        spam = self.is_spam(text, hashtags)
        if spam and self.cfg.drop_spam:
            return None

        created = getattr(t, "date", None)
        if created is not None and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        user = getattr(t, "user", None)
        reply_to = getattr(t, "inReplyToTweetId", None)
        return {
            "tweet_id": str(t.id),
            "created_at": created.astimezone(WIB).isoformat() if created else "",
            "username": self._user(user.username) if user else "",
            "raw_text": text,
            "reply_count": getattr(t, "replyCount", 0),
            "retweet_count": getattr(t, "retweetCount", 0),
            "like_count": getattr(t, "likeCount", 0),
            "quote_count": getattr(t, "quoteCount", 0),
            "conversation_id": str(getattr(t, "conversationId", "") or ""),
            "search_query_used": query,
            "scrape_type": scrape_type,
            "root_tweet_id": str(root_id) if root_id else "",
            "is_reply": reply_to is not None,
            "in_reply_to_tweet_id": str(reply_to) if reply_to else "",
            "lang": getattr(t, "lang", ""),
            "hashtags": "|".join(hashtags),
            "is_spam_suspect": spam,
            "sentiment_label": "",
            "scraped_at": datetime.now(WIB).isoformat(timespec="seconds"),
        }


# =============================================================================
# 5. STORAGE (checkpointing + resume)
# =============================================================================
class StorageManager:
    """Append inkremental ke CSV (UTF-8-SIG) & JSONL, dedup tweet_id, catat query selesai."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.csv_path = Path(cfg.output_csv)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.seen: set[str] = set()
        self.done: set[str] = set()
        self.new_rows = 0
        self._pending = 0

        has_data = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        if has_data:
            with open(self.csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != FIELDS:
                    raise RuntimeError(
                        f"Header {self.csv_path} berbeda dari skema script. "
                        "Gunakan nama file output lain atau migrasikan datanya."
                    )
                self.seen = {r["tweet_id"] for r in reader}
            log.info("Resume: %d tweet sudah tersimpan di %s", len(self.seen), self.csv_path)

        self._csv = open(self.csv_path, "a", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._csv, fieldnames=FIELDS, extrasaction="ignore")
        if not has_data:
            self._writer.writeheader()

        self._jsonl = None
        if cfg.output_jsonl:
            Path(cfg.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
            self._jsonl = open(cfg.output_jsonl, "a", encoding="utf-8")

        self.done_path = Path(cfg.done_file)
        if self.done_path.exists():
            self.done = set(self.done_path.read_text(encoding="utf-8").splitlines())

    def add(self, row: dict) -> bool:
        tid = row["tweet_id"]
        if tid in self.seen:
            return False
        self.seen.add(tid)
        self._writer.writerow(row)
        if self._jsonl:
            self._jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.new_rows += 1
        self._pending += 1
        if self._pending >= self.cfg.checkpoint_every:
            self.flush()
        return True

    def flush(self) -> None:
        self._csv.flush()
        if self._jsonl:
            self._jsonl.flush()
        self._pending = 0

    def is_done(self, key: str) -> bool:
        return key in self.done

    def mark_done(self, key: str) -> None:
        self.flush()
        self.done.add(key)
        with open(self.done_path, "a", encoding="utf-8") as f:
            f.write(key + "\n")

    def close(self) -> None:
        self.flush()
        self._csv.close()
        if self._jsonl:
            self._jsonl.close()


# =============================================================================
# 6. ENGINE
# =============================================================================
class ScraperEngine:
    def __init__(self, cfg: Config, api: API, parser: TweetParser, storage: StorageManager, qb: QueryBuilder):
        self.cfg, self.api, self.parser, self.storage, self.qb = cfg, api, parser, storage, qb
        self.stats = {"queries": 0, "tweets": 0, "replies": 0, "skipped": 0}
        self.total_raw = 0  # total hasil mentah dari X selama run ini (bukti pipeline berfungsi)
        self.empty_streak = 0  # jumlah query berturut-turut yang 0 hasil

    async def _jitter(self, lo: float | None = None, hi: float | None = None) -> None:
        lo = self.cfg.delay_min if lo is None else lo
        hi = self.cfg.delay_max if hi is None else hi
        await asyncio.sleep(random.uniform(lo, hi))

    async def _with_backoff(self, label: str, factory) -> bool:
        """Jalankan coroutine dengan exponential backoff + jitter. True kalau sukses."""
        for attempt in range(self.cfg.max_retries + 1):
            try:
                await factory()
                return True
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as e:  # 429, putus koneksi, timeout, dll.
                if attempt >= self.cfg.max_retries:
                    log.error("[%s] gagal permanen: %r", label, e)
                    return False
                wait = min(self.cfg.backoff_cap, self.cfg.backoff_base * 2**attempt)
                wait += random.uniform(0, self.cfg.backoff_base)
                log.warning("[%s] error %r | retry %d/%d dalam %.0fs", label, e, attempt + 1, self.cfg.max_retries, wait)
                await asyncio.sleep(wait)
        return False

    async def _expand_replies(self, root: Any, query: str) -> None:
        cfg = self.cfg
        if not cfg.expand_replies_per_tweet or (getattr(root, "replyCount", 0) or 0) < cfg.min_replies_to_expand:
            return

        async def run() -> None:
            n = 0
            async for r in self.api.tweet_replies(root.id, limit=cfg.expand_replies_per_tweet):
                row = self.parser.parse(r, query, scrape_type="reply", root_id=root.id)
                if row is None:
                    self.stats["skipped"] += 1
                elif self.storage.add(row):
                    self.stats["replies"] += 1
                n += 1
                if n % cfg.page_size == 0:
                    await self._jitter()

        await self._with_backoff(f"replies {root.id}", run)
        await self._jitter(cfg.reply_delay_min, cfg.reply_delay_max)

    async def _scrape_query(self, query: str) -> None:
        n = 0

        async def run() -> None:
            nonlocal n
            n = 0
            async for t in self.api.search(query, limit=self.cfg.per_window_limit, kv={"product": self.cfg.product}):
                n += 1
                self.total_raw += 1
                row = self.parser.parse(t, query)
                if row is None:
                    self.stats["skipped"] += 1
                elif self.storage.add(row):
                    self.stats["tweets"] += 1
                    await self._expand_replies(t, query)
                if n % self.cfg.page_size == 0:  # jeda "per halaman" dengan jitter
                    log.info("     ... %d hasil mentah | baru di run ini: %d tweet, %d reply",
                             n, self.stats["tweets"], self.stats["replies"])
                    await self._jitter()

        ok = await self._with_backoff(f"search {query[:60]}...", run)
        if ok and n == 0 and self.total_raw == 0:
            # 0 hasil dan belum pernah ada hasil sama sekali -> kemungkinan masalah akun/query, BUKAN jendela kosong.
            # Jangan ditandai selesai supaya tidak ikut ter-skip saat resume.
            self.empty_streak += 1
            log.warning("  -> 0 hasil (belum pernah ada hasil di run ini; tidak ditandai selesai)")
            if self.empty_streak >= self.cfg.max_empty_streak:
                raise RuntimeError(
                    f"{self.empty_streak} query berturut-turut 0 hasil. Kemungkinan akun/cookies bermasalah atau query "
                    "terlalu ketat. Jalankan: python scraper_x_perpres111.py --diagnose"
                )
            return
        if ok:
            self.empty_streak = 0
            self.storage.mark_done(query)  # sukses (termasuk jendela yang memang kosong setelah pipeline terbukti jalan)
        log.info("  -> %d hasil mentah | total baru: %d tweet, %d reply", n, self.stats["tweets"], self.stats["replies"])

    def plan(self) -> list[str]:
        windows = list(self.qb.date_windows(self.cfg.since, self.cfg.until, self.cfg.window_days))
        return [self.qb.build(g, s, e) for g in self.cfg.query_groups for (s, e) in windows]

    async def run(self) -> None:
        queries = self.plan()
        todo = [q for q in queries if not self.storage.is_done(q)]
        log.info("Total query: %d | sudah selesai: %d | akan dijalankan: %d", len(queries), len(queries) - len(todo), len(todo))
        for i, q in enumerate(todo, start=1):
            log.info("[%d/%d] %s", i, len(todo), q)
            await self._scrape_query(q)
            self.stats["queries"] += 1
            await self._jitter()
        log.info("SELESAI. Baris baru: %d tweet + %d reply | dilewati (retweet/spam): %d", self.stats["tweets"], self.stats["replies"], self.stats["skipped"])


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scraper X: Perpres 111/2025 & isu LGBTQ+ (text mining)")
    p.add_argument("--since", help="YYYY-MM-DD (inklusif)")
    p.add_argument("--until", help="YYYY-MM-DD (eksklusif)")
    p.add_argument("--window-days", type=int)
    p.add_argument("--limit-per-window", type=int, help="maks tweet per query x jendela (-1 = tanpa batas)")
    p.add_argument("--expand-replies", type=int, help="replies per tweet (0 = nonaktif)")
    p.add_argument("--product", choices=["Latest", "Top"])
    p.add_argument("--cookies", help="path cookies.json")
    p.add_argument("--out", help="path CSV output")
    p.add_argument("--delay-min", type=float, help="jeda minimum antar halaman (detik)")
    p.add_argument("--delay-max", type=float, help="jeda maksimum antar halaman (detik)")
    p.add_argument("--min-replies", type=int, help="ekspansi replies hanya untuk tweet dengan reply_count >= N")
    p.add_argument("--drop-spam", action="store_true", help="buang (bukan sekadar flag) tweet yang dicurigai spam")
    p.add_argument("--raw-username", action="store_true", help="simpan handle mentah (tanpa hash)")
    p.add_argument("--reset-account", action="store_true", help="hapus & daftarkan ulang akun dari cookies.json")
    p.add_argument("--dry-run", action="store_true", help="hanya cetak daftar query")
    p.add_argument("--diagnose", action="store_true", help="uji akun & pencarian (user, query sederhana, query inti)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def apply_overrides(cfg: Config, a: argparse.Namespace) -> Config:
    if a.since: cfg.since = a.since
    if a.until: cfg.until = a.until
    if a.window_days: cfg.window_days = a.window_days
    if a.limit_per_window is not None: cfg.per_window_limit = a.limit_per_window
    if a.expand_replies is not None: cfg.expand_replies_per_tweet = a.expand_replies
    if a.product: cfg.product = a.product
    if a.cookies: cfg.cookies_file = a.cookies
    if a.out: cfg.output_csv = a.out
    if a.delay_min is not None: cfg.delay_min = a.delay_min
    if a.delay_max is not None: cfg.delay_max = a.delay_max
    if a.min_replies is not None: cfg.min_replies_to_expand = a.min_replies
    if a.drop_spam: cfg.drop_spam = True
    if a.raw_username: cfg.hash_username = False
    return cfg


async def diagnose(cfg: Config, api: API) -> None:
    """Uji bertahap untuk memisahkan masalah akun/cookies, library, dan query."""
    qb = QueryBuilder(cfg)
    set_log_level("DEBUG")  # tampilkan pesan internal twscrape (request, "pagination stalled", dll.)

    async def raw_probe(q: str, product: str = "Latest") -> None:
        """Lihat respons mentah SearchTimeline: status HTTP, jumlah entri, dan error dari X."""
        got = 0
        async for r in api.search_raw(q, limit=1, kv={"product": product}):
            got += 1
            obj = r.json()
            print(f"   raw[{product}]: HTTP {r.status_code}, entri={len(api._gql_entries(obj))}, errors={obj.get('errors')}")
            break
        if got == 0:
            print(f"   raw[{product}]: tidak ada halaman berisi entri (X mengembalikan timeline kosong / request ditolak / "
                  "pagination stalled). Lihat baris log DEBUG di atas.")

    async def count(q: str, limit: int = 5) -> list:
        out = []
        async for t in api.search(q, limit=limit, kv={"product": cfg.product}):
            out.append(t)
        return out

    print("\n== 1. Status akun di pool ==")
    for a in await api.pool.accounts_info():
        print("  ", a)

    print("\n== 2. Panggilan GraphQL dasar (user_by_login) ==")
    try:
        u = await api.user_by_login("nasa")
        print("   OK, id =", getattr(u, "id", None) if u else None, "(None = autentikasi/endpoint bermasalah)")
    except Exception as e:
        print("   ERROR:", repr(e))

    print("\n== 2b. Respons mentah SearchTimeline untuk 'Indonesia' ==")
    for product in ("Latest", "Top"):
        try:
            await raw_probe("Indonesia", product)
        except Exception as e:
            print("   ERROR:", repr(e))

    tests = [
        ("3. Search paling sederhana: 'Indonesia'", "Indonesia"),
        ("4. Search topik tanpa operator & tanggal", '"Perpres 111" LGBT'),
        ("5. Query inti 1 jendela (persis seperti script)", qb.build(cfg.query_groups[0], "2026-07-05", "2026-07-08")),
        ("6. Query 'loose' 1 jendela", qb.build(cfg.query_groups[1], "2026-07-05", "2026-07-08")),
    ]
    for title, q in tests:
        print(f"\n== {title} ==\n   query ({len(q)} karakter): {q[:140]}{'...' if len(q) > 140 else ''}")
        try:
            res = await count(q)
            print(f"   -> {len(res)} hasil")
            for t in res[:2]:
                print("     *", (t.rawContent or "")[:90].replace("\n", " "))
        except Exception as e:
            print("   ERROR:", repr(e))

    print(
        "\nCara baca: (2b) HTTP 200 + entri=0 dan errors=None -> X sendiri mengembalikan hasil kosong untuk sesi ini "
        "(akun dibatasi/baru; coba akun lain atau 'Top'). errors berisi pesan -> kirim pesannya. "
        "entri>0 tetapi (3) 0 hasil -> parser twscrape gagal (versi/ perubahan X). "
        "\n(2) None/ERROR -> cookies/akun. (3) 0 -> akun/library/endpoint bermasalah. "
        "(3) ada tapi (4) 0 -> kata kunci/operator. (4) ada tapi (5) 0 -> query inti terlalu rumit/ketat; "
        "pangkas istilah atau pakai grup 'loose'."
    )


async def amain(args: argparse.Namespace) -> None:
    cfg = apply_overrides(Config(), args)
    qb = QueryBuilder(cfg)

    if args.dry_run:
        windows = list(qb.date_windows(cfg.since, cfg.until, cfg.window_days))
        for g in cfg.query_groups:
            print(f"# grup: {g.name} ({len(windows)} jendela)")
            for s, e in windows[:3]:
                print("  ", qb.build(g, s, e))
            if len(windows) > 3:
                print("   ...")
        print(f"\nTotal query: {len(windows) * len(cfg.query_groups)}")
        return

    set_log_level("DEBUG" if args.verbose else "INFO")  # INFO menampilkan pesan "No account available / next available at"
    api = API(cfg.db_path)
    await AuthManager(cfg, api).setup(reset=args.reset_account)

    if args.diagnose:
        await diagnose(cfg, api)
        return

    storage = StorageManager(cfg)
    try:
        engine = ScraperEngine(cfg, api, TweetParser(cfg), storage, qb)
        await engine.run()
    finally:
        storage.close()  # data yang sudah ditulis selalu ter-flush, juga saat Ctrl+C


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        log.warning("Dihentikan pengguna. Jalankan ulang perintah yang sama untuk melanjutkan (resume).")
        sys.exit(130)


if __name__ == "__main__":
    main()