# Scraper X: Perpres 111/2025 & Isu LGBTQ+ (Text Mining)

## 1. Instalasi

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Menyiapkan cookies X (aman)

Script memakai **cookies sesi** (bukan password), jadi tidak perlu login otomatis yang rawan challenge.

1. **Gunakan akun sekunder** khusus scraping, jangan akun utama. Scraping dengan login melanggar ToS X
   dan akun bisa dibatasi atau diblokir.
2. Login ke https://x.com di browser (sebaiknya jendela/profil terpisah).
3. Ambil dua cookie, pilih salah satu cara:
   - **DevTools:** `F12` → tab **Application** (Chrome) / **Storage** (Firefox) → **Cookies** → `https://x.com`
     → salin nilai `auth_token` dan `ct0`.
   - **Ekstensi Cookie-Editor:** buka x.com → *Export* → *JSON*, simpan sebagai `cookies.json`
     (format ini langsung didukung script).
4. Buat `cookies.json` di folder script (jika tidak memakai ekspor ekstensi):

   ```json
   { "username": "akun_scrape", "auth_token": "ISI_AUTH_TOKEN", "ct0": "ISI_CT0" }
   ```

   Untuk beberapa akun (rotasi otomatis saat kena rate limit), gunakan array berisi objek seperti di atas.
5. **Jangan klik Logout di browser** setelah menyalin cookies. Logout membatalkan `auth_token`.
   Cukup tutup jendelanya.
6. **Keamanan:**
   - `cookies.json`, `accounts.db`, dan folder `output/` sudah ada di `.gitignore`. Jangan pernah di-commit atau dibagikan.
   - Cookies bisa kedaluwarsa. Kalau muncul error autentikasi, ambil cookies baru lalu jalankan dengan `--reset-account`.

## 3. Menjalankan

```bash
# Cek query yang akan dijalankan (tanpa scraping)
python scraper_x_perpres111.py --dry-run

# Uji coba kecil dulu (1 jendela pendek, tanpa replies)
python scraper_x_perpres111.py --since 2026-07-05 --until 2026-07-08 --limit-per-window 100 --expand-replies 0

# Jalankan penuh (aman di-Ctrl+C, jalankan ulang perintah yang sama untuk resume)
python scraper_x_perpres111.py
```

Opsi lain: `--window-days`, `--product Top|Latest`, `--drop-spam`, `--raw-username`, `--out`, `--verbose`.
Semua default ada di dataclass `Config` (bagian atas script).

**Output** (folder `output/`):

| File | Isi |
|---|---|
| `perpres111_raw.csv` | Dataset utama, UTF-8-SIG (aman dibuka di Excel) |
| `perpres111_raw.jsonl` | Salinan JSONL (nonaktifkan dengan `output_jsonl=None`) |
| `.done_queries.txt` | Query/jendela yang sudah selesai (dipakai untuk resume) |
| `.salt` | Salt hash username (simpan; kalau hilang, hash lama tidak bisa direproduksi) |

Baca di pandas dengan `pd.read_csv("output/perpres111_raw.csv", dtype={"tweet_id": str, "conversation_id": str})`.

## 4. Skema kolom

`tweet_id`, `created_at` (ISO 8601, WIB/UTC+7), `username` (hash SHA-256 + salt), `raw_text`,
`reply_count`, `retweet_count`, `like_count`, `quote_count`, `conversation_id`, `search_query_used`,
lalu tambahan: `scrape_type` (search/reply), `root_tweet_id`, `is_reply`, `in_reply_to_tweet_id`, `lang`,
`hashtags`, `is_spam_suspect`, `sentiment_label` (kosong, untuk anotasi manual), `scraped_at`.

## 5. Menyesuaikan query

Query dibangun dari `Config.query_groups`: `(istilah kebijakan) (istilah isu) lang:id -filter:retweets since: until:`.
Spasi antar-grup berarti **AND**, sedangkan di dalam grup memakai **OR**.

**Kalau volume terlalu sedikit**
- Aktifkan/perluas grup `loose` (mis. tambah `"Prabowo"`, `"Yusril"`, `"Kemhan"` pada istilah kebijakan).
- Tambahkan variasi yang Anda temukan di data (hashtag, singkatan, bahasa gaul) ke `policy_terms` / `issue_terms`.
- Longgarkan `lang:id` (set `lang=""`), karena tweet pendek atau campur bahasa kadang terdeteksi `und`.
- Perluas rentang `since`/`until` dan coba `--product Top`.
- Perkecil `window_days` (mis. 1) supaya tiap jendela tidak mentok batas kedalaman pencarian X.

**Kalau terlalu banyak noise**
- Hapus istilah yang terlalu umum dari `policy_terms` (mis. `"pertahanan negara"`, `"Perpres"` di grup `loose`).
- Tambahkan operator negatif langsung di `QueryBuilder.build`, misalnya `-slot -judi -kripto`, atau `-from:akun_berita`
  untuk membuang akun media agar yang tersisa opini pengguna.
- Gunakan `--drop-spam`, atau filter `is_spam_suspect == False` saat analisis.
- Ambil sampel 100 baris acak dan hitung berapa persen yang relevan. Itu patokan presisi query.

## 6. Catatan penting

- Anotasi: `sentiment_label` perlu diisi manual. Tentukan sejak awal bahwa sentimen diukur **terhadap Perpres/kebijakan**,
  bukan terhadap kelompok LGBTQ+, dan sebaiknya ada lebih dari satu anotator untuk mengukur kesepakatan.
- Sebelum melatih Multinomial NB, buang duplikat dan near-duplicate (tweet copy-paste) dan lakukan split train/test
  setelah itu, supaya tidak terjadi kebocoran data.
- Privasi: username di-hash, displayname dan profil tidak disimpan. Jangan mempublikasikan teks tweet utuh beserta handle.

## 7. Membersihkan dump teks

Untuk menggabungkan dan menormalisasi empat dump teks menjadi CSV sesuai skema di atas:

```bash
python clean_data.py
```

Default membaca `replies-output.txt`, `replies.txt`, `tambahan-replies.txt`, dan `tweet-batch.txt`,
lalu menulis `cleaned_data.csv`. Baris dideduplikasi berdasarkan `tweet_id`. Username di-hash memakai
salt `output/.salt` yang sama dengan scraper; simpan file salt tersebut agar pseudonim tetap konsisten.
Kolom yang tidak tersedia pada dump teks (misalnya `lang` dan `search_query_used`) dibiarkan kosong.
Gunakan `--drop-spam` untuk membuang baris yang ditandai spam atau `--out nama.csv` untuk mengganti nama keluaran.
