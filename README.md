pip install -r requirements.txt

# 1. Tambah akun (cookies lebih andal daripada login password)
python x_scraper.py add-account --username akun_scrape --cookies "auth_token=...; ct0=..."

# 2. Scrape sesuai target
python x_scraper.py tweet https://x.com/NASA/status/123... --limit 500
python x_scraper.py search "#topik lang:id since:2026-01-01" --limit 300 --expand-replies 20
python x_scraper.py user nasa --limit 100

# 3. Kombinasi sekaligus
python x_scraper.py batch --config targets.json --out output/dataset.csv
