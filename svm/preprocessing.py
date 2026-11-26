"""Text filtering, Indonesian/English X normalization, and sklearn transformer."""
from __future__ import annotations

import re
from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin

ARTIFACT_RE = re.compile(r"python\s+scripts/xpost\.py.*", re.IGNORECASE | re.DOTALL)
HEADER_ARTIFACT_RE = re.compile(
    r"(?:^|\n)\s*(?:###\s+.*|URL:\s*https?://\S+|created:\s*.*)",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#(\w+)")
LAUGH_RE = re.compile(
    r"\b(?:w*k+w+k+\w*|k+w+k+\w*|h*a+h+a+\w*|xixixi+\w*|hehe+\w*|hihi+\w*|o+a*k+w+o+a*k+\w*)\b",
    re.IGNORECASE,
)
ELONGATION_RE = re.compile(r"(.)\1{2,}")
NON_ALPHA_RE = re.compile(r"[^a-z0-9_\s]+")
WHITESPACE_RE = re.compile(r"\s+")

SPAM_PATTERNS = re.compile(
    r"(?:shopee\.co\.id|shp\.ee|tokopedia\.link|s\.shopee|\bslot\b|\bgacor\b|"
    r"\bmaxwin\b|\bpinjol\b|\btogel\b|\bcasino\b|\bsbobet\b|\bbandar\b)",
    re.IGNORECASE,
)

FOREIGN_NOISE_RE = re.compile(
    r"\b(?:gobierno|siempre|hacernos|pelada|porque|estoy|seguro|pudiera|"
    r"paraguay|espa[nñ]a|obrigado|voc[eê]|tamb[eé]m|n[aã]o)\b",
    re.IGNORECASE,
)

ID_ANCHOR_RE = re.compile(
    r"\b(?:yang|yg|dan|di|ke|dari|ini|itu|tidak|gak|ga|gk|ada|bisa|untuk|"
    r"dengan|sama|kalau|kalo|orang|negara|perpres|prabowo|pemerintah|kaum|"
    r"ancaman|rakyat|indonesia|mau|jadi|udah|aja|banget|bukan)\b",
    re.IGNORECASE,
)

EMOJI_TOKEN_MAP: dict[str, str] = {
    # Laughter / sarcasm / amusement
    "🤣": " emo_tawa ",
    "😂": " emo_tawa ",
    "😹": " emo_tawa ",
    "😆": " emo_tawa ",
    "😝": " emo_tawa ",
    "🤪": " emo_tawa ",
    "🤭": " emo_tawa ",
    "😅": " emo_tawa ",
    "😁": " emo_tawa ",
    "😜": " emo_tawa ",
    # Crying / distress / sadness
    "😭": " emo_nangis ",
    "😢": " emo_nangis ",
    "😥": " emo_nangis ",
    "😓": " emo_nangis ",
    "😔": " emo_sedih ",
    "💔": " emo_sedih ",
    "🥺": " emo_sedih ",
    "😞": " emo_sedih ",
    "🥀": " emo_sedih ",
    "🥲": " emo_sedih ",
    "🥹": " emo_sedih ",
    # Anger / outrage
    "😡": " emo_marah ",
    "🤬": " emo_marah ",
    "💢": " emo_marah ",
    "🔥": " emo_marah ",
    "👊": " emo_marah ",
    "😤": " emo_marah ",
    "🖕": " emo_marah ",
    # Fear / anxiety
    "😰": " emo_takut ",
    "😱": " emo_takut ",
    "😨": " emo_takut ",
    "🫣": " emo_takut ",
    "⚠": " emo_takut ",
    "🚨": " emo_takut ",
    "😬": " emo_takut ",
    # Disgust
    "🤢": " emo_jijik ",
    "🤮": " emo_jijik ",
    "💩": " emo_jijik ",
    "🤡": " emo_jijik ",
    # Surprise / confusion
    "😲": " emo_kaget ",
    "😳": " emo_kaget ",
    "🤔": " emo_kaget ",
    "👀": " emo_kaget ",
    "❓": " emo_kaget ",
    "⁉": " emo_kaget ",
    "🤯": " emo_kaget ",
    "🧐": " emo_kaget ",
    # Positive / support
    "👍": " emo_positif ",
    "👏": " emo_positif ",
    "🙌": " emo_positif ",
    "💯": " emo_positif ",
    "🥰": " emo_positif ",
    "✨": " emo_positif ",
    "🙏": " emo_positif ",
    "❤": " emo_positif ",
    "💚": " emo_positif ",
    "🫶": " emo_positif ",
    "🫡": " emo_positif ",
}

SLANG_DICT: dict[str, str] = {
    "yg": "yang",
    "yng": "yang",
    "ga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "enggak": "tidak",
    "kagak": "tidak",
    "kgk": "tidak",
    "ndak": "tidak",
    "tak": "tidak",
    "tdk": "tidak",
    "g": "tidak",
    "udh": "sudah",
    "udah": "sudah",
    "sdh": "sudah",
    "dah": "sudah",
    "blm": "belum",
    "blom": "belum",
    "krn": "karena",
    "karna": "karena",
    "soalnya": "karena",
    "tp": "tetapi",
    "tpi": "tetapi",
    "bgt": "banget",
    "bgtt": "banget",
    "bener": "benar",
    "bnr": "benar",
    "beneran": "benar",
    "kl": "kalau",
    "klo": "kalau",
    "kalo": "kalau",
    "kalok": "kalau",
    "sm": "sama",
    "sma": "sama",
    "dgn": "dengan",
    "dg": "dengan",
    "utk": "untuk",
    "untk": "untuk",
    "dlm": "dalam",
    "pdhl": "padahal",
    "emg": "memang",
    "emang": "memang",
    "mmg": "memang",
    "skrg": "sekarang",
    "skg": "sekarang",
    "bkn": "bukan",
    "jgn": "jangan",
    "knp": "kenapa",
    "knpa": "kenapa",
    "gmn": "bagaimana",
    "gmna": "bagaimana",
    "gimana": "bagaimana",
    "bikin": "membuat",
    "liat": "lihat",
    "lht": "lihat",
    "gw": "saya",
    "gue": "saya",
    "gua": "saya",
    "ane": "saya",
    "aku": "saya",
    "ak": "saya",
    "sy": "saya",
    "lu": "kamu",
    "lo": "kamu",
    "elu": "kamu",
    "elo": "kamu",
    "klean": "kalian",
    "klian": "kalian",
    "kalian": "kalian",
    "wowo": "prabowo",
    "prabski": "prabowo",
    "bowo": "prabowo",
    "elgbt": "lgbt",
    "elggbtee": "lgbt",
    "elgbttee": "lgbt",
    "lgbtq": "lgbt",
    "lgbtqi": "lgbt",
    "lgbtqia": "lgbt",
    "omon": "omong",
    "omon2": "omong",
    "sbnrnya": "sebenarnya",
    "sbnernya": "sebenarnya",
    "sprti": "seperti",
    "kyk": "seperti",
    "kayak": "seperti",
    "kek": "seperti",
    "mreka": "mereka",
    "mrk": "mereka",
    "org": "orang",
    "org2": "orang",
    "hrs": "harus",
    "hrus": "harus",
    "bs": "bisa",
    "bsa": "bisa",
    "jd": "jadi",
    "jdi": "jadi",
    "tau": "tahu",
    "tw": "tahu",
    "dpt": "dapat",
    "jg": "juga",
    "jga": "juga",
    "thn": "tahun",
    "uu": "undang_undang",
    "aja": "saja",
    "aj": "saja",
    "sj": "saja",
    "doang": "saja",
    "dng": "dong",
    "smpe": "sampai",
    "ampe": "sampai",
    "sampe": "sampai",
    "bnyk": "banyak",
    "byk": "banyak",
    "bbrp": "beberapa",
    "brp": "berapa",
    "dr": "dari",
    "dri": "dari",
    "pd": "pada",
    "kpd": "kepada",
    "tsb": "tersebut",
    "ttg": "tentang",
    "tntg": "tentang",
    "soal": "tentang",
    "skrng": "sekarang",
    "nnt": "nanti",
    "ntar": "nanti",
    "ntr": "nanti",
    "trus": "terus",
    "trs": "terus",
    "msh": "masih",
    "msih": "masih",
    "dl": "dulu",
    "dlu": "dulu",
    "lg": "lagi",
    "lgi": "lagi",
    "gt": "begitu",
    "gtu": "begitu",
    "gitu": "begitu",
    "gini": "begini",
    "gni": "begini",
    "bgtu": "begitu",
    "kpn": "kapan",
    "dmn": "dimana",
    "dmna": "dimana",
    "kmrn": "kemarin",
    "bsk": "besok",
    "pke": "pakai",
    "pake": "pakai",
    "pk": "pakai",
    "mksd": "maksud",
    "mksdnya": "maksudnya",
    "jls": "jelas",
    "bgtnya": "banget",
    "pny": "punya",
    "py": "punya",
    "bwt": "buat",
    "bt": "buat",
    "bg": "bagi",
    "stuju": "setuju",
    "stj": "setuju",
    "bgus": "bagus",
    "bgs": "bagus",
    "jlk": "jelek",
    "prh": "parah",
    "gblg": "goblok",
    "gblk": "goblok",
    "bgo": "bego",
    "tll": "tolol",
    "anj": "anjing",
    "anjg": "anjing",
    "anjir": "anjing",
    "njir": "anjing",
    "bngst": "bangsat",
    "bgst": "bangsat",
    "kntl": "kontol",
    "mmk": "memek",
    "tai": "taik",
    "sakit": "sakit",
    "skt": "sakit",
    "tkut": "takut",
    "tkt": "takut",
    "kcm": "kejam",
    "kcewa": "kecewa",
    "kcwa": "kecewa",
    "sdh2": "sudah",
    "mngkn": "mungkin",
    "mgkn": "mungkin",
    "plg": "paling",
    "slh": "salah",
    "mnrt": "menurut",
    "mnurut": "menurut",
    "pemerintah": "pemerintah",
    "pmrnth": "pemerintah",
    "pemerentah": "pemerintah",
    "pres": "presiden",
    "rkyt": "rakyat",
    "masy": "masyarakat",
    "indo": "indonesia",
    "konoha": "indonesia",
    "wakanda": "indonesia",
    "boti": "boti",
    "homok": "homo",
    "banci": "waria",
    "bencong": "waria",
    "sjw": "sjw",
    "fr": "nyata",
    "ngl": "jujur",
    "tbh": "jujur",
    "imo": "menurut_saya",
    "wtf": "apa_an",
    "omg": "astaga",
    "smh": "geleng_kepala",
    "lol": "tawa_wkwk",
    "lmao": "tawa_wkwk",
    "lmfao": "tawa_wkwk",
    "rofl": "tawa_wkwk",
    "btw": "ngomong_ngomong",
    "irl": "nyata",
    "ytta": "yang_tahu_saja",
    "asbun": "asal_bunyi",
    "fomo": "ikut_ikutan",
    "baper": "bawa_perasaan",
    "mbg": "makan_bergizi_gratis",
    "ham": "hak_asasi_manusia",
    "dpr": "dpr",
    "kuhp": "kuhp",
    "mk": "mahkamah_konstitusi",
    "tni": "tni",
    "polri": "polri",
    "akmil": "akmil",
    "npd": "narsistik",
    "bacot": "omong_kosong",
    "bcot": "omong_kosong",
    "dungu": "bodoh",
    "tolol": "tolol",
    "goblog": "goblok",
    "goblok": "goblok",
    "idiot": "idiot",
    "bloon": "bodoh",
    "kocak": "lucu",
    "ngakak": "tawa_wkwk",
    "wkwk": "tawa_wkwk",
    "wkwkwk": "tawa_wkwk",
    "kwkw": "tawa_wkwk",
    "haha": "tawa_wkwk",
    "hehe": "tawa_wkwk",
    "wkw": "tawa_wkwk",
}

NEGATION_WORDS = {"tidak", "bukan", "jangan", "belum", "kurang", "tanpa", "not", "never"}

STOPWORDS_ID_EN = {
    "yang",
    "di",
    "ke",
    "dari",
    "dan",
    "atau",
    "ini",
    "itu",
    "pada",
    "untuk",
    "dengan",
    "adalah",
    "sebagai",
    "dalam",
    "oleh",
    "saja",
    "juga",
    "sudah",
    "akan",
    "telah",
    "tersebut",
    "bahwa",
    "agar",
    "supaya",
    "serta",
    "para",
    "sebuah",
    "suatu",
    "itu",
    "iya",
    "ya",
    "nih",
    "sih",
    "dong",
    "deh",
    "lah",
    "kah",
    "tah",
    "mah",
    "tuh",
    "pun",
    "the",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "to",
    "of",
    "in",
    "and",
    "a",
    "an",
    "that",
    "this",
    "for",
    "on",
    "with",
    "as",
    "at",
    "by",
    "it",
    "its",
    "from",
    "or",
    "so",
    "if",
    "they",
    "them",
    "we",
    "you",
    "i",
    "my",
    "your",
    "their",
    "our",
    "he",
    "she",
    "his",
    "her",
}

PROTECTED_ROOTS = {
    "perpres",
    "presiden",
    "prabowo",
    "militer",
    "nonmiliter",
    "negara",
    "agama",
    "pelangi",
    "setuju",
    "kecewa",
    "bahaya",
    "serius",
    "fitnah",
    "ancaman",
    "kebijakan",
    "pemerintah",
    "korupsi",
    "konflik",
    "masyarakat",
    "peraturan",
    "kasihan",
    "prihatin",
    "khawatir",
    "terkejut",
    "bingung",
    "menjijikkan",
    "persekusi",
    "pengalihan",
    "undang_undang",
    "tawa_wkwk",
    "omong_kosong",
    "hak_asasi_manusia",
    "lgbt",
    "indonesia",
    "rakyat",
    "hukum",
    "pasal",
    "pidana",
    "kampanye",
    "buzzer",
    "normalisasi",
    "diskriminasi",
    "kemanusiaan",
    "perempuan",
    "keluarga",
    "ketimuran",
    "moral",
    "budaya",
    "seksual",
    "orientasi",
    "minoritas",
    "demo",
    "demonstrasi",
    "politik",
    "penguasa",
    "rezim",
    "anjing",
    "bangsat",
    "goblok",
    "tolol",
    "bodoh",
    "sampah",
    "hama",
    "jijik",
    "takut",
    "sedih",
    "marah",
    "senang",
    "kaget",
    "heran",
    "aneh",
    "lucu",
    "ironi",
    "sarkas",
    "dukung",
    "tolak",
    "bagus",
    "hebat",
    "mantap",
    "keren",
    "parah",
    "rusak",
    "hancur",
    "panik",
    "ngeri",
    "miris",
    "menangis",
    "sudi",
}


def strip_scraper_artifacts(text: str) -> str:
    """Remove leaked xpost/fxtwitter CLI dumps from scraped raw text."""
    if not isinstance(text, str):
        return ""
    cleaned = ARTIFACT_RE.sub("", text)
    cleaned = HEADER_ARTIFACT_RE.sub(" ", cleaned)
    return cleaned.strip()


def is_valid_opinion_tweet(raw_text: str, is_spam_suspect: bool = False) -> tuple[bool, str]:
    """Filter spam, affiliate links, empty mention-only tweets, and foreign noise."""
    if is_spam_suspect:
        return False, "spam_suspect"

    stripped = strip_scraper_artifacts(raw_text)
    if not stripped:
        return False, "empty_after_artifact_strip"

    if SPAM_PATTERNS.search(stripped):
        return False, "spam_keyword_or_link"

    no_url_mention = URL_RE.sub(" ", stripped)
    no_url_mention = MENTION_RE.sub(" ", no_url_mention)
    alpha_chars = re.findall(r"[a-zA-Z]", no_url_mention)
    if len(alpha_chars) < 3:
        return False, "empty_or_too_short"

    if FOREIGN_NOISE_RE.search(no_url_mention) and not ID_ANCHOR_RE.search(no_url_mention):
        return False, "foreign_noise"

    return True, "ok"


def _replace_emojis(text: str) -> str:
    text = text.replace("\ufe0f", "").replace("\ufe0e", "")
    for emoji_char, token in EMOJI_TOKEN_MAP.items():
        if emoji_char in text:
            text = text.replace(emoji_char, token)
    return text


def stem_indonesian(token: str) -> str:
    """Lightweight deterministic affix stripper preserving key domain/emotion roots."""
    if (
        len(token) <= 4
        or "_" in token
        or token in PROTECTED_ROOTS
        or token.startswith("emo_")
        or token.startswith("tawa_")
    ):
        return token

    word = token
    # Strip common possessive/particle suffixes carefully
    for suffix in ("nya", "lah", "kah", "tah"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break

    if word in PROTECTED_ROOTS:
        return word

    # Strip derivational suffixes
    for suffix in ("kan", "an", "i"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            cand = word[: -len(suffix)]
            if cand in PROTECTED_ROOTS or len(cand) >= 4:
                word = cand
            break

    if word in PROTECTED_ROOTS:
        return word

    # Strip common prefixes safely
    for prefix in ("meng", "meny", "mend", "ment", "mem", "men", "peng", "peny", "pem", "pen", "ber", "ter", "per", "di", "ke", "se"):
        if word.startswith(prefix) and len(word) - len(prefix) >= 4:
            cand = word[len(prefix) :]
            if cand in PROTECTED_ROOTS:
                return cand
            if prefix in ("di", "ber", "ter", "se") and len(cand) >= 4:
                word = cand
                break
    return word


def normalize_for_lexicon(text: str) -> str:
    """Normalize text while retaining punctuation hints for the rule-based labeler."""
    cleaned = strip_scraper_artifacts(text)
    cleaned = _replace_emojis(cleaned)
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = MENTION_RE.sub(" ", cleaned)
    cleaned = HASHTAG_SYMBOL_RE.sub(r" \1 ", cleaned)
    cleaned = cleaned.lower()
    cleaned = LAUGH_RE.sub(" tawa_wkwk ", cleaned)
    cleaned = ELONGATION_RE.sub(r"\1", cleaned)
    tokens = re.findall(r"[a-z0-9_]+|[!?]+", cleaned)
    norm_tokens: list[str] = []
    for tok in tokens:
        if tok.startswith("!") or tok.startswith("?"):
            norm_tokens.append(tok)
        else:
            norm_tokens.append(SLANG_DICT.get(tok, tok))
    return WHITESPACE_RE.sub(" ", " ".join(norm_tokens)).strip()


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline for TF-IDF + SVM: clean, normalize, negation-join, stem."""
    cleaned = strip_scraper_artifacts(text)
    if not cleaned:
        return ""
    cleaned = _replace_emojis(cleaned)
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = MENTION_RE.sub(" ", cleaned)
    cleaned = HASHTAG_SYMBOL_RE.sub(r" \1 ", cleaned)
    cleaned = cleaned.lower()
    cleaned = LAUGH_RE.sub(" tawa_wkwk ", cleaned)
    cleaned = ELONGATION_RE.sub(r"\1", cleaned)
    cleaned = NON_ALPHA_RE.sub(" ", cleaned)

    raw_tokens = cleaned.split()
    mapped_tokens = [SLANG_DICT.get(tok, tok) for tok in raw_tokens]

    processed: list[str] = []
    i = 0
    while i < len(mapped_tokens):
        tok = mapped_tokens[i]
        if tok in NEGATION_WORDS:
            # Look ahead up to 1 non-stopword content token to form negation bigram
            j = i + 1
            while j < len(mapped_tokens) and mapped_tokens[j] in STOPWORDS_ID_EN and j - i <= 2:
                j += 1
            if j < len(mapped_tokens) and mapped_tokens[j] not in NEGATION_WORDS:
                next_tok = stem_indonesian(mapped_tokens[j])
                processed.append(f"{tok}_{next_tok}")
                i = j + 1
                continue
            processed.append(tok)
            i += 1
            continue

        if tok in STOPWORDS_ID_EN or len(tok) <= 1 or tok.isdigit():
            i += 1
            continue

        stemmed = stem_indonesian(tok)
        if stemmed not in STOPWORDS_ID_EN and len(stemmed) > 1:
            processed.append(stemmed)
        i += 1

    return " ".join(processed).strip()


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer calling `preprocess_text`."""

    def fit(self, X: Iterable[str], y: object = None) -> "TextCleanerTransformer":
        return self

    def transform(self, X: Iterable[str]) -> list[str]:
        return [preprocess_text(str(item) if item is not None else "") for item in X]
