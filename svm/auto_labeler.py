"""Multi-signal Indonesian/English emotion auto-labeler for Perpres 111/2025 tweets."""
from __future__ import annotations

import re
from dataclasses import dataclass

from svm.preprocessing import normalize_for_lexicon, strip_scraper_artifacts

CLASSES_9 = [
    "marah",
    "sarkas",
    "kecewa",
    "senang",
    "takut",
    "jijik",
    "sedih",
    "terkejut",
    "netral",
]

CLASSES_EKMAN7 = [
    "marah",
    "senang",
    "sedih",
    "takut",
    "jijik",
    "terkejut",
    "netral",
]

CLASSES_MAIN5 = [
    "marah",
    "sedih",
    "takut",
    "senang",
    "netral",
]

# Layer A: Multi-word idioms and domain phrases (regex -> (emotion, weight))
PHRASE_RULES: list[tuple[re.Pattern[str], str, float]] = [
    # --- MARAH (anger, outrage, distraction accusations, political fury) ---
    (re.compile(r"\b(?:pengalihan isu|ngepecah belah|memecah belah)\b"), "marah", 3.2),
    (re.compile(r"\b(?:mengalihkan|alihkan|alihin)\s+(?:perhatian|isu|fokus)\b"), "marah", 3.2),
    (re.compile(r"\b(?:kambing hitam|tumbal|ditumbalkan|tumbalin)\b"), "marah", 3.0),
    (re.compile(r"\b(?:harus diturunin|lengserkan|turunkan prabowo|stop ruling)\b"), "marah", 3.2),
    (re.compile(r"\b(?:omong kosong|bacot|mulutnya|congornya|tutup mulut)\b"), "marah", 2.8),
    (re.compile(r"\b(?:tidak tahu malu|tidak punya malu|muka tembok)\b"), "marah", 2.9),
    (re.compile(r"\b(?:penjahat ham|anti ham|tidak mengerti ham|langgar ham|represif)\b"), "marah", 2.8),
    (re.compile(r"\b(?:penipu|manipulatif|psikopat|korup|perampokan|dzalim|zalim)\b"), "marah", 2.8),
    (re.compile(r"\b(?:goblok|tolol|bangsat|anjing|babi|keparat|brengsek|bajingan|kontol|dobol)\b"), "marah", 2.7),
    (re.compile(r"\b(?:muak|geram|murka|emosi|kesal|benci banget|yaelah)\b"), "marah", 2.6),
    (re.compile(r"\b(?:melawan|lawan terus|perangi|basmi|berantas|tangkap semua|hajar)\b"), "marah", 2.3),
    (re.compile(r"\b(?:gila kekuasaan|gila hormat|rezim bobrok|pemerintah bobrok)\b"), "marah", 3.0),
    (re.compile(r"\b(?:anak haram|bikin emosi|naik darah|sakit jiwa)\b"), "marah", 2.8),
    # --- SARKAS (irony, mockery, political hypocrisy, satirical laughter) ---
    (re.compile(r"\bgiliran\s+(?:pas\s+)?kampanye\b"), "sarkas", 3.2),
    (re.compile(r"\btumben\s+(?:benar|bener|pintar|waras)\b"), "sarkas", 2.8),
    (re.compile(r"\bsame same but different\b|\bsi paling\b"), "sarkas", 2.9),
    (re.compile(r"\bmalah backfire\b|\boh irony\b|\bpantesan\b"), "sarkas", 2.8),
    (re.compile(r"\bhidup jokowi\b"), "sarkas", 2.8),
    (re.compile(r"\b(?:playing victim|bikin sendiri nuduh|posting sendiri)\b"), "sarkas", 2.9),
    (re.compile(r"\b(?:closeted gay|mas didit|teddy|pasangan.*paris|pakai dasi pelangi)\b"), "sarkas", 2.9),
    (re.compile(r"\b(?:ceraikan dulu|suamimu|bapaknya sendiri|ngaca dulu|bercermin)\b"), "sarkas", 2.8),
    (re.compile(r"\b(?:selamat menikmati|kado istimewa|kado terindah|mantap kali bah)\b"), "sarkas", 2.5),
    (re.compile(r"\b(?:non brain|otak udang|iq jongkok|sdm rendah|buzzer bayaran|caper)\b"), "sarkas", 2.6),
    (re.compile(r"\b(?:paling suci|sok suci|suci banget|malaikat|open minded)\b"), "sarkas", 2.6),
    (re.compile(r"\b(?:lucu banget|kocak banget|lawak|dagelan|komedi|sirkus)\b"), "sarkas", 2.4),
    (re.compile(r"\byang_tahu_saja\b|\bapa yang kamu harepin\b"), "sarkas", 2.4),
    # --- KECEWA (misplaced priorities, hollow policy, unfulfilled expectations) ---
    (re.compile(r"\b(?:urusin gaji|urus dulu|mending urus|fokus ke ekonomi|rupiah anjlok|set our priorities)\b"), "kecewa", 3.0),
    (re.compile(r"\bkurang kerjaan\b|\btidak ada kerjaan\b"), "kecewa", 2.9),
    (re.compile(r"\b(?:omong doang|omon doang|cuma omong|tidak ada sanksi|tanpa sanksi|macan ompong)\b"), "kecewa", 3.0),
    (re.compile(r"\b(?:harusnya pemerintah|seharusnya pemerintah|mending belajar|belajar lagi)\b"), "kecewa", 2.7),
    (re.compile(r"\bwaktunya\s+(?:tidak|ga|gak)\s+tepat\b"), "kecewa", 2.9),
    (re.compile(r"\b(?:buang anggaran|buang waktu|buang duit|tidak berguna|percuma|sia sia)\b"), "kecewa", 2.6),
    (re.compile(r"\b(?:gagal fokus|salah sasaran|salah fokus|tidak relevan|receh)\b"), "kecewa", 2.6),
    (re.compile(r"\b(?:sayang sekali|mengecewakan|kecewa berat|bikin kecewa|hilang respek)\b"), "kecewa", 3.0),
    (re.compile(r"\b(?:mbg tidak jelas|korupsi malah|lapangan kerja|kemiskinan|di gusur|digusur)\b"), "kecewa", 2.5),
    # --- SENANG (approval, agreement, praise, relief, support) ---
    (re.compile(r"\b(?:saya setuju|kami setuju|aku setuju|setuju banget|setuju pak|sangat setuju)\b"), "senang", 3.3),
    (re.compile(r"\b(?:dukung penuh|mendukung penuh|kami dukung|siap dukung|setuju sekali)\b"), "senang", 3.3),
    (re.compile(r"\b(?:kebijakan yang benar|keputusan tepat|langkah tepat|langkah bagus|hal positif)\b"), "senang", 3.1),
    (re.compile(r"\b(?:terima kasih pak|terima kasih presiden|hidup prabowo|mantap pak|bagus pak)\b"), "senang", 3.0),
    (re.compile(r"\b(?:alhamdulillah|syukurlah|puji tuhan|bersyukur|akhirnya ada|senang hati)\b"), "senang", 3.0),
    (re.compile(r"\b(?:semoga istiqomah|semoga terlaksana|gas terus|lanjutkan pak|keren banget)\b"), "senang", 2.8),
    (re.compile(r"\b(?:menjaga moral|selamatkan generasi|lindung[ia] anak|nilai ketimuran)\b"), "senang", 2.3),
    # --- TAKUT (fear, worry, threat anxiety, persecution risk) ---
    (re.compile(r"\bkonflik horizontal\b|\bslippery slope\b"), "takut", 3.1),
    (re.compile(r"\b(?:tetap mengkhawatirkan|sangat mengkhawatirkan|bikin khawatir|cemas|takutnya)\b"), "takut", 3.2),
    (re.compile(r"\bancaman nyata\b|\bbahaya besar\b|\bmerusak generasi\b|\bsomething worse\b"), "takut", 2.8),
    (re.compile(r"\b(?:dipersekusi|persekusi|razia|dirazia|kriminalisasi|ditangkap|dipukul|kekerasan)\b"), "takut", 2.9),
    (re.compile(r"\b(?:tidak aman|merasa terancam|waspada|hati hati|bahaya laten|darurat|dibunuh|saling bunuh)\b"), "takut", 2.8),
    (re.compile(r"\b(?:ngeri banget|seram banget|mengerikan|menakutkan|panik)\b"), "takut", 3.0),
    (re.compile(r"\b(?:azab|kiamat|kehancuran|bencana moral|rusak moral)\b"), "takut", 2.5),
    # --- JIJIK (disgust, revulsion, moral contempt, dehumanization) ---
    (re.compile(r"\b(?:amit amit|tidak sudi|ga sudi|gak sudi|ogah banget|mual lihat)\b"), "jijik", 3.3),
    (re.compile(r"\b(?:manusia sampah|kaum sampah|nyampah banget|sampah masyarakat)\b"), "jijik", 3.3),
    (re.compile(r"\b(?:menjijikkan|sejijik itu|bikin jijik|jijik banget|jijik sekali|najis)\b"), "jijik", 3.4),
    (re.compile(r"\b(?:penyakit masyarakat|penyakit menular|kelainan jiwa|gangguan jiwa|sakit mental)\b"), "jijik", 2.9),
    (re.compile(r"\b(?:hama|kaum taik|kaum laknat|laknatullah|kaum sodo[mt]|perilaku menyimpang)\b"), "jijik", 3.0),
    (re.compile(r"\b(?:kotor banget|permainan kotor|bejat banget|memalukan|aib bangsa)\b"), "jijik", 2.7),
    # --- SEDIH (sadness, sympathy, heartbreak, pity) ---
    (re.compile(r"\b(?:kasihan banget|kasihan sekali|sungguh kasihan|turut prihatin|sangat prihatin)\b"), "sedih", 3.2),
    (re.compile(r"\b(?:genuinely horrible|i m so sorry|im so sorry|so sad|heartbreaking)\b"), "sedih", 3.2),
    (re.compile(r"\b(?:tidak berdaya|tak berdaya|dibully|dirundung|korban terus|tertindas)\b"), "sedih", 2.8),
    (re.compile(r"\b(?:bikin nangis|mau nangis|menangis|air mata|sedih banget|pilu|miris banget)\b"), "sedih", 3.2),
    (re.compile(r"\b(?:innalillahi|duka cita|menyedihkan|memilukan|nasib rakyat|nasib minoritas)\b"), "sedih", 2.9),
    (re.compile(r"\b(?:hancur hati|patah hati|nyesek|nestapa|terpuruk)\b"), "sedih", 3.0),
    # --- TERKEJUT (surprise, disbelief, sudden timing confusion) ---
    (re.compile(r"\b(?:kok tiba tiba|kenapa tiba tiba|tiba tiba banget|mendadak banget)\b"), "terkejut", 3.2),
    (re.compile(r"\b(?:baru heboh|kenapa baru heboh|anehnya kenapa|kok bisa|bisa bisanya)\b"), "terkejut", 3.0),
    (re.compile(r"\b(?:jaman apa ini|zaman apa ini|seriusan ini|benarkah ini|beneran nih)\b"), "terkejut", 3.1),
    (re.compile(r"\b(?:apa tidak kebalik|apa ga kebalik|ini tah yang dimaksud|maksudnya gimana)\b"), "terkejut", 2.9),
    (re.compile(r"\b(?:kaget banget|terkejut|syok|astaga|ya ampun|masa sih|hah serius)\b"), "terkejut", 2.9),
    (re.compile(r"\b(?:tidak nyangka|ga nyangka|gak nyangka|diluar nalar|tidak masuk akal)\b"), "terkejut", 2.8),
    # --- NETRAL (factual reporting, statutory references) ---
    (re.compile(r"\b(?:undang_undang nomor|pasal \d+ ayat \d+|peraturan presiden nomor 111)\b"), "netral", 2.6),
    (re.compile(r"\b(?:kebijakan umum pertahanan negara|jakumhaneg|tahun 2025 2029)\b"), "netral", 2.0),
]

# Layer B: Unigram lexicons (token -> weight)
UNIGRAM_LEXICON: dict[str, dict[str, float]] = {
    "marah": {
        "marah": 2.2, "emosi": 1.8, "kesal": 1.8, "sebel": 1.7, "geram": 2.2, "murka": 2.4,
        "benci": 2.0, "muak": 2.2, "fitnah": 2.0, "korup": 2.0, "korupsi": 1.7,
        "penipu": 2.2, "manipulatif": 2.1, "psikopat": 2.3, "narsistik": 1.8, "goblok": 2.3,
        "tolol": 2.3, "bodoh": 1.8, "dungu": 2.1, "idiot": 2.1, "bangsat": 2.5, "anjing": 2.2,
        "babi": 2.1, "keparat": 2.5, "brengsek": 2.4, "bajingan": 2.5, "setan": 2.0,
        "iblis": 2.0, "biadab": 2.4, "beradab": 1.5, "laknat": 2.0, "lawan": 1.5,
        "perangi": 1.9, "musnahkan": 2.3, "hancurkan": 2.1, "lengser": 2.1, "turun": 1.3,
        "demo": 1.4, "protes": 1.6, "tolak": 1.6, "boikot": 1.9, "jahat": 1.9, "kejam": 2.0,
        "zalim": 2.2, "dzalim": 2.2, "penindas": 2.2, "fasis": 2.1, "otoriter": 2.0,
        "hipokrit": 2.0, "munafik": 2.1, "bitch": 2.2, "damn": 1.9, "shit": 2.0,
        "fuck": 2.3, "fucking": 2.4, "bastard": 2.3, "stupid": 1.8, "evil": 1.9,
        "rapist": 2.2, "genocida": 2.0, "genosida": 1.9, "racist": 2.0,
        "homophobic": 1.8, "queerphobic": 1.8, "represif": 1.9, "dobol": 2.1,
        "kontol": 2.3, "yaelah": 1.6, "hajar": 1.8, "bacot": 2.2, "bacod": 2.2,
        "congor": 2.2, "ngehek": 2.0, "taik": 1.9, "emo_marah": 2.6,
    },
    "sarkas": {
        "tawa_wkwk": 1.6, "sarkas": 2.5, "ironi": 2.3, "irony": 2.3, "nyindir": 2.2,
        "sindir": 2.0, "sindiran": 2.0, "kocak": 1.9, "lucu": 1.5, "lawak": 2.1,
        "dagelan": 2.2, "komedi": 2.0, "sirkus": 2.0, "badut": 2.2, "clown": 2.2,
        "tumben": 1.8, "padahal": 1.2, "giliran": 1.3, "katanya": 1.1, "cie": 1.9,
        "ciee": 1.9, "ytta": 1.9, "yang_tahu_saja": 2.0, "buzzer": 1.6, "buzzers": 1.6,
        "denial": 1.6, "halu": 1.8, "delulu": 1.9, "halusinasi": 1.7, "karma": 1.5,
        "backfire": 2.1, "karma": 1.5, "sok": 1.4, "pura": 1.3, "palsu": 1.6,
        "kocak": 1.8, "ngakak": 1.8, "lol": 1.5, "emo_tawa": 1.7,
    },
    "kecewa": {
        "kecewa": 2.6, "mengecewakan": 2.7, "sayang": 1.1, "percuma": 2.2, "sia": 1.5,
        "gagal": 1.8, "buruk": 1.7, "parah": 1.6, "payah": 2.1, "bobrok": 2.1,
        "rusak": 1.6, "hancur": 1.6, "mending": 1.5, "seharusnya": 1.7, "harusnya": 1.7,
        "urusin": 1.9, "urus": 1.3, "prioritas": 1.8, "anggaran": 1.2, "gaji": 1.3,
        "guru": 1.1, "kemiskinan": 1.4, "anjlok": 1.7, "pengangguran": 1.5, "mubazir": 2.0,
        "receh": 1.8, "asbun": 1.9, "omong": 1.3, "janji": 1.4, "bohong": 1.8,
        "boong": 1.8, "hoax": 1.6, "hoaks": 1.6, "kurang": 1.0, "lemah": 1.5,
        "lambat": 1.4, "lemas": 1.3, "males": 1.4, "malas": 1.4, "capek": 1.6,
        "lelah": 1.6, "bosan": 1.5, "disappointed": 2.5, "useless": 2.2, "pathetik": 2.0,
        "pathetic": 2.1, "waste": 1.9,
    },
    "senang": {
        "setuju": 2.4, "dukung": 2.3, "mendukung": 2.3, "dukungan": 1.8, "bagus": 2.1,
        "baik": 1.5, "hebat": 2.2, "mantap": 2.3, "keren": 2.1, "luar_biasa": 2.3,
        "positif": 2.0, "tepat": 1.9, "benar": 1.5, "senang": 2.5, "bahagia": 2.5,
        "gembira": 2.5, "bangga": 2.3, "apresiasi": 2.3, "terimakasih": 2.1,
        "terima_kasih": 2.1, "alhamdulillah": 2.4, "syukur": 2.2, "bersyukur": 2.3,
        "puji": 1.8, "berkah": 2.0, "barokah": 2.0, "salut": 2.2, "respek": 2.0,
        "respect": 2.0, "juara": 2.0, "sukses": 2.0, "semangat": 1.7, "lanjutkan": 2.0,
        "gas": 1.4, "top": 1.6, "nice": 1.8, "good": 1.7, "great": 1.9, "agree": 2.1,
        "support": 1.9, "proud": 2.1, "happy": 2.3, "love": 1.8, "cinta": 1.6,
        "damai": 1.6, "sehat": 1.4, "selamat": 1.7, "emo_positif": 2.4,
    },
    "takut": {
        "takut": 2.6, "khawatir": 2.5, "mengkhawatirkan": 2.7, "cemas": 2.5, "waswas": 2.5,
        "ngeri": 2.4, "seram": 2.3, "serem": 2.3, "mengerikan": 2.6, "menakutkan": 2.6,
        "panik": 2.3, "bahaya": 2.1, "berbahaya": 2.2, "terancam": 2.4, "ancam": 1.5,
        "waspada": 2.1, "persekusi": 2.4, "dipersekusi": 2.5, "razia": 1.9, "kriminalisasi": 2.3,
        "ditangkap": 1.8, "penjara": 1.6, "kekerasan": 2.0, "dipukul": 1.8, "intimidasi": 2.3,
        "teror": 2.2, "darurat": 2.0, "kiamat": 1.8, "azab": 2.0, "bencana": 2.0,
        "laten": 1.9, "konflik": 1.6, "horor": 2.2, "horror": 2.2, "scary": 2.4,
        "afraid": 2.5, "fear": 2.4, "worried": 2.4, "worry": 2.2, "anxious": 2.4,
        "danger": 2.1, "threat": 1.7, "nervous": 2.1, "frightened": 2.5, "emo_takut": 2.6,
    },
    "jijik": {
        "jijik": 2.8, "menjijikkan": 2.9, "sejijik": 2.8, "najis": 2.7, "sudi": 2.2,
        "ogah": 2.2, "mual": 2.4, "muntah": 2.5, "eneg": 2.4, "enek": 2.4, "ilfil": 2.3,
        "sampah": 2.2, "nyampah": 2.3, "hama": 2.4, "kotor": 2.0, "bejat": 2.3,
        "busuk": 2.1, "bangkai": 2.3, "mesum": 2.1, "cabul": 2.2, "asusila": 2.0,
        "penyimpangan": 1.9, "menyimpang": 1.9, "kelainan": 2.0, "penyakit": 1.8,
        "menular": 1.7, "aib": 2.1, "memalukan": 2.0, "malu": 1.5, "cuih": 2.6,
        "cih": 2.4, "disgusting": 2.8, "gross": 2.5, "nasty": 2.4, "filthy": 2.5,
        "trash": 2.1, "scum": 2.5, "sickening": 2.6, "emo_jijik": 2.8,
    },
    "sedih": {
        "sedih": 2.6, "menyedihkan": 2.7, "nangis": 2.4, "menangis": 2.5, "tangis": 2.3,
        "kasihan": 2.4, "prihatin": 2.3, "miris": 2.3, "pilu": 2.5, "memilukan": 2.6,
        "duka": 2.4, "berduka": 2.5, "innalillahi": 2.3, "malang": 2.2, "nestapa": 2.5,
        "nyesek": 2.4, "terpuruk": 2.3, "menderita": 2.3, "derita": 2.2, "sengsara": 2.3,
        "korban": 1.7, "dibully": 2.2, "bully": 2.0, "dirundung": 2.3, "tertindas": 2.3,
        "berdaya": 1.4, "maaf": 1.5, "sorry": 2.0, "horrible": 2.2, "sad": 2.5,
        "tragic": 2.3, "tragedy": 2.3, "painful": 2.2, "cry": 2.3, "crying": 2.3,
        "tears": 2.2, "heartbroken": 2.6, "hopeless": 2.3, "miserable": 2.4,
        "emo_sedih": 2.5, "emo_nangis": 1.9,
    },
    "terkejut": {
        "kaget": 2.6, "terkejut": 2.7, "syok": 2.6, "shock": 2.5, "heran": 2.3,
        "bingung": 2.1, "aneh": 2.1, "anehnya": 2.3, "ajaib": 2.0, "mendadak": 2.1,
        "heboh": 1.9, "geger": 2.0, "astaga": 2.3, "ya_ampun": 2.2, "waduh": 2.0,
        "hah": 2.1, "lah": 1.1, "loh": 2.1, "lho": 2.1, "kok": 1.6, "kenapa": 1.5,
        "mengapa": 1.6, "bagaimana": 1.4, "benarkah": 2.2, "seriusan": 2.3, "serius": 1.4,
        "massa": 1.3, "nyangka": 2.1, "kebalik": 1.9, "nalar": 1.8, "mempertanyakan": 2.1,
        "confused": 2.3, "surprised": 2.5, "shocked": 2.6, "what": 1.2, "why": 1.2,
        "weird": 2.1, "strange": 2.1, "unbelievable": 2.4, "emo_kaget": 2.6,
    },
}

POSITIVE_TOKENS = set(UNIGRAM_LEXICON["senang"].keys())
FEAR_TOKENS = set(UNIGRAM_LEXICON["takut"].keys())
NEG_WORDS = {"tidak", "bukan", "jangan", "belum", "kurang", "tanpa", "not", "never"}
INTERROGATIVE_LEAD_RE = re.compile(r"^\s*(?:kenapa|mengapa|kok|apa|bagaimana|gimana|emangnya|benarkah|lah|loh|lho)\b")
INFO_CITATION_RE = re.compile(
    r"\b(?:peraturan presiden|perpres nomor 111|kebijakan umum pertahanan|undang_undang|pasal|ayat|source|sumber|berita|tempo|kompas|dpr)\b"
)


@dataclass
class EmotionLabelResult:
    label_9class: str
    label_ekman7: str
    label_main5: str
    confidence: float
    matched_cues: str
    scores: dict[str, float]


def score_emotions(raw_text: str) -> EmotionLabelResult:
    """Compute multi-layer emotion scores and derive 9-class, Ekman-7, and 5-class labels."""
    raw_clean = strip_scraper_artifacts(raw_text)
    norm = normalize_for_lexicon(raw_clean)

    scores: dict[str, float] = {
        "marah": 0.0,
        "sarkas": 0.0,
        "kecewa": 0.0,
        "senang": 0.0,
        "takut": 0.0,
        "jijik": 0.0,
        "sedih": 0.0,
        "terkejut": 0.0,
        "netral": 0.45,  # small prior for short objective statements
    }
    cues: list[str] = []

    # Layer A: Multi-word phrase matching
    for pattern, emo, weight in PHRASE_RULES:
        if pattern.search(norm):
            scores[emo] += weight
            cues.append(f"phrase:{emo}")

    # Layer B + C: Token matching with negation window inversion
    tokens = norm.split()
    alpha_tokens = [t for t in tokens if not t.startswith("!") and not t.startswith("?")]
    for idx, tok in enumerate(alpha_tokens):
        negated = any(alpha_tokens[k] in NEG_WORDS for k in range(max(0, idx - 2), idx))

        for emo, lex in UNIGRAM_LEXICON.items():
            if tok in lex:
                w = lex[tok]
                if negated:
                    if tok in POSITIVE_TOKENS:
                        # "tidak setuju", "tidak bagus", "bukan hal positif" -> kecewa / marah
                        scores["kecewa"] += w * 0.9
                        scores["marah"] += w * 0.6
                        cues.append(f"neg_{tok}->kecewa/marah")
                    elif tok in FEAR_TOKENS:
                        scores["netral"] += 0.8
                        cues.append(f"neg_{tok}->neutralize")
                    elif emo == "terkejut":
                        scores["terkejut"] += w * 0.5
                    else:
                        scores[emo] += w * 0.25
                else:
                    scores[emo] += w
                    cues.append(f"{emo}:{tok}")

    # Layer D: Pragmatic, punctuation, sarcasm co-occurrence, and ALL-CAPS boosts
    has_laugh = ("tawa_wkwk" in alpha_tokens) or ("emo_tawa" in alpha_tokens)
    has_political_target = any(
        w in norm
        for w in (
            "prabowo", "pemerintah", "presiden", "kampanye", "akmil", "didit",
            "teddy", "gibran", "jokowi", "buzzer", "rezim", "undang_undang",
            "perpres", "padahal", "katanya", "otak", "brain", "ironi", "irony",
            "sanksi", "pengalihan", "isu", "konflik", "hukum",
        )
    )
    if has_laugh and has_political_target:
        scores["sarkas"] += 2.3
        cues.append("pragmatic:laugh+political_target->sarkas")
    elif has_laugh and scores["marah"] >= 1.5:
        scores["sarkas"] += 1.6
        cues.append("pragmatic:laugh+anger->sarkas")
    elif has_laugh and scores["jijik"] >= 1.5:
        scores["sarkas"] += 1.2
    elif "emo_tawa" in alpha_tokens and not has_political_target and scores["marah"] == 0 and scores["jijik"] == 0:
        scores["senang"] += 1.2

    # Question marks & interrogatives -> terkejut (if not dominated by pure hostility)
    q_count = raw_clean.count("?")
    excl_count = raw_clean.count("!")
    if q_count >= 1:
        q_boost = min(2.2, 0.9 + 0.5 * q_count)
        if INTERROGATIVE_LEAD_RE.search(norm):
            q_boost += 0.9
        scores["terkejut"] += q_boost
        cues.append("pragmatic:question->terkejut")

    # ALL-CAPS intensity boost
    caps_words = [w for w in re.findall(r"\b[A-Z]{4,}\b", raw_clean) if w not in {"LGBT", "LGBTQ", "HTTP", "HTTPS", "KUHP", "PRABOWO", "YTTA", "AKMIL"}]
    if len(caps_words) >= 3 or excl_count >= 2:
        if scores["marah"] >= scores["terkejut"] and scores["marah"] > 0:
            scores["marah"] += 1.2
        elif scores["senang"] > 1.5:
            scores["senang"] += 0.8
        elif scores["sarkas"] > 1.5:
            scores["sarkas"] += 0.8
        else:
            scores["marah"] += 0.7
            scores["terkejut"] += 0.7
        cues.append("pragmatic:caps_or_excl")

    # Neutral factual/news boost when emotional signals are weak
    non_neutral_max = max(v for k, v in scores.items() if k != "netral")
    if INFO_CITATION_RE.search(norm) and non_neutral_max < 2.1:
        scores["netral"] += 1.8
        cues.append("pragmatic:citation->netral")
    if non_neutral_max < 0.75:
        scores["netral"] += 0.8

    # Tie-breaking preference order for 9-class
    priority_9 = ["marah", "takut", "jijik", "sedih", "senang", "kecewa", "sarkas", "terkejut", "netral"]
    label_9 = max(priority_9, key=lambda c: (round(scores[c], 4), -priority_9.index(c)))

    # Derive Ekman 7-class (marah, senang, sedih, takut, jijik, terkejut, netral)
    ekman_scores = {
        "marah": scores["marah"],
        "senang": scores["senang"],
        "sedih": scores["sedih"] + scores["kecewa"] * 0.85,
        "takut": scores["takut"],
        "jijik": scores["jijik"],
        "terkejut": scores["terkejut"],
        "netral": scores["netral"],
    }
    # Map kecewa / sarkas explicitly into underlying Ekman basic emotion
    if label_9 == "kecewa":
        if scores["marah"] > ekman_scores["sedih"] + 0.5:
            label_ekman7 = "marah"
        else:
            label_ekman7 = "sedih"
    elif label_9 == "sarkas":
        if scores["jijik"] > scores["marah"] and scores["jijik"] >= 2.0:
            label_ekman7 = "jijik"
        elif scores["terkejut"] > scores["marah"] + 1.0 and not has_political_target:
            label_ekman7 = "terkejut"
        elif scores["senang"] > scores["marah"] + 1.2 and not has_political_target:
            label_ekman7 = "senang"
        else:
            label_ekman7 = "marah"
    elif label_9 == "netral":
        label_ekman7 = "netral"
    else:
        priority_ekman = ["marah", "takut", "jijik", "sedih", "senang", "terkejut", "netral"]
        label_ekman7 = max(priority_ekman, key=lambda c: (round(ekman_scores[c], 4), -priority_ekman.index(c)))

    # Derive Condensed 5 Main Classes (marah, sedih, takut, senang, netral)
    if label_ekman7 in ("marah", "jijik"):
        label_main5 = "marah"
    elif label_ekman7 == "sedih":
        label_main5 = "sedih"
    elif label_ekman7 == "takut":
        label_main5 = "takut"
    elif label_ekman7 == "senang":
        label_main5 = "senang"
    elif label_ekman7 == "terkejut":
        # Route valenced surprise if strong secondary emotion exists, else netral
        secondary = max(["marah", "takut", "sedih", "senang"], key=lambda c: ekman_scores[c])
        if ekman_scores[secondary] >= 1.5:
            label_main5 = secondary
        else:
            label_main5 = "netral"
    else:
        label_main5 = "netral"

    sorted_vals = sorted(scores.values(), reverse=True)
    total_pos = sum(max(0.0, v) for v in scores.values()) or 1.0
    margin = (sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) > 1 else sorted_vals[0]
    confidence = min(0.99, max(0.35, 0.5 * (sorted_vals[0] / total_pos) + 0.5 * min(1.0, margin / 3.0)))

    return EmotionLabelResult(
        label_9class=label_9,
        label_ekman7=label_ekman7,
        label_main5=label_main5,
        confidence=round(confidence, 4),
        matched_cues=";".join(cues[:8]),
        scores={k: round(v, 3) for k, v in scores.items()},
    )
