"""
enrich_manifest.py
==================
Enriches the MIDI manifest with conditioning metadata:
  - genre   : macro-genre label assigned via artist/filename heuristics
  - bpm_bucket : BPM rounded to nearest 10, clamped to [60, 200]

Reads:  data/json/manifest.jsonl
Writes: data/json/manifest_enriched.jsonl  (same format, two new fields)
"""

import json as js
import re

INPUT_PATH  = "data/json/manifest.jsonl"
OUTPUT_PATH = "data/json/manifest_enriched.jsonl"


GENRES = [
    "rock", "pop", "funk_disco", "jazz", "blues",
    "latin", "electronic", "classical", "country", "reggae",
    "unknown",
]

ARTIST_GENRE: dict[str, str] = {

    "ac dc": "rock", "acdc": "rock", "aerosmith": "rock",
    "alice cooper": "rock", "allman brothers": "rock",
    "arctic monkeys": "rock", "avril lavigne": "rock",
    "bachman turner overdrive": "rock", "beatles": "rock",
    "the beatles": "rock", "the beach boys": "rock", "beach boys": "rock",
    "black sabbath": "rock", "blink182": "rock",
    "bon jovi": "rock", "boston": "rock",
    "bruce springsteen": "rock", "bryan adams": "rock",
    "camel": "rock", "chicago": "rock",
    "coldplay": "rock", "cream": "rock",
    "creedence clearwater revival": "rock", "crowded house": "rock",
    "deep purple": "rock", "def leppard": "rock",
    "devo": "rock",
    "dire straits": "rock", "doors": "rock", "the doors": "rock",
    "eagles": "rock", "the eagles": "rock", "eddie money": "rock",
    "electric light orchestra": "rock",
    "elvis presley": "rock", "elvis": "rock",
    "eric clapton": "blues", "europe": "rock",
    "fleetwood mac": "rock", "foo fighters": "rock",
    "foreigner": "rock", "genesis": "rock",
    "grateful dead": "rock", "green day": "rock",
    "guns n roses": "rock", "guns n' roses": "rock", "heart": "rock",
    "hendrix": "rock", "jimi hendrix": "rock", "iron maiden": "rock",
    "jefferson airplane": "rock", "jethro tull": "rock",
    "jimmy eat world": "rock", "joan jett": "rock",
    "john lennon": "rock", "journey": "rock",
    "kansas": "rock", "king crimson": "rock",
    "kiss": "rock", "led zeppelin": "rock",
    "linkin park": "rock", "lynyrd skynyrd": "rock",
    "metallica": "rock", "muse": "rock",
    "nirvana": "rock", "oasis": "rock",
    "offspring": "rock", "ozzy osbourne": "rock",
    "pearl jam": "rock", "pink floyd": "rock",
    "queen": "rock", "r.e.m.": "rock", "rem": "rock",
    "radiohead": "rock", "rage against the machine": "rock",
    "ramones": "rock", "red hot chili peppers": "rock",
    "rolling stones": "rock", "the rolling stones": "rock", "rush": "rock",
    "scorpions": "rock",
    "simple minds": "rock", "simple plan": "rock",
    "smashing pumpkins": "rock", "soundgarden": "rock",
    "status quo": "rock", "steely dan": "jazz",
    "steppenwolf": "rock", "steve miller band": "rock",
    "styx": "rock",
    "supertramp": "rock", "the clash": "rock",
    "the police": "rock", "the who": "rock", "who": "rock",
    "thin lizzy": "rock", "toto": "rock",
    "u2": "rock", "van halen": "rock",
    "van morrison": "rock", "weezer": "rock",
    "whitesnake": "rock", "yes": "rock",
    "zz top": "blues",
    "alanis morissette": "rock", "alanis morrisette": "rock",
    "alan parsons project": "rock",
    "billy idol": "rock",
    "collective soul": "rock", "david bowie": "rock",
    "edgar winter band": "rock",
    "mike oldfield": "rock", "nightwish": "rock",
    "emerson lake & palmer": "rock", "emerson lake and palmer": "rock",
    "the corrs": "rock", "corrs": "rock",
    "stranglers": "rock", "the stranglers": "rock",
    "the cure": "rock", "cure": "rock",
    "inxs": "rock", "the cranberries": "rock", "cranberries": "rock",
    "neil young": "rock", "neil diamond": "pop",
    "tom petty": "rock", "the tragically hip": "rock",
    "tragically hip": "rock",
    "guess who": "rock", "the guess who": "rock",
    "judas priest": "rock", "rammstein": "rock",
    "limp bizkit": "rock",
    "frank zappa": "rock",
    "bob dylan": "rock",
    "sheryl crow": "rock",
    "joe cocker": "rock",
    "everly brothers": "rock", "the everly brothers": "rock",
    "dave clark five": "rock",
    "todd rundgren": "rock",
    "monkees": "rock", "the monkees": "rock",
    "the shadows": "rock", "shadows": "rock",
    "kate bush": "rock",
    "the yardbirds": "rock", "yardbirds": "rock",
    "young rascals": "rock", "the young rascals": "rock",
    "zombies": "rock", "the zombies": "rock",
    "bread": "pop",
    "chris rea": "rock",
    "del amitri": "rock",
    "eddie & hot rods": "rock",
    "electric prunes": "rock",
    "badfinger": "rock",
    "busted": "rock",
    "cock robin": "pop",
    "clout": "pop",
    "double": "pop",
    "alphaville": "pop",
    "bwitched": "pop",
    "daddy cool": "rock",
    "diesel": "rock",
    "dion": "pop",
    "don mclean": "pop",
    "daryl braithwaite": "pop",
    "abba": "pop", "ace of base": "pop",
    "a-ha": "pop", "aqua": "pop",
    "backstreet boys": "pop", "bee gees": "pop",
    "belinda carlisle": "pop", "beyonce": "pop",
    "britney spears": "pop", "carpenters": "pop",
    "cascada": "pop", "celine dion": "pop",
    "christina aguilera": "pop", "culture club": "pop",
    "cyndi lauper": "pop", "daniel bedingfield": "pop",
    "destiny's child": "pop", "dido": "pop",
    "diana ross": "pop",
    "eamon": "pop", "elliott yamin": "pop",
    "elton john": "pop", "billy joel": "pop",
    "cliff richard": "pop", "clouseau": "pop",
    "duran duran": "pop", "sting": "pop",
    "madonna": "pop", "michael jackson": "pop",
    "pet shop boys": "pop", "phil collins": "pop",
    "robbie williams": "pop", "spice girls": "pop",
    "whitney houston": "pop", "michael buble": "pop",
    "lionel richie": "pop", "tom jones": "pop",
    "george michael": "pop", "roy orbison": "pop",
    "frank sinatra": "pop",
    "tina turner": "pop",
    "prince": "pop",
    "marco borsato": "pop", "frans bauer": "pop",
    "dana winner": "pop",
    "andre hazes": "pop",
    "erasure": "electronic",
    "k3": "pop",
    "kinderen voor kinderen": "pop",
    "daniel powter": "pop",
    "david hasselhoff": "pop",
    "shania twain": "country",
    "anne murray": "country",
    "sarah mclachlan": "pop",
    "patsy cline": "country",
    "dionne warwick": "pop",
    "debbie gibson": "pop",
    "bobby vee": "pop",
    "barry manilow": "pop",
    "band aid": "pop",
    "charlotte nilsson": "pop",
    "angel": "pop",
    "chingy": "pop",
    "danny elfman": "rock",
    "average white band": "funk_disco",
    "boney m": "funk_disco", "chaka khan": "funk_disco",
    "chic": "funk_disco", "earth wind & fire": "funk_disco",
    "earth wind and fire": "funk_disco",
    "james brown": "funk_disco", "kool & the gang": "funk_disco",
    "parliament": "funk_disco", "rick james": "funk_disco",
    "sly and the family stone": "funk_disco",
    "tower of power": "funk_disco",
    "donna summer": "funk_disco",
    "aaliyah": "funk_disco", "akon": "pop",
    "alicia keys": "funk_disco", "alicia keyes": "funk_disco",
    "aretha franklin": "funk_disco",
    "bobby caldwell": "funk_disco",
    "corinne bailey rae": "funk_disco",
    "coolio": "pop",
    "stevie wonder": "funk_disco",
    "temptations": "funk_disco", "the temptations": "funk_disco",
    "the four tops": "funk_disco", "four tops": "funk_disco",
    "jamiroquai": "funk_disco",
    "ray charles": "funk_disco",
    "level 42": "funk_disco",
    "chi-lites": "funk_disco",
    "dells": "funk_disco",
    "billy ocean": "pop",
    "bobby hachey": "pop",
    "antonio carlos jobim": "jazz", "brecker brothers": "jazz",
    "chick corea": "jazz", "dave brubeck": "jazz",
    "donald fagan": "jazz", "donald fagen": "jazz",
    "duke ellington": "jazz", "ella fitzgerald": "jazz",
    "herbie hancock": "jazz", "john coltrane": "jazz",
    "miles davis": "jazz", "pat metheny": "jazz",
    "thelonious monk": "jazz", "weather report": "jazz",
    "al jarreau": "jazz", "david sanborn": "jazz",
    "amy winehouse": "jazz",
    "john mclaughlin": "jazz",
    "david torkanowsky": "jazz",
    "bb king": "blues", "buddy guy": "blues",
    "howlin wolf": "blues", "john lee hooker": "blues",
    "muddy waters": "blues", "robert johnson": "blues",
    "stevie ray vaughan": "blues",
    "blues brother": "blues", "blues brothers": "blues",
    "chuck berry": "rock",
    "chambao": "latin", "ricky martin": "latin",
    "shakira": "latin", "carlos santana": "latin",
    "marc anthony": "latin", "santana": "latin",
    "gloria estefan": "latin",
    "dj bobo": "electronic", "dj casper": "electronic",
    "dj sammy": "electronic", "bob sinclar": "electronic",
    "depeche mode": "electronic",
    "jean michel jarre": "electronic", "kraftwerk": "electronic",
    "bach": "classical", "beethoven": "classical",
    "brahms": "classical", "mozart": "classical",
    "vivaldi": "classical", "chopin": "classical",
    "bethoven": "classical",
    "charlie brown": "jazz",
    "final fantasy": "classical",
    "alan jackson": "country", "buck owens": "country",
    "charlie pride": "country", "clint black": "country",
    "conway twitty": "country", "dan seals": "country",
    "dolly parton": "country", "dwight yoakam": "country",
    "garth brooks": "country", "george jones": "country",
    "george strait": "country", "hank williams": "country",
    "johnny cash": "country", "kenny chesney": "country",
    "kenny rogers": "country", "merle haggard": "country",
    "reba mcentire": "country", "tim mcgraw": "country",
    "toby keith": "country", "vince gill": "country",
    "waylon jennings": "country", "willie nelson": "country",
    "brad paisley": "country", "brooks and dunn": "country",
    "billy ray cyrus": "country", "carrie underwood": "country",
    "shania twain": "country", "anne murray": "country",
    "patsy cline": "country",
    "david lee murphy": "country", "daryle singletary": "country",
    "david kersh": "country",
    "billy swan": "country",
    "bob marley": "reggae",
    "edith piaf": "pop", "charles aznavour": "pop",
    "ginette reno": "pop", "isabelle boulay": "pop",
    "gerry boulet": "rock",
    "salvatore adamo": "pop", "adamo": "pop",
    "sweet people": "pop",
    "claude nougaro": "jazz",
    "boris vian": "jazz",
    "beau dommage": "rock",
    "eminem": "pop",
    "busta rhymes": "pop", "bhusta rhymes": "pop",
    "celtic": "country",
    "bluegrass": "country",
    "cajun": "country",
    "zemlja obecana": "pop",
    "doe maar": "rock",
}


KEYWORD_RULES: list[tuple[list[str], str]] = [
    (["blues"], "blues"),
    (["jazz", "bossa", "swing"], "jazz"),
    (["reggae", "ska"], "reggae"),
    (["funk", "disco", "soul", "motown", "r&b", "rnb"], "funk_disco"),
    (["country", "bluegrass", "cajun", "nashville"], "country"),
    (["house", "techno", "trance", "edm", "electro", "synth"], "electronic"),
    (["latin", "salsa", "samba", "bossa", "rumba", "cha cha", "reggaeton", "merengue"], "latin"),
    (["classical", "bach", "mozart", "beethoven", "chopin", "brahms", "vivaldi",
      "sonata", "symphony", "concerto", "fugue", "prelude", "etude", "waltz",
      "minuet", "nocturne", "aria"], "classical"),
    (["rock", "metal", "punk", "grunge", "hard"], "rock"),
    (["pop", "boy band"], "pop"),
]


def normalize_artist(song_id: str) -> str:
    """Extract and normalize artist name from song_id."""
    s = song_id.strip()

    if " - " in s:
        s = s.split(" - ")[0].strip()

    elif s.count("_") >= 2 and not s[0].isdigit():
        parts = s.split("_")
        s = " ".join(parts[:2])

    # Remove common suffixes
    for suffix in [" midi only", " met tekst", " nvt", " kar", " ver2", " v2",
                   ".mid", ".kar", " sequenced by"]:
        idx = s.lower().find(suffix)
        if idx > 0:
            s = s[:idx]

    return s.strip().lower()


def classify_genre(song_id: str) -> str:
    """Assign a macro-genre to a song_id via artist lookup + keyword fallback."""
    artist = normalize_artist(song_id)
    sid_lower = song_id.lower()

    if artist in ARTIST_GENRE:
        return ARTIST_GENRE[artist]

    for known_artist, genre in ARTIST_GENRE.items():
        if artist.startswith(known_artist) or known_artist.startswith(artist):
            return genre

    for known_artist, genre in ARTIST_GENRE.items():
        if known_artist in sid_lower:
            return genre

    for keywords, genre in KEYWORD_RULES:
        for kw in keywords:
            if kw in sid_lower:
                return genre

    if "_natant" in sid_lower or "natant" in sid_lower:
        return "classical"

    return "unknown"


def bucket_bpm(bpm: float) -> int:
    """
    Round BPM to nearest 10, clamp to [60, 200].
    BPMs above 200 are likely octave errors — halve them first.
    """
    b = float(bpm)
    if b > 200:
        b = b / 2.0
    b = max(60.0, min(200.0, b))
    return int(round(b / 10.0) * 10)


def main():
    rows = []
    genre_counts: dict[str, int] = {}

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = js.loads(line)

            genre = classify_genre(row.get("song_id", ""))
            bpm_b = bucket_bpm(row.get("bpm", 120))

            row["genre"] = genre
            row["bpm_bucket"] = bpm_b
            rows.append(row)

            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(js.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] Enriched {len(rows)} rows → {OUTPUT_PATH}")
    print(f"[OK] Genre distribution:")
    for g in sorted(genre_counts.keys()):
        print(f"     {g:15s} : {genre_counts[g]:5d}")

    unknown_pct = genre_counts.get("unknown", 0) / max(1, len(rows)) * 100
    print(f"\n[INFO] Unknown rate: {unknown_pct:.1f}%")
    if unknown_pct > 30:
        print("[WARN] High unknown rate — consider adding more artist mappings")


if __name__ == "__main__":
    main()
