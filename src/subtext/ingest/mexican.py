"""Build the Mexican-Spanish subset of the parallel corpus.

The corpus does not record where a subtitle was translated. It has to be inferred, and the
only evidence available is the Spanish itself: which regional words the translator chose.
This module holds that inference end to end, so `subtext build-mx` reproduces every table
from a freshly loaded `aligned_lines` with no notebook, no shell history and no API key.

The shape of it:

1. score every 25-line bucket of the corpus against four regional word lists   (one scan)
2. keep 1000-line windows with >= 10 Mexican markers and twice as many as peninsular
3. merge contiguous windows into seed documents
4. refine each document's edges -- see `subtext.boundaries`, where the windows shrink and
   the threshold falls with them
5. copy every line inside the refined ranges into `mx_corpus`
6. index the English side into `phrase_index` -- see `subtext.tokenizer`

Step 1 is the only full scan. The whole build is about 90 seconds on a laptop.

The word lists are accent-stripped and matched whole-word, so ` vale ` cannot fire inside
"equivale". They must be stored stripped too: the haystack has its accents removed before
matching, so an entry like "ándale" could never fire. One such dead entry was removed on
2026-09-04 -- it changed no count, because it had never matched anything. `tests/` pins this. They are incomplete and always will be: a statistical hunt over the finished
corpus keeps surfacing misses. Re-run that hunt after widening any list -- the rebuild is
cheap enough that there is no excuse not to.
"""

from __future__ import annotations

from typing import Sequence

#: Mexican register markers. Presence of these is the positive signal.
MEXICAN: tuple[str, ...] = (
    "ahorita", "ahoritita", "alberca", "albercas",
    "andale", "andele", "antojitos", "apapachar",
    "apapacho", "banqueta", "banquetas", "botana",
    "botanas", "chafa", "chale", "chamaco",
    "chamacos", "chamarra", "chamarras", "chamba",
    "chambear", "chava", "chavo", "chavos",
    "chido", "chidos", "chin", "chinga",
    "chingada", "chingaderas", "chingado", "chingar",
    "chingo", "chingon", "chingona", "chingue",
    "codo", "cuate", "cuates", "culero",
    "culeros", "desmadre", "elote", "elotes",
    "escuincle", "escuincles", "feria", "flojera",
    "fregon", "fregona", "fuchi", "gacha",
    "gacho", "guacala", "guey", "hijole",
    "hueva", "huevon", "jalar", "jale",
    "lana", "luegito", "madrear", "madriza",
    "mames", "manches", "mande", "morra",
    "morro", "morros", "naco", "nacos",
    "nel", "neta", "orale", "padrisimo",
    "pendeja", "pendejo", "pendejos", "pinche",
    "pinches", "platica", "platicamos", "platicando",
    "platicar", "platicas", "popote", "popotes",
    "sepa", "simon", "tantita", "tantito",
    "troca", "trocas", "varo", "vato",
    "vatos", "wey", "zafo",
)

#: Peninsular (Spain) markers. A window rich in these is not Mexican, whatever else it has.
PENINSULAR: tuple[str, ...] = (
    "alquilar", "alquiler", "aparcamiento", "aparcar",
    "ascensor", "billete", "billetes", "cabreado",
    "cabrear", "capullo", "cazadora", "chaval",
    "chavala", "chavales", "chorrada", "coche",
    "coches", "cono", "currar", "curro",
    "cutre", "dormitorio", "flipante", "flipar",
    "flipas", "follar", "fontanero", "gafas",
    "gilipollas", "gilipolleces", "guay", "guisantes",
    "hostia", "hostias", "joder", "jodida",
    "jodido", "judias", "maja", "majo",
    "melocoton", "mogollon", "mola", "molar",
    "movil", "moviles", "ordenador", "pajita",
    "patatas", "pija", "pijo", "piso",
    "pisos", "tarta", "tia", "tias",
    "tio", "tios", "vale", "vales",
    "vosotros", "vuestra", "vuestras", "vuestro",
    "vuestros", "zumo",
)

#: Pan-American: common across Latin America, so useful context but not discriminating.
LATIN_AMERICAN: tuple[str, ...] = (
    "agarrar", "boleto", "boletos", "carro",
    "carros", "celular", "celulares", "cerillo",
    "cerillos", "checa", "checar", "chicharos",
    "computadora", "computadoras", "departamento", "departamentos",
    "durazno", "elevador", "estacionamiento", "estacionar",
    "frijoles", "jugo", "lentes", "manejando",
    "manejar", "papas", "platano", "plomero",
    "recamara", "renta", "rentar", "saco",
    "ustedes", "valija",
)

#: Markers of *other* Latin-American varieties. Tracked to show the detector is not merely
#: catching "anything that is not Spain" -- these stay at 0.10% in the finished corpus.
OTHER_LATAM: tuple[str, ...] = (
    "acha", "bacano", "boluda", "boludo",
    "bondi", "broder", "carajito", "chabon",
    "chamo", "che", "chevere", "forro",
    "gil", "guagua", "laburar", "laburo",
    "mae", "pana", "parcera", "parcero",
    "pelotudo", "piba", "pibe", "posta",
    "quilombo", "tuanis", "vaina", "vos",
)

#: Used only to score recall. Every word here is also in MEXICAN, so it measures how much
#: marked content the document threshold KEEPS -- not whether the detector generalises to
#: unseen markers. It is not a held-out set and must not be reported as one.
GOLD: tuple[str, ...] = (
    "alberca", "andale", "banqueta",
    "chafa", "chamba", "chido",
    "escuincle", "gacho", "guey",
    "hijole", "neta", "no mames",
    "no manches", "orale", "popote",
    "tantito", "vato", "wey",
)

#: Bucket size the corpus is scored at. Every refinement window is a multiple of this.
RESOLUTION = 25

#: Seed windows: 1000 lines, >= 10 Mexican markers, >= twice as many Mexican as peninsular.
#: Loosening the threshold shrinks the corpus 7x to buy ~8 points of recall, a bad trade.
SEED_WINDOW = 1000
SEED_THRESHOLD = 10
MX_OVER_ES = 2

_ACCENTS = ("'áéíóúñüÁÉÍÓÚÑÜ','aeiounuaeiounu'")


def normalised(column: str) -> str:
    """Lowercase, strip accents, punctuation to spaces, and pad both ends.

    The padding is what turns a substring test into a whole-word one.
    """
    return (
        "concat(' ', replaceRegexpAll(translateUTF8(lowerUTF8(" + column + "),"
        + _ACCENTS + "), '[^a-z0-9]+', ' '), ' ')"
    )


def any_of(words: Sequence[str], expr: str = "t") -> str:
    """multiSearchAny over space-padded words -- i.e. whole-word matching."""
    literals = ", ".join("' " + w.replace("'", "''") + " '" for w in words)
    return "multiSearchAny(" + expr + ", [" + literals + "])"


def win_scores_sql(table: str, source: str = "aligned_lines",
                   column: str = "target_text", resolution: int = RESOLUTION) -> str:
    """One pass over the corpus, counting every word list per bucket."""
    return (
        "INSERT INTO " + table + "\n"
        "SELECT intDiv(pair_id, " + str(resolution) + ") AS b, count() AS n_rows,\n"
        "       countIf(" + any_of(MEXICAN) + ") AS mx,\n"
        "       countIf(" + any_of(LATIN_AMERICAN) + ") AS latam,\n"
        "       countIf(" + any_of(PENINSULAR) + ") AS es,\n"
        "       countIf(" + any_of(OTHER_LATAM) + ") AS other\n"
        "FROM (SELECT pair_id, " + normalised(column) + " AS t FROM " + source + ")\n"
        "GROUP BY b"
    )


def gold_sql(table: str, source: str = "aligned_lines",
             column: str = "target_text", resolution: int = RESOLUTION) -> str:
    """Recall scoring set, at the same resolution as the scores."""
    return (
        "INSERT INTO " + table + "\n"
        "SELECT intDiv(pair_id, " + str(resolution) + ") AS b,\n"
        "       countIf(" + any_of(GOLD, normalised(column)) + ") AS gold\n"
        "FROM " + source + " GROUP BY b"
    )


def seed_sql(scores: str, resolution: int = RESOLUTION) -> str:
    """Windows that pass the threshold, as (lo, hi) pair_id ranges.

    Scores are stored per `resolution` bucket, so they are re-aggregated up to
    SEED_WINDOW here rather than scanned again.
    """
    per = SEED_WINDOW // resolution
    return (
        "SELECT w * " + str(SEED_WINDOW) + " AS lo, w * " + str(SEED_WINDOW)
        + " + " + str(SEED_WINDOW - 1) + " AS hi\n"
        "FROM (SELECT intDiv(b, " + str(per) + ") AS w, sum(mx) AS mx, sum(es) AS es\n"
        "      FROM " + scores + " GROUP BY w)\n"
        "WHERE mx >= " + str(SEED_THRESHOLD) + " AND mx >= " + str(MX_OVER_ES) + " * es\n"
        "ORDER BY w"
    )


def corpus_sql(target: str, docs: str, source: str = "aligned_lines") -> str:
    """Every line inside a document range -- the step that turns markers into a corpus."""
    return (
        "INSERT INTO " + target + "\n"
        "SELECT a.pair_id, toUInt32(intDiv(a.pair_id, 1000)), d.doc_id,\n"
        "       a.source_text, a.target_text\n"
        "FROM " + source + " a\n"
        "INNER JOIN " + docs + " d ON 1 = 1\n"
        "WHERE a.pair_id BETWEEN d.lo AND d.hi"
    )


# --------------------------------------------------------------------------------------
# The build itself
# --------------------------------------------------------------------------------------

TABLES: dict[str, str] = {
    "win_scores": "(b UInt32, n_rows UInt32, mx UInt32, latam UInt32, es UInt32, other UInt32)"
                  " ENGINE = MergeTree ORDER BY b",
    "gold_scores": "(b UInt32, gold UInt32) ENGINE = MergeTree ORDER BY b",
    "mx_docs": "(doc_id UInt32, lo UInt64, hi UInt64, n_lines UInt64,"
               " mx UInt64, es UInt64, other UInt64) ENGINE = MergeTree ORDER BY doc_id",
    "mx_corpus": "(pair_id UInt64, blk UInt32, doc_id UInt32, en String, es String)"
                 " ENGINE = MergeTree ORDER BY pair_id",
}


def _recreate(ch, name: str) -> None:
    ch.command(f"DROP TABLE IF EXISTS {name}")
    ch.command(f"CREATE TABLE {name} {TABLES[name.split('.')[-1]]}")


def load_scores(ch, table: str = "win_scores"):
    """Pull the per-bucket marker counts back as prefix sums for the refinement.

    Only buckets carrying a marker are fetched: a bucket with neither contributes zero to
    every window sum, so its absence is the same as its presence.
    """
    import numpy as np

    from ..boundaries import Scores

    rows = ch.query(f"SELECT b, mx, es FROM {table} WHERE mx > 0 OR es > 0").result_rows
    size = max((r[0] for r in rows), default=0) + 2
    mx = np.zeros(size, np.int64)
    es = np.zeros(size, np.int64)
    for b, m, e in rows:
        mx[b] = m
        es[b] = e
    return Scores(mx=np.concatenate([[0], np.cumsum(mx)]),
                  es=np.concatenate([[0], np.cumsum(es)]))


def read_index(path) -> list[tuple[int, int]]:
    """Read a shipped boundary index: a TSV with a header, `lo` and `hi` columns."""
    from pathlib import Path

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    lo_i, hi_i = header.index("lo"), header.index("hi")
    return [(int(p[lo_i]), int(p[hi_i]))
            for p in (line.split("\t") for line in lines[1:] if line.strip())]


def write_index(ch, path, docs_table: str = "mx_docs") -> int:
    """Write the boundary index out as TSV: integers only, so it is safe to redistribute."""
    from pathlib import Path

    rows = ch.query(
        f"SELECT doc_id, lo, hi, n_lines, mx, es, other FROM {docs_table} ORDER BY doc_id"
    ).result_rows
    body = "\n".join("\t".join(str(v) for v in r) for r in rows)
    Path(path).write_text("doc_id\tlo\thi\tn_lines\tmx\tes\tother\n" + body + "\n",
                          encoding="utf-8")
    return len(rows)


def statements(sql: str) -> list[str]:
    """Split a script into executable statements, ignoring semicolons inside comments."""
    stripped = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    return [s.strip() for s in stripped.split(";") if s.strip()]


def build(*, from_index=None, skip_scan: bool = False, log=print) -> dict:
    """Rebuild the Mexican corpus from a loaded `aligned_lines`.

    `from_index` uses shipped document boundaries instead of deriving them, which skips
    the detection entirely. The result is identical because OPUS v2024 is checksum-pinned
    and `pair_id` is a deterministic counter, so a range means the same lines everywhere.
    """
    import time

    from .. import db
    from ..boundaries import merge, refine
    from ..tokenizer import phrase_index_sql

    ch = db.client()
    stats: dict[str, object] = {}
    t_all = time.perf_counter()

    def step(name):
        log(f"  {name} ...")
        return time.perf_counter()

    if not skip_scan:
        t = step("scoring the corpus (one full scan)")
        _recreate(ch, "win_scores")
        ch.command(win_scores_sql("win_scores"))
        _recreate(ch, "gold_scores")
        ch.command(gold_sql("gold_scores"))
        stats["scan_secs"] = round(time.perf_counter() - t, 1)

    if from_index is not None:
        t = step(f"reading shipped boundaries from {from_index}")
        ranges = read_index(from_index)
        stats["source"] = "shipped index"
    else:
        t = step("seeding documents and refining their edges")
        seeds = [(int(lo), int(hi)) for lo, hi in ch.query(seed_sql("win_scores")).result_rows]
        stats["seed_windows"] = len(seeds)
        scores = load_scores(ch)
        merged_seeds = merge(seeds)
        ranges = [tuple(r) for r in merge(refine(lo, hi, scores) for lo, hi in merged_seeds)]
        stats["source"] = "derived"
    stats["documents"] = len(ranges)
    stats["boundary_secs"] = round(time.perf_counter() - t, 2)

    t = step(f"materialising {len(ranges)} documents into mx_corpus")
    _recreate(ch, "mx_docs")
    ch.insert("mx_docs",
              [[i, lo, hi, hi - lo + 1, 0, 0, 0] for i, (lo, hi) in enumerate(ranges, 1)],
              column_names=["doc_id", "lo", "hi", "n_lines", "mx", "es", "other"])
    _recreate(ch, "mx_corpus")
    ch.command(corpus_sql("mx_corpus", "mx_docs"))
    stats["lines"] = ch.query("SELECT count() FROM mx_corpus").result_rows[0][0]
    stats["corpus_secs"] = round(time.perf_counter() - t, 1)

    t = step("counting markers per document")
    counts = ch.query(f"""
        SELECT doc_id,
               countIf({any_of(MEXICAN)}) AS mx,
               countIf({any_of(PENINSULAR)}) AS es,
               countIf({any_of(OTHER_LATAM)}) AS other
        FROM (SELECT doc_id, {normalised('es')} AS t FROM mx_corpus)
        GROUP BY doc_id
    """).result_rows
    by_doc = {r[0]: r[1:] for r in counts}
    ch.command("TRUNCATE TABLE mx_docs")
    ch.insert("mx_docs",
              [[i, lo, hi, hi - lo + 1, *by_doc.get(i, (0, 0, 0))]
               for i, (lo, hi) in enumerate(ranges, 1)],
              column_names=["doc_id", "lo", "hi", "n_lines", "mx", "es", "other"])
    stats["mexican_markers"] = sum(v[0] for v in by_doc.values())
    stats["peninsular_markers"] = sum(v[1] for v in by_doc.values())

    t = step("indexing English phrases")
    for statement in statements(phrase_index_sql(source="mx_corpus", table="phrase_index")):
        ch.command(statement)
    stats["phrases"] = ch.query("SELECT count() FROM phrase_index").result_rows[0][0]
    stats["index_secs"] = round(time.perf_counter() - t, 1)

    stats["total_secs"] = round(time.perf_counter() - t_all, 1)
    return stats
