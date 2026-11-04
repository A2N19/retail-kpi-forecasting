"""Verify references against Crossref (DOI), DataCite (dataset), DBLP (NeurIPS) and arXiv (preprints).

Builds paper/references.bib ONLY from metadata returned by these services and writes
paper/references_check.csv with the comparison against the expected citation (column expected_citation).
Scopus / Web of Science indexing cannot be verified through a public API: columns are left "TO CHECK".

Run from repo root: .venv/Scripts/python.exe paper/verify_refs.py
"""

from __future__ import annotations

import csv
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "thesis-reference-check/1.0 (python urllib)"}

# key, source, identifier, expected (the citation as planned for the paper; for discrepancy check)
REFS = [
    ("chen2012dataset", "datacite", "10.24432/C5CG6D", "Chen (2012), Online Retail II, UCI"),
    ("makridakis2018", "crossref", "10.1371/journal.pone.0194889", "PLOS ONE 13(3), e0194889, 2018"),
    ("hewamalage2023", "crossref_query", "Forecast evaluation for data scientists: common pitfalls and best practices",
     "Data Mining and Knowledge Discovery 37, 788-832, 2023"),
    ("hyndman2006", "crossref", "10.1016/j.ijforecast.2006.03.001", "IJF 22(4), 679-688, 2006"),
    ("ke2017lightgbm", "openalex", "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", "NeurIPS 30, 2017"),
    ("lundberg2017shap", "arxiv", "1705.07874", "NeurIPS 30, 2017"),
    ("arantes2025", "crossref", "10.3934/era.2025092", "Electronic Research Archive 33(4), 2092-2117, 2025"),
    ("diebold1995", "crossref", "10.1080/07350015.1995.10524599", "JBES 13(3), 253-263, 1995"),
    ("harvey1997", "crossref", "10.1016/S0169-2070(96)00719-4", "IJF 13(2), 281-291, 1997"),
    ("tashman2000", "crossref", "10.1016/S0169-2070(00)00065-0", "IJF 16(4), 437-450, 2000"),
    ("bergmeir2018", "crossref", "10.1016/j.csda.2017.11.003", "CSDA 120, 70-83, 2018"),
    ("fildes2022", "crossref", "10.1016/j.ijforecast.2019.06.004", "IJF 38(4), 1283-1318, 2022"),
    ("kim2026drivers", "arxiv", "2608.02911",
     "Kim, McCarthy & Lee, Forecasting Revenue with its Customer-Base Drivers: When and Why Coordination Helps"),
    ("hyndman2002", "crossref", "10.1016/S0169-2070(01)00110-8", "Hyndman et al., IJF 2002 (ETS state space)"),
    ("hoerl1970", "crossref", "10.1080/00401706.1970.10488634", "Hoerl & Kennard, Technometrics 1970"),
    ("breiman2001", "crossref", "10.1023/A:1010933404324", "Breiman, Machine Learning 45, 2001"),
    ("chen2012rfm", "crossref", "10.1057/dbm.2012.17", "Chen, Sain & Guo, 2012 (Online Retail case study)"),
    ("grinsztajn2022", "crossref", "10.52202/068431-0037", "Grinsztajn et al., NeurIPS 2022"),
    ("kaufman2012leakage", "crossref", "10.1145/2382577.2382579", "Kaufman et al., Leakage in data mining, TKDD 2012"),
    # Not in Crossref; OpenAlex lists DOI 10.2307/4615733 but it does not resolve at doi.org (404) -> DOI not used.
    ("holm1979", "openalex_work", "W2121044470", "Holm, Scandinavian Journal of Statistics 6(2), 65-70, 1979"),
    # 5th edition has no DOI found; 10.1002/9781118619193 is the 4th edition (2008, without Ljung) -> not used.
    ("box2015", "book_isbn", "9781118675021|10.1111/jtsa.12194",
     "Box, Jenkins, Reinsel & Ljung, Time Series Analysis, 5th ed., Wiley, 2015"),
]


# What has been read beyond metadata (recorded manually, by ref_id), written to the "note" column.
READ_NOTES = {
    "arantes2025": "full text read 2026-11-03 (publisher PDF): daily sales of Rossmann stores; XGBoost vs. linear "
                   "regression; hybrid LR XGBoost; accuracy-interpretability trade-off",
    "kim2026drivers": "arXiv abstract read 2026-11-03; full text not read",
    "chen2012rfm": "not read; context (intro paper of UCI 'Online Retail', id 352) from UCI API metadata",
}


def get(url: str, accept: str | None = None, attempts: int = 3) -> bytes:
    h = dict(UA)
    if accept:
        h["Accept"] = accept
    for k in range(attempts):  # retry transient network errors (e.g. arXiv timeouts)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
                return r.read()
        except (TimeoutError, urllib.error.URLError) as ex:
            if k == attempts - 1 or (isinstance(ex, urllib.error.HTTPError) and ex.code == 404):
                raise
            time.sleep(5 * (k + 1))
    raise RuntimeError("unreachable")


def crossref_doi(doi: str) -> dict:
    return json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]


def crossref_query(q: str) -> dict:
    url = "https://api.crossref.org/works?rows=1&query.bibliographic=" + urllib.parse.quote(q)
    return json.loads(get(url))["message"]["items"][0]


def _author(a: dict) -> str:
    if a.get("family"):
        return f"{a['family']}, {a.get('given', '')}".strip(", ")
    return a.get("name", "")


def from_crossref(m: dict) -> dict:
    al = m.get("author", [])
    # Some Crossref records list affiliations as name-only "authors": keep persons (family name) when present.
    if any(a.get("family") for a in al):
        al = [a for a in al if a.get("family")]
    authors = " and ".join(_author(a) for a in al)
    year = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
    return {"title": html.unescape(m["title"][0]), "author": authors,
            "journal": html.unescape((m.get("container-title") or [""])[0]),
            "volume": m.get("volume", ""), "number": m.get("issue", ""), "pages": m.get("page", ""),
            "year": str(year), "doi": m.get("DOI", ""), "publisher": m.get("publisher", ""),
            "type": m.get("type", "")}


def from_datacite(doi: str) -> dict:
    m = json.loads(get("https://api.datacite.org/dois/" + urllib.parse.quote(doi)))["data"]["attributes"]
    return {"title": m["titles"][0]["title"], "author": " and ".join(c["name"] for c in m["creators"]),
            "journal": "", "volume": "", "number": "", "pages": "", "year": str(m["publicationYear"]),
            "doi": m["doi"], "publisher": m.get("publisher") if isinstance(m.get("publisher"), str)
            else (m.get("publisher") or {}).get("name", ""), "type": "dataset"}


def from_dblp(title: str) -> dict:
    url = "https://dblp.org/search/publ/api?format=json&h=5&q=" + urllib.parse.quote(title)
    hits = json.loads(get(url))["result"]["hits"].get("hit", [])
    hits = [h["info"] for h in hits if h["info"].get("venue") in {"NIPS", "NeurIPS"}]
    i = hits[0]
    a = i["authors"]["author"]
    a = a if isinstance(a, list) else [a]
    names = " and ".join(re.sub(r"\s\d{4}$", "", x["text"]) for x in a)
    return {"title": i["title"].rstrip("."), "author": names, "journal": "Advances in Neural Information Processing Systems",
            "volume": i.get("volume", ""), "number": "", "pages": i.get("pages", ""), "year": i["year"],
            "doi": i.get("doi", ""), "publisher": "", "type": "proceedings", "url": i.get("ee", "")}


def from_arxiv(aid: str) -> dict:
    x = get("http://export.arxiv.org/api/query?id_list=" + aid)
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    e = ET.fromstring(x).find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        raise LookupError("arXiv id not found")
    title = " ".join(e.find("a:title", ns).text.split())
    authors = " and ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns))
    year = e.find("a:published", ns).text[:4]
    jref = e.find("arxiv:journal_ref", ns)
    comment = e.find("arxiv:comment", ns)
    venue = " ".join(t.text for t in (jref, comment) if t is not None and t.text)
    if re.search(r"NIPS|NeurIPS|Neural Information Processing", venue, re.I):
        return {"title": title, "author": authors, "journal": "Advances in Neural Information Processing Systems",
                "volume": "30" if year == "2017" else "", "number": "", "pages": "", "year": year, "doi": "",
                "publisher": "", "type": "proceedings", "url": f"https://arxiv.org/abs/{aid}",
                "note": f"venue from arXiv metadata: {venue}"}
    return {"title": title, "author": authors, "journal": f"arXiv preprint arXiv:{aid}", "volume": "",
            "number": "", "pages": "", "year": year, "doi": "", "publisher": "arXiv", "type": "preprint",
            "url": f"https://arxiv.org/abs/{aid}", "note": "preprint, not peer reviewed"}


def from_openalex(title: str) -> dict:
    """Work without DOI: take the OpenAlex record whose title matches exactly and has the most authors."""
    res = json.loads(get("https://api.openalex.org/works?per-page=10&search=" + urllib.parse.quote(title)))["results"]
    res = [w for w in res if (w.get("title") or "").strip().lower() == title.lower()]
    w = max(res, key=lambda w: len(w["authorships"]))
    names = " and ".join(a["author"]["display_name"] for a in w["authorships"])
    return {"title": w["title"], "author": names, "journal": "Advances in Neural Information Processing Systems",
            "volume": "30", "number": "", "pages": "", "year": str(w["publication_year"]), "doi": "",
            "publisher": "", "type": "proceedings", "url": "",
            "note": f"no DOI; metadata from OpenAlex {w['id']}; pages NOT verified"}


def from_openalex_work(wid: str) -> dict:
    """Journal article without a resolvable DOI: metadata from a specific OpenAlex record; DOI is NOT copied."""
    w = json.loads(get("https://api.openalex.org/works/" + wid))
    src = (w.get("primary_location") or {}).get("source") or {}
    b = w.get("biblio") or {}
    pages = f"{b['first_page']}-{b['last_page']}" if b.get("first_page") and b.get("last_page") else ""
    note = f"metadata from OpenAlex {wid}; issue not given by OpenAlex"
    if w.get("doi"):
        note += f"; OpenAlex DOI {w['doi']} not used (does not resolve at doi.org)"
    return {"title": w["title"], "author": " and ".join(a["author"]["display_name"] for a in w["authorships"]),
            "journal": src.get("display_name", ""), "volume": b.get("volume") or "", "number": "",
            "pages": pages, "year": str(w["publication_year"]), "doi": "", "publisher": "",
            "type": "journal-article", "note": note}


def from_book(spec: str) -> dict:
    """Book: ISBN record from Open Library (title, publisher, date) + authors/edition from a Crossref review title."""
    isbn, review_doi = spec.split("|")
    ol = json.loads(get(f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"))
    rec = ol[f"ISBN:{isbn}"]
    review = re.sub(r"\s+", " ", html.unescape(crossref_doi(review_doi)["title"][0]))  # NBSP etc.
    authors = ["George E. P. Box", "Gwilym M. Jenkins", "Gregory C. Reinsel", "Greta M. Ljung"]
    missing = [a for a in authors if a not in review]
    if missing:
        raise LookupError(f"authors not confirmed by review title: {missing}")
    if "5th Edition" not in review or rec["title"] not in review:
        raise LookupError("edition/title not confirmed")
    return {"title": "Time Series Analysis: Forecasting and Control", "author": " and ".join(authors),
            "journal": "", "volume": "", "number": "", "pages": "", "year": rec["publish_date"][-4:], "doi": "",
            "publisher": rec["publishers"][0]["name"], "type": "book", "edition": "5th", "isbn": isbn,
            "note": f"Open Library ISBN {isbn} + Crossref review {review_doi}; no DOI for 5th ed. "
                    f"(10.1002/9781118619193 is the 4th ed.)"}


def bib_entry(key: str, r: dict) -> str:
    kind = {"journal-article": "article", "dataset": "misc", "preprint": "misc", "proceedings": "inproceedings",
            "proceedings-article": "inproceedings", "book": "book"}.get(r["type"], "article")
    r = {k: (v.replace("‐", "-").replace("&", r"\&") if isinstance(v, str) and k not in {"doi", "url"} else v)
         for k, v in r.items()}
    f = {"author": r["author"], "title": "{" + r["title"] + "}", "year": r["year"]}
    if kind == "article":
        f.update(journal=r["journal"], volume=r["volume"], number=r["number"], pages=r["pages"].replace("-", "--"))
    elif kind == "inproceedings":
        f.update(booktitle=r["journal"], volume=r["volume"], pages=r["pages"].replace("-", "--"))
    elif kind == "book":
        f.update(edition=r["edition"], publisher=r["publisher"], isbn=r["isbn"])
    else:
        f.update(howpublished=r["journal"] or r["publisher"])
    if r.get("doi"):
        f["doi"] = r["doi"]
        if kind == "misc":  # IEEEtran.bst does not print doi; datasets must show the DOI link (UCI citation)
            f["url"] = "https://doi.org/" + r["doi"].upper()
    if r.get("url") and not r.get("doi"):
        f["url"] = r["url"]
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items() if v)
    return f"@{kind}{{{key},\n{body}\n}}\n"


def main() -> None:
    rows, bib = [], []
    for key, src, ident, expected in REFS:
        try:
            if src == "crossref":
                r = from_crossref(crossref_doi(ident))
            elif src == "crossref_query":
                r = from_crossref(crossref_query(ident))
            elif src == "datacite":
                r = from_datacite(ident)
            elif src == "dblp":
                r = from_dblp(ident)
            elif src == "openalex":
                r = from_openalex(ident)
            elif src == "arxiv":
                r = from_arxiv(ident)
            elif src == "openalex_work":
                r = from_openalex_work(ident)
            elif src == "book_isbn":
                r = from_book(ident)
            else:
                raise LookupError("no DOI and not indexed in Crossref; verify manually")
            status = "found"
            bib.append(bib_entry(key, r))
        except Exception as ex:  # noqa: BLE001
            r, status = {}, f"NOT FOUND: {type(ex).__name__}: {ex}"
        rows.append({
            "ref_id": key, "source_checked": src, "query": ident, "status": status,
            "doi": r.get("doi", ""), "authors": r.get("author", ""), "year": r.get("year", ""),
            "title": r.get("title", ""), "journal": r.get("journal", ""), "volume": r.get("volume", ""),
            "issue": r.get("number", ""), "pages": r.get("pages", ""),
            "note": "; ".join(x for x in (r.get("note", ""), READ_NOTES.get(key, "")) if x),
            "expected_citation": expected,
            "scopus_yes_no": "TO CHECK", "wos_yes_no": "TO CHECK", "checked_date": date.today().isoformat(),
        })
        time.sleep(0.5)
    # Never silently drop a reference: if any lookup failed, keep the existing files and stop.
    failed = [r["ref_id"] for r in rows if r["status"] != "found"]
    if failed:
        print(f"ABORT: lookups failed for {failed}; references.bib and references_check.csv were NOT overwritten")
        for r in rows:
            print(f"{r['ref_id']:20s} {r['status'][:80]}")
        sys.exit(1)
    with open(ROOT / "paper" / "references_check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (ROOT / "paper" / "references.bib").write_text(
        "% Generated by paper/verify_refs.py from Crossref / DataCite / DBLP / arXiv metadata. Do not edit by hand.\n\n"
        + "\n".join(bib), encoding="utf-8")
    for r in rows:
        print(f"{r['ref_id']:20s} {r['status'][:60]:60s} | {r['year']} | {r['journal'][:40]} {r['volume']}({r['issue']}) {r['pages']} | {r['title'][:70]}")


if __name__ == "__main__":
    main()
