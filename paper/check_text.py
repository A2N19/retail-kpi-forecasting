"""Compare the visible text of main.tex with the text extracted from main.pdf.

Checks
 1. inline % comments whose tail is prose (text after the last source-like token) -> possibly hidden text;
    comments that start with "source:" or "numbers from" are intentional notes and are skipped
 2. word runs of the running text (floats removed) that are missing from the PDF body
 3. float captions (tables, figures) whose words are not found as a contiguous run in the PDF
 4. 6-word sequences repeated within 40 words in the .tex visible text (accidental duplicates)
 5. 6-word sequences repeated in the PDF body more often than in the .tex running text

Text extraction uses pypdf (pdftotext/poppler is not installed). Ligatures are expanded; words are lowercased,
hyphens removed and only alphabetic tokens of length >= 3 kept, so hyphenation and math cause no false alarms.
Usage: .venv/Scripts/python.exe paper/check_text.py   (exit code 1 if a problem is found)
"""

from __future__ import annotations

import difflib
import re
import sys
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

PAPER = Path(__file__).resolve().parent
LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
NOTE_PREFIXES = ("source:", "numbers from")
FLOAT_RE = re.compile(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", re.S)
CAPTION_RE = re.compile(r"\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", re.S)


def strip_comment(line: str) -> tuple[str, str]:
    m = re.search(r"(?<!\\)%", line)
    return (line, "") if m is None else (line[:m.start()], line[m.start() + 1:])


def prose_tail(comment: str) -> list[str]:
    toks = comment.split()
    last = -1
    for k, t in enumerate(toks):
        if re.search(r"[_./=:()\d~\\]", t):
            last = k
    tail = toks[last + 1:]
    return tail if len(tail) >= 2 and all(re.fullmatch(r"[A-Za-z,]+", t) for t in tail) else []


def read_tex() -> tuple[str, list[tuple[int, str]]]:
    lines = (PAPER / "main.tex").read_text(encoding="utf-8").splitlines()
    body, flagged, inside = [], [], False
    for i, raw in enumerate(lines, 1):
        if r"\begin{document}" in raw:
            inside = True
            continue
        if r"\end{document}" in raw:
            break
        if not inside:
            continue
        vis, com = strip_comment(raw)
        if com.strip() and vis.strip() and not com.strip().lower().startswith(NOTE_PREFIXES):
            tail = prose_tail(com)
            if tail:
                flagged.append((i, " ".join(tail)))
        inp = re.match(r"\s*\\input\{(.+?)\}", vis)
        if inp:
            body.extend(strip_comment(tl)[0] for tl in (PAPER / inp.group(1)).read_text(encoding="utf-8").splitlines())
            continue
        body.append(vis)
    return "\n".join(body), flagged


def norm(t: str) -> str:
    for k, v in LIGATURES.items():
        t = t.replace(k, v)
    return t


def tex_to_words(t: str) -> list[str]:
    t = re.sub(r"\\(cite|ref|label|includegraphics|bibliography|bibliographystyle)(\[[^\]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\\(begin|end)\{[^}]*\}(\{[^}]*\})?", " ", t)
    t = re.sub(r"\$([A-Za-z])\$-", r"\1-", t)  # "$p$-values" is printed as "p-values"
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\CHECK\[([^\]]*)\]", r" CHECK \1 ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    return words(t)


def words(t: str) -> list[str]:
    t = norm(t).replace("-", "").replace("\u2010", "").replace("\u2013", " ").replace("\u2014", " ")
    return [w for w in re.findall(r"[a-z]+", t.lower()) if len(w) >= 3]


def ngrams(ws: list[str], n: int = 6) -> list[tuple]:
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def contains_run(hay: list[str], needle: list[str]) -> bool:
    if not needle:
        return True
    s = " ".join(hay)
    return (" " + " ".join(needle) + " ") in (" " + s + " ")


def main() -> int:
    tex, flagged = read_tex()
    floats = [m.group(0) for m in FLOAT_RE.finditer(tex)]
    running = FLOAT_RE.sub(" ", tex)
    tw = tex_to_words(running)
    pdf = norm("\n".join(p.extract_text() for p in PdfReader(PAPER / "main.pdf").pages))
    body_pdf = re.split(r"\nREFERENCES\n", pdf)[0]
    pw_all, pw = words(pdf), words(body_pdf)
    problems = 0

    def report(title, items):
        nonlocal problems
        print(title)
        for it in items:
            print("   " + it)
        if not items:
            print("   none")
        problems += len(items)

    report("1) Inline comments ending in prose (possibly hidden text):",
           [f"main.tex:{i}: ...{t}" for i, t in flagged])

    sm = difflib.SequenceMatcher(None, tw, pw, autojunk=False)
    miss = [f"...{' '.join(tw[max(0, a - 5):a])} [[{' '.join(tw[a:b])}]] {' '.join(tw[b:b + 5])}..."
            for op, a, b, _, _ in sm.get_opcodes() if op in ("delete", "replace") and b - a >= 3]
    report("2) Running-text word runs missing from the PDF (>= 3 words):", miss)

    caps = []
    for f in floats:
        for c in CAPTION_RE.findall(f):
            cw = tex_to_words(c)
            if contains_run(pw_all, cw):
                continue
            # table captions are set in small caps, which pypdf splits ("HOLM" -> "h olm"): fuzzy match instead
            blocks = difflib.SequenceMatcher(None, cw, pw_all, autojunk=False).get_matching_blocks()
            ratio = sum(b.size for b in blocks) / max(1, len(cw))
            if ratio < 0.85:
                caps.append(f"match {ratio:.2f}: " + " ".join(cw[:12]) + " ...")
    report("3) Float captions not found in the PDF (exact, or fuzzy >= 0.85 for small caps):", caps)

    g = ngrams(tw)
    first: dict = {}
    near = []
    for k, x in enumerate(g):
        if x in first and k - first[x] <= 40:
            near.append(f"'{' '.join(x)}' at word {first[x]} and {k}")
        first.setdefault(x, k)
    report("4) Accidental repetitions in main.tex (same 6 words within 40 words):", near)

    tc, pc = Counter(ngrams(tex_to_words(tex))), Counter(ngrams(pw))  # tex incl. floats vs PDF body incl. floats
    dup = [f"x{c} in PDF vs x{tc.get(x, 0)} in tex: {' '.join(x)}" for x, c in pc.items() if c > 1 and c > tc.get(x, 0)]
    report("5) 6-word sequences repeated in the PDF body more often than in main.tex:", dup)

    cited = set(re.findall(r"\\cite\{([^}]+)\}", tex))
    keys = {k.strip() for c in cited for k in c.split(",")}
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    unresolved = [f"'{m}' in PDF" for m in ("[?]", "??") if m in pdf]
    unresolved += [f"\\cite key missing in references.bib: {k}" for k in sorted(keys) if f"{{{k}," not in bib]
    report("6) Unresolved citations or references:", unresolved)

    print(f"running-text words: tex {len(tw)}, pdf body {len(pw)}; problems: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
