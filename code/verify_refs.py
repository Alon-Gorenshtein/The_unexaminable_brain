"""Verify every manuscript reference against Crossref by DOI, emit AMA 11th-edition
text + a Zotero-importable BibTeX file. (ama-citation-zotero skill discipline.)"""
import json, os, urllib.request, urllib.parse, time, sys

# (key, DOI, stated short description for the diff check)
REFS = [
    ("teasdale1974", "10.1016/S0140-6736(74)91639-0", "Teasdale Jennett GCS Lancet 1974"),
    ("sharshar2014", "10.1007/s00134-014-3214-y", "Sharshar neuro exam ICM 2014"),
    ("oddo2016", "10.1186/s13054-016-1294-5", "Oddo sedation brain injury Crit Care 2016"),
    ("wijdicks2005", "10.1002/ana.20611", "Wijdicks FOUR score Ann Neurol 2005"),
    ("brennan2020", "10.3171/2020.6.JNS20992", "Brennan missing verbal GCS J Neurosurg 2020"),
    ("zhang2025", "10.1111/jocn.17729", "Zhang etGCS MIMIC J Clin Nurs 2025"),
    ("sterne2009", "10.1136/bmj.b2393", "Sterne multiple imputation BMJ 2009"),
    ("collins2015", "10.7326/M14-0697", "Collins TRIPOD Ann Intern Med 2015"),
    ("johnson2023", "10.1038/s41597-022-01899-x", "Johnson MIMIC-IV Sci Data 2023"),
    ("vonelm2007", "10.1016/S0140-6736(07)61602-X", "von Elm STROBE Lancet 2007"),
    ("benchimol2015", "10.1371/journal.pmed.1001885", "Benchimol RECORD PLoS Med 2015"),
]


def crossref(doi):
    # Optional contact address for the Crossref polite pool, read from the environment.
    # When CROSSREF_MAILTO is unset, no contact address is sent.
    mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    agent = "ama-verify/1.0"
    if mailto:
        url += "?mailto=" + urllib.parse.quote(mailto)
        agent += " (mailto:" + mailto + ")"
    req = urllib.request.Request(url, headers={"User-Agent": agent})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]


def ama_authors(authors):
    names = []
    for a in authors or []:
        fam = a.get("family", "")
        given = a.get("given", "")
        initials = "".join(p[0] for p in given.replace("-", " ").split() if p) if given else ""
        names.append((fam + " " + initials).strip())
    if len(names) > 6:
        return ", ".join(names[:6]) + ", et al"
    return ", ".join(names)


def fmt_ama(n, m):
    auth = ama_authors(m.get("author"))
    title = (m.get("title") or [""])[0].strip().rstrip(".")
    jrnl = (m.get("short-container-title") or m.get("container-title") or [""])
    jrnl = jrnl[0] if jrnl else ""
    yr = ""
    for k in ("published-print", "published-online", "published", "issued"):
        if k in m and m[k].get("date-parts", [[None]])[0][0]:
            yr = str(m[k]["date-parts"][0][0]); break
    vol = m.get("volume", ""); iss = m.get("issue", ""); pg = m.get("page", "")
    doi = m.get("DOI", "")
    cite = f"{n}. {auth}. {title}. *{jrnl}*. {yr}"
    if vol:
        cite += f";{vol}"
        if iss:
            cite += f"({iss})"
        if pg:
            cite += f":{pg}"
    cite += f". doi:{doi}"
    return cite, dict(auth=auth, title=title, jrnl=jrnl, yr=yr, vol=vol, iss=iss, pg=pg, doi=doi,
                      type=m.get("type"))


def bibtex(key, m):
    meta = fmt_ama(0, m)[1]
    auths = " and ".join(
        f"{a.get('family','')}, {a.get('given','')}".strip(", ") for a in (m.get("author") or []))
    fields = [("author", auths), ("title", meta["title"]), ("journal", meta["jrnl"]),
              ("year", meta["yr"]), ("volume", meta["vol"]), ("number", meta["iss"]),
              ("pages", meta["pg"]), ("doi", meta["doi"])]
    body = ",\n  ".join(f'{k} = {{{v}}}' for k, v in fields if v)
    return "@article{" + key + ",\n  " + body + "\n}"


ama, bib, diffs = [], [], []
for i, (key, doi, stated) in enumerate(REFS, 1):
    try:
        m = crossref(doi)
        cite, meta = fmt_ama(i, m)
        ama.append(cite)
        bib.append(bibtex(key, m))
        # diff check: does a stated surname appear in canonical authors?
        stated_surnames = [w for w in stated.split() if w[0].isupper() and len(w) > 2][:1]
        ok = any(s.lower() in meta["auth"].lower() for s in stated_surnames) if stated_surnames else True
        diffs.append(f"[{ 'OK ' if ok else 'CHECK'}] {i}. {meta['auth'][:45]} | {meta['jrnl']} {meta['yr']};{meta['vol']} | {doi}")
    except Exception as e:
        ama.append(f"{i}. [VERIFY FAILED for {doi}: {e}]")
        diffs.append(f"[FAIL] {i}. {stated} | {doi} | {e}")
    time.sleep(0.4)

open("output/refs_AMA.txt", "w").write("\n".join(ama))
open("output/refs_for_zotero.bib", "w").write("\n\n".join(bib))
print("=== DIFF REPORT ===")
print("\n".join(diffs))
print("\n=== AMA BIBLIOGRAPHY ===")
print("\n".join(ama))
