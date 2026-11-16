#!/usr/bin/env python3
"""RunbookDrift miner.

Builds one benchmark window from the public GitLab runbooks repository:
  snapshot T0, look-back snapshot TM = T0 - 12 months, label horizon T1 = T0 + 12 months.

Steps   = actionable lines in docs/**/*.md at T0 (shell/code lines, or lines with
          inline code or external URLs), excluding auto-generated sections.
Labels  = DRIFTED if, by T1, the line was changed or removed AND at least one of its
          references (URL, host, command token) no longer appears in the file;
          VALID if the exact line is still present at T1;
          otherwise REWORDED (excluded from evaluation).
Features use only information available at T0 (no look-ahead).

Usage: mine.py <repo> <T0 YYYY-MM-DD> <out.csv>
"""
import csv, os, re, subprocess, sys, datetime as dt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

REPO, T0_DATE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                          errors="replace").stdout


def rev_before(date):
    # pin to midnight UTC so the snapshot does not depend on the time of day the script runs
    return git("rev-list", "-1", f"--before={date} 00:00:00 +0000", "HEAD").strip()


def shift(date, months):
    d = dt.date.fromisoformat(date)
    return d.replace(year=d.year + months // 12).isoformat()


T0 = rev_before(T0_DATE)
TM = rev_before(shift(T0_DATE, -12))
T1 = rev_before(shift(T0_DATE, 12))
T0_TS = int(git("show", "-s", "--format=%ct", T0).strip())


def md_files(rev):
    return [p for p in git("ls-tree", "-r", "--name-only", rev, "docs").splitlines()
            if p.endswith(".md")]


def read(rev, path):
    return git("show", f"{rev}:{path}")


# ---------- reference (entity) extraction ----------
URL_RE = re.compile(r"https?://[^\s)<>\]\"'`|]+")
CODE_SPAN_RE = re.compile(r"`([^`]+)`")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@=+-]*[A-Za-z0-9]")
SKIP_HEADS = {"sudo", "then", "do", "done", "fi", "echo", "export", "cd", "for", "if"}


def norm_url(u):
    u = u.rstrip(".,;:")
    u = re.sub(r"[?#].*$", "", u)
    m = re.match(r"https?://([^/]+)(/.*)?", u)
    if not m:
        return []
    host, path = m.group(1).lower(), (m.group(2) or "").rstrip("/")
    ents = [f"host:{host}"]
    if path:
        ents.append(f"url:{host}{path.lower()}")
    return ents


def specific(tok):
    if len(tok) < 4 or tok.startswith("-") or not re.search(r"[A-Za-z]", tok):
        return False
    if not re.search(r"[./_:@=-]", tok):
        return False
    if re.match(r"^(e\.g|i\.e)$", tok, re.I):
        return False
    return True


def entities(line, is_code):
    ents = set()
    text = line
    for u in URL_RE.findall(text):
        ents.update(norm_url(u))
    text = URL_RE.sub(" ", text)
    spans = CODE_SPAN_RE.findall(text)
    chunks = spans if not is_code else [text]
    for ch in chunks:
        if re.search(r"<[^>]+>|\$\{?[A-Z_]+|\{\{", ch) and len(ch) < 6:
            continue
        toks = TOKEN_RE.findall(ch)
        for t in toks:
            if specific(t):
                ents.add(f"tok:{t.lower()}")
        words = [w for w in ch.strip().lstrip("$# ").split() if w]
        while words and words[0] in SKIP_HEADS:
            words = words[1:]
        if len(words) >= 2 and re.match(r"^[a-z][a-z0-9_-]+$", words[0]) and \
                re.match(r"^[a-z][a-z0-9_-]+$", words[1]):
            ents.add(f"cmd:{words[0]} {words[1]}")
    return ents


# ---------- step extraction ----------
FENCE_RE = re.compile(r"^\s*(```|~~~)")


CODE_LANGS = {"sh", "bash", "shell", "console", "zsh", "shell-session", "sql", "ruby", "rb",
              "promql", "python", "py", "plaintext-cmd"}
CMD_START = re.compile(r"^(\$ |sudo |kubectl|gcloud|knife |gitlab-|ssh |curl |git |chef|helm |"
                       r"terraform |gsutil|vault |psql|redis-cli|consul |systemctl|journalctl|"
                       r"docker |glab |tsh |aws |az |bundle |rails |pgbouncer|patronictl|gitaly)")


def steps(text):
    out, in_code, in_gen, lang = [], False, False, ""
    for i, raw in enumerate(text.splitlines()):
        line = raw.rstrip()
        if "<!-- MARKER" in line:
            in_gen = True
        if "END_MARKER" in line:
            in_gen = False
            continue
        m = FENCE_RE.match(line)
        if m:
            in_code = not in_code
            lang = line.strip()[3:].strip().lower().split()[0] if in_code and line.strip()[3:].strip() else ""
            continue
        if in_gen:
            continue
        s = line.strip()
        if len(s) < 8:
            continue
        if in_code:
            if lang and lang not in CODE_LANGS:
                continue  # diagrams, yaml/json config dumps, output
            if not lang and not CMD_START.match(s):
                continue  # untagged blocks: keep only lines that look like commands
            if s.startswith("#") and not s.startswith("#!"):
                continue
            ents = entities(s, True)
            kind = "code"
        else:
            if not (URL_RE.search(s) or "`" in s):
                continue
            ents = entities(s, False)
            kind = "text"
        if ents:
            out.append((i + 1, s, kind, ents))
    return out


def ent_df(rev, files):
    """number of documents mentioning each entity at rev (all actionable lines)."""
    df = defaultdict(set)
    for p in files:
        for _, _, _, ents in steps(read(rev, p)):
            for e in ents:
                df[e].add(p)
    return df


def blame(path):
    """line number -> committer timestamp at T0."""
    out = git("blame", "--line-porcelain", T0, "--", path)
    times, ln, cur = {}, 0, None
    for l in out.splitlines():
        if re.match(r"^[0-9a-f]{40} \d+ (\d+)", l):
            ln = int(l.split()[2])
        elif l.startswith("committer-time "):
            cur = int(l.split()[1])
        elif l.startswith("\t"):
            times[ln] = cur
    return times


def main():
    files0 = md_files(T0)
    filesM = md_files(TM)
    files1 = set(md_files(T1))
    # rename map T0 -> T1
    ren = {}
    for l in git("diff", "-M", "--name-status", T0, T1, "--", "docs").splitlines():
        parts = l.split("\t")
        if parts[0].startswith("R") and len(parts) == 3:
            ren[parts[1]] = parts[2]

    df0 = ent_df(T0, files0)
    dfM = ent_df(TM, filesM)

    with ThreadPoolExecutor(8) as ex:
        blames = dict(zip(files0, ex.map(blame, files0)))

    # per-entity last confirmation time across the whole corpus at T0
    last_conf = defaultdict(int)
    parsed = {}
    for p in files0:
        st = steps(read(T0, p))
        parsed[p] = st
        for ln, _, _, ents in st:
            t = blames[p].get(ln) or 0
            for e in ents:
                last_conf[e] = max(last_conf[e], t)

    doc_commits = {}
    since = dt.datetime.fromtimestamp(T0_TS - 365 * 86400).strftime("%Y-%m-%d")
    for p in files0:
        doc_commits[p] = len(git("rev-list", f"--since={since}", T0, "--", p).split())

    # change propagation: entity retirement events in the 12 months before T0
    retire = defaultdict(list)  # entity -> [(doc, ts)]
    log = git("log", "-p", "--no-color", "-M", "--format=@@@C %ct", f"{TM}..{T0}", "--", "docs")
    # A retirement = entity removed from a document in a commit and not re-added anywhere
    # in that same commit (so content moved between documents is not a retirement).
    cur_ts, cur_file, rem, add = 0, None, set(), set()
    commit_rem, commit_add = [], set()

    def flush():
        if cur_file and cur_file.endswith(".md"):
            commit_rem.append((cur_file, set(rem)))
            commit_add.update(add)

    def flush_commit():
        for f, r in commit_rem:
            for e in r - commit_add:
                retire[e].append((f, cur_ts))

    for l in log.splitlines():
        if l.startswith("@@@C "):
            flush(); flush_commit()
            cur_file, rem, add, commit_rem, commit_add = None, set(), set(), [], set()
            cur_ts = int(l.split()[1])
        elif l.startswith("diff --git "):
            flush(); rem, add = set(), set()
            cur_file = l.split(" b/", 1)[-1]
        elif l.startswith("-") and not l.startswith("---"):
            t = l[1:].strip()
            rem |= entities(t, False) | (entities(t, True) if CMD_START.match(t) else set())
        elif l.startswith("+") and not l.startswith("+++"):
            t = l[1:].strip()
            add |= entities(t, False) | (entities(t, True) if CMD_START.match(t) else set())
    flush(); flush_commit()

    rows = []
    for p in files0:
        p1 = ren.get(p, p)
        if p1 not in files1:
            continue  # whole document deleted or moved out: ambiguous, skip
        t1_text = read(T1, p1)
        t1_lines = {l.strip() for l in t1_text.splitlines()}
        t1_low = t1_text.lower()
        bl = blames[p]
        doc_last = max([t for t in bl.values() if t] or [T0_TS])
        surv = sum(s in t1_lines for _, s, _, _ in parsed[p]) / max(1, len(parsed[p]))
        rewritten = surv < 0.5
        for ln, s, kind, ents in parsed[p]:
            if s in t1_lines:
                label = "valid"
            else:
                def present(e):
                    v = e.split(":", 1)[1]
                    return v in t1_low
                label = "drifted" if any(not present(e) for e in ents) else "reworded"
            lt = bl.get(ln) or T0_TS
            ages = [(T0_TS - last_conf[e]) / 86400 for e in ents]
            dfo = [len(df0[e] - {p}) for e in ents]
            decay = [(len(dfM.get(e, ())) - len(df0[e])) / (len(dfM.get(e, ())) + 1) for e in ents]
            hosts = [e for e in ents if e.startswith("host:")]
            spec = [e for e in ents if not e.startswith("host:")] or list(ents)
            r_other = [len({d for d, _ in retire.get(e, []) if d != p}) for e in spec]
            r_rate = [len({d for d, _ in retire.get(e, []) if d != p}) / (len(dfM.get(e, ())) + 1)
                      for e in spec]
            r_same = [sum(1 for d, _ in retire.get(e, []) if d == p) for e in spec]
            r_last = [min([(T0_TS - t) / 86400 for d, t in retire.get(e, []) if d != p] or [9999])
                      for e in spec]
            rows.append(dict(
                window=T0_DATE, doc=p, line=ln, kind=kind, text=s[:300], label=label,
                doc_rewritten=int(rewritten), doc_survival=round(surv, 3),
                n_ents=len(ents), n_urls=sum(e.startswith("url:") for e in ents),
                n_hosts=len(hosts),
                line_age=(T0_TS - lt) / 86400,
                doc_age=(T0_TS - doc_last) / 86400,
                doc_commits_1y=doc_commits[p],
                ent_conf_age_max=max(ages), ent_conf_age_mean=sum(ages) / len(ages),
                frac_unconfirmed_1y=sum(a > 365 for a in ages) / len(ages),
                df_other_min=min(dfo), df_other_mean=sum(dfo) / len(dfo),
                frac_orphan=sum(d == 0 for d in dfo) / len(dfo),
                decay_max=max(decay), decay_mean=sum(decay) / len(decay),
                retired_other_docs_max=max(r_other), retire_rate_max=max(r_rate),
                retired_same_doc=int(max(r_same) > 0),
                frac_ents_retired_elsewhere=sum(x > 0 for x in r_other) / len(r_other),
                days_since_retired_elsewhere=min(r_last),
                entities="|".join(sorted(ents))[:600],
            ))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    c = Counter(r["label"] for r in rows if not r["doc_rewritten"])
    nrw = len({r["doc"] for r in rows if r["doc_rewritten"]})
    print(f"window {T0_DATE}: T0={T0[:10]} TM={TM[:10]} T1={T1[:10]} docs={len(files0)} "
          f"steps={len(rows)} rewritten_docs={nrw} labels(non-rewritten)={dict(c)}")


if __name__ == "__main__":
    main()
