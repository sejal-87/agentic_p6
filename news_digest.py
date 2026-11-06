#!/usr/bin/env python3
"""M1-P6: AI-Based Personalized News Digest Generator.

Usage:
  python3 news_digest.py                      # interactive interest selection
  python3 news_digest.py -i tech,science      # choose interests directly
  python3 news_digest.py -i tech --offline    # use built-in sample articles
Optional: set GROQ_API_KEY to get LLM-written summaries (otherwise extractive NLP is used).
Pipeline: fetch RSS -> clean -> categorize -> de-duplicate -> summarize -> digest.
"""
import argparse, datetime, html, json, math, os, re, sys, urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

FEEDS = {
    "tech": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "health": "https://feeds.bbci.co.uk/news/health/rss.xml",
    "world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "sports": "https://feeds.bbci.co.uk/sport/rss.xml",
    "entertainment": "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
}
PREFS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digest_prefs.json")
STOP = set("""a an the and or but if of to in on at for with by from as is are was were be been it its this that these
those he she they we you i his her their our your has have had will would can could not no new says said after
over into about than more most also up out who what when where how why which""".split())

SAMPLE = {
    "tech": [
        ("Tech giant unveils new AI chip for data centres", "The company announced a new AI chip that it says doubles training speed for large language models. Analysts expect strong demand from cloud providers. The chip ships next year."),
        ("New AI chip from tech giant doubles training speed", "A newly announced AI chip from the tech giant promises to double training speed for large language models, with cloud providers expected to buy heavily."),
        ("Regulators open probe into social media app over teen safety", "Regulators have opened an investigation into a popular social media app over concerns about teenage users. The company says it will cooperate fully."),
    ],
    "science": [
        ("Astronomers spot distant planet with water vapour in atmosphere", "Astronomers using a space telescope detected water vapour in the atmosphere of a distant rocky planet. The finding raises hopes of habitable worlds. Further observations are planned."),
        ("Scientists discover water vapour on faraway planet", "Scientists have detected water vapour in the atmosphere of a faraway rocky planet using a space telescope, raising hopes about habitable worlds."),
    ],
    "business": [
        ("Central bank holds interest rates steady amid inflation worries", "The central bank kept interest rates unchanged on Thursday, citing persistent inflation. Markets had widely expected the decision. Governors signalled cuts may come later."),
        ("Retail sales rise for third month as shoppers return", "Retail sales grew for a third straight month as consumers spent more on clothing and electronics, according to official figures released today."),
    ],
    "health": [
        ("Trial shows new vaccine cuts severe illness by 80 percent", "A large clinical trial found a new vaccine reduced severe illness by 80 percent. Researchers called the results encouraging and said regulators will review the data."),
    ],
    "world": [
        ("Leaders meet for climate summit as pressure mounts", "World leaders gathered for a major climate summit, facing pressure to agree stronger emission cuts. Talks are expected to run through the weekend."),
    ],
    "sports": [
        ("Home side beat rivals 2-1 in dramatic derby", "The home side came from behind to beat their rivals 2-1 in a dramatic derby, with the winner arriving in stoppage time."),
        ("Dramatic derby: home team defeats rivals 2-1 late on", "The home team defeated their rivals 2-1 in a dramatic derby after a late stoppage-time winner."),
    ],
    "entertainment": [
        ("Acclaimed director announces new sci-fi film", "An acclaimed director has announced a new science fiction film, with filming set to begin in the spring and a star-studded cast attached."),
    ],
}

def clean(text):
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()

def fetch_category(cat, limit=12, timeout=8):
    req = urllib.request.Request(FEEDS[cat], headers={"User-Agent": "Mozilla/5.0 NewsDigest/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        root = ET.fromstring(r.read())
    items = []
    for it in root.iter("item"):
        title = clean(it.findtext("title"))
        if title:
            thumb = it.find("{http://search.yahoo.com/mrss/}thumbnail")
            small = thumb.get("url", "") if thumb is not None else ""
            large = small.replace("/ace/standard/240/", "/news/624/") + ".webp" if "/ace/standard/240/" in small else small
            items.append({"title": title, "text": clean(it.findtext("description")),
                          "link": (it.findtext("link") or "").strip(), "category": cat,
                          "image": large, "image_small": small})
        if len(items) >= limit:
            break
    return items

def load_articles(cats, offline):
    arts, used_offline = [], offline
    for c in cats:
        got = []
        if not offline:
            try:
                got = fetch_category(c)
            except Exception as e:
                print(f"  [warn] could not fetch '{c}' ({type(e).__name__}); using sample data", file=sys.stderr)
        if not got:
            used_offline = True
            got = [{"title": t, "text": d, "link": "(sample)", "category": c} for t, d in SAMPLE.get(c, [])]
        arts += got
    return arts, used_offline

def tokens(s):
    return [w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP and len(w) > 2]

def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w

def vectors(docs):
    toks = [[stem(w) for w in tokens(d)] for d in docs]
    df = Counter(w for t in toks for w in set(t))
    n = len(docs)
    vecs = []
    for t in toks:
        tf = Counter(t)
        vecs.append({w: c * (math.log((1 + n) / (1 + df[w])) + 1) for w, c in tf.items()})
    return vecs

def cosine(a, b):
    num = sum(v * b.get(k, 0) for k, v in a.items())
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0

def dedupe(arts, threshold=0.5):
    """Greedy clustering: an article too similar to an already-kept one is merged into it."""
    vecs = vectors([a["title"] + " " + a["text"] for a in arts])
    kept, kvecs, removed = [], [], 0
    for a, v in zip(arts, vecs):
        for k, kv in zip(kept, kvecs):
            if cosine(v, kv) >= threshold:
                k.setdefault("also", []).append(a["link"])
                removed += 1
                break
        else:
            kept.append(a); kvecs.append(v)
    return kept, removed

def extractive_summary(art, max_sents=2, max_chars=260):
    text = art["text"] or art["title"]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
    if not sents:
        return art["title"]
    freq = Counter(stem(w) for w in tokens(art["title"] + " " + text))
    scored = []
    for i, s in enumerate(sents):
        ws = [stem(w) for w in tokens(s)]
        score = sum(freq[w] for w in ws) / (len(ws) or 1) + (0.5 if i == 0 else 0)
        scored.append((score, i, s))
    top = sorted(sorted(scored, reverse=True)[:max_sents], key=lambda x: x[1])
    out = " ".join(s for _, _, s in top)
    return out if len(out) <= max_chars else out[: max_chars - 1].rsplit(" ", 1)[0] + "…"

def llm_available():
    return bool(os.environ.get("GROQ_API_KEY"))

def llm_chat(prompt, max_tokens=300, timeout=40):
    payload = {"model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"), "max_tokens": max_tokens + 1500,
               "reasoning_effort": "low", "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"], "Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 ai-project/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  [LLM unavailable: {type(e).__name__}; using fallback]", file=sys.stderr)
        return None

_llm_failed = False
def llm_summary(art):
    global _llm_failed
    if _llm_failed or not llm_available():
        return None
    out = llm_chat(f"Summarize this news item in at most 2 concise sentences. Reply with the summary only.\nTitle: {art['title']}\nText: {art['text']}", 120, timeout=20)
    if not out: _llm_failed = True
    return out

def summarize(art):
    return llm_summary(art) or extractive_summary(art)

def choose_interests(arg):
    saved = []
    if os.path.exists(PREFS_FILE):
        try: saved = json.load(open(PREFS_FILE)).get("interests", [])
        except Exception: pass
    if arg:
        raw = arg
    elif sys.stdin.isatty():
        print("Available categories:", ", ".join(FEEDS))
        raw = input(f"Your interests (comma separated){' [Enter = ' + ','.join(saved) + ']' if saved else ''}: ").strip()
        raw = raw or ",".join(saved)
    else:
        raw = ",".join(saved)
    cats = [c.strip().lower() for c in raw.split(",") if c.strip().lower() in FEEDS]
    cats = list(dict.fromkeys(cats)) or ["tech", "science", "business"]
    json.dump({"interests": cats}, open(PREFS_FILE, "w"))
    return cats

def build_digest(cats, per_cat, offline):
    arts, used_offline = load_articles(cats, offline)
    total = len(arts)
    arts, removed = dedupe(arts)
    today = datetime.date.today().strftime("%A, %d %B %Y")
    lines = [f"# Your Personalized News Digest — {today}", "",
             f"_Interests: {', '.join(cats)} | {total} articles scanned, {removed} duplicate(s) removed"
             + (" | sample data (offline)_" if used_offline else "_"), ""]
    for c in cats:
        group = [a for a in arts if a["category"] == c][:per_cat]
        if not group:
            continue
        lines += [f"## {c.title()}", ""]
        for a in group:
            lines += [f"**{a['title']}**", summarize(a)]
            if a["link"] != "(sample)":
                lines.append(f"Read more: {a['link']}")
            if a.get("also"):
                lines.append(f"_(+{len(a['also'])} similar report(s) merged)_")
            lines.append("")
    return "\n".join(lines)

def main():
    p = argparse.ArgumentParser(description="Personalized AI news digest")
    p.add_argument("-i", "--interests", help="comma separated, e.g. tech,science,sports")
    p.add_argument("-n", "--per-category", type=int, default=3)
    p.add_argument("--offline", action="store_true", help="use sample data, no internet")
    a = p.parse_args()
    cats = choose_interests(a.interests)
    digest = build_digest(cats, a.per_category, a.offline)
    print("\n" + digest)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"digest_{datetime.date.today()}.md")
    open(out, "w", encoding="utf-8").write(digest)
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
