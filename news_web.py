#!/usr/bin/env python3
"""Web UI for M1-P6 Personalized News Digest (reuses the engine in news_digest.py).
Run:  GROQ_API_KEY=... python3 news_web.py      then open http://127.0.0.1:8766"""
import json, os
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import news_digest as nd

PORT = int(os.environ.get("PORT", "8766"))

def generate(cats, per_cat, offline):
    cats = [c for c in dict.fromkeys(cats) if c in nd.FEEDS] or ["tech"]
    json.dump({"interests": cats}, open(nd.PREFS_FILE, "w"))
    arts, used_offline = nd.load_articles(cats, offline)
    total = len(arts)
    arts, removed = nd.dedupe(arts)
    picked = []
    for c in cats:
        picked += [a for a in arts if a["category"] == c][:per_cat]
    nd._llm_failed = False
    with ThreadPoolExecutor(max_workers=6) as ex:
        sums = list(ex.map(nd.summarize, picked))
    sections = []
    for c in cats:
        items = [{"title": a["title"], "summary": s, "link": "" if a["link"] == "(sample)" else a["link"],
                  "image": a.get("image", ""), "image_small": a.get("image_small", ""),
                  "merged": len(a.get("also", []))} for a, s in zip(picked, sums) if a["category"] == c]
        if items: sections.append({"category": c, "items": items})
    return {"stats": {"scanned": total, "removed": removed, "offline": used_offline, "llm": nd.llm_available() and not nd._llm_failed},
            "sections": sections}

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>The Daily Digest</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--paper:#f6f2e9;--ink:#191714;--mut:#6f685c;--red:#b0261d;--rule:#cfc7b6;--serif:Georgia,"Times New Roman",serif;--sans:-apple-system,"Helvetica Neue",Arial,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);line-height:1.55}
.page{max-width:1040px;margin:0 auto;padding:0 26px 70px}
.mast{text-align:center;padding:30px 0 14px;border-bottom:4px double var(--ink)}
.mast h1{margin:0;font-size:3.1rem;font-weight:700;letter-spacing:-.01em;line-height:1.05}
.mast .meta{display:flex;justify-content:space-between;margin-top:14px;padding-top:8px;border-top:1px solid var(--ink);font:600 .7rem var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--mut)}
.setup{padding:22px 0;border-bottom:1px solid var(--rule)}
.setup h2{margin:0 0 12px;font:700 .72rem var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--mut)}
.chips{display:flex;flex-wrap:wrap;gap:8px}.chips label{cursor:pointer}.chips input{display:none}
.chips span{display:block;padding:7px 15px;border:1px solid var(--ink);font:600 .82rem var(--sans);letter-spacing:.06em;text-transform:uppercase;border-radius:2px;transition:background .15s}
.chips span:hover{background:#ebe5d6}.chips input:checked+span{background:var(--ink);color:var(--paper)}
.row{display:flex;gap:22px;align-items:center;flex-wrap:wrap;margin-top:16px;font:.85rem var(--sans);color:var(--mut)}
.row select{font:inherit;color:var(--ink);background:var(--paper);border:1px solid var(--rule);padding:5px 8px;border-radius:2px}
.row label{display:flex;gap:8px;align-items:center;cursor:pointer}.row input[type=checkbox]{accent-color:var(--ink)}
#go{margin-left:auto;font:700 .82rem var(--sans);letter-spacing:.12em;text-transform:uppercase;background:var(--red);color:#fff;border:0;padding:13px 26px;cursor:pointer;border-radius:2px}
#go:hover{background:#8f1c15}#go:disabled{opacity:.6;cursor:wait}
.status{padding:14px 0;border-bottom:1px solid var(--rule);font:.8rem var(--sans);color:var(--mut)}
.status b{color:var(--ink)}.status span+span::before{content:"\00b7";margin:0 10px}
.sec{margin-top:34px}
.sec h3{margin:0 0 18px;padding-top:7px;border-top:3px solid var(--ink);font:800 .85rem var(--sans);letter-spacing:.2em;text-transform:uppercase;color:var(--red)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:0 34px}
.story{padding:0 0 24px;margin:0 0 24px;border-bottom:1px solid var(--rule);animation:in .5s both}
.story figure{margin:0 0 12px;aspect-ratio:16/9;background:#e6dfcf;overflow:hidden}
.story figure img{width:100%;height:100%;object-fit:cover;display:block;filter:saturate(.92)}
.story h4{margin:0 0 8px;font-size:1.3rem;line-height:1.22;font-weight:700}
.story p{margin:0 0 10px;color:#3a362f;font-size:1rem}
.story .src{font:.78rem var(--sans);color:var(--mut)}.story .src a{color:var(--red);font-weight:700;text-decoration:none;letter-spacing:.04em}.story .src a:hover{text-decoration:underline}
.story .also{font:italic .8rem var(--sans);color:var(--mut);margin-left:10px}
.lead{grid-column:1/-1;display:grid;grid-template-columns:1.15fr 1fr;gap:30px;align-items:center}
.lead.noimg{grid-template-columns:1fr}.lead figure{margin:0}.lead h4{font-size:2rem;line-height:1.15}.lead p{font-size:1.1rem}
.wait{padding:60px 0;text-align:center;font-style:italic;color:var(--mut);font-size:1.1rem}.wait::after{content:"";animation:dots 1.4s steps(4,end) infinite}
@keyframes dots{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}}
@keyframes in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.empty{padding:60px 0;text-align:center;color:var(--mut);font-style:italic}.err{color:var(--red);padding:30px 0;font:600 .95rem var(--sans)}
@media(max-width:760px){.grid{grid-template-columns:1fr}.lead{grid-template-columns:1fr}.mast h1{font-size:2.1rem}.lead h4{font-size:1.5rem}#go{width:100%;margin:0}}
</style></head><body><div class="page">
<div class="mast"><h1>The Daily Digest</h1><div class="meta"><span id="date"></span><span>Personalised edition</span><span>Summaries by AI</span></div></div>
<div class="setup"><h2>Choose your sections</h2><div class="chips" id="chips"></div>
 <div class="row"><label>Stories per section <select id="n"><option>2</option><option selected>3</option><option>4</option><option>5</option></select></label>
 <label><input type="checkbox" id="off"> Use offline sample data</label><button id="go">Build my digest</button></div></div>
<div id="out"><div class="empty">Select your sections and build today's edition.</div></div>
</div>
<script>
const CATS=['tech','science','business','health','world','sports','entertainment'];const $=id=>document.getElementById(id);
$('date').textContent=new Date().toLocaleDateString(undefined,{weekday:'long',day:'numeric',month:'long',year:'numeric'});
CATS.forEach(c=>{const l=document.createElement('label');l.innerHTML='<input type="checkbox" value="'+c+'"'+(['tech','science','business'].includes(c)?' checked':'')+'><span>'+c+'</span>';$('chips').appendChild(l)});
function el(t,c,x){const e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e}
$('go').onclick=async()=>{
 const cats=[...document.querySelectorAll('#chips input:checked')].map(i=>i.value);
 if(!cats.length){alert('Pick at least one section');return}
 $('go').disabled=true;$('go').textContent='Setting the type...';const o=$('out');o.textContent='';o.appendChild(el('div','wait',"Gathering today's stories"));
 try{
  const r=await fetch('/api/digest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cats,n:+$('n').value,offline:$('off').checked})});
  const d=await r.json();if(d.error)throw new Error(d.error);render(d);
 }catch(e){o.textContent='';o.appendChild(el('p','err','Something went wrong: '+e.message))}
 finally{$('go').disabled=false;$('go').textContent='Build my digest'}};
function figure(a,card){if(!a.image)return null;const f=el('figure');const im=document.createElement('img');im.alt='';im.loading='lazy';im.referrerPolicy='no-referrer';let tried=false;
 im.onerror=()=>{if(!tried&&a.image_small){tried=true;im.src=a.image_small}else{f.remove();card.classList.add('noimg')}};im.src=a.image;f.appendChild(im);return f}
function render(d){const o=$('out');o.textContent='';const s=d.stats;
 const st=el('div','status');[[s.scanned,' articles read'],[s.removed,' repeated stories removed'],[null,s.llm?'summaries written by AI':'summaries by text analysis'],[null,s.offline?'sample data':'live BBC feeds']].forEach(([v,t])=>{const sp=el('span');if(v!==null)sp.appendChild(el('b','',String(v)));sp.appendChild(document.createTextNode(t));st.appendChild(sp)});o.appendChild(st);
 let k=0;d.sections.forEach(sec=>{const S=el('div','sec');S.appendChild(el('h3','',sec.category));const g=el('div','grid');
  sec.items.forEach((a,i)=>{const c=el('div','story'+(i===0?' lead':''));c.style.animationDelay=(k++*70)+'ms';const f=figure(a,c);if(f)c.appendChild(f);else if(i===0)c.classList.add('noimg');
   const t=el('div');t.appendChild(el('h4','',a.title));t.appendChild(el('p','',a.summary));const src=el('div','src');
   if(a.link){const l=el('a','','READ THE FULL STORY');l.href=a.link;l.target='_blank';l.rel='noopener';src.appendChild(l)}
   if(a.merged)src.appendChild(el('span','also','+'+a.merged+' similar report'+(a.merged>1?'s':'')+' merged'));t.appendChild(src);c.appendChild(t);g.appendChild(c)});
  S.appendChild(g);o.appendChild(S)});
 window.scrollTo({top:o.offsetTop-10,behavior:'smooth'})}
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body, ctype="application/json"):
        data = body.encode(); self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        self.send(200, PAGE, "text/html") if self.path in ("/", "/index.html") else self.send(404, "{}")
    def do_POST(self):
        if self.path != "/api/digest": return self.send(404, "{}")
        try:
            p = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            self.send(200, json.dumps(generate(list(p.get("cats", [])), max(1, min(8, int(p.get("n", 3)))), bool(p.get("offline")))))
        except Exception as e:
            self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"News digest running at http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    try: srv.serve_forever()
    except KeyboardInterrupt: print("\nStopped.")
