# server.py — MCP server for sageitinc.com (STRICT domain only everywhere)
# Python 3.14-ready edition (asyncio.run only; no deprecated loop APIs)
# Site/WordPress tools unchanged in spirit; global search restricted to sageitinc.com.
# Advanced RAG + local PDF gateway (ungated) retained.

import os, sys, re, asyncio, xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urlparse, urljoin, urldefrag, quote_plus, parse_qs as _parse_qs, unquote_plus as _unquote_plus
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import httpx, numpy as np
from bs4 import BeautifulSoup
from fastmcp import FastMCP

# optional: trafilatura can be heavy; keep optional to avoid import death
try:
    import trafilatura
    _TRAF_OK = True
except Exception:
    _TRAF_OK = False

# Stronger desktop headers to avoid bot blocks
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0) Gecko/20100101 Firefox/118.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# -------------------- Config --------------------
STRICT_DOMAIN   = "sageitinc.com"
SITE_BASE       = f"https://{STRICT_DOMAIN}/"
SITEMAP_INDEX   = f"https://{STRICT_DOMAIN}/sitemap_index.xml"
RSS_FEED        = f"https://{STRICT_DOMAIN}/feed/"
WP_API_BASE     = f"https://{STRICT_DOMAIN}/wp-json/wp/v2"
BFS_START       = SITE_BASE

# Embedding model (override by env EMB_MODEL if you want)
EMB_MODEL_NAME  = os.environ.get("EMB_MODEL", "all-MiniLM-L6-v2")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

def _log(msg: str): print(str(msg), file=sys.stderr)
_log(f"[MCP] starting for {STRICT_DOMAIN} with model={EMB_MODEL_NAME}")

ALLOWED_PATHS    = ("/",)
MAX_URLS         = 2000
CHUNK_TOKENS     = 900
CHUNK_OVERLAP    = 120
TOP_K            = 6
CRAWL_TIMEOUT    = 25
TIMEOUT          = 25

SKIP_EXT_RE = re.compile(r".*\.(?:jpg|jpeg|png|gif|svg|webp|ico|zip|rar|7z|mp4|mp3|mov|avi|css|js|xml|woff2?)(\?.*)?$", re.I)
IMAGE_EXT_RE = re.compile(r".*\.(?:jpg|jpeg|png|gif|svg|webp)(\?.*)?$", re.I)
PDF_EXT_RE   = re.compile(r".*\.pdf(\?.*)?$", re.I)

AUTOMATION_LINKS = [
    {"title": "Automation Services Overview", "url": f"https://{STRICT_DOMAIN}/services/automation-anywhere", "type": "url"},
    {"title": "RPA & Hyperautomation — Manufacturing Supplier Automation (PDF)", "url": f"https://{STRICT_DOMAIN}/resources/case_study/digital-modernization-manufacturing-supplier-automation.pdf", "type": "pdf"},
    {"title": "Automation Center of Excellence for a Global Bank (PDF)", "url": f"https://{STRICT_DOMAIN}/resources/case_study/Automation_Center_of_Excellence_for_a_Global_Bank.pdf", "type": "pdf"},
]

HEALTHCARE_PDFS = [
    {"title": "AI Quality Control & Document Management — Healthcare Case Study (PDF)", "url": f"https://{STRICT_DOMAIN}/resources/case_study/ai-quality-control-document-management.pdf", "type": "pdf"},
    {"title": "CMS & HL7 FHIR Interoperability for Healthcare (PDF)", "url": f"https://{STRICT_DOMAIN}/resources/case_study/cms-hl7-fhir-interoperability-for-healthcare.pdf", "type": "pdf"},
    {"title": "Member Experience & CMS Interoperability — Case Study (PDF)", "url": f"https://{STRICT_DOMAIN}/resources/case_study/Member_Experience_and_CMS_Interoperability_case_study.pdf", "type": "pdf"},
    {"title": "Ambulance Service Migration, Support & Automation (PDF)", "url": f"https://{STRICT_DOMAIN}/resources/case_study/ambulance-service-migration-support-automation.pdf", "type": "pdf"},
    {"title": "Veterinary License Renewal Automation (PDF)", "url": f"https://{STRICT_DOMAIN}/resources/case_study/veterinary-license-renewal-automation.pdf", "type": "pdf"},
]

CLOUD_PDFS = [
    {"title": "Cloud Migration & Platform Modernization — Case Study (PDF)", "url": f"https://{STRICT_DOMAIN}/resources/case_study/cloud-migration-and-platform-modernization.pdf", "type": "pdf"},
    {"title": "On-Prem to TIBCO Cloud Integration — Case Study (PDF)", "url": f"https://{STRICT_DOMAIN}/resources/case_study/on-prem-to-tibco-cloud-integration.pdf", "type": "pdf"},
]

# --- Query detectors ---
_AUTOMATION_RE = re.compile(r"\b(automation|automations|hyperautomation|rpa|bot|bots|coe)\b", re.I)
_HEALTHCARE_RE = re.compile(r"\b(health\s*care|healthcare|medical|clinical|provider|hospital|payer|emr|ehr|hl7|fhir|ambulance|veterinar(?:y|ian))\b", re.I)
_CLOUD_RE      = re.compile(r"\b(cloud|aws|azure|gcp|migration|modernization|cloud[-\s]?native|saas|paas|iaas|tibco\s*cloud)\b", re.I)
_AERO_RE       = re.compile(r"\b(aero(?:space|nautics))\b", re.I)  # explicit stop (not on site)

def _mentions_automation(q: str) -> bool: return bool(_AUTOMATION_RE.search(q or ""))
def _mentions_healthcare(q: str) -> bool: return bool(_HEALTHCARE_RE.search(q or ""))
def _mentions_cloud(q: str) -> bool: return bool(_CLOUD_RE.search(q or ""))
def _mentions_aero(q: str) -> bool: return bool(_AERO_RE.search(q or ""))

_PDF_INTENT_RE = re.compile(r"\b(pdf|download|case\s*study|white\s*paper|whitepaper|brochure)\b", re.I)
def _asks_for_pdf(q: str) -> bool: return bool(_PDF_INTENT_RE.search(q or ""))

mcp = FastMCP("sageit-site-wp")

# -------------------- Embeddings --------------------
_MODEL = None
def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMB_MODEL_NAME)
        _log(f"[embeddings] loaded: {EMB_MODEL_NAME} (dim={_MODEL.get_sentence_embedding_dimension()})")
    return _MODEL

def _embed_texts(texts: List[str]) -> np.ndarray:
    if not texts: 
        return np.zeros((0, _model().get_sentence_embedding_dimension()), dtype="float32")
    arr = _model().encode(texts, normalize_embeddings=True)
    return np.asarray(arr, dtype="float32")

# -------------------- Vectors: Website index --------------------
CHUNKS: List[str] = []
EMBS:   List[np.ndarray] = []
METAS:  List[Dict[str, Any]] = []

def _canon(u: str) -> str:
    try:
        if not u: return ""
        u, _ = urldefrag(u)
        p = urlparse(u)
        if p.netloc.lower() != STRICT_DOMAIN: return ""
        scheme = "https"; path = p.path or "/"; query = f"?{p.query}" if p.query else ""
        return f"{scheme}://{STRICT_DOMAIN}{path}{query}"
    except Exception:
        return ""

def _same_host(u: str) -> bool:
    try: return urlparse(u or "").netloc.lower() == STRICT_DOMAIN
    except: return False

def _allowed_path(u: str) -> bool:
    path = urlparse(u).path or "/"
    return any(path.startswith(p) for p in ALLOWED_PATHS)

def _dedup(seq: List[str]) -> List[str]:
    s: Set[str] = set(); out: List[str] = []
    for x in seq:
        if x and x not in s:
            s.add(x); out.append(x)
    return out

def _clean_html_to_text(html: str) -> str:
    if not html: return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script","style","noscript"]): t.decompose()
        text = soup.get_text("\n", strip=True)
        return re.sub(r"\n{3,}", "\n\n", text)
    except Exception:
        return html or ""

def _extract_images_and_links(base_url: str, html: str) -> Tuple[List[str], List[str]]:
    imgs, lnks = [], []
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        for m in soup.find_all("meta", property="og:image", content=True):
            src = urljoin(base_url, m["content"]); cu = _canon(src)
            if cu and IMAGE_EXT_RE.match(cu): imgs.append(cu)
        for img in soup.find_all("img", src=True):
            src = urljoin(base_url, img["src"]); cu = _canon(src)
            if cu and IMAGE_EXT_RE.match(cu): imgs.append(cu)
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"]); cu = _canon(href)
            if not cu: continue
            if SKIP_EXT_RE.match(cu) and not PDF_EXT_RE.match(cu): continue
            lnks.append(cu)
    except Exception:
        pass
    return _dedup(imgs), _dedup(lnks)

async def _aget(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        r = await client.get(url, timeout=CRAWL_TIMEOUT, follow_redirects=True, headers=HTTP_HEADERS)
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        if ("text/html" not in ctype) and ("xml" not in ctype) and ("rss" not in ctype):
            return None
        return r.text
    except Exception as e:
        _log(f"[fetch error] {url} {e}"); return None

def _extract_text(html: str) -> Tuple[str, str]:
    if not html:
        return "", ""
    if _TRAF_OK:
        try:
            md = trafilatura.extract_metadata(html)
            title = (md.title or "") if md else ""
        except Exception:
            title = ""
        try:
            text = trafilatura.extract(html) or ""
        except Exception:
            text = ""
        return (title.strip() or ""), text
    # Fallback without trafilatura
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string if soup.title and soup.title.string else "").strip()
    for t in soup(["script","style","noscript"]): t.decompose()
    text = soup.get_text("\n", strip=True)
    return title, text

def _extract_links(base_url: str, html: str) -> List[str]:
    out = []
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            if SKIP_EXT_RE.match(href): continue
            if not _same_host(href): continue
            cu = _canon(href)
            if not cu or not _allowed_path(cu): continue
            out.append(cu)
    except Exception:
        pass
    return _dedup(out)

def _chunk_words(text: str, size=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    words = (text or "").split()
    if not words: return
    i, n = 0, len(words)
    while i < n:
        j = min(n, i + size); yield " ".join(words[i:j])
        i = j - overlap if (j - overlap) > 0 else j

def _add_chunks(new_chunks: List[str], url: str, title: str, meta: Optional[Dict[str, Any]] = None) -> int:
    if not new_chunks: return 0
    with_headers = [f"[SOURCE] {url}\n[TITLE] {title or url}\n\n{ck}".strip() for ck in new_chunks]
    embs = _embed_texts(with_headers); start = len(CHUNKS)
    for i, ck in enumerate(with_headers):
        CHUNKS.append(ck)
        base = {"url": url, "title": title or url, "chunk_index": start + i, "authority": STRICT_DOMAIN}
        if meta: base.update(meta)
        METAS.append(base); EMBS.append(embs[i])
    return len(with_headers)

# -------------------- Discovery (site) --------------------
async def _fetch_urlset_urls(client: httpx.AsyncClient, sitemap_url: str) -> List[str]:
    try:
        xml = await _aget(client, sitemap_url); 
        if not xml: return []
        root = ET.fromstring(xml)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        locs = [n.text for n in root.findall(".//sm:url/sm:loc", ns) if n.text] or \
               [n.text for n in root.findall(".//url/loc") if n.text]
        out = []
        for u in locs:
            if not _same_host(u): continue
            cu = _canon(u); 
            if not cu: continue
            if SKIP_EXT_RE.match(cu): continue
            if not _allowed_path(cu): continue
            out.append(cu)
        return _dedup(out)
    except Exception as e:
        _log(f"[urlset parse error] {sitemap_url} {e}"); return []

async def _parse_sitemap_index_or_urlset() -> List[str]:
    try:
        async with httpx.AsyncClient(follow_redirects=True, headers=HTTP_HEADERS) as c:
            xml = await _aget(c, SITEMAP_INDEX); 
            if not xml: return []
            root = ET.fromstring(xml)
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            smaps = [n.text for n in root.findall(".//sm:sitemap/sm:loc", ns) if n.text] or \
                    [n.text for n in root.findall(".//sitemap/loc") if n.text]
            urls: List[str] = []
            if smaps:
                for sm in smaps:
                    if not _same_host(sm): continue
                    urls.extend(await _fetch_urlset_urls(c, sm))
            else:
                locs = [n.text for n in root.findall(".//sm:url/sm:loc", ns) if n.text] or \
                       [n.text for n in root.findall(".//url/loc") if n.text]
                for u in locs:
                    if not _same_host(u): continue
                    cu = _canon(u)
                    if cu and not SKIP_EXT_RE.match(cu) and _allowed_path(cu):
                        urls.append(cu)
            return _dedup(urls)
    except Exception as e:
        _log(f"[sitemap index parse error] {e}"); return []

async def _bfs_crawl(start_url: str, limit=MAX_URLS) -> List[str]:
    seen: Set[str] = set(); q: List[str] = [start_url]; out: List[str] = []
    async with httpx.AsyncClient(follow_redirects=True, headers=HTTP_HEADERS) as c:
        while q and len(out) < limit:
            url = _canon(q.pop(0)); 
            if not url or url in seen: continue
            seen.add(url)
            html = await _aget(c, url)
            if not html: continue
            out.append(url)
            for nxt in _extract_links(url, html):
                if nxt not in seen and len(out) + len(q) < limit:
                    q.append(nxt)
    return _dedup(out)

async def _rss_collect_urls() -> List[str]:
    try:
        async with httpx.AsyncClient(follow_redirects=True, headers=HTTP_HEADERS) as c:
            xml = await _aget(c, RSS_FEED); 
            if not xml: return []
        root = ET.fromstring(xml)
        items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
        urls = []
        for it in items:
            link_el = it.find("link")
            if link_el is not None and link_el.text:
                cu = _canon(link_el.text.strip()); 
                if cu: urls.append(cu)
            else:
                href = it.find("{http://www.w3.org/2005/Atom}link")
                if href is not None and href.get("href"):
                    cu = _canon(href.get("href").strip())
                    if cu: urls.append(cu)
        return _dedup(urls)
    except Exception as e:
        _log(f"[rss parse error] {e}"); return []
    

# -------------------- WordPress REST --------------------
def _client(): return httpx.Client(follow_redirects=True, timeout=TIMEOUT, headers=HTTP_HEADERS)

def _wp_get(endpoint: str, params: Dict[str, Any] = None) -> Any:
    url = f"{WP_API_BASE}/{endpoint.lstrip('/')}"
    with _client() as c:
        r = c.get(url, params=params or {}); r.raise_for_status(); return r.json()

def _wp_get_post_by_id(kind: str, pid: int) -> Optional[Dict[str, Any]]:
    try: 
        if pid <= 0: return None
        return _wp_get(f"{kind}/{pid}")
    except Exception as e: 
        _log(f"[wp_get_post_by_id error] {kind}/{pid} {e}"); 
        return None

def _wp_find_by_url(url: str) -> Optional[Dict[str, Any]]:
    cu = _canon(url)
    if not cu: return None
    for kind in ("posts", "pages"):
        try:
            page = 1
            while page <= 10:
                arr = _wp_get(kind, params={"search": "", "per_page": 100, "page": page})
                if not arr: break
                for it in arr:
                    link = it.get("link","")
                    if _canon(link) == cu:
                        it["_kind"] = kind; 
                        return it
                page += 1
        except Exception:
            continue
    return None

def _wp_taxonomy(kind: str, ids: List[int]) -> List[Dict[str, Any]]:
    if not ids: return []
    ids = [int(i) for i in ids if str(i).isdigit() and int(i) > 0]
    out: List[Dict[str, Any]] = []
    for i in ids:
        try:
            it = _wp_get(f"{kind}/{i}", params={"_fields":"id,name,slug,link"})
            link = _canon(it.get("link",""))
            out.append({"id": it.get("id"), "name": it.get("name"),
                        "slug": it.get("slug"), "link": link or "",
                        "authority": STRICT_DOMAIN if link else ""})
        except Exception as e:
            _log(f"[wp_taxonomy {kind}/{i}] {e}")
    return out

def _wp_search_posts_and_pages(query: str, per_page: int = 10, page: int = 1) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    q = (query or "").strip()
    if not q: return []
    per_page = max(1, min(100, per_page)); page = max(1, page)

    for kind in ("posts", "pages"):
        try:
            arr = _wp_get(kind, params={
                "search": q, "per_page": per_page, "page": page,
                "_fields": "id,date,modified,link,title,excerpt,type"
            })
            for it in arr:
                out.append({
                    "id": it.get("id"),
                    "type": it.get("type") or (kind[:-1]),
                    "title": (it.get("title") or {}).get("rendered") or "",
                    "excerpt_html": (it.get("excerpt") or {}).get("rendered") or "",
                    "date": it.get("date") or "",
                    "modified": it.get("modified") or "",
                    "link": it.get("link") or "",
                })
        except Exception as e:
            _log(f"[wp_search {kind}] {e}")

    seen: Set[str] = set(); items: List[Dict[str, Any]] = []
    for it in out:
        cu = _canon(it.get("link","")); 
        if not cu or cu in seen: continue
        seen.add(cu)
        title = BeautifulSoup(it["title"], "html.parser").get_text(" ", strip=True)
        excerpt_txt = _clean_html_to_text(it.pop("excerpt_html", ""))
        items.append({
            "id": it["id"], "type": it["type"], "title": title,
            "excerpt": excerpt_txt, "date": it["date"], "modified": it["modified"],
            "link": cu, "source_url": cu, "snippet": f"{excerpt_txt}\n\nRead more: {cu}",
            "authority": STRICT_DOMAIN
        })
    return items

def _wp_recent(limit: int = 10, include_pages: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    limit = max(1, min(50, limit))
    kinds = [("posts","post")] + ([("pages","page")] if include_pages else [])
    for api, tname in kinds:
        try:
            arr = _wp_get(api, params={"per_page": min(100, limit), "page": 1,
                                       "_fields": "id,date,modified,link,title,excerpt,type"})
            for it in arr:
                cu = _canon(it.get("link","")); 
                if not cu: continue
                out.append({
                    "id": it.get("id"), "type": tname,
                    "title": BeautifulSoup(((it.get("title") or {}).get("rendered") or ""), "html.parser").get_text(" ", strip=True),
                    "excerpt": _clean_html_to_text(((it.get("excerpt") or {}).get("rendered") or "")),
                    "date": it.get("date") or "", "modified": it.get("modified") or "",
                    "link": cu, "source_url": cu, "authority": STRICT_DOMAIN
                })
        except Exception as e:
            _log(f"[wp_recent {api}] {e}")
    seen: Set[str] = set(); items: List[Dict[str, Any]] = []
    for it in out:
        if it["link"] in seen: continue
        seen.add(it["link"]); items.append(it)
        if len(items) >= limit: break
    return items

# ---------- NEW: category + PDF helpers ----------
def _wp_categories_search_raw(query: str, per_page: int = 20) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    try:
        arr = _wp_get("categories", params={
            "search": q, "per_page": max(1, min(100, per_page)),
            "_fields": "id,name,slug,link,count"
        })
        out = []
        for it in arr or []:
            link = _canon(it.get("link",""))
            out.append({
                "id": it.get("id"),
                "name": it.get("name",""),
                "slug": it.get("slug",""),
                "count": it.get("count", 0),
                "link": link,
                "authority": STRICT_DOMAIN if link else ""
            })
        return out
    except Exception as e:
        _log(f"[wp_categories_search_raw] {e}")
        return []

_PDF_LINK_RE = re.compile(r'https?://[^"\']+\.pdf(\?[^"\']*)?$', re.I)

def _extract_pdf_urls_from_html(base_url: str, html: str) -> List[str]:
    urls = []
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            cu = _canon(href)
            if not cu:
                continue
            if PDF_EXT_RE.match(cu) or _PDF_LINK_RE.match(cu):
                urls.append(cu)
    except Exception:
        pass
    return _dedup(urls)

# -------------------- Indexing (site) --------------------
async def _index_urls(urls: List[str], limit=MAX_URLS, meta_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    cnt_pages, cnt_chunks = 0, 0
    async with httpx.AsyncClient(follow_redirects=True, headers=HTTP_HEADERS) as c:
        for url in urls[:limit]:
            html = await _aget(c, url)
            if not html: continue
            title, text = _extract_text(html)
            pieces = list(_chunk_words(text))
            if not pieces: continue
            meta = {}
            if meta_map and url in meta_map: meta["modified"] = meta_map[url]
            added = _add_chunks(pieces, url, title or url, meta=meta)
            cnt_pages += 1; cnt_chunks += added
    return {"pages": cnt_pages, "chunks": cnt_chunks, "total_index": len(CHUNKS)}

def _search_unique_urls(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    if not CHUNKS: return []
    q = _embed_texts([query])[0]; qn = float(np.linalg.norm(q)) + 1e-9
    scored = []
    for i, e in enumerate(EMBS):
        sn = float(np.linalg.norm(e)) + 1e-9
        s = float(np.dot(q, e) / (qn * sn)); scored.append((s, i))
    scored.sort(reverse=True)

    best_by_url: Dict[str, Tuple[float, int]] = {}
    for s, idx in scored:
        m = METAS[idx]; u = _canon(m["url"])
        if not u: continue
        cur = best_by_url.get(u)
        if (cur is None) or (s > cur[0]): best_by_url[u] = (s, idx)

    top = sorted(best_by_url.items(), key=lambda x: x[1][0], reverse=True)[:k]
    out = []
    for rank, (url, (s, idx)) in enumerate(top, start=1):
        m = METAS[idx]; base = CHUNKS[idx][:600].replace("\n", " ")
        snippet = f"{base}\n\nRead more: {url}"
        out.append({
            "rank": rank, "score": round(float(s), 4), "url": url, "source_url": url,
            "title": m.get("title") or url, "snippet": snippet,
            "authority": STRICT_DOMAIN, "modified": m.get("modified","")
        })
    return out

# -------------------- External search (domain-locked) --------------------
DDG_HTML = "https://html.duckduckgo.com/html/?q={q}&kl=wt-wt"

def _ddg_parse(html: str, max_items: int) -> List[Dict[str, Any]]:
    out = []
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        for r in soup.select("a.result__a"):
            title = r.get_text(" ", strip=True)
            href = r.get("href") or ""
            if not href: continue
            out.append({"title": title, "url": href})
            if len(out) >= max_items: break
    except Exception:
        pass
    return out

def _generic_web_search_domain_locked(query: str, num: int) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q: return []
    q = f"site:{STRICT_DOMAIN} {q}"
    url = DDG_HTML.format(q=quote_plus(q))
    items = []
    try:
        with httpx.Client(follow_redirects=True, timeout=TIMEOUT, headers={"User-Agent":"Mozilla/5.0"}) as c:
            r = c.get(url); r.raise_for_status()
            items = _ddg_parse(r.text, max_items=max(1, min(50, num)))
    except Exception as e:
        _log(f"[ddg search error] {e}")
    results = []
    seen = set()
    for it in items:
        cu = _canon(it.get("url",""))
        if not cu:
            continue
        if cu in seen: 
            continue
        seen.add(cu)
        title = it.get("title") or cu
        snippet = f"{title}\n\nRead more: {cu}"
        results.append({
            "title": title, "url": cu, "source_url": cu, "snippet": snippet,
            "authority": STRICT_DOMAIN, "source_type": "site"
        })
    return results

def _social_domain_locked(query: str, num: int) -> List[Dict[str, Any]]:
    return _generic_web_search_domain_locked(query, num)

# -------------------- Advanced RAG: Local Knowledge Base --------------------
try:
    from pypdf import PdfReader
    _PDF_OK = True
except Exception:
    _PDF_OK = False

try:
    from rank_bm25 import BM25Okapi as _BM25
    _BM25_OK = True
except Exception:
    _BM25_OK = False

KB_CHUNKS: List[str] = []
KB_EMBS:   List[np.ndarray] = []
KB_METAS:  List[Dict[str, Any]] = []
KB_TOKENS: List[List[str]] = []
KB_BM25   = None

def _kb_clear():
    KB_CHUNKS.clear(); KB_EMBS.clear(); KB_METAS.clear(); KB_TOKENS.clear()
    global KB_BM25; KB_BM25 = None

def _kb_tokenize(txt: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", (txt or "").lower())

def _kb_add_chunks(chunks: List[str], source_id: str, title: str, meta: Optional[Dict[str,Any]]=None) -> int:
    if not chunks: return 0
    headered = [f"[SOURCE] {source_id}\n[TITLE] {title}\n\n{ck}".strip() for ck in chunks]
    embs = _embed_texts(headered)
    base_idx = len(KB_CHUNKS)
    for i, (ck, emb) in enumerate(zip(headered, embs)):
        KB_CHUNKS.append(ck)
        KB_EMBS.append(emb)
        KB_TOKENS.append(_kb_tokenize(ck))
        m = {"kb_source": source_id, "title": title, "kb_chunk_index": base_idx + i, "authority": "kb"}
        if meta: m.update(meta)
        KB_METAS.append(m)
    return len(chunks)

def _kb_chunk(text: str, size=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    words = (text or "").split()
    if not words: return
    i, n = 0, len(words)
    while i < n:
        j = min(n, i + size)
        yield " ".join(words[i:j])
        i = j - overlap if (j - overlap) > 0 else j

def _kb_read_pdf(fp: Path) -> str:
    if not _PDF_OK: return ""
    try:
        reader = PdfReader(str(fp))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    except Exception as e:
        _log(f"[kb pdf read] {fp} {e}")
        return ""

def _kb_read_html(fp: Path) -> str:
    try:
        html = fp.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script","style","noscript"]): t.decompose()
        return soup.get_text("\n", strip=True)
    except Exception as e:
        _log(f"[kb html read] {fp} {e}")
        return ""

def _kb_read_text(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        _log(f"[kb text read] {fp} {e}")
        return ""

def _kb_ingest_path(path: Path) -> Tuple[int, int]:
    if not path.exists(): return (0, 0)
    added_chunks, files_cnt = 0, 0
    exts = {".md",".txt",".html",".htm",".pdf"}
    files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    for fp in files:
        ext = fp.suffix.lower()
        if ext == ".pdf":
            text = _kb_read_pdf(fp)
        elif ext in {".html",".htm"}:
            text = _kb_read_html(fp)
        else:
            text = _kb_read_text(fp)
        if not text: continue
        chunks = list(_kb_chunk(text))
        source_id = f"kb://{fp.name}"
        added_chunks += _kb_add_chunks(chunks, source_id=source_id, title=fp.stem, meta={"filepath": str(fp)})
        files_cnt += 1
    global KB_BM25
    if _BM25_OK and KB_TOKENS:
        KB_BM25 = _BM25(KB_TOKENS)
        _log(f"[kb] BM25 built over {len(KB_TOKENS)} chunks")
    return (added_chunks, files_cnt)

def _kb_search_internal(query: str, k: int, hybrid: bool = True, alpha: float = 0.65) -> List[Dict[str, Any]]:
    if not KB_CHUNKS: return []
    qvec = _embed_texts([query])[0]; qn = float(np.linalg.norm(qvec)) + 1e-9
    cos_scores: List[Tuple[float, int]] = []
    for i, e in enumerate(KB_EMBS):
        sn = float(np.linalg.norm(e)) + 1e-9
        cos_scores.append((float(np.dot(qvec, e) / (qn * sn)), i))

    bm25_scores: Dict[int, float] = {}
    if hybrid and _BM25_OK and KB_BM25 is not None:
        toks = _kb_tokenize(query)
        raw = KB_BM25.get_scores(toks)
        if len(raw):
            mn, mx = float(np.min(raw)), float(np.max(raw))
            rng = (mx - mn) or 1.0
            for idx, val in enumerate(raw):
                bm25_scores[idx] = (float(val) - mn) / rng

    combined: List[Tuple[float, int]] = []
    for cos, idx in cos_scores:
        if hybrid and bm25_scores:
            score = alpha * ((cos + 1.0) / 2.0) + (1.0 - alpha) * bm25_scores.get(idx, 0.0)
        else:
            score = (cos + 1.0) / 2.0
        combined.append((score, idx))
    combined.sort(reverse=True)

    out = []
    seen_sources: Set[str] = set()
    for rank, (s, idx) in enumerate(combined[:max(1,k)]):
        m = KB_METAS[idx]
        src = m.get("kb_source","kb://")
        if src in seen_sources:
            continue
        seen_sources.add(src)
        base = KB_CHUNKS[idx][:600].replace("\n"," ")
        snippet = f"{base}\n\nRead more: {src}"
        out.append({
            "rank": rank+1, "score": round(float(s),4),
            "url": src, "source_url": src, "title": m.get("title") or src,
            "snippet": snippet, "authority": "kb", "source_type": "kb"
        })
        if len(out) >= k: break
    return out

# -------------------- MCP tools: Site-wide --------------------
@mcp.tool()
def refresh_crawl(limit: int = MAX_URLS, use_sitemap: bool = True, include_wp_api: bool = True, include_rss: bool = True) -> Dict[str, Any]:
    global CHUNKS, EMBS, METAS
    CHUNKS, EMBS, METAS = [], [], []

    urls: List[str] = []
    wp_meta: Dict[str, str] = {}

    if use_sitemap:
        sm = asyncio.run(_parse_sitemap_index_or_urlset()); _log(f"[sitemap urls] {len(sm)}")
        urls.extend(sm)

    if include_wp_api:
        try:
            page = 1
            while True:
                arr = _wp_get("posts", params={"per_page": 100, "page": page, "_fields": "link,modified"})
                if not arr: break
                for it in arr:
                    cu = _canon(it.get("link","")); 
                    if cu:
                        urls.append(cu)
                        if it.get("modified"): wp_meta[cu] = it["modified"]
                page += 1
        except Exception as e:
            _log(f"[wp posts urls] {e}")
        try:
            page = 1
            while True:
                arr = _wp_get("pages", params={"per_page": 100, "page": page, "_fields": "link,modified"})
                if not arr: break
                for it in arr:
                    cu = _canon(it.get("link","")); 
                    if cu:
                        urls.append(cu)
                        if it.get("modified"): wp_meta[cu] = it["modified"]
                page += 1
        except Exception as e:
            _log(f"[wp pages urls] {e}")

    if include_rss:
        rss = asyncio.run(_rss_collect_urls()); _log(f"[rss urls] {len(rss)}"); urls.extend(rss)

    if not urls:
        bfs = asyncio.run(_bfs_crawl(BFS_START, limit=limit)); _log(f"[bfs urls] {len(bfs)} (fallback)")
        urls.extend(bfs)

    urls = _dedup([_canon(u) for u in urls if _canon(u)])
    if limit: urls = urls[:limit]
    stats = asyncio.run(_index_urls(urls, limit=limit, meta_map=wp_meta))
    _log(f"[indexed] {stats}")
    return {"ok": True, "sources": len(urls), "stats": stats}

@mcp.tool()
def search_corpus(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if _mentions_aero(q):
        return [{"rank": 1, "score": 1.0, "url": "", "source_url": "",
                 "title": "Not Found", "snippet": "I dont have that in the site content.",
                 "authority": STRICT_DOMAIN, "modified": "", "source_type": "site"}]

    if _wants_pdf(q):
        pdfs = getpdf(q, k=max(6, k))
        if pdfs and not (len(pdfs) == 1 and ("Not Found" in (pdfs[0].get("title") or ""))):
            return pdfs[:k]

    if _mentions_automation(q):
        bundle = []
        bundle.extend(AUTOMATION_LINKS)
        if _mentions_healthcare(q): bundle.extend(HEALTHCARE_PDFS)
        if _mentions_cloud(q):      bundle.extend(CLOUD_PDFS)

        seen = set(); out_items = []
        for it in bundle:
            u = it["url"]
            if u in seen: continue
            seen.add(u)
            u_display = _to_direct_pdf_if_available(u) if it.get("type") == "pdf" or u.lower().endswith(".pdf") else u
            base = f"[SOURCE] {u}\n[TITLE] {it['title']}\n\nOn-site resource ({it['type'].upper()})."
            snippet = f"{base}\n\nRead more: {u}"
            out_items.append({
                "rank": len(out_items)+1,
                "score": 1.0 - (len(out_items) * 0.001),
                "url": u_display,
                "source_url": u,
                "title": it["title"],
                "snippet": snippet,
                "authority": STRICT_DOMAIN,
                "modified": "",
                "source_type": "site",
            })
        if out_items:
            return out_items[:max(1, min(k, len(out_items)))]

    base = _search_unique_urls(q, max(1, k))
    for b in base:
        b["source_type"] = "site"

    if not base:
        return [{"rank": 1, "score": 1.0, "url": "", "source_url": "",
                 "title": "Not Found", "snippet": "I dont have that in the site content.",
                 "authority": STRICT_DOMAIN, "modified": "", "source_type": "site"}]

    return base

@mcp.tool()
def search_corpus_plus_media(query: str, k: int = TOP_K, images_per_result: int = 5) -> List[Dict[str, Any]]:
    base = search_corpus(query or "", max(1, k))
    if not base: return base
    out: List[Dict[str, Any]] = []
    with httpx.Client(follow_redirects=True, timeout=CRAWL_TIMEOUT, headers=HTTP_HEADERS) as client:
        for r in base:
            url = r.get("url"); imgs, lnks = [], []
            try:
                if url:
                    resp = client.get(url); resp.raise_for_status()
                    html = resp.text; imgs, lnks = _extract_images_and_links(url, html)
            except Exception as e:
                _log(f"[media fetch error] {url} {e}")
            r2 = dict(r); r2["images"] = imgs[:max(0, images_per_result)]; r2["links"] = lnks; out.append(r2)
    return out

@mcp.tool()
def get_assets(url: str, images_limit: int = 20) -> Dict[str, Any]:
    try:
        if not _same_host(url):
            return {"ok": False, "url": url, "error": "off-domain blocked"}
        cu = _canon(url)
        if not cu:
            return {"ok": False, "url": url, "error": "off-domain blocked"}
        r = httpx.get(cu, timeout=30, follow_redirects=True, headers=HTTP_HEADERS)
        r.raise_for_status()
        imgs, lnks = _extract_images_and_links(cu, r.text)
        return {"ok": True, "url": cu, "images": imgs[:max(0, images_limit)], "links": lnks}
    except Exception as e:
        _log(f"[get_assets error] {url} {e}")
        return {"ok": False, "url": url, "error": str(e)}

@mcp.tool()
def get_page(url: str) -> Dict[str, Any]:
    try:
        if not _same_host(url): return {"ok": False, "url": url, "error": "off-domain blocked"}
        cu = _canon(url)
        if not cu:
            return {"ok": False, "url": url, "error": "off-domain blocked"}
        r = httpx.get(cu, timeout=30, follow_redirects=True, headers=HTTP_HEADERS); r.raise_for_status()
        title, text = _extract_text(r.text)
        content_with_source = f"[SOURCE] {cu}\n[TITLE] {title or cu}\n\n{text}".strip()
        text_with_source    = f"{text}\n\nRead more: {cu}"
        return {"ok": True, "url": cu, "source_url": cu, "title": title or cu,
                "text": text, "content_with_source": content_with_source,
                "text_with_source": text_with_source, "authority": STRICT_DOMAIN,
                "source_type": "site"}
    except Exception as e:
        _log(f"[get_page error] {url} {e}"); return {"ok": False, "url": url, "error": str(e)}

def _parse_wp_iso(s: str) -> Optional[datetime]:
    if not s: return None
    # robust ISO parser for '2025-10-30T12:34:56' or with 'Z'
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s[:-1]).replace(tzinfo=timezone.utc)
        if "+" in s[10:] or "-" in s[10:]:
            return datetime.fromisoformat(s)
        return datetime.fromisoformat(s)
    except Exception:
        return None

@mcp.tool()
def get_site_updates(since_days: int = 30, max_items: int = 50) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for m in METAS:
        u = _canon(m.get("url","")); 
        if not u: continue
        mod = m.get("modified","")
        if u not in latest or (mod and mod > latest[u].get("modified","")):
            latest[u] = {"url": u, "title": m.get("title", u), "modified": mod,
                         "authority": STRICT_DOMAIN, "source_type": "site"}
    cutoff = datetime.utcnow() - timedelta(days=max(0, since_days))
    out = []
    for v in latest.values():
        keep = True; mod = v.get("modified","")
        dt = _parse_wp_iso(mod)
        if dt:
            keep = dt.replace(tzinfo=None) >= cutoff
        if keep: out.append(v)
    out.sort(key=lambda x: x.get("modified",""), reverse=True)
    return out[:max(1, max_items)]

# -------------------- WordPress tools --------------------
@mcp.tool()
def wp_posts_search(query: str, per_page: int = 10, page: int = 1) -> List[Dict[str, Any]]:
    return _wp_search_posts_and_pages(query, per_page=per_page, page=page)

@mcp.tool()
def wp_recent_posts(limit: int = 10, include_pages: bool = True) -> List[Dict[str, Any]]:
    return _wp_recent(limit=limit, include_pages=include_pages)

@mcp.tool()
def wp_get_post(id: Optional[int] = None, url: Optional[str] = None) -> Dict[str, Any]:
    try:
        item = None; kind = "posts"
        if id is not None:
            item = _wp_get_post_by_id("posts", int(id)); kind = "posts"
            if not item: item = _wp_get_post_by_id("pages", int(id)); kind = "pages"
        elif url:
            item = _wp_find_by_url(url); 
            if item: kind = item.get("_kind","posts")
        else:
            return {"ok": False, "error": "Provide 'id' or 'url'."}

        if not item: return {"ok": False, "error": "Post not found on sageitinc.com."}

        link = _canon(item.get("link",""))
        title = BeautifulSoup(((item.get("title") or {}).get("rendered") or ""), "html.parser").get_text(" ", strip=True)
        date, mod = item.get("date",""), item.get("modified","")
        html = (item.get("content") or {}).get("rendered") or ""
        if not html:
            full = _wp_get(f"{kind}/{item.get('id')}"); html = (full.get("content") or {}).get("rendered") or ""
        text = _clean_html_to_text(html); imgs, lnks = _extract_images_and_links(link or SITE_BASE, html)

        return {"ok": True, "id": item.get("id"), "type": (item.get("type") or ("post" if kind=="posts" else "page")),
                "title": title, "link": link, "date": date, "modified": mod,
                "text": text, "images": imgs, "links": lnks,
                "text_with_source": f"{text}\n\nRead more: {link}" if link else text,
                "authority": STRICT_DOMAIN, "source_type": "site"}
    except Exception as e:
        _log(f"[wp_get_post error] {e}"); return {"ok": False, "error": str(e)}

@mcp.tool()
def wp_categories(ids: List[int]) -> List[Dict[str, Any]]:
    return _wp_taxonomy("categories", ids or [])

@mcp.tool()
def wp_tags(ids: List[int]) -> List[Dict[str, Any]]:
    return _wp_taxonomy("tags", ids or [])

# -------------------- NEW: category search + posts by category + case study finder --------------------
@mcp.tool()
def wp_categories_search(query: str, per_page: int = 20) -> List[Dict[str, Any]]:
    return _wp_categories_search_raw(query, per_page=per_page)

@mcp.tool()
def wp_posts_by_category(category: str, limit: int = 50, pdfs_only: bool = True) -> List[Dict[str, Any]]:
    if not category:
        return []
    cat_id = None
    cat_name = str(category).strip()
    if str(cat_name).isdigit():
        cat_id = int(cat_name)
    else:
        cats = _wp_categories_search_raw(cat_name, per_page=50)
        for c in cats:
            if c.get("slug","").lower() == cat_name.lower() or c.get("name","").lower() == cat_name.lower():
                cat_id = c["id"]; cat_name = c.get("name", cat_name); break
        if cat_id is None and cats:
            cat_id = cats[0]["id"]; cat_name = cats[0].get("name", cat_name)

    if cat_id is None:
        return []

    items: List[Dict[str, Any]] = []
    page = 1
    fetched_posts = 0
    while True:
        per = 100
        try:
            arr = _wp_get("posts", params={
                "categories": cat_id, "per_page": per, "page": page,
                "_fields": "id,date,modified,link,title,content"
            })
            if not arr:
                break
            for it in arr:
                link = _canon(it.get("link",""))
                if not link: 
                    continue
                title = BeautifulSoup(((it.get("title") or {}).get("rendered") or ""), "html.parser").get_text(" ", strip=True)
                html  = (it.get("content") or {}).get("rendered") or ""
                if pdfs_only:
                    pdfs = _extract_pdf_urls_from_html(link, html)
                    for purl in pdfs:
                        items.append({
                            "type": "pdf",
                            "title": title or purl,
                            "source_url": link,
                            "url": purl,
                            "post_id": it.get("id"),
                            "date": it.get("date",""),
                            "modified": it.get("modified",""),
                            "category": cat_name,
                            "authority": STRICT_DOMAIN,
                            "source_type": "site"
                        })
                else:
                    items.append({
                        "type": "post",
                        "title": title or link,
                        "url": link,
                        "post_id": it.get("id"),
                        "date": it.get("date",""),
                        "modified": it.get("modified",""),
                        "category": cat_name,
                        "authority": STRICT_DOMAIN,
                        "source_type": "site"
                    })
                fetched_posts += 1
                if fetched_posts >= max(1, limit):
                    break
            if fetched_posts >= max(1, limit):
                break
            page += 1
        except Exception as e:
            _log(f"[wp_posts_by_category] {e}")
            break

    if pdfs_only and items and _pdf_gateway_running():
        for it in items:
            if it.get("type") == "pdf":
                try:
                    from urllib.parse import urlparse as _urlparse
                    fname = Path(_urlparse(it["url"]).path).name
                except Exception:
                    fname = ""
                kb_path = (_PDF_GATE_BASE / fname)
                if fname and kb_path.exists() and kb_path.suffix.lower() == ".pdf":
                    it["download_url"] = _pdf_gateway_url_for(fname)

    return items

@mcp.tool()
def case_study_find(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    q = (query or "").lower()
    hints = ["healthcare","automation","cloud","boomi","servicenow","cybersecurity","data","ai","integration","manufacturing","retail","finance","insurance"]
    cand = None
    for h in hints:
        if h in q:
            cand = h
            break
    if cand is None and "case" in q and "study" in q:
        parts = re.findall(r"[a-zA-Z0-9]+", q)
        if parts:
            cand = parts[-1]

    items = []
    if cand:
        items = wp_posts_by_category(cand, limit=max(10, limit), pdfs_only=True)
        if items:
            return items[:limit]

    if cand and not items:
        posts = wp_posts_by_category(cand, limit=max(10, limit), pdfs_only=False)
        return posts[:limit]

    site = search_corpus(f"case study {query}", k=max(50, limit))
    out = []
    for r in site:
        url = r.get("url","")
        if PDF_EXT_RE.match(url):
            out.append({
                "type": "pdf",
                "title": r.get("title", url),
                "url": url,
                "source_url": url,
                "authority": STRICT_DOMAIN,
                "source_type": "site"
            })
        if len(out) >= limit: 
            break
    return out

# -------------------- External tools (domain-locked) --------------------
@mcp.tool()
def open_web_search(query: str, num: int = 10) -> List[Dict[str, Any]]:
    return _generic_web_search_domain_locked(query or "", max(1, min(50, int(num))))

@mcp.tool()
def social_search(query: str, num: int = 10) -> List[Dict[str, Any]]:
    return _social_domain_locked(query or "", max(1, min(50, int(num))))

# -------------------- KB / RAG tools --------------------
@mcp.tool()
def kb_refresh(dir: str = "./kb") -> Dict[str, Any]:
    _kb_clear()
    p = Path(dir)
    added_chunks, files_cnt = _kb_ingest_path(p)
    return {"ok": True, "dir": str(p.resolve()), "chunks": added_chunks, "files": files_cnt, "bm25": bool(_BM25_OK and KB_BM25 is not None)}

@mcp.tool()
def kb_add_text(id: str, text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not id: return {"ok": False, "error": "id required"}
    sid = id if id.startswith("kb://") else f"kb://{id}"
    chunks = list(_kb_chunk(text or ""))
    cnt = _kb_add_chunks(chunks, source_id=sid, title=id, meta=meta or {})
    global KB_BM25
    if _BM25_OK and KB_TOKENS:
        KB_BM25 = _BM25(KB_TOKENS)
    return {"ok": True, "id": sid, "chunks_added": cnt, "bm25": bool(_BM25_OK and KB_BM25 is not None)}

@mcp.tool()
def kb_search(query: str, k: int = 6, hybrid: bool = True, alpha: float = 0.65) -> List[Dict[str, Any]]:
    return _kb_search_internal(query or "", max(1, k), hybrid=bool(hybrid), alpha=float(alpha))

@mcp.tool()
def kb_status() -> Dict[str, Any]:
    return {"chunks": len(KB_CHUNKS), "files": len({m.get('kb_source') for m in KB_METAS}),
            "bm25": bool(_BM25_OK and KB_BM25 is not None), "pdf_support": bool(_PDF_OK)}

@mcp.tool()
def kb_list_files() -> List[Dict[str, Any]]:
    seen = {}
    for m in KB_METAS:
        sid = m.get("kb_source")
        if sid and sid not in seen:
            seen[sid] = {"source": sid, "title": m.get("title",""), "authority":"kb"}
    return list(seen.values())

@mcp.tool()
def kb_delete_source(source_id: str) -> Dict[str, Any]:
    if not source_id: return {"ok": False, "error": "source_id required"}
    sid = source_id if source_id.startswith("kb://") else f"kb://{source_id}"
    keep_idx = [i for i,m in enumerate(KB_METAS) if m.get("kb_source") != sid]
    if len(keep_idx) == len(KB_METAS):
        return {"ok": False, "error": "source not found"}
    new_chunks = [KB_CHUNKS[i] for i in keep_idx]
    new_embs   = [KB_EMBS[i] for i in keep_idx]
    new_metas  = [KB_METAS[i] for i in keep_idx]
    new_tokens = [KB_TOKENS[i] for i in keep_idx]
    KB_CHUNKS[:] = new_chunks
    KB_EMBS[:]   = new_embs
    KB_METAS[:]  = new_metas
    KB_TOKENS[:] = new_tokens
    global KB_BM25
    if _BM25_OK and KB_TOKENS:
        KB_BM25 = _BM25(KB_TOKENS)
    else:
        KB_BM25 = None
    return {"ok": True, "remaining_chunks": len(KB_CHUNKS), "bm25": bool(_BM25_OK and KB_BM25 is not None)}

# -------------------- Router (site → web → KB) --------------------
@mcp.tool()
def find_content(query: str, k: int = 8) -> Dict[str, Any]:
    q = (query or "").strip()
    site_results = search_corpus(q, k=max(1, k))
    web_results = []
    if not site_results:
        web_results = open_web_search(q, num=max(5, k))
    kb_results = []
    if not site_results and not web_results:
        kb_results = kb_search(q, k=max(5, k), hybrid=True, alpha=0.65)
    return {"site_results": site_results, "web_results": web_results, "kb_results": kb_results}

# -------------------- NEW: RAG PDF helpers + secure local download gateway --------------------
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import threading
from http import cookies
import secrets, hmac, hashlib

_PDF_GATE_HOST = "127.0.0.1"
_PDF_GATE_PORT = int(os.environ.get("PDF_GATE_PORT", "8788"))
_PDF_GATE_BASE = Path("./kb").resolve()
_PDF_HTTPD: Optional[ThreadingHTTPServer] = None
_PDF_THREAD: Optional[threading.Thread] = None

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_TOKEN_TTL = 60 * 60
_COOKIE_PREFIX = "sageit_pdf_"
_SECRET = os.environ.get("PDF_GATE_SECRET", secrets.token_hex(16)).encode("utf-8")

_LEADS: List[Dict[str, Any]] = []
_TOKENS: Dict[str, Dict[str, Any]] = {}

def _kb_is_pdf_meta(m: Dict[str, Any]) -> bool:
    fp = (m.get("filepath") or "").strip()
    return bool(fp and Path(fp).suffix.lower() == ".pdf")

def _kb_meta_by_source(source_id: str) -> Optional[Dict[str, Any]]:
    sid = source_id if source_id.startswith("kb://") else f"kb://{source_id}"
    for m in KB_METAS:
        if m.get("kb_source") == sid:
            return m
    return None

def _pdf_gateway_running() -> bool:
    return _PDF_HTTPD is not None

def _pdf_gateway_url_for(filename: str) -> str:
    return f"http://{_PDF_GATE_HOST}:{_PDF_GATE_PORT}/pdf?file={quote_plus(filename)}"

def _sign_token(raw: str) -> str:
    sig = hmac.new(_SECRET, raw.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"

def _verify_token(tok: str) -> Optional[Dict[str, Any]]:
    try:
        raw, sig = tok.rsplit(".", 1)
        good = hmac.new(_SECRET, raw.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(good, sig):
            return None
        data = _TOKENS.get(tok)
        if not data: 
            return None
        if datetime.utcnow().timestamp() - data["issued"] > _TOKEN_TTL:
            _TOKENS.pop(tok, None)
            return None
        return data
    except Exception:
        return None

def _make_token(filename: str, name: str, email: str) -> str:
    raw = f"{filename}|{name}|{email}|{secrets.token_hex(8)}|{int(datetime.utcnow().timestamp())}"
    tok = _sign_token(raw)
    _TOKENS[tok] = {"file": filename, "name": name, "email": email, "issued": datetime.utcnow().timestamp()}
    return tok

def _cookie_name_for(filename: str) -> str:
    base = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]
    return f"{_COOKIE_PREFIX}{base}"

def _lead_add(filename: str, name: str, email: str, ua: str = "", ip: str = ""):
    _LEADS.append({
        "file": filename,
        "name": name,
        "email": email,
        "ts": datetime.utcnow().isoformat() + "Z",
        "ip": ip,
        "ua": ua,
    })

def _read_body(environ_len: Optional[int], rfile) -> bytes:
    try:
        ln = int(environ_len or 0)
        if ln <= 0: return b""
        return rfile.read(ln)
    except Exception:
        return b""

def _parse_form(body: bytes) -> Dict[str, str]:
    try:
        s = body.decode("utf-8", errors="ignore")
        pairs = s.split("&")
        out = {}
        for p in pairs:
            if not p: continue
            if "=" in p:
                k, v = p.split("=", 1)
            else:
                k, v = p, ""
            out[_unquote_plus(k)] = _unquote_plus(v)
        return out
    except Exception:
        return {}

# Simple UNGATED PDF handler
class _PdfOnlyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            pr = urlparse(self.path)
            if pr.path != "/pdf":
                self.send_response(404); self.end_headers(); return

            qs = _parse_qs(pr.query)
            filename = (qs.get("file", [""])[0] or "").strip()
            if not filename:
                self.send_response(400); self.end_headers(); self.wfile.write(b"file parameter required"); return

            target = (_PDF_GATE_BASE / filename).resolve()
            if not str(target).startswith(str(_PDF_GATE_BASE)) or target.suffix.lower() != ".pdf":
                self.send_response(404); self.end_headers(); self.wfile.write(b"pdf not found"); return
            if not target.exists():
                self.send_response(404); self.end_headers(); self.wfile.write(b"pdf not found"); return

            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            _log(f"[pdf gateway GET] {e}")
            try:
                self.send_response(500); self.end_headers(); self.wfile.write(b"internal error")
            except Exception:
                pass

    def log_message(self, fmt, *args):  # quiet
        return

@mcp.tool()
def pdf_direct_link(file_or_source: str) -> Dict[str, Any]:
    if not _pdf_gateway_running():
        return {"ok": False, "error": "PDF gateway not running. Call pdf_gateway_start first."}
    filename = ""
    if file_or_source.startswith("kb://"):
        m = _kb_meta_by_source(file_or_source)
        if not m or not _kb_is_pdf_meta(m):
            return {"ok": False, "error": "PDF source not found in KB"}
        filename = Path(m.get("filepath") or "").name
    else:
        filename = Path(file_or_source).name

    if not filename.lower().endswith(".pdf"):
        return {"ok": False, "error": "Please provide a .pdf filename or a kb://<file>.pdf source id"}

    target = (_PDF_GATE_BASE / filename).resolve()
    if not str(target).startswith(str(_PDF_GATE_BASE)) or not target.exists():
        return {"ok": False, "error": f"PDF not found in gateway directory: {filename}"}

    return {"ok": True, "filename": filename, "download_url": _pdf_gateway_url_for(filename)}

@mcp.tool()
def pdf_gateway_start(host: str = "127.0.0.1", port: int = 8788, kb_dir: str = "./kb") -> Dict[str, Any]:
    global _PDF_HTTPD, _PDF_THREAD, _PDF_GATE_HOST, _PDF_GATE_PORT, _PDF_GATE_BASE
    if _PDF_HTTPD is not None:
        return {"ok": True, "host": _PDF_GATE_HOST, "port": _PDF_GATE_PORT, "kb_dir": str(_PDF_GATE_BASE)}
    _PDF_GATE_HOST = host
    _PDF_GATE_PORT = int(port)
    _PDF_GATE_BASE = Path(kb_dir).resolve()
    _PDF_GATE_BASE.mkdir(parents=True, exist_ok=True)
    try:
        _PDF_HTTPD = ThreadingHTTPServer((_PDF_GATE_HOST, _PDF_GATE_PORT), _PdfOnlyHandler)
        _PDF_THREAD = threading.Thread(target=_PDF_HTTPD.serve_forever, daemon=True)
        _PDF_THREAD.start()
        return {"ok": True, "host": _PDF_GATE_HOST, "port": _PDF_GATE_PORT, "kb_dir": str(_PDF_GATE_BASE)}
    except Exception as e:
        _PDF_HTTPD = None; _PDF_THREAD = None
        return {"ok": False, "error": str(e)}

@mcp.tool()
def pdf_gateway_status() -> Dict[str, Any]:
    return {"running": _pdf_gateway_running(), "host": _PDF_GATE_HOST, "port": _PDF_GATE_PORT, "kb_dir": str(_PDF_GATE_BASE)}

@mcp.tool()
def pdf_gateway_stop() -> Dict[str, Any]:
    global _PDF_HTTPD, _PDF_THREAD
    try:
        if _PDF_HTTPD is None:
            return {"ok": True, "stopped": False}
        _PDF_HTTPD.shutdown()
        _PDF_HTTPD.server_close()
        _PDF_HTTPD = None
        _PDF_THREAD = None
        return {"ok": True, "stopped": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    
# --- PDF intent helpers (kept for search_corpus) ---
_PDF_REQ_RE = re.compile(r"\b(pdf|download|case\s*study\s*pdf|whitepaper|brochure)\b", re.I)
def _wants_pdf(q: str) -> bool:
    return bool(_PDF_REQ_RE.search(q or ""))

def _filename_from_url(u: str) -> str:
    try:
        from urllib.parse import urlparse as _urlparse
        return Path(_urlparse(u).path).name
    except Exception:
        return ""

def _to_direct_pdf_if_available(url: str) -> str:
    fname = _filename_from_url(url)
    if not fname or Path(fname).suffix.lower() != ".pdf":
        return url
    if _pdf_gateway_running():
        kb_file = (_PDF_GATE_BASE / fname)
        if kb_file.exists():
            return _pdf_gateway_url_for(fname)
    return url

@mcp.tool()
def getpdf(query: str, k: int = 10) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    bundle: List[Dict[str, Any]] = []

    if _mentions_automation(q):
        for it in AUTOMATION_LINKS:
            if (it.get("type") == "pdf") or it.get("url","").lower().endswith(".pdf"):
                bundle.append(it)
    if _mentions_healthcare(q):
        bundle.extend(HEALTHCARE_PDFS)
    if _mentions_cloud(q):
        bundle.extend(CLOUD_PDFS)

    seen = set()
    out: List[Dict[str, Any]] = []
    for it in bundle:
        u = it.get("url","")
        if not u or u in seen:
            continue
        seen.add(u)
        direct = _to_direct_pdf_if_available(u)
        out.append({
            "rank": len(out)+1,
            "type": "pdf",
            "title": it.get("title") or _filename_from_url(u) or "PDF",
            "url": direct,
            "source_url": u,
            "authority": STRICT_DOMAIN,
            "source_type": "site"
        })

    if not out:
        site = search_corpus(q, k=max(30, k))
        for r in site:
            u = r.get("url","")
            if PDF_EXT_RE.match(u):
                direct = _to_direct_pdf_if_available(u)
                out.append({
                    "rank": len(out)+1,
                    "type": "pdf",
                    "title": r.get("title") or _filename_from_url(u) or "PDF",
                    "url": direct,
                    "source_url": u,
                    "authority": STRICT_DOMAIN,
                    "source_type": "site"
                })
            if len(out) >= k:
                break

    if not out:
        return [{
            "rank": 1, "type": "pdf", "title": "Not Found",
            "url": "", "source_url": "",
            "authority": STRICT_DOMAIN, "source_type": "site",
            "snippet": "I dont have that in the site content."
        }]

    return out[:max(1, k)]

@mcp.tool()
def kb_list_pdfs() -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for m in KB_METAS:
        if not _kb_is_pdf_meta(m): 
            continue
        sid = m.get("kb_source")
        if sid in seen: 
            continue
        seen.add(sid)
        fp = Path(m.get("filepath")).name if m.get("filepath") else ""
        item = {
            "source": sid,
            "title": m.get("title",""),
            "filename": fp,
            "authority": "kb",
        }
        if fp and _pdf_gateway_running():
            item["download_url"] = _pdf_gateway_url_for(fp)
        out.append(item)
    return out

@mcp.tool()
def kb_find_pdfs(query: str, k: int = 20) -> List[Dict[str, Any]]:
    results = kb_search(query or "", k=max(50, k), hybrid=True, alpha=0.65)
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for r in results:
        m = _kb_meta_by_source(r.get("url",""))
        if not m or not _kb_is_pdf_meta(m): 
            continue
        sid = m["kb_source"]
        if sid in seen: 
            continue
        seen.add(sid)
        fp = Path(m.get("filepath")).name if m.get("filepath") else ""
        item = {
            "rank": len(out)+1,
            "score": r.get("score", 0.0),
            "source": sid,
            "title": m.get("title",""),
            "filename": fp,
            "authority": "kb",
        }
        if fp and _pdf_gateway_running():
            item["download_url"] = _pdf_gateway_url_for(fp)
        out.append(item)
        if len(out) >= k:
            break
    return out

@mcp.tool()
def kb_pdf_download_link(source_id: str) -> Dict[str, Any]:
    m = _kb_meta_by_source(source_id)
    if not m or not _kb_is_pdf_meta(m):
        return {"ok": False, "error": "PDF source not found in KB"}
    fp = Path(m.get("filepath")).name
    if not _pdf_gateway_running():
        return {"ok": False, "error": "PDF gateway not running. Call pdf_gateway_start first."}
    return {"ok": True, "download_url": _pdf_gateway_url_for(fp), "filename": fp}

# --- simple lead tools (left intact) ---
@mcp.tool()
def pdf_leads_list(limit: int = 100) -> Dict[str, Any]:
    return {"ok": True, "count": min(len(_LEADS), limit), "items": _LEADS[-max(1, limit):]}

@mcp.tool()
def pdf_leads_clear() -> Dict[str, Any]:
    _LEADS.clear()
    return {"ok": True}

# -------------------- Grounding prompts --------------------
@mcp.prompt("grounded_only")
def grounded_only():
    """
    System:
      You are STRICTLY grounded to sageitinc.com.
      Always call search_corpus(query) first.
      Use ONLY text from search_corpus and (if needed) get_page(url).
      Do NOT use outside knowledge or other domains.
      If nothing is found, reply EXACTLY: "I dont have that in the site content."
      Include the source link visibly (use 'snippet' which already contains 'Read more: <url>').
    """
    return "Ready."

@mcp.prompt("grounded_wp_only")
def grounded_wp_only():
    """
    System:
      You are STRICTLY grounded to sageitinc.com WordPress content.
      Use ONLY wp_* tools above.
      If nothing relevant is found, reply EXACTLY: "I dont have that in the site content".
      Always include a visible source URL using 'link' or 'snippet'.
    """
    return "Ready."

# -------------------- Main --------------------
if __name__ == "__main__":
    # FastMCP handles stdio; no legacy loop calls (3.14-safe)
    mcp.run(transport="stdio")
