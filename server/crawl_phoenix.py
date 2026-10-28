#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polite public-site crawler for fnx.co.il.

Creates JSONL records compatible with the Phoenix chatbot corpus:
  {"url": ..., "title": ..., "text": ..., "collected_at": ...}

The crawler checks robots.txt, uses a delay between requests, stays on the
Phoenix domain by default, supports sitemap indexes and PDF text extraction,
and never overwrites the existing corpus unless --output explicitly points
there. If --merge-existing is supplied, existing records are merged safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import (
    parse_qsl,
    urlencode,
    urldefrag,
    urljoin,
    urlsplit,
    urlunsplit,
)
from urllib.robotparser import RobotFileParser

import truststore
truststore.inject_into_ssl()

import requests
from bs4 import BeautifulSoup

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # PDF URLs will be reported as skipped if pypdf is not installed.


USER_AGENT = "PhoenixPublicKnowledgeCrawler/1.0"
BASE_URL = "https://www.fnx.co.il/"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "pages_crawled.jsonl"
DEFAULT_REPORT = Path(__file__).resolve().parent / "data" / "crawl_report.json"
MAX_TEXT_CHARS = 80_000
MIN_TEXT_CHARS = 80
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "msclkid", "yclid", "referrer", "source", "campaignid",
}
UNSTABLE_QUERY_KEYS = {
    "q", "query", "search", "searchterm", "filter", "sort", "session",
    "sid", "token", "access_token", "returnurl", "redirect", "callback",
    "page", "pageindex", "page_number", "page_size", "pagesize", "limit", "offset",
}
SKIP_PATH_PARTS = (
    "/login", "/log-in", "/signin", "/sign-in", "/logout", "/log-out",
    "/personal-area", "/personalarea", "/my-account", "/myaccount",
    "/account/", "/customer-area", "/customerarea", "/checkout", "/cart",
    "/wp-admin", "/admin/", "/search/", "/search?", "/api/", "/graphql",
)
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".css",
    ".js", ".mjs", ".map", ".woff", ".woff2", ".ttf", ".eot", ".mp4",
    ".mov", ".mp3", ".wav", ".zip", ".rar", ".7z", ".exe", ".dmg",
    ".xml", ".json", ".csv", ".xlsx", ".xls", ".doc", ".docx", ".ppt",
    ".pptx", ".ics",
}
RETRY_STATUSES = {429, 500, 502, 503, 504}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_for_hash(text: str) -> str:
    text = normalize_space(text).casefold()
    text = re.sub(r"[^\w\u0590-\u05ff]+", " ", text, flags=re.UNICODE)
    return normalize_space(text)


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def host_allowed(host: str, include_subdomains: bool) -> bool:
    host = (host or "").lower().rstrip(".")
    if host in {"fnx.co.il", "www.fnx.co.il"}:
        return True
    return include_subdomains and host.endswith(".fnx.co.il")


def normalize_url(raw_url: str, base: str = BASE_URL, include_subdomains: bool = False) -> str | None:
    if not raw_url:
        return None
    raw_url = raw_url.strip()
    if raw_url.startswith(("mailto:", "tel:", "javascript:", "data:", "blob:")):
        return None
    absolute = urljoin(base, raw_url)
    absolute, _fragment = urldefrag(absolute)
    parts = urlsplit(absolute)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return None

    host = parts.hostname.lower().rstrip(".")
    if not host_allowed(host, include_subdomains):
        return None

    # The apex domain redirects to www on the public website; normalize it to
    # avoid storing the same page twice under two hostnames.
    if host == "fnx.co.il":
        host = "www.fnx.co.il"
    scheme = "https"
    path = re.sub(r"/{2,}", "/", parts.path or "/")

    # Drop tracking parameters but retain stable functional identifiers.
    query_items = []
    for key, value in parse_qsl(parts.query, keep_blank_values=False):
        key_lower = key.lower()
        if key_lower in TRACKING_PARAMS or key_lower.startswith("utm_"):
            continue
        if key_lower in UNSTABLE_QUERY_KEYS:
            return None
        query_items.append((key, value))
    query_items.sort(key=lambda item: (item[0].lower(), item[1]))
    query = urlencode(query_items, doseq=True)

    path_lower = path.lower()
    if any(part in path_lower for part in SKIP_PATH_PARTS):
        return None
    if path_lower.endswith("/robots.txt") or "sitemap" in path_lower and path_lower.endswith(".xml"):
        return None

    last_segment = path_lower.rsplit("/", 1)[-1]
    dot = last_segment.rfind(".")
    if dot >= 0:
        extension = last_segment[dot:]
        if extension in SKIP_EXTENSIONS:
            return None
        if extension not in {".pdf", ".html", ".htm", ".aspx", ".php", ".asp"}:
            # Avoid downloading unknown binary file types.
            return None

    return urlunsplit((scheme, host, path, query, ""))


class PhoenixCrawler:
    def __init__(
        self,
        start_url: str,
        include_subdomains: bool,
        delay: float,
        timeout: float,
        max_pages: int,
        max_sitemap_urls: int,
        use_sitemaps: bool = True,
    ) -> None:
        self.start_url = normalize_url(start_url, include_subdomains=include_subdomains) or BASE_URL
        self.include_subdomains = include_subdomains
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.max_pages = max_pages
        self.max_sitemap_urls = max_sitemap_urls
        self.use_sitemaps = use_sitemaps
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/pdf,application/xml,text/xml;q=0.9,*/*;q=0.7",
            "Accept-Language": "he-IL,he;q=0.9,en;q=0.6",
        })
        self.robots_cache: dict[str, RobotFileParser | None] = {}
        self.robots_allowed_cache: dict[str, bool] = {}
        self.last_request_at = 0.0
        self.records: list[dict[str, str]] = []
        self.visited: set[str] = set()
        self.pending: deque[str] = deque()
        self.pending_set: set[str] = set()
        self.sitemap_urls_seen: set[str] = set()
        self.content_hashes: set[str] = set()
        self.stats: dict[str, Any] = {
            "started_at": utc_now(),
            "start_url": self.start_url,
            "sitemaps_found": 0,
            "sitemaps_read": 0,
            "urls_discovered_from_sitemaps": 0,
            "urls_discovered_from_links": 0,
            "fetched_html": 0,
            "fetched_pdf": 0,
            "saved_records": 0,
            "duplicate_content_skipped": 0,
            "short_or_empty_pages_skipped": 0,
            "robots_disallowed": 0,
            "http_errors": 0,
            "network_errors": 0,
            "unsupported_content_skipped": 0,
            "noindex_pages_skipped": 0,
            "pdf_extraction_errors": 0,
            "errors": [],
            "max_pages": max_pages,
            "max_pages_reached": False,
        }

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def _request(self, url: str, *, allow_robots_file: bool = False) -> requests.Response | None:
        if not allow_robots_file and not self.can_fetch(url):
            self.stats["robots_disallowed"] += 1
            return None

        for attempt in range(4):
            self._wait_for_rate_limit()
            try:
                response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
                self.last_request_at = time.monotonic()
                if response.status_code in RETRY_STATUSES and attempt < 3:
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        wait = min(20.0, max(1.0, float(retry_after)))
                    except ValueError:
                        wait = min(12.0, 1.5 * (2 ** attempt))
                    logging.warning("HTTP %s for %s; retrying in %.1fs", response.status_code, url, wait)
                    time.sleep(wait)
                    continue
                if response.status_code >= 400:
                    self.stats["http_errors"] += 1
                    if len(self.stats["errors"]) < 200:
                        self.stats["errors"].append({"url": url, "error": f"HTTP {response.status_code}"})
                    return None
                # Validate redirects by host without applying page-only URL filters;
                # sitemap.xml and robots-related endpoints are intentionally filtered
                # from the corpus but must still be fetchable by the crawler.
                final_parts = urlsplit(response.url)
                if (final_parts.scheme.lower() not in {"http", "https"} or
                        not final_parts.hostname or
                        not host_allowed(final_parts.hostname, self.include_subdomains)):
                    self.stats["unsupported_content_skipped"] += 1
                    return None
                return response
            except requests.RequestException as exc:
                self.last_request_at = time.monotonic()
                if attempt < 3:
                    time.sleep(min(12.0, 1.5 * (2 ** attempt)))
                    continue
                self.stats["network_errors"] += 1
                if len(self.stats["errors"]) < 200:
                    self.stats["errors"].append({"url": url, "error": str(exc)[:300]})
                logging.warning("Network error for %s: %s", url, exc)
                return None
        return None

    def can_fetch(self, url: str) -> bool:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if host not in self.robots_allowed_cache:
            robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                # robots.txt must be fetched without recursively checking robots.
                self._wait_for_rate_limit()
                response = self.session.get(robots_url, timeout=self.timeout, allow_redirects=True)
                self.last_request_at = time.monotonic()
                if response.status_code == 404:
                    self.robots_cache[host] = None
                    self.robots_allowed_cache[host] = True
                elif response.status_code == 200:
                    parser.parse(response.text.splitlines())
                    self.robots_cache[host] = parser
                    self.robots_allowed_cache[host] = True
                else:
                    # If robots.txt cannot be read reliably, fail closed rather
                    # than accidentally crawling a disallowed section.
                    self.robots_cache[host] = None
                    self.robots_allowed_cache[host] = False
                    logging.warning("Cannot verify robots.txt for %s (HTTP %s); skipping host", host, response.status_code)
            except requests.RequestException as exc:
                self.robots_cache[host] = None
                self.robots_allowed_cache[host] = False
                logging.warning("Cannot verify robots.txt for %s (%s); skipping host", host, exc)

        if not self.robots_allowed_cache.get(host, False):
            return False
        parser = self.robots_cache.get(host)
        if parser is None:
            return True
        return parser.can_fetch(USER_AGENT, url)

    def _robots_sitemaps(self, host_url: str) -> list[str]:
        parts = urlsplit(host_url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        # can_fetch() already caches and reads robots; reuse it if possible.
        host = parts.hostname or ""
        if host not in self.robots_allowed_cache:
            self.can_fetch(host_url)
        parser = self.robots_cache.get(host)
        result: list[str] = []
        if parser is not None:
            for item in getattr(parser, "sitemaps", []) or []:
                if item:
                    result.append(item.strip())
        # Common locations are tried even if robots.txt does not declare them.
        result.extend([
            urljoin(robots_url, "/sitemap.xml"),
            urljoin(robots_url, "/sitemap_index.xml"),
            urljoin(robots_url, "/sitemap-index.xml"),
        ])
        normalized: list[str] = []
        for item in result:
            parts2 = urlsplit(item)
            if not parts2.hostname or not host_allowed(parts2.hostname, self.include_subdomains):
                continue
            cleaned = urlunsplit((parts2.scheme or "https", parts2.netloc, parts2.path, "", ""))
            if cleaned not in normalized:
                normalized.append(cleaned)
        return normalized

    @staticmethod
    def _xml_locs(root: ET.Element) -> tuple[list[str], list[str]]:
        sitemap_locs: list[str] = []
        page_locs: list[str] = []
        for child in list(root):
            local = child.tag.rsplit("}", 1)[-1].lower()
            if local == "sitemap":
                for descendant in child.iter():
                    if descendant.tag.rsplit("}", 1)[-1].lower() == "loc" and descendant.text:
                        sitemap_locs.append(descendant.text.strip())
                        break
            elif local == "url":
                for descendant in child.iter():
                    if descendant.tag.rsplit("}", 1)[-1].lower() == "loc" and descendant.text:
                        page_locs.append(descendant.text.strip())
                        break
        return sitemap_locs, page_locs

    def discover_sitemaps(self) -> list[str]:
        if not self.use_sitemaps:
            return []
        roots = [self.start_url]
        discovered: list[str] = []
        for root_url in roots:
            for sitemap in self._robots_sitemaps(root_url):
                if sitemap not in discovered:
                    discovered.append(sitemap)
        self.stats["sitemaps_found"] = len(discovered)
        page_urls: list[str] = []
        sitemap_queue = deque(discovered)

        while sitemap_queue and len(page_urls) < self.max_sitemap_urls:
            sitemap_url = sitemap_queue.popleft()
            if sitemap_url in self.sitemap_urls_seen:
                continue
            self.sitemap_urls_seen.add(sitemap_url)
            if not self.can_fetch(sitemap_url):
                continue
            response = self._request(sitemap_url)
            if response is None:
                continue
            content_type = response.headers.get("Content-Type", "").lower()
            body = response.content
            if "xml" not in content_type and not body.lstrip().startswith(b"<?xml") and b"<urlset" not in body[:1000] and b"<sitemapindex" not in body[:1000]:
                continue
            try:
                root = ET.fromstring(body)
            except ET.ParseError as exc:
                logging.warning("Could not parse sitemap %s: %s", sitemap_url, exc)
                continue
            self.stats["sitemaps_read"] += 1
            nested, pages = self._xml_locs(root)
            for nested_url in nested:
                parts = urlsplit(nested_url)
                if parts.hostname and host_allowed(parts.hostname, self.include_subdomains):
                    cleaned = urlunsplit((parts.scheme or "https", parts.netloc, parts.path, "", ""))
                    if cleaned not in self.sitemap_urls_seen:
                        sitemap_queue.append(cleaned)
            for page_url in pages:
                normalized = normalize_url(page_url, include_subdomains=self.include_subdomains)
                if normalized:
                    page_urls.append(normalized)
            logging.info("Read sitemap %s (%d page URLs so far)", sitemap_url, len(page_urls))

        # Keep stable order and cap seed URLs to avoid unbounded sitemap inputs.
        unique = list(dict.fromkeys(page_urls))[: self.max_sitemap_urls]
        self.stats["urls_discovered_from_sitemaps"] = len(unique)
        return unique

    def enqueue(self, url: str) -> None:
        normalized = normalize_url(url, include_subdomains=self.include_subdomains)
        if not normalized or normalized in self.visited or normalized in self.pending_set:
            return
        self.pending.append(normalized)
        self.pending_set.add(normalized)

    def _extract_links(self, soup: BeautifulSoup, page_url: str) -> list[str]:
        found: list[str] = []
        for anchor in soup.find_all("a", href=True):
            normalized = normalize_url(anchor.get("href", ""), base=page_url, include_subdomains=self.include_subdomains)
            if normalized:
                found.append(normalized)
        return found

    @staticmethod
    def _page_title(soup: BeautifulSoup, fallback_url: str) -> str:
        h1 = soup.find("h1")
        if h1:
            title = normalize_space(h1.get_text(" ", strip=True))
            if title:
                return title[:300]
        title_tag = soup.find("title")
        if title_tag:
            title = normalize_space(title_tag.get_text(" ", strip=True))
            if title:
                return title[:300]
        last = urlsplit(fallback_url).path.rstrip("/").rsplit("/", 1)[-1]
        return (last or "הפניקס")[:300]

    @staticmethod
    def _has_noindex(soup: BeautifulSoup, headers: requests.structures.CaseInsensitiveDict) -> bool:
        xrobots = (headers.get("X-Robots-Tag", "") or "").lower()
        if "noindex" in xrobots:
            return True
        for meta in soup.find_all("meta"):
            name = (meta.get("name") or meta.get("property") or "").lower()
            content = (meta.get("content") or "").lower()
            if name in {"robots", "googlebot", "bingbot"} and "noindex" in content:
                return True
        return False

    @staticmethod
    def _extract_html_text(soup: BeautifulSoup) -> str:
        # Remove repeated site chrome and non-content elements before extracting.
        for tag in soup.find_all(["script", "style", "noscript", "svg", "iframe", "nav", "footer", "aside", "form"]):
            tag.decompose()
        for tag in soup.find_all(["header"]):
            # Preserve page-level headers if they contain the only main heading;
            # most global headers have navigation links and little useful text.
            if tag.find("nav") or len(normalize_space(tag.get_text(" ", strip=True))) < 500:
                tag.decompose()
        for selector in [
            ".cookie", ".cookies", ".cookie-banner", ".cookie-consent", ".cookieConsent",
            ".accessibility-widget", ".accessibility-menu", ".skip-link", ".social-share",
        ]:
            for tag in soup.select(selector):
                tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"}) or soup.body or soup
        text = normalize_space(main.get_text(" ", strip=True))
        return text[:MAX_TEXT_CHARS]

    def _save_record(self, url: str, title: str, text: str) -> None:
        text = normalize_space(text)
        if len(text) < MIN_TEXT_CHARS:
            self.stats["short_or_empty_pages_skipped"] += 1
            return
        digest = content_hash(text)
        if len(normalize_for_hash(text)) >= 180 and digest in self.content_hashes:
            self.stats["duplicate_content_skipped"] += 1
            return
        self.content_hashes.add(digest)
        self.records.append({
            "url": url,
            "title": normalize_space(title)[:300] or "הפניקס",
            "text": text,
            "collected_at": utc_now(),
        })

    def _process_pdf(self, response: requests.Response, url: str) -> None:
        if PdfReader is None:
            self.stats["unsupported_content_skipped"] += 1
            if len(self.stats["errors"]) < 200:
                self.stats["errors"].append({"url": url, "error": "PDF skipped: install pypdf"})
            return
        try:
            import io
            reader = PdfReader(io.BytesIO(response.content))
            chunks: list[str] = []
            for page in reader.pages[:150]:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    chunks.append(page_text)
            text = normalize_space(" ".join(chunks))[:MAX_TEXT_CHARS]
            metadata_title = ""
            try:
                metadata_title = str((reader.metadata or {}).get("/Title") or "").strip()
            except Exception:
                pass
            url_name = urlsplit(url).path.rsplit("/", 1)[-1]
            fallback = re.sub(r"[_-]+", " ", re.sub(r"\.pdf$", "", url_name, flags=re.I))
            title = metadata_title or fallback or "מסמך PDF של הפניקס"
            if not text:
                self.stats["pdf_extraction_errors"] += 1
                if len(self.stats["errors"]) < 200:
                    self.stats["errors"].append({"url": url, "error": "No extractable text in PDF (possibly scanned image)"})
                return
            self.stats["fetched_pdf"] += 1
            self._save_record(url, title, text)
        except Exception as exc:
            self.stats["pdf_extraction_errors"] += 1
            if len(self.stats["errors"]) < 200:
                self.stats["errors"].append({"url": url, "error": f"PDF extraction failed: {str(exc)[:250]}"})
            logging.warning("Could not extract PDF %s: %s", url, exc)

    def _process_html(self, response: requests.Response, requested_url: str) -> None:
        final_url = normalize_url(response.url, include_subdomains=self.include_subdomains) or requested_url
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "application/pdf" in content_type or urlsplit(final_url).path.lower().endswith(".pdf"):
            self._process_pdf(response, final_url)
            return
        if "html" not in content_type and not response.text.lstrip().lower().startswith(("<!doctype html", "<html", "<head", "<body")):
            self.stats["unsupported_content_skipped"] += 1
            return
        self.stats["fetched_html"] += 1
        soup = BeautifulSoup(response.content, "html.parser")
        links = self._extract_links(soup, final_url)
        for link in links:
            if link not in self.visited and link not in self.pending_set:
                self.enqueue(link)
                self.stats["urls_discovered_from_links"] += 1

        if self._has_noindex(soup, response.headers):
            self.stats["noindex_pages_skipped"] += 1
            return

        canonical_tag = soup.find("link", rel=lambda value: value and "canonical" in [v.lower() for v in (value if isinstance(value, list) else str(value).split())])
        record_url = final_url
        if canonical_tag and canonical_tag.get("href"):
            canonical = normalize_url(canonical_tag.get("href"), base=final_url, include_subdomains=self.include_subdomains)
            if canonical:
                record_url = canonical
        title = self._page_title(soup, record_url)
        text = self._extract_html_text(soup)
        self._save_record(record_url, title, text)

    def crawl(self) -> list[dict[str, str]]:
        sitemap_pages = self.discover_sitemaps()
        for url in sitemap_pages:
            self.enqueue(url)
        self.enqueue(self.start_url)

        logging.info("Starting crawl; %d URL(s) seeded from sitemaps", len(sitemap_pages))
        while self.pending:
            if self.max_pages > 0 and len(self.visited) >= self.max_pages:
                self.stats["max_pages_reached"] = True
                logging.warning("Reached --max-pages=%d; stopping", self.max_pages)
                break
            url = self.pending.popleft()
            self.pending_set.discard(url)
            if url in self.visited:
                continue
            self.visited.add(url)
            response = self._request(url)
            if response is None:
                continue
            try:
                self._process_html(response, url)
            except Exception as exc:
                logging.exception("Unexpected error processing %s", url)
                if len(self.stats["errors"]) < 200:
                    self.stats["errors"].append({"url": url, "error": f"Processing failed: {str(exc)[:250]}"})
            if len(self.visited) % 25 == 0:
                logging.info("Progress: visited=%d, saved=%d, queued=%d", len(self.visited), len(self.records), len(self.pending))

        self.stats["visited_urls"] = len(self.visited)
        self.stats["queued_urls_remaining"] = len(self.pending)
        self.stats["saved_records"] = len(self.records)
        self.stats["finished_at"] = utc_now()
        return self.records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("url") and obj.get("text"):
                    records.append({
                        "url": str(obj["url"]),
                        "title": str(obj.get("title") or "הפניקס"),
                        "text": str(obj["text"]),
                        "collected_at": str(obj.get("collected_at") or utc_now()),
                    })
            except json.JSONDecodeError:
                logging.warning("Skipping invalid JSONL row %s:%d", path, line_no)
    return records


def merge_records(existing: Iterable[dict[str, Any]], crawled: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    # Crawled content wins for the same URL; old records are retained if the
    # crawler could not rediscover them. Then remove exact duplicate long text.
    by_url: dict[str, dict[str, str]] = {}
    for record in existing:
        by_url[str(record["url"])] = {
            "url": str(record["url"]),
            "title": str(record.get("title") or "הפניקס"),
            "text": str(record.get("text") or ""),
            "collected_at": str(record.get("collected_at") or utc_now()),
        }
    for record in crawled:
        by_url[str(record["url"])] = {
            "url": str(record["url"]),
            "title": str(record.get("title") or "הפניקס"),
            "text": str(record.get("text") or ""),
            "collected_at": str(record.get("collected_at") or utc_now()),
        }

    result: list[dict[str, str]] = []
    seen_content: set[str] = set()
    for record in sorted(by_url.values(), key=lambda item: item["url"]):
        normalized = normalize_for_hash(record["text"])
        if len(normalized) >= 180:
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if digest in seen_content:
                continue
            seen_content.add(digest)
        result.append(record)
    return result


def write_jsonl_atomic(path: Path, records: list[dict[str, str]]) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.name}.bak_{stamp}")
        backup_path.write_bytes(path.read_bytes())
        logging.info("Existing output backed up to %s", backup_path)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)
    return backup_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl public Phoenix (fnx.co.il) pages into chatbot JSONL.")
    parser.add_argument("--start-url", default=BASE_URL, help="Start URL (default: https://www.fnx.co.il/)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output JSONL (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--merge-existing", type=Path, default=None, help="Merge this existing JSONL into the output; crawled pages replace same-URL records")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help=f"Report JSON path (default: {DEFAULT_REPORT})")
    parser.add_argument("--max-pages", type=int, default=10000, help="Maximum pages/PDFs to visit; use 0 for no limit (default: 10000)")
    parser.add_argument("--max-sitemap-urls", type=int, default=100000, help="Maximum sitemap URLs to seed (default: 100000)")
    parser.add_argument("--delay", type=float, default=0.6, help="Minimum seconds between requests (default: 0.6)")
    parser.add_argument("--timeout", type=float, default=25.0, help="HTTP timeout in seconds (default: 25)")
    parser.add_argument("--include-subdomains", action="store_true", help="Also crawl subdomains of fnx.co.il (each host's robots.txt is checked)")
    parser.add_argument("--no-sitemaps", action="store_true", help="Disable sitemap discovery and use internal links only")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.max_pages < 0:
        parser.error("--max-pages must be 0 (unlimited) or a positive integer")
    if args.max_sitemap_urls < 1:
        parser.error("--max-sitemap-urls must be positive")
    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)

    existing_records: list[dict[str, Any]] = []
    if args.merge_existing:
        existing_records = read_jsonl(args.merge_existing)
        logging.info("Loaded %d existing records from %s", len(existing_records), args.merge_existing)

    crawler = PhoenixCrawler(
        start_url=args.start_url,
        include_subdomains=args.include_subdomains,
        delay=args.delay,
        timeout=args.timeout,
        max_pages=args.max_pages,
        max_sitemap_urls=args.max_sitemap_urls,
        use_sitemaps=not args.no_sitemaps,
    )
    crawled_records = crawler.crawl()
    if args.merge_existing:
        output_records = merge_records(existing_records, crawled_records)
    else:
        output_records = merge_records([], crawled_records)

    backup = write_jsonl_atomic(args.output, output_records)
    crawler.stats["final_record_count"] = len(output_records)
    crawler.stats["output_file"] = str(args.output.resolve())
    crawler.stats["merge_existing_file"] = str(args.merge_existing.resolve()) if args.merge_existing else None
    crawler.stats["backup_file"] = str(backup.resolve()) if backup else None
    crawler.stats["finished_at"] = utc_now()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(crawler.stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Phoenix crawl summary ===")
    print(f"Visited URLs:                {crawler.stats.get('visited_urls', 0)}")
    print(f"HTML pages fetched:           {crawler.stats.get('fetched_html', 0)}")
    print(f"PDFs extracted:               {crawler.stats.get('fetched_pdf', 0)}")
    print(f"New records kept:             {len(crawled_records)}")
    print(f"Final records in output:       {len(output_records)}")
    print(f"Duplicate content skipped:    {crawler.stats.get('duplicate_content_skipped', 0)}")
    print(f"Robots disallowed:             {crawler.stats.get('robots_disallowed', 0)}")
    print(f"HTTP/network errors:           {crawler.stats.get('http_errors', 0) + crawler.stats.get('network_errors', 0)}")
    print(f"Max pages reached:             {crawler.stats.get('max_pages_reached', False)}")
    print(f"Output:                        {args.output.resolve()}")
    print(f"Report:                        {args.report.resolve()}")
    if backup:
        print(f"Backup:                        {backup.resolve()}")
    if PdfReader is None:
        print("NOTE: pypdf is not installed, so PDF text extraction is disabled. Install it with: python -m pip install pypdf")
    if crawler.stats.get("max_pages_reached"):
        print("NOTE: The page limit was reached. Increase --max-pages or use --max-pages 0 if you want an unlimited crawl.")
    if crawler.stats.get("errors"):
        print(f"Some errors were recorded in the report ({len(crawler.stats['errors'])} listed). Review crawl_report.json.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCrawl interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
