import truststore

truststore.inject_into_ssl()

import json
import time
import xml.etree.ElementTree as ET

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.fnx.co.il"
USER_AGENT = "PhoenixHackathonBot/0.1"
MAX_PAGES = 100
DELAY_SECONDS = 1.5

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = DATA_DIR / "pages.jsonl"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def is_official_domain(url):
    host = (urlparse(url).hostname or "").lower()
    return host == "fnx.co.il" or host.endswith(".fnx.co.il")


def load_robots():
    robots_url = "https://fnx.co.il/robots.txt"

    response = session.get(robots_url, timeout=20)
    response.raise_for_status()

    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(response.text.splitlines())

    sitemap_urls = [
        line.split(":", 1)[1].strip()
        for line in response.text.splitlines()
        if line.lower().startswith("sitemap:")
    ]

    if not sitemap_urls:
        sitemap_urls = [f"{BASE_URL}/sitemap.xml"]

    return parser, sitemap_urls


def read_sitemap(url, visited=None, depth=0):
    if visited is None:
        visited = set()

    if (
        depth > 4
        or len(visited) >= 50
        or url in visited
        or not is_official_domain(url)
    ):
        return []

    visited.add(url)

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError) as error:
        print(f"Could not read sitemap: {url} ({error})")
        return []

    if root.tag.endswith("sitemapindex"):
        results = []

        for element in root.findall(".//{*}loc"):
            if element.text:
                results.extend(
                    read_sitemap(
                        element.text.strip(), visited, depth + 1
                    )
                )

        return results

    if root.tag.endswith("urlset"):
        return [
            element.text.strip()
            for element in root.findall(".//{*}loc")
            if element.text and is_official_domain(element.text.strip())
        ]

    return []


def clean_page(html, url):
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    for element in soup.select(
        "script, style, noscript, svg, form, nav, footer, header"
    ):
        element.decompose()

    main = soup.find("main") or soup.find("article") or soup.body

    if main is None:
        return None

    text = main.get_text(" ", strip=True)
    text = " ".join(text.split())

    if len(text) < 200:
        return None

    return {
        "url": url,
        "title": title,
        "text": text,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    print("Checking robots.txt...")

    try:
        robots, sitemap_urls = load_robots()
    except requests.RequestException as error:
        print(f"Could not verify robots.txt. Stopping: {error}")
        return

    candidates = []

    for sitemap_url in sitemap_urls:
        candidates.extend(read_sitemap(sitemap_url))

    # Remove duplicates and skip non-HTML files and query URLs.
    urls = []
    seen = set()
    excluded_extensions = (
        ".pdf", ".jpg", ".jpeg", ".png", ".webp",
        ".svg", ".zip", ".doc", ".docx", ".xlsx"
    )

    for url in candidates:
        url, _ = urldefrag(url)
        parsed = urlparse(url)

        if not is_official_domain(url):
            continue

        if parsed.query:
            continue

        if parsed.path.lower().endswith(excluded_extensions):
            continue

        if url not in seen:
            seen.add(url)
            urls.append(url)

    allowed_urls = [
        url for url in urls
        if robots.can_fetch(USER_AGENT, url)
    ][:MAX_PAGES]

    if not allowed_urls:
        print("No permitted pages found in the sitemap.")
        print("Check the sitemap URL and robots.txt before proceeding.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for index, url in enumerate(allowed_urls, start=1):
            print(f"[{index}/{len(allowed_urls)}] Fetching {url}")

            try:
                response = session.get(url, timeout=25)

                if response.status_code == 429:
                    print("Rate limit received. Stopping the crawl.")
                    break

                response.raise_for_status()

                content_type = response.headers.get(
                    "Content-Type", ""
                ).lower()

                if "text/html" not in content_type:
                    print("Skipping non-HTML content.")
                    continue

                page = clean_page(response.text, url)

                if page is None:
                    print("Skipping page with insufficient text.")
                    continue

                output.write(
                    json.dumps(page, ensure_ascii=False) + "\n"
                )
                output.flush()
                saved += 1

            except requests.RequestException as error:
                print(f"Could not fetch page: {error}")

            if index < len(allowed_urls):
                time.sleep(DELAY_SECONDS)

    print(f"\nFinished. Saved {saved} pages.")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
