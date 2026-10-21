"""Single-page, polite HTML metadata scraper for a user-provided public listing URL."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

USER_AGENT = "RentVsSellStudentProject/1.0 (+educational single-page fetch)"


def robots_allows(url: str) -> bool:
    parsed = urlparse(url)
    robots_url = urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))
    response = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=15)
    if response.status_code == 404:
        return True
    response.raise_for_status()
    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(response.text.splitlines())
    return parser.can_fetch(USER_AGENT, url)


def jsonld_records(soup: BeautifulSoup) -> list[dict]:
    records = []

    def add_nodes(value) -> None:
        if isinstance(value, list):
            for item in value:
                add_nodes(item)
        elif isinstance(value, dict):
            records.append(value)
            add_nodes(value.get("@graph", []))

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            parsed = json.loads(script.string or script.get_text())
        except (json.JSONDecodeError, TypeError):
            continue
        add_nodes(parsed)
    return records


def first_value(*values):
    return next((str(value).strip() for value in values if value is not None and str(value).strip()), None)


def scrape(url: str) -> dict:
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("網址必須以 http:// 或 https:// 開頭")
    if not robots_allows(url):
        raise PermissionError("robots.txt 不允許讀取此網址；已停止請求。")

    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    records = jsonld_records(soup)

    offer = next((record.get("offers") for record in records if record.get("offers")), None)
    if isinstance(offer, list):
        offer = offer[0] if offer else None
    if not isinstance(offer, dict):
        offer = {}

    og_title = soup.find("meta", property="og:title")
    og_description = soup.find("meta", property="og:description")
    description = soup.find("meta", attrs={"name": "description"})
    item_price = soup.select_one('[itemprop="price"]')
    item_currency = soup.select_one('[itemprop="priceCurrency"]')
    title = first_value(
        og_title.get("content") if og_title else None,
        soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else None,
        soup.title.get_text(" ", strip=True) if soup.title else None,
    )
    price = first_value(
        item_price.get("content") if item_price else None,
        item_price.get_text(" ", strip=True) if item_price else None,
        offer.get("price"),
        offer.get("lowPrice"),
    )
    return {
        "url": response.url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "http_status": response.status_code,
        "title": title,
        "description": first_value(
            og_description.get("content") if og_description else None,
            description.get("content") if description else None,
        ),
        "price": price,
        "currency": first_value(
            item_currency.get("content") if item_currency else None,
            offer.get("priceCurrency"),
        ),
        "json_ld_types": sorted({str(record.get("@type")) for record in records if record.get("@type")}),
        "content_type": response.headers.get("Content-Type"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取單一公開房屋刊登頁的靜態 HTML metadata")
    parser.add_argument("url", help="公開刊登頁網址；一次只請求一個頁面")
    parser.add_argument("--output", default="output/crawled_listing.json", help="JSON 輸出路徑")
    args = parser.parse_args()

    result = scrape(args.url)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n已儲存：{output}")


if __name__ == "__main__":
    main()

