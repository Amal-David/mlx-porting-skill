#!/usr/bin/env python3
"""Collect GitHub contributor pages with receipts for review-only research sweeps."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, load_structured

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
API_VERSION = "2022-11-28"
USER_AGENT = "mlx-model-porting-skill/0.1"
LINK_RE = re.compile(r'\s*<([^>]+)>;\s*rel="([^"]+)"')
RATE_LIMIT_HEADERS = {
    "X-RateLimit-Limit": "limit",
    "X-RateLimit-Remaining": "remaining",
    "X-RateLimit-Used": "used",
    "X-RateLimit-Reset": "reset",
    "X-RateLimit-Resource": "resource",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect GitHub contributor pages with audit receipts")
    parser.add_argument("--repo", default="ml-explore/mlx", help="Repository in owner/name form")
    parser.add_argument("--requested-count", type=int, default=1000, help="Maximum contributors to retain")
    parser.add_argument("--per-page", type=int, default=100, help="GitHub contributors per_page value")
    parser.add_argument("--api-base", default="https://api.github.com", help="GitHub API base URL")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--token-env", default="GITHUB_TOKEN", help="Environment variable containing a GitHub token")
    parser.add_argument("--access-date", help="Access date to record, defaults to today's UTC date")
    parser.add_argument("--offline-fixture", help="Fixture JSON with linked_pages and anonymous_pages; intended for tests")
    parser.add_argument(
        "--output",
        default=str(SKILL_ROOT / "assets" / "contributor-refresh.json"),
        help="Output JSON report path",
    )
    return parser.parse_args()


def get_header(headers: dict[str, str], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


def parse_link_header(value: str | None) -> dict[str, str]:
    links: dict[str, str] = {}
    if not value:
        return links
    for part in value.split(","):
        match = LINK_RE.match(part)
        if match:
            links[match.group(2)] = match.group(1)
    return links


def page_from_url(url: str) -> int | None:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    values = query.get("page")
    if not values:
        return None
    try:
        return int(values[0])
    except ValueError:
        return None


def contributors_url(api_base: str, repo: str, per_page: int, page: int = 1, anon: bool = False) -> str:
    params: dict[str, str | int] = {"per_page": per_page, "page": page}
    if anon:
        params["anon"] = "true"
    query = urllib.parse.urlencode(params)
    return f"{api_base.rstrip('/')}/repos/{repo}/contributors?{query}"


def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.urlencode(sorted(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)))
    return urllib.parse.urlunparse(parsed._replace(query=query))


class GitHubClient:
    def __init__(self, timeout: float, token: str | None):
        self.timeout = timeout
        self.token = token

    def fetch_json(self, url: str) -> tuple[list[Any], dict[str, str], int, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": API_VERSION,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                if not isinstance(body, list):
                    raise SkillError(f"GitHub contributors response is not a list for {url}")
                return body, dict(response.headers.items()), int(response.status), response.url
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise SkillError(f"GitHub request failed for {url}: HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SkillError(f"GitHub request failed for {url}: {exc}") from exc


class FixtureClient:
    def __init__(self, fixture: dict[str, Any], api_base: str, repo: str, per_page: int):
        self.pages: dict[str, dict[str, Any]] = {}
        for key, anon in (("linked_pages", False), ("anonymous_pages", True)):
            for index, page in enumerate(fixture.get(key, []), start=1):
                url = page.get("url") or contributors_url(api_base, repo, per_page, index, anon=anon)
                self.pages[url] = page
                self.pages[normalize_url(url)] = page

    def fetch_json(self, url: str) -> tuple[list[Any], dict[str, str], int, str]:
        page = self.pages.get(url) or self.pages.get(normalize_url(url))
        if page is None:
            raise SkillError(f"Offline contributor fixture has no page for {url}")
        body = page.get("body", [])
        if not isinstance(body, list):
            raise SkillError(f"Offline contributor fixture page is not a list: {url}")
        headers = {str(k): str(v) for k, v in page.get("headers", {}).items()}
        for direct in ("Link", "ETag", "Last-Modified", "X-GitHub-Api-Version-Selected", *RATE_LIMIT_HEADERS):
            if direct in page and direct not in headers:
                headers[direct] = str(page[direct])
        return body, headers, int(page.get("status_code", 200)), url


def rate_limit_receipt(headers: dict[str, str]) -> dict[str, str]:
    receipt: dict[str, str] = {}
    for header, key in RATE_LIMIT_HEADERS.items():
        value = get_header(headers, header)
        if value is not None:
            receipt[key] = value
    return receipt


def page_receipt(url: str, status_code: int, headers: dict[str, str], count: int, retained_count: int) -> dict[str, Any]:
    link_header = get_header(headers, "Link")
    receipt: dict[str, Any] = {
        "page": page_from_url(url),
        "url": url,
        "status_code": status_code,
        "count": count,
        "retained_count": retained_count,
        "link_header": link_header,
        "etag": get_header(headers, "ETag"),
        "last_modified": get_header(headers, "Last-Modified"),
        "rate_limit": rate_limit_receipt(headers),
        "link_rels": sorted(parse_link_header(link_header)),
    }
    selected = get_header(headers, "X-GitHub-Api-Version-Selected")
    if selected:
        receipt["selected_api_version"] = selected
    return receipt


def collect_pages(
    client: GitHubClient | FixtureClient,
    start_url: str,
    requested_count: int,
    *,
    keep_logins: bool,
) -> dict[str, Any]:
    url: str | None = start_url
    count = 0
    logins: list[str] = []
    pages: list[dict[str, Any]] = []
    stop_reason = "not_started"
    seen_urls: set[str] = set()
    while url and count < requested_count:
        normalized = normalize_url(url)
        if normalized in seen_urls:
            raise SkillError(f"Contributor pagination loop detected at {url}")
        seen_urls.add(normalized)
        items, headers, status_code, resolved_url = client.fetch_json(url)
        remaining = requested_count - count
        retained_items = items[:remaining]
        if keep_logins:
            for item in retained_items:
                if isinstance(item, dict) and item.get("login"):
                    logins.append(str(item["login"]))
        retained_count = len(retained_items)
        count += retained_count
        pages.append(page_receipt(resolved_url, status_code, headers, len(items), retained_count))

        links = parse_link_header(get_header(headers, "Link"))
        if count >= requested_count:
            stop_reason = "requested_count_reached"
            break
        next_url = links.get("next")
        if not next_url:
            stop_reason = "link_header_exhausted"
            break
        url = urllib.parse.urljoin(resolved_url, next_url)
    if not pages:
        stop_reason = "no_pages"
    return {
        "count": count,
        "logins": logins,
        "pages": pages,
        "pages_fetched": len(pages),
        "stop_reason": stop_reason,
    }


def validate_args(args: argparse.Namespace) -> None:
    if "/" not in args.repo or args.repo.count("/") != 1:
        raise SkillError("--repo must be in owner/name form")
    if args.requested_count < 1:
        raise SkillError("--requested-count must be at least 1")
    if not 1 <= args.per_page <= 100:
        raise SkillError("--per-page must be between 1 and 100")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    token = os.environ.get(args.token_env) if not args.offline_fixture else None
    if args.offline_fixture:
        client: GitHubClient | FixtureClient = FixtureClient(
            load_structured(args.offline_fixture),
            args.api_base,
            args.repo,
            args.per_page,
        )
    else:
        client = GitHubClient(args.timeout, token)

    linked_start_url = contributors_url(args.api_base, args.repo, args.per_page)
    anonymous_start_url = contributors_url(args.api_base, args.repo, args.per_page, anon=True)
    linked = collect_pages(client, linked_start_url, args.requested_count, keep_logins=True)
    anonymous = collect_pages(client, anonymous_start_url, args.requested_count, keep_logins=False)
    access_date = args.access_date or datetime.now(timezone.utc).date().isoformat()
    generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "review_only": True,
        "repo": args.repo,
        "source": linked_start_url,
        "retrieved": access_date,
        "requested_count": args.requested_count,
        "per_page": args.per_page,
        "linked_user_count": linked["count"],
        "anonymous_author_count": anonymous["count"],
        "pages_fetched": linked["pages_fetched"],
        "anonymous_pages_fetched": anonymous["pages_fetched"],
        "top_logins": linked["logins"],
        "api_receipt": {
            "api_version": API_VERSION,
            "linked_start_url": linked_start_url,
            "anonymous_start_url": anonymous_start_url,
            "linked_stop_reason": linked["stop_reason"],
            "anonymous_stop_reason": anonymous["stop_reason"],
            "linked_pages": linked["pages"],
            "anonymous_pages": anonymous["pages"],
        },
        "instructions": [
            "Treat contributor collection as source-selection evidence, not implementation guidance.",
            "Do not persist raw anonymous author identities; keep only aggregate anon=true counts and page receipts.",
            "Promote a contributor-derived learning only after pinned source review, local validation gates, and rollback conditions.",
            "If repository or code search is rate-limited or incomplete, keep long-tail rescreening open in the backlog.",
        ],
    }


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args)
        dump_json(report, args.output)
        print(
            f"wrote {args.output}: "
            f"{report['linked_user_count']} linked contributors, "
            f"{report['anonymous_author_count']} anon=true author buckets, "
            f"{report['pages_fetched']} linked pages"
        )
        return 0
    except (SkillError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
