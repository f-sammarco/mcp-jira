"""The shape of a whole project in one call: issues, epics, parents and links, cached.

Finding what a project already holds is the slow part of most agent workflows: the search
tool returns 50 issues a page, in sequence, with every field the API has, so reading a project
of a thousand issues costs twenty round trips and tens of thousands of tokens, and the
answer is only "complete" if the agent pages to the very end.

This does the same walk once, on the server, and keeps the result:

* the project is split in windows of issue numbers (`key >= P-1 AND key <= P-100`), each one
  page, and the windows are fetched in parallel;
* only the few fields that describe the shape are asked for, and only the fields the agent
  needs go back, as rows, not as one JSON object per issue;
* the snapshot is cached; a later call fetches only what changed since (`updated >= -Nm`)
  and a full walk is repeated now and then, so a deleted or moved issue does not linger;
* every query (text, epic, type, status, label) is answered from the snapshot, so asking
  again with a different filter costs nothing and the answer is complete by construction.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .client import JiraClient
from .utils import quote_jql_identifier_if_needed

logger = logging.getLogger("mcp-jira")

#: What describes the shape of a project; nothing heavy (no description, no comments).
GRAPH_FIELDS = [
    "summary",
    "status",
    "issuetype",
    "labels",
    "parent",
    "components",
    "issuelinks",
    "updated",
]
#: Issue numbers per window: a window holds at most this many issues, so one page each.
WINDOW_KEYS = 100
PAGE_SIZE = 100
MAX_WORKERS = 6
#: A walk of the whole project is repeated after this long, so deletions and moves show up.
FULL_REFRESH_SECONDS = 6 * 3600
#: A snapshot younger than this is answered without asking Jira at all.
DEFAULT_MAX_AGE_SECONDS = 300
#: Safety net: a project larger than this is cut and reported as incomplete.
MAX_ISSUES = 20000
DEFAULT_LIMIT = 200
MAX_LIMIT = 1000

_PROJECT_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_KEY_NUMBER = re.compile(r"-(\d+)$")


@dataclass
class _Snapshot:
    """Everything the last walks of one project returned."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: set[tuple[str, str, str]] = field(default_factory=set)
    synced_at: float = 0.0
    full_at: float = 0.0
    truncated: bool = False


_CACHE: dict[tuple[str, str], _Snapshot] = {}
_LOCK = threading.Lock()


def _number(key: str) -> int:
    found = _KEY_NUMBER.search(key)
    return int(found.group(1)) if found else 0


def _node(issue: dict[str, Any]) -> dict[str, Any]:
    fields = issue.get("fields") or {}
    status = fields.get("status") or {}
    kind = fields.get("issuetype") or {}
    return {
        "key": issue.get("key", ""),
        "type": kind.get("name", ""),
        "epic": kind.get("hierarchyLevel") == 1 or kind.get("name") == "Epic",
        "status": status.get("name", ""),
        "category": (status.get("statusCategory") or {}).get("name", ""),
        "summary": fields.get("summary") or "",
        "parent": (fields.get("parent") or {}).get("key", ""),
        "labels": list(fields.get("labels") or []),
        "components": [c.get("name", "") for c in fields.get("components") or []],
        "updated": fields.get("updated") or "",
    }


def _edges(issue: dict[str, Any]) -> set[tuple[str, str, str]]:
    """Each link once, however many of its ends were fetched: (outward issue, label, inward issue)."""
    key = issue.get("key", "")
    found: set[tuple[str, str, str]] = set()
    for link in (issue.get("fields") or {}).get("issuelinks") or []:
        label = (link.get("type") or {}).get("outward") or (link.get("type") or {}).get(
            "name", "relates to"
        )
        if link.get("outwardIssue"):
            found.add((key, label, link["outwardIssue"].get("key", "")))
        elif link.get("inwardIssue"):
            found.add((link["inwardIssue"].get("key", ""), label, key))
    return found


def _csv(value: str | list[str] | None) -> list[str]:
    if not value:
        return []
    items = value.split(",") if isinstance(value, str) else value
    return [item.strip() for item in items if item and item.strip()]


class ProjectGraphMixin(JiraClient):
    """Whole-project reads from one cached snapshot."""

    if TYPE_CHECKING:
        # Provided by SearchMixin in the composed JiraFetcher: every query here goes through it.
        def _apply_projects_filter(
            self, jql: str, projects_filter: str | None = None
        ) -> str: ...

    # -- walking Jira ---------------------------------------------------------

    def _run_query(self, jql: str) -> list[dict[str, Any]]:
        """Every issue of one JQL query, all pages, with only the graph fields."""
        issues: list[dict[str, Any]] = []
        if self.config.is_cloud:
            body: dict[str, Any] = {
                "jql": jql,
                "fields": GRAPH_FIELDS,
                "maxResults": PAGE_SIZE,
            }
            token: str | None = None
            while True:
                if token:
                    body["nextPageToken"] = token
                response = self.jira.post("rest/api/3/search/jql", json=body)
                if not isinstance(response, dict):
                    raise TypeError(f"Unexpected search response: {type(response)}")
                issues.extend(response.get("issues") or [])
                token = response.get("nextPageToken")
                if not token or len(issues) >= MAX_ISSUES:
                    return issues
        start = 0
        while True:
            response = self.jira.jql(
                jql, fields=",".join(GRAPH_FIELDS), start=start, limit=50
            )
            if not isinstance(response, dict):
                raise TypeError(f"Unexpected search response: {type(response)}")
            page = response.get("issues") or []
            issues.extend(page)
            start += len(page)
            if len(page) < 50 or len(issues) >= MAX_ISSUES:
                return issues

    def _bounds(self, project: str) -> tuple[int, int] | None:
        """The lowest and highest issue numbers the project holds, or None when it is empty."""
        quoted = quote_jql_identifier_if_needed(project)
        ends: list[int] = []
        for order in ("ASC", "DESC"):
            found = self._first_key(f"project = {quoted} ORDER BY key {order}")
            if not found:
                return None
            ends.append(_number(found))
        return ends[0], ends[1]

    def _first_key(self, jql: str) -> str:
        # Like every other query here, inside the operator's project allowlist.
        jql = self._apply_projects_filter(jql)
        if self.config.is_cloud:
            response = self.jira.post(
                "rest/api/3/search/jql",
                json={"jql": jql, "fields": ["key"], "maxResults": 1},
            )
        else:
            response = self.jira.jql(jql, fields="key", start=0, limit=1)
        issues = response.get("issues") if isinstance(response, dict) else None
        return (issues or [{}])[0].get("key", "") if issues else ""

    def _windows(self, project: str, low: int, high: int) -> list[str]:
        quoted = quote_jql_identifier_if_needed(project)
        return [
            f"project = {quoted} AND key >= {project}-{start} "
            f"AND key <= {project}-{min(start + WINDOW_KEYS - 1, high)} ORDER BY key ASC"
            for start in range(low, high + 1, WINDOW_KEYS)
        ]

    def _fetch_all(self, queries: list[str]) -> list[dict[str, Any]]:
        """The queries run in parallel; one that fails is tried once more, then raised."""

        def run(jql: str) -> list[dict[str, Any]]:
            try:
                return self._run_query(self._apply_projects_filter(jql))
            except Exception:  # noqa: BLE001 - a transient failure gets one more try
                time.sleep(0.5)
                return self._run_query(self._apply_projects_filter(jql))

        with ThreadPoolExecutor(
            max_workers=min(MAX_WORKERS, len(queries) or 1)
        ) as pool:
            pages = list(pool.map(run, queries))
        return [issue for page in pages for issue in page]

    # -- the snapshot ---------------------------------------------------------

    def _sync(
        self, project: str, refresh: bool, max_age: int
    ) -> tuple[_Snapshot, str, int]:
        """The snapshot, brought up to date as little as possible: (snapshot, how, queries made)."""
        cache_key = (self.config.url, project.upper())
        now = time.time()
        with _LOCK:
            snapshot = _CACHE.get(cache_key)
        if snapshot and not refresh and now - snapshot.synced_at < max_age:
            return snapshot, "cache", 0
        full = (
            refresh or snapshot is None or now - snapshot.full_at > FULL_REFRESH_SECONDS
        )
        if full:
            bounds = self._bounds(project)
            fresh = _Snapshot(synced_at=now, full_at=now)
            queries = self._windows(project, *bounds) if bounds else []
            issues = self._fetch_all(queries)
            self._absorb(fresh, issues)
            snapshot, how = fresh, "full"
            queried = len(queries) + 2
        else:
            assert snapshot is not None
            minutes = int((now - snapshot.synced_at) / 60) + 5
            quoted = quote_jql_identifier_if_needed(project)
            issues = self._fetch_all(
                [f"project = {quoted} AND updated >= -{minutes}m ORDER BY key ASC"]
            )
            self._absorb(snapshot, issues)
            snapshot.synced_at = now
            how, queried = "incremental", 1
        with _LOCK:
            _CACHE[cache_key] = snapshot
        return snapshot, how, queried

    @staticmethod
    def _absorb(snapshot: _Snapshot, issues: list[dict[str, Any]]) -> None:
        for issue in issues:
            if not issue.get("key"):
                continue
            snapshot.nodes[issue["key"]] = _node(issue)
            snapshot.edges |= _edges(issue)
        if len(snapshot.nodes) >= MAX_ISSUES:
            snapshot.truncated = True

    # -- the answer -----------------------------------------------------------

    def get_project_graph(
        self,
        project_key: str,
        view: str = "outline",
        epic_keys: str | list[str] | None = None,
        query: str | list[str] | None = None,
        types: str | list[str] | None = None,
        statuses: str | list[str] | None = None,
        labels: str | list[str] | None = None,
        include_links: bool = False,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        refresh: bool = False,
        max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    ) -> dict[str, Any]:
        """What a project holds, completely, in as few tokens as it can be said.

        Args:
            project_key: The project, e.g. ``MD``.
            view: ``outline`` (the epics with their child counts, no issues) or ``issues``
                (the issues that match, as rows).
            epic_keys: Only the issues under these epics (and the epics themselves).
            query: Words; an issue matches when its key, summary, labels or components
                contain any of them (case-insensitive).
            types: Only these issue types.
            statuses: Only these statuses (name or category: To Do, In Progress, Done).
            labels: Only issues with one of these labels.
            include_links: Add the issue links that touch the returned rows.
            limit: Rows per call (``next_offset`` continues).
            offset: First row.
            refresh: Walk the whole project again instead of trusting the cache.
            max_age_seconds: A snapshot younger than this is not refreshed at all.

        Returns:
            A dict: how fresh the data is (``as_of``, ``source``), whether it is
            ``complete``, the counts, the epics, and the matching rows.

        Raises:
            ValueError: If the project key is not a Jira project key.
        """
        if not _PROJECT_KEY.match(project_key or ""):
            raise ValueError(f"'{project_key}' is not a Jira project key.")
        project = project_key.upper()
        started = time.time()
        snapshot, how, queried = self._sync(project, refresh, max(0, max_age_seconds))
        nodes = sorted(snapshot.nodes.values(), key=lambda n: _number(n["key"]))

        children: dict[str, int] = {}
        for node in nodes:
            if node["parent"]:
                children[node["parent"]] = children.get(node["parent"], 0) + 1
        by_type: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for node in nodes:
            by_type[node["type"]] = by_type.get(node["type"], 0) + 1
            by_category[node["category"]] = by_category.get(node["category"], 0) + 1
        epics = [
            {
                "key": n["key"],
                "summary": n["summary"],
                "status": n["status"],
                "children": children.get(n["key"], 0),
            }
            for n in nodes
            if n["epic"]
        ]

        result: dict[str, Any] = {
            "project": project,
            "complete": not snapshot.truncated,
            "as_of": datetime.fromtimestamp(snapshot.synced_at, timezone.utc).isoformat(
                timespec="seconds"
            ),
            "source": how,
            "jira_queries": queried,
            "elapsed_ms": int((time.time() - started) * 1000),
            "total_issues": len(nodes),
            "counts": {
                "epics": len(epics),
                "by_type": by_type,
                "by_status_category": by_category,
            },
        }
        if snapshot.truncated:
            result["incomplete_reason"] = (
                f"the project has more than {MAX_ISSUES} issues"
            )
        if view == "outline":
            result["epics"] = epics
            return result

        wanted_epics = {k.upper() for k in _csv(epic_keys)}
        words = [w.lower() for w in _csv(query)]
        wanted_types = {t.lower() for t in _csv(types)}
        wanted_status = {s.lower() for s in _csv(statuses)}
        wanted_labels = {x.lower() for x in _csv(labels)}

        def matches(node: dict[str, Any], terms: list[str]) -> bool:
            if wanted_epics and not (
                node["key"] in wanted_epics or node["parent"] in wanted_epics
            ):
                return False
            if wanted_types and node["type"].lower() not in wanted_types:
                return False
            if wanted_status and not (
                node["status"].lower() in wanted_status
                or node["category"].lower() in wanted_status
            ):
                return False
            if wanted_labels and not wanted_labels & {
                x.lower() for x in node["labels"]
            }:
                return False
            if terms:
                haystack = " ".join(
                    [node["key"], node["summary"], *node["labels"], *node["components"]]
                ).lower()
                return any(term in haystack for term in terms)
            return True

        matched = [n for n in nodes if matches(n, words)]
        note = ""
        if words and not matched and len(words) == 1 and " " in words[0]:
            # Alternatives listed with spaces instead of commas arrive as ONE phrase, and no issue holds
            # the whole phrase. A search that finds nothing for that reason would read as "it does not exist".
            split = words[0].split()
            retried = [n for n in nodes if matches(n, split)]
            if retried:
                words, matched = split, retried
                note = (
                    "The query matched nothing as one phrase, so it was split on spaces. "
                    "Separate alternatives with commas."
                )
        if words and not matched and not note:
            note = (
                "No issue has any of these terms in its key, summary, labels or components "
                "(each comma-separated term is a phrase, case-insensitive)."
            )
        if words:
            result["terms"] = words
        if note:
            result["note"] = note
        limit = max(1, min(limit, MAX_LIMIT))
        offset = max(0, offset)
        page = matched[offset : offset + limit]
        result["matched"] = len(matched)
        result["returned"] = len(page)
        result["next_offset"] = (
            offset + limit if offset + limit < len(matched) else None
        )
        result["columns"] = ["key", "type", "status", "summary", "parent", "labels"]
        result["rows"] = [
            [
                n["key"],
                n["type"],
                n["status"],
                n["summary"],
                n["parent"],
                ",".join(n["labels"]),
            ]
            for n in page
        ]
        if include_links:
            keys = {n["key"] for n in page}
            result["links"] = sorted(
                [
                    list(edge)
                    for edge in snapshot.edges
                    if edge[0] in keys or edge[2] in keys
                ]
            )
        return result
