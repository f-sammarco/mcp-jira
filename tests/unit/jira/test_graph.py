"""Tests for the Jira ProjectGraphMixin: one walk, one snapshot, every question answered from it."""

import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from mcp_atlassian.jira import graph
from mcp_atlassian.jira import JiraFetcher


def issue(
    number: int,
    *,
    kind: str = "Story",
    status: str = "Da fare",
    category: str = "To Do",
    parent: str = "",
    labels: tuple[str, ...] = (),
    summary: str | None = None,
    links: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    return {
        "key": f"MD-{number}",
        "fields": {
            "summary": summary or f"Issue {number}",
            "status": {"name": status, "statusCategory": {"name": category}},
            "issuetype": {"name": kind, "hierarchyLevel": 1 if kind == "Epic" else 0},
            "labels": list(labels),
            "parent": {"key": parent} if parent else None,
            "components": [],
            "issuelinks": list(links),
            "updated": "2026-11-16T10:00:00.000+0000",
        },
    }


class FakeCloud:
    """A Jira Cloud search endpoint over a list of issues; records every JQL it was asked."""

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        self.issues = {i["key"]: i for i in issues}
        self.changed: list[dict[str, Any]] = []
        self.queries: list[str] = []

    def number(self, key: str) -> int:
        return int(key.rsplit("-", 1)[1])

    def post(self, path: str, json: dict[str, Any]) -> dict[str, Any]:
        jql = json["jql"]
        self.queries.append(jql)
        ordered = sorted(self.issues.values(), key=lambda i: self.number(i["key"]))
        if "updated >=" in jql:
            return {"issues": self.changed}
        window = re.search(r"key >= MD-(\d+) AND key <= MD-(\d+)", jql)
        if window:
            low, high = int(window.group(1)), int(window.group(2))
            found = [i for i in ordered if low <= self.number(i["key"]) <= high]
            return {"issues": found}
        if "DESC" in jql:
            return {"issues": ordered[-1:]}
        return {"issues": ordered[:1]}


@pytest.fixture(autouse=True)
def empty_cache():
    graph._CACHE.clear()
    yield
    graph._CACHE.clear()


@pytest.fixture
def project() -> list[dict[str, Any]]:
    """Two epics (1, 2) and 248 issues; every third one under epic 1, the rest under epic 2."""
    issues = [
        issue(1, kind="Epic", summary="Gallery"),
        issue(
            2, kind="Epic", summary="Player", status="In corso", category="In Progress"
        ),
    ]
    for number in range(3, 251):
        issues.append(
            issue(
                number,
                parent="MD-1" if number % 3 == 0 else "MD-2",
                kind="Bug" if number % 50 == 0 else "Story",
                labels=("DEV-FE",) if number % 2 == 0 else (),
                summary=f"Sticky playlist {number}"
                if number == 120
                else f"Issue {number}",
            )
        )
    return issues


@pytest.fixture
def cloud(jira_fetcher: JiraFetcher, project):
    fake = FakeCloud(project)
    jira_fetcher.config.url = "https://test.atlassian.net"
    jira_fetcher.config.projects_filter = None
    jira_fetcher.config.url = "https://test.atlassian.net"
    jira_fetcher.jira = MagicMock()
    jira_fetcher.jira.post.side_effect = fake.post
    return fake


def test_walks_the_whole_project_in_parallel_windows(jira_fetcher, cloud):
    result = jira_fetcher.get_project_graph("MD")

    assert result["complete"] is True
    assert result["source"] == "full"
    assert result["total_issues"] == 250
    # Two queries find the ends of the project, then one query per window of 100 numbers.
    assert result["jira_queries"] == 2 + 3
    assert len(cloud.queries) == 2 + 3
    assert result["counts"]["epics"] == 2
    assert result["counts"]["by_type"] == {"Epic": 2, "Story": 243, "Bug": 5}
    assert "rows" not in result


def test_outline_lists_every_epic_with_its_size(jira_fetcher, cloud):
    result = jira_fetcher.get_project_graph("MD", view="outline")

    by_key = {epic["key"]: epic for epic in result["epics"]}
    assert by_key["MD-1"]["summary"] == "Gallery"
    assert by_key["MD-1"]["children"] + by_key["MD-2"]["children"] == 248
    assert by_key["MD-2"]["status"] == "In corso"


def test_issues_view_answers_filters_from_the_snapshot_without_asking_jira(
    jira_fetcher, cloud
):
    jira_fetcher.get_project_graph("MD")
    asked = len(cloud.queries)

    under_epic = jira_fetcher.get_project_graph(
        "MD", view="issues", epic_keys="MD-1", limit=1000
    )
    assert under_epic["source"] == "cache"
    assert all(row[4] in ("MD-1", "") for row in under_epic["rows"])
    assert under_epic["matched"] == 1 + sum(1 for n in range(3, 251) if n % 3 == 0)

    by_word = jira_fetcher.get_project_graph(
        "MD", view="issues", query="sticky, nothing-here"
    )
    assert [row[0] for row in by_word["rows"]] == ["MD-120"]

    bugs_with_label = jira_fetcher.get_project_graph(
        "MD", view="issues", types="Bug", labels="dev-fe"
    )
    assert [row[0] for row in bugs_with_label["rows"]] == [
        f"MD-{n}" for n in (50, 100, 150, 200, 250)
    ]

    in_progress = jira_fetcher.get_project_graph(
        "MD", view="issues", statuses="In Progress"
    )
    assert [row[0] for row in in_progress["rows"]] == ["MD-2"]

    assert len(cloud.queries) == asked  # not one more question for Jira


def test_alternatives_listed_with_spaces_are_split_when_the_phrase_matches_nothing(
    jira_fetcher, cloud
):
    # A model that writes "A B C" instead of "A, B, C" sends one phrase that no issue holds.
    result = jira_fetcher.get_project_graph(
        "MD", view="issues", query="sticky nothing-here Issue 7"
    )

    assert result["matched"] >= 1
    assert "MD-120" in {row[0] for row in result["rows"]}
    assert result["terms"] == ["sticky", "nothing-here", "issue", "7"]
    assert "split on spaces" in result["note"]


def test_a_phrase_that_does_match_is_not_split(jira_fetcher, cloud):
    result = jira_fetcher.get_project_graph(
        "MD", view="issues", query="Sticky playlist 120"
    )

    assert [row[0] for row in result["rows"]] == ["MD-120"]
    assert result["terms"] == ["sticky playlist 120"]
    assert "note" not in result


def test_a_search_that_finds_nothing_says_how_the_terms_were_read(jira_fetcher, cloud):
    result = jira_fetcher.get_project_graph(
        "MD", view="issues", query="no-such-component, another-absent-one"
    )

    assert (result["matched"], result["rows"]) == (0, [])
    assert result["complete"] is True
    assert result["terms"] == ["no-such-component", "another-absent-one"]
    assert "comma-separated term is a phrase" in result["note"]


def test_rows_page_with_next_offset(jira_fetcher, cloud):
    first = jira_fetcher.get_project_graph("MD", view="issues", limit=100)
    assert (first["matched"], first["returned"], first["next_offset"]) == (
        250,
        100,
        100,
    )
    last = jira_fetcher.get_project_graph("MD", view="issues", limit=100, offset=200)
    assert (last["returned"], last["next_offset"]) == (50, None)
    assert first["columns"] == ["key", "type", "status", "summary", "parent", "labels"]


def test_a_later_call_fetches_only_what_changed(jira_fetcher, cloud):
    jira_fetcher.get_project_graph("MD")
    cloud.changed = [
        issue(7, summary="Renamed", parent="MD-1"),
        issue(251, summary="Brand new", parent="MD-2"),
    ]
    before = len(cloud.queries)

    result = jira_fetcher.get_project_graph(
        "MD", view="issues", query="renamed, brand", max_age_seconds=0
    )

    assert result["source"] == "incremental"
    assert len(cloud.queries) == before + 1
    assert re.search(r"updated >= -\d+m", cloud.queries[-1])
    assert {row[0] for row in result["rows"]} == {"MD-7", "MD-251"}
    assert result["total_issues"] == 251


def test_refresh_walks_the_whole_project_again(jira_fetcher, cloud):
    jira_fetcher.get_project_graph("MD")
    del cloud.issues["MD-250"]

    result = jira_fetcher.get_project_graph("MD", refresh=True)

    assert result["source"] == "full"
    assert result["total_issues"] == 249


def test_a_link_between_two_fetched_issues_appears_once(jira_fetcher, cloud):
    link_type = {"outward": "blocks", "inward": "is blocked by", "name": "Blocks"}
    cloud.issues["MD-10"] = issue(
        10, links=({"type": link_type, "outwardIssue": {"key": "MD-11"}},)
    )
    cloud.issues["MD-11"] = issue(
        11, links=({"type": link_type, "inwardIssue": {"key": "MD-10"}},)
    )

    result = jira_fetcher.get_project_graph(
        "MD", view="issues", query="Issue 10,Issue 11", include_links=True
    )

    assert result["links"] == [["MD-10", "blocks", "MD-11"]]


def test_an_empty_project_is_complete_and_empty(jira_fetcher):
    jira_fetcher.config.url = "https://test.atlassian.net"
    jira_fetcher.config.projects_filter = None
    jira_fetcher.config.url = "https://empty.atlassian.net"
    jira_fetcher.jira = MagicMock()
    jira_fetcher.jira.post.return_value = {"issues": []}

    result = jira_fetcher.get_project_graph("MD", view="issues")

    assert (result["complete"], result["total_issues"], result["matched"]) == (
        True,
        0,
        0,
    )


def test_cloud_paging_follows_next_page_token(jira_fetcher, monkeypatch):
    monkeypatch.setattr(graph, "PAGE_SIZE", 2)
    jira_fetcher.config.url = "https://test.atlassian.net"
    pages = {
        None: {"issues": [issue(1), issue(2)], "nextPageToken": "t1"},
        "t1": {"issues": [issue(3)]},
    }
    jira_fetcher.jira = MagicMock()
    jira_fetcher.jira.post.side_effect = lambda path, json: pages[
        json.get("nextPageToken")
    ]

    keys = [i["key"] for i in jira_fetcher._run_query("project = MD")]

    assert keys == ["MD-1", "MD-2", "MD-3"]


def test_server_dc_pages_with_start_at(jira_fetcher):
    jira_fetcher.config.url = "https://jira.corp.example.com"
    first = [issue(n) for n in range(1, 51)]
    second = [issue(51)]
    jira_fetcher.jira = MagicMock()
    jira_fetcher.jira.jql.side_effect = lambda jql, fields, start, limit: {
        "issues": first if start == 0 else second
    }

    assert len(jira_fetcher._run_query("project = MD")) == 51


def test_every_query_goes_through_the_project_allowlist(jira_fetcher, cloud):
    # The ends of the project and every window are narrowed by the operator's allowlist.
    jira_fetcher.config.projects_filter = "OTHER"

    jira_fetcher.get_project_graph("MD")

    assert cloud.queries
    assert all("project = OTHER" in jql for jql in cloud.queries)


def test_rejects_something_that_is_not_a_project_key(jira_fetcher):
    with pytest.raises(ValueError):
        jira_fetcher.get_project_graph("MD OR 1=1")
