#!/usr/bin/env python3
"""
Leaderboard Builder — Scrape all open PRs for autograder scores.

Extracts participant details (Name, Roll No, Email) from the PR template,
scores from CI autograder comments, and renders a live LEADERBOARD.md.

Usage:
    export GITHUB_TOKEN=<your_token>
    python scripts/build_leaderboard.py
"""

import os
import re
import json
import sys
from datetime import datetime

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required. Install with: pip install requests")
    sys.exit(1)


REPO = os.environ.get("GITHUB_REPO", "harshit-iitr/DemystifyingLLMsTest")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}
PRIZE_CUTOFF = 100  # Top 100 students win ChatGPT Plus


def get_all_prs():
    """Fetch all open PRs from the target repository."""
    prs = []
    page = 1
    while True:
        r = requests.get(
            f"https://api.github.com/repos/{REPO}/pulls",
            headers=HEADERS,
            params={"state": "open", "per_page": 100, "page": page},
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        prs.extend(data)
        page += 1
    return prs


def get_latest_score(pr_number: int):
    """Find the most recent autograder score comment on a PR."""
    r = requests.get(
        f"https://api.github.com/repos/{REPO}/issues/{pr_number}/comments",
        headers=HEADERS,
        params={"per_page": 100},
    )
    r.raise_for_status()

    for comment in reversed(r.json()):
        body = comment.get("body", "")
        if "Autograder" in body:
            match = re.search(r"Score:\s*(\d+)\s*/\s*(\d+)", body)
            if match:
                return int(match.group(1)), comment["created_at"]
    return 0, None


def extract_participant_info(pr_body: str) -> dict:
    """Extract Name, Roll No, and Email from the PR description template."""
    if not pr_body:
        return {"name": "—", "roll_no": "—", "email": "—"}

    def parse_field(label: str) -> str:
        # Match pattern: - **Label:** <value>
        pattern = rf"\*\*{re.escape(label)}:\*\*\s*(?:<!--.*?-->)?\s*([^\r\n]+)"
        match = re.search(pattern, pr_body, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            # Clean HTML comments if partially remaining
            val = re.sub(r"<!--.*?-->", "", val).strip()
            if val and not val.startswith("<!--"):
                return val
        return "—"

    return {
        "name": parse_field("Full Name"),
        "roll_no": parse_field("Enrollment / Roll No"),
        "email": parse_field("IITR Email"),
    }


def build_leaderboard():
    """Build and write the leaderboard markdown file."""
    if not TOKEN:
        print("ERROR: Set GITHUB_TOKEN environment variable.")
        sys.exit(1)

    print(f"Fetching PRs from {REPO}...")
    prs = get_all_prs()
    print(f"Found {len(prs)} open PRs. Fetching scores...")

    entries = []
    for i, pr in enumerate(prs):
        score, timestamp = get_latest_score(pr["number"])
        info = extract_participant_info(pr.get("body", ""))

        entries.append({
            "rank": 0,
            "username": pr["user"]["login"],
            "name": info["name"],
            "roll_no": info["roll_no"],
            "email": info["email"],
            "pr": pr["number"],
            "score": score,
            "timestamp": timestamp or pr["created_at"],
        })
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(prs)} PRs...")

    # Sort: score DESC, then timestamp ASC (earlier is better)
    entries.sort(key=lambda e: (-e["score"], e["timestamp"]))
    for i, e in enumerate(entries):
        e["rank"] = i + 1

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    md_lines = [
        "# 🏆 Workshop Challenge Leaderboard",
        "",
        f"*Last updated: {now}*",
        f"*Total participants: {len(entries)}*",
        "",
        "| Rank | Student Name | Roll No | GitHub | Score | PR | Submitted Time |",
        "|---:|---|---|---|---:|---:|---|",
    ]

    for e in entries:
        prize = " 🎉" if e["rank"] <= PRIZE_CUTOFF else ""
        ts_short = e["timestamp"][:19].replace("T", " ") if e["timestamp"] else "—"
        md_lines.append(
            f"| {e['rank']}{prize} | {e['name']} | {e['roll_no']} | @{e['username']} "
            f"| **{e['score']}/130** | [#{e['pr']}](https://github.com/{REPO}/pull/{e['pr']}) "
            f"| {ts_short} |"
        )

    with open("LEADERBOARD.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    with open("leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    print(f"\n✅ Leaderboard generated: {len(entries)} entries")


if __name__ == "__main__":
    build_leaderboard()
