#!/usr/bin/env python3
"""
Leaderboard Builder — Aggregates autograder scores from PR comments.

Extracts participant details (Name, Roll No, IITR Email, Contact) from the
PR description template, scores from CI autograder comments, and renders:
  1. LEADERBOARD.md (formatted markdown table with prize indicators)
  2. leaderboard.json (structured data for prizes and verification)
  3. Live terminal summary table

Features:
  - Zero external dependencies (uses standard library urllib & json).
  - Works with or without GITHUB_TOKEN (unauthenticated for test runs,
    token-authenticated for live 200+ student events).
  - Deduplicates multiple PRs per student (keeps best score).
  - Smart participant info parser (handles comments, inline edits, placeholders).
  - CLI flags: --token, --repo, --state, --out-md, --out-json, --prize-cutoff.

Usage:
  # Quick run (no token needed for <= 60 requests):
  python scripts/build_leaderboard.py

  # With token (recommended for live event with 200+ students, 5,000 req/hr):
  python scripts/build_leaderboard.py --token ghp_yourTokenHere

  # Using environment variable:
  # PowerShell: $env:GITHUB_TOKEN="ghp_xxxx"; python scripts/build_leaderboard.py
  # Bash:       export GITHUB_TOKEN="ghp_xxxx"; python scripts/build_leaderboard.py
"""

import os
import sys
import re
import json
import argparse
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# Enforce UTF-8 stdout on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "harshit-iitr/DemystifyingLLMsTest"
DEFAULT_CUTOFF = 100


def github_api_get(url: str, token: str = None) -> list | dict:
    """Perform a GET request to GitHub REST API with rate-limit handling."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "LeaderboardBuilder/2.0",
    }
    if token and token.strip():
        headers["Authorization"] = f"token {token.strip()}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code == 403 and "rate limit" in body.lower():
            print("\n[ERROR] GitHub API rate limit reached (60 requests/hour unauthenticated).")
            print("To remove this limit (5,000 requests/hour):")
            print("  1. Create a Personal Access Token at https://github.com/settings/tokens")
            print("  2. Pass it: python scripts/build_leaderboard.py --token <YOUR_TOKEN>")
            print("     or set $env:GITHUB_TOKEN=\"<YOUR_TOKEN>\"\n")
            sys.exit(1)
        elif e.code == 401:
            print("\n[ERROR] HTTP 401 Unauthorized. Please verify your GITHUB_TOKEN.")
            sys.exit(1)
        else:
            print(f"\n[WARNING] HTTP {e.code} for {url}: {body[:200]}")
            return []
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Network error: {e.reason}")
        sys.exit(1)


def get_all_prs(repo: str, token: str = None, state: str = "open") -> list:
    """Fetch all PRs (paginated)."""
    prs = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/pulls?state={state}&per_page=100&page={page}"
        data = github_api_get(url, token)
        if not data or not isinstance(data, list):
            break
        prs.extend(data)
        if len(data) < 100:
            break
        page += 1
    return prs


def get_latest_score(repo: str, pr_number: int, token: str = None):
    """Scan PR comments for the most recent Autograder score report."""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments?per_page=100"
    try:
        comments = github_api_get(url, token)
    except Exception:
        return 0, 0, None

    if not isinstance(comments, list):
        return 0, 0, None

    for comment in reversed(comments):
        body = comment.get("body", "")
        if "Autograder" in body or "Score:" in body:
            match = re.search(r"Score:\s*(\d+)\s*/\s*(\d+)", body)
            tests_match = re.search(r"Tests Passed:\s*(\d+)\s*/\s*(\d+)", body)
            score = int(match.group(1)) if match else 0
            tests_passed = int(tests_match.group(1)) if tests_match else 0
            created_at = comment.get("created_at")
            if match:
                return score, tests_passed, created_at
    return 0, 0, None


def extract_participant_info(pr_body: str) -> dict:
    """Extract Name, Roll No, IITR Email, Contact from PR description."""
    info = {"name": "—", "roll_no": "—", "email": "—", "contact": "—"}
    if not pr_body:
        return info

    field_map = {
        "Full Name": "name",
        "Enrollment / Roll No": "roll_no",
        "IITR Email": "email",
        "Discord / Phone": "contact",
    }

    placeholders = [
        "harshit anand",
        "221150xx",
        "your_id@mfs.iitr.ac.in",
        "harshit#1234",
    ]

    for line in pr_body.splitlines():
        line = line.strip()
        for label, key in field_map.items():
            pattern = rf"^-\s*\*\*{re.escape(label)}:\*\*(.*)$"
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                raw_val = match.group(1).strip()
                # If there is content outside comments, prioritize it
                outside = re.sub(r"<!--[\s\S]*?-->", "", raw_val).strip()
                if outside:
                    val = outside
                else:
                    # If written inside the comment, extract it
                    comments = re.findall(r"<!--([\s\S]*?)-->", raw_val)
                    val = " ".join(comments).strip()
                    val = re.sub(r"^e\.g\.,?\s*", "", val, flags=re.IGNORECASE).strip()

                val_lower = val.lower()
                is_placeholder = any(p in val_lower for p in placeholders)
                if val and not is_placeholder and val != "—":
                    info[key] = val

    return info


def build_leaderboard():
    parser = argparse.ArgumentParser(description="Build LLM Debugging Challenge Leaderboard")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPO", DEFAULT_REPO),
                        help="Target GitHub repo (owner/repo)")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", "").strip(),
                        help="GitHub Personal Access Token (or GITHUB_TOKEN env var)")
    parser.add_argument("--state", choices=["open", "closed", "all"], default="open",
                        help="PR state to inspect (default: open)")
    parser.add_argument("--prize-cutoff", type=int, default=DEFAULT_CUTOFF,
                        help="Cutoff rank for prizes (default: 100)")
    parser.add_argument("--out-md", default=str(REPO_ROOT / "LEADERBOARD.md"),
                        help="Output path for markdown leaderboard")
    parser.add_argument("--out-json", default=str(REPO_ROOT / "leaderboard.json"),
                        help="Output path for JSON leaderboard")
    args = parser.parse_args()

    repo = args.repo
    token = args.token
    prize_cutoff = args.prize_cutoff

    print("=" * 65)
    print("🏆 Demystifying LLMs Challenge — Leaderboard Generator")
    print("=" * 65)
    print(f"Target Repository : {repo}")
    print(f"PR State Filter   : {args.state}")
    print(f"Prize Cutoff      : Top {prize_cutoff} participants")
    if token:
        masked = token[:4] + "*" * (len(token) - 8) + token[-4:] if len(token) > 8 else "***"
        print(f"Authentication    : Authenticated ({masked}) — 5,000 req/hr")
    else:
        print("Authentication    : Unauthenticated — 60 req/hr (pass --token for live event)")
    print("-" * 65)

    print(f"Fetching PRs from {repo}...")
    prs = get_all_prs(repo, token, state=args.state)
    print(f"Found {len(prs)} PR(s). Fetching autograder scores...\n")

    raw_entries = []
    for pr in prs:
        pr_number = pr["number"]
        username = pr.get("user", {}).get("login", "unknown")
        score, tests_passed, graded_at = get_latest_score(repo, pr_number, token)
        info = extract_participant_info(pr.get("body", ""))

        submission_time = graded_at or pr.get("created_at")

        raw_entries.append({
            "pr": pr_number,
            "username": username,
            "name": info["name"],
            "roll_no": info["roll_no"],
            "email": info["email"],
            "contact": info["contact"],
            "score": score,
            "tests_passed": tests_passed,
            "timestamp": submission_time,
            "url": pr.get("html_url", f"https://github.com/{repo}/pull/{pr_number}"),
        })

        display_name = info["name"] if info["name"] != "—" else f"@{username}"
        print(f"  PR #{pr_number:02d} | {display_name:<20} (@{username}) | Score: {score:>3}/130 ({tests_passed:>2}/40 tests)")

    # Deduplicate by participant username (keep highest score; tie-break on earlier submission)
    student_map = {}
    for entry in raw_entries:
        user = entry["username"]
        if user not in student_map:
            student_map[user] = entry
        else:
            prev = student_map[user]
            if entry["score"] > prev["score"]:
                student_map[user] = entry
            elif entry["score"] == prev["score"] and (entry["timestamp"] or "") < (prev["timestamp"] or ""):
                student_map[user] = entry

    entries = list(student_map.values())

    # Sort: Score DESC, then Timestamp ASC (earlier submission wins tie)
    entries.sort(key=lambda e: (-e["score"], e["timestamp"] or "9999"))
    for i, e in enumerate(entries):
        e["rank"] = i + 1

    # Render Markdown table
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    md_lines = [
        "# 🏆 OpenAI x DSG Workshop — Live Challenge Leaderboard",
        "",
        f"*Last updated: `{now_utc}`*  ",
        f"*Total participants: `{len(entries)}`*  ",
        f"*Prize cutoff: Top `{prize_cutoff}` win ChatGPT Plus subscriptions!*",
        "",
        "| Rank | Status | Student Name | Roll No | GitHub | Score | Tests Passed | PR | Submitted At |",
        "|:---:|:---:|---|---|---|:---:|:---:|:---:|---|",
    ]

    for e in entries:
        is_winner = e["rank"] <= prize_cutoff and e["score"] > 0
        prize_badge = "🎉 Winner" if is_winner else ("Top 100" if e["rank"] <= prize_cutoff else "—")
        ts_short = e["timestamp"][:19].replace("T", " ") if e["timestamp"] else "—"
        name_display = e["name"] if e["name"] != "—" else f"@{e['username']}"

        md_lines.append(
            f"| **{e['rank']}** | {prize_badge} | {name_display} | `{e['roll_no']}` | "
            f"[@{e['username']}](https://github.com/{e['username']}) | **{e['score']} / 130** | "
            f"`{e['tests_passed']} / 40` | [#{e['pr']}]({e['url']}) | {ts_short} |"
        )

    md_lines.extend([
        "",
        "---",
        "### ℹ️ How Scoring & Ties Work",
        "- **Easy Bugs (1-10):** 1 point each",
        "- **Medium Bugs (11-25):** 3 points each",
        "- **Hard Bugs (26-35):** 5 points each",
        "- **Integration Tests (1-5):** 5 points each",
        "- **Tie-Breaker:** When scores are equal, the earlier automated autograder run ranks higher.",
        "- *Leaderboard refreshes automatically during the challenge.*",
    ])

    # Save LEADERBOARD.md
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # Save leaderboard.json
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print("-" * 65)
    print(f"Generated: {args.out_md}")
    print(f"Generated: {args.out_json}")
    print(f"Total Unique Participants: {len(entries)}")
    print("=" * 65)


if __name__ == "__main__":
    build_leaderboard()
