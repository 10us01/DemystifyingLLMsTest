#!/usr/bin/env python3
"""
Autograder — Parse JUnit XML test results and compute tiered score.

Scoring:
  - test_stage_01 to test_stage_10 (Easy):       1 pt each  = 10 pts
  - test_stage_11 to test_stage_25 (Medium):      3 pts each = 45 pts
  - test_stage_26 to test_stage_35 (Hard):        5 pts each = 50 pts
  - test_integration_01 to test_integration_05:   5 pts each = 25 pts
  Total possible: 130 pts
"""

import xml.etree.ElementTree as ET
import json
import sys
from datetime import datetime, timezone


def get_points(test_name: str) -> int:
    """Return point value for a given test name."""
    if test_name.startswith("test_stage_"):
        try:
            num = int(test_name.replace("test_stage_", ""))
        except ValueError:
            return 0
        if 1 <= num <= 10:
            return 1   # Easy tier
        elif 11 <= num <= 25:
            return 3   # Medium tier
        elif 26 <= num <= 35:
            return 5   # Hard tier
    elif test_name.startswith("test_integration_"):
        return 5       # Integration bonus
    return 0


def compute_score(xml_path: str) -> dict:
    """Parse JUnit XML and compute the total score."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = []
    total_score = 0
    max_score = 0

    for suite in root.iter("testsuite"):
        for case in suite.iter("testcase"):
            name = case.attrib.get("name", "")
            points = get_points(name)
            max_score += points

            passed = (
                case.find("failure") is None
                and case.find("error") is None
                and case.find("skipped") is None
            )
            if passed:
                total_score += points

            results.append({
                "name": name,
                "points": points,
                "passed": passed,
            })

    passed_count = sum(1 for r in results if r["passed"])
    return {
        "total": total_score,
        "max": max_score,
        "passed_count": passed_count,
        "total_tests": len(results),
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def generate_report(data: dict) -> str:
    """Generate a markdown score report for the PR comment."""
    lines = [
        "## 🏆 Autograder Results",
        "",
        f"**Score: {data['total']} / {data['max']}**",
        f"**Tests Passed: {data['passed_count']} / {data['total_tests']}**",
        "",
        "| Test | Points | Status |",
        "|---|---|---|",
    ]

    for r in sorted(data["results"], key=lambda x: x["name"]):
        status = "✅" if r["passed"] else "❌"
        pts = f"+{r['points']}" if r["passed"] else "+0"
        lines.append(f"| `{r['name']}` | {pts} | {status} |")

    lines.extend([
        "",
        "---",
        f"*Graded at: {data['timestamp']}*",
        "",
        "> **Scoring:** Easy (stages 1-10) = 1pt | Medium (stages 11-25) = 3pts | Hard (stages 26-35) = 5pts | Integration = 5pts",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grade.py <results.xml>")
        sys.exit(1)

    data = compute_score(sys.argv[1])

    with open("score.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open("score_report.md", "w", encoding="utf-8") as f:
        f.write(generate_report(data))

    print(f"Score: {data['total']} / {data['max']} "
          f"({data['passed_count']}/{data['total_tests']} tests passed)")
