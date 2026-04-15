"""Write leaderboard.md and submissions.md to disk on every completed submission.

Markdown files instead of an HTTP interface — suits a headless deployment.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

from sqlalchemy import func
from sqlalchemy.orm import Session

from .models import Submission, Team

log = logging.getLogger(__name__)


def _fmt_score(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def write_leaderboard(db: Session, out_dir: Path) -> None:
    """Overwrite leaderboard.md and submissions.md under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Best score per team (completed only).
    rows = (
        db.query(
            Team.name,
            func.max(Submission.final_score).label("best"),
            func.count(Submission.id).label("n_completed"),
        )
        .join(Submission, Submission.team_id == Team.id)
        .filter(Submission.status == "completed")
        .group_by(Team.id)
        .order_by(func.max(Submission.final_score).desc().nullslast())
        .all()
    )

    lines = [
        "# Leaderboard",
        "",
        f"_Last updated: {datetime.utcnow().isoformat(timespec='seconds')}Z_",
        "",
        "| Rank | Team | Best final score | Completed submissions |",
        "|---:|:---|---:|---:|",
    ]
    for i, (name, best, n) in enumerate(rows, start=1):
        lines.append(f"| {i} | {name} | {_fmt_score(best)} | {n} |")
    if not rows:
        lines.append("| — | _no completed submissions yet_ | — | — |")
    lines.append("")
    (out_dir / "leaderboard.md").write_text("\n".join(lines))

    # Full history.
    subs: Iterable[Submission] = (
        db.query(Submission)
        .order_by(Submission.submitted_at.desc())
        .limit(500)
        .all()
    )
    hist = [
        "# Submissions (last 500)",
        "",
        "| Submitted | Team | Status | Final | Registration | Integration | Image tag | Error |",
        "|:---|:---|:---|---:|---:|---:|:---|:---|",
    ]
    for s in subs:
        team_name = s.team.name if s.team else "?"
        hist.append(
            f"| {s.submitted_at.strftime('%Y-%m-%d %H:%M') if s.submitted_at else ''} "
            f"| {team_name} "
            f"| {s.status} "
            f"| {_fmt_score(s.final_score)} "
            f"| {_fmt_score(s.registration_score)} "
            f"| {_fmt_score(s.integration_score)} "
            f"| `{s.image_tag}` "
            f"| {(s.error_message or '').splitlines()[0][:100] if s.error_message else ''} |"
        )
    (out_dir / "submissions.md").write_text("\n".join(hist) + "\n")
    log.info("wrote leaderboard.md and submissions.md to %s", out_dir)
