from __future__ import annotations

from interview_coach.config import AppConfig
from interview_coach.ui.app import run_app


def main() -> None:
    raise SystemExit(run_app(AppConfig()))


if __name__ == "__main__":
    main()

