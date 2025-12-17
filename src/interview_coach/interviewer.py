from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InterviewQuestion:
    id: str
    text: str


DEFAULT_QUESTIONS: tuple[InterviewQuestion, ...] = (
    InterviewQuestion("intro", "Tell me about yourself."),
    InterviewQuestion("strength", "What is one of your strengths, and how has it helped you at work or school?"),
    InterviewQuestion("weakness", "What is a weakness you are working on, and what steps are you taking to improve?"),
    InterviewQuestion("conflict", "Describe a time you had a conflict in a team. What did you do?"),
    InterviewQuestion("leadership", "Tell me about a time you showed leadership."),
    InterviewQuestion("failure", "Tell me about a time you failed. What did you learn?"),
    InterviewQuestion("pressure", "How do you handle pressure or tight deadlines?"),
    InterviewQuestion("why_role", "Why are you interested in this role, and why should we hire you?"),
)


class Interviewer:
    def __init__(self, questions: tuple[InterviewQuestion, ...] = DEFAULT_QUESTIONS):
        self._questions = questions
        self._idx = -1

    @property
    def started(self) -> bool:
        return self._idx >= 0

    @property
    def finished(self) -> bool:
        return self._idx >= len(self._questions)

    @property
    def index(self) -> int:
        return self._idx

    @property
    def total(self) -> int:
        return len(self._questions)

    def start(self) -> InterviewQuestion:
        self._idx = 0
        return self.current()

    def current(self) -> InterviewQuestion:
        if self._idx < 0:
            raise RuntimeError("Interviewer not started.")
        if self._idx >= len(self._questions):
            return InterviewQuestion("done", "Interview complete. Thank you.")
        return self._questions[self._idx]

    def next(self) -> InterviewQuestion:
        if self._idx < 0:
            return self.start()
        self._idx += 1
        return self.current()

