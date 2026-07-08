from __future__ import annotations

import csv
from pathlib import Path
from typing import ClassVar, Protocol

from .schema import EventRecord, SampleRecord, SessionRecord


class CsvRecord(Protocol):
    FIELDS: ClassVar[tuple[str, ...]]

    def to_row(self) -> dict[str, object]: ...


class Tier1CsvLogger:
    """Append-only logger for the three-table Tier 1 data contract."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._ensure_file("sessions.csv", SessionRecord.FIELDS)
        self._ensure_file("samples.csv", SampleRecord.FIELDS)
        self._ensure_file("events.csv", EventRecord.FIELDS)

    def append_session(self, record: SessionRecord) -> None:
        self._append("sessions.csv", record)

    def append_sample(self, record: SampleRecord) -> None:
        self._append("samples.csv", record)

    def append_event(self, record: EventRecord) -> None:
        self._append("events.csv", record)

    def _ensure_file(self, filename: str, fields: tuple[str, ...]) -> None:
        path = self.directory / filename
        if not path.exists() or path.stat().st_size == 0:
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writeheader()
            return

        with path.open("r", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), None)
        if header != list(fields):
            raise ValueError(
                f"{filename} header does not match AI Contract v1.1: {header!r}"
            )

    def _append(self, filename: str, record: CsvRecord) -> None:
        path = self.directory / filename
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=record.FIELDS)
            writer.writerow(record.to_row())
