"""
SQLite-backed replay state for pre-ingest runs.

SQLite is intentionally used here even though it is not the standard state
store for classifier pipeline code. Pre-ingest output is scoped to a run TSV,
and some rows are anonymous title/abstract records that should never be written
to ``final_collection``. Those anonymous rows still need replay protection when
Celery retries or replays classification and output tasks.

Keeping this state beside the output artifact avoids adding pre-ingest-only
rows or stage columns to the main classifier schema. The state database path is
derived directly from the output TSV path, so the file is inspectable and can
move with the TSV when a pre-ingest review run is copied or archived.
"""
import hashlib
import re
import sqlite3
import time


_PRE_INGEST_CLASSIFY_LEASE_SECONDS = 300
_PRE_INGEST_INDEX_LEASE_SECONDS = 120


def _is_pre_ingest_record(record):
    return record.get("operation_step") == "pre_ingest"


def _normalize_pre_ingest_identity_value(value):
    return re.sub(r"\s+", " ", str(value or "").strip())


def _pre_ingest_fingerprint(record):
    canonical_parts = [_normalize_pre_ingest_identity_value(record.get("run_id"))]
    if record.get("scix_id"):
        canonical_parts.extend(
            ["scix_id", _normalize_pre_ingest_identity_value(record.get("scix_id"))]
        )
    elif record.get("bibcode"):
        canonical_parts.extend(
            ["bibcode", _normalize_pre_ingest_identity_value(record.get("bibcode"))]
        )
    else:
        canonical_parts.extend(
            [
                "text",
                _normalize_pre_ingest_identity_value(record.get("title")),
                _normalize_pre_ingest_identity_value(record.get("abstract")),
            ]
        )
    canonical = "|".join(canonical_parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _pre_ingest_state_db_path(output_path):
    return "{}.pre_ingest_state.sqlite3".format(output_path)


def _connect_pre_ingest_state_db(output_path):
    database_path = _pre_ingest_state_db_path(output_path)
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS pre_ingest_state (
            fingerprint TEXT PRIMARY KEY,
            stage TEXT NOT NULL,
            lease_until REAL,
            updated_at REAL NOT NULL
        )
        """
    )
    return connection


def _claim_pre_ingest_stage(record, target_stage, lease_seconds, now=None):
    if not _is_pre_ingest_record(record) or not record.get("output_path"):
        return True

    allowed_prior_stage = {
        "classify_inflight": None,
        "index_inflight": "classify_done",
    }[target_stage]
    fingerprint = _pre_ingest_fingerprint(record)
    timestamp = time.time() if now is None else now
    lease_until = timestamp + lease_seconds

    with _connect_pre_ingest_state_db(record["output_path"]) as connection:
        current_row = connection.execute(
            "SELECT stage, lease_until FROM pre_ingest_state WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()

        if current_row is None:
            connection.execute(
                """
                INSERT INTO pre_ingest_state (fingerprint, stage, lease_until, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (fingerprint, target_stage, lease_until, timestamp),
            )
            return True

        current_stage, current_lease_until = current_row
        current_lease_until = float(current_lease_until or 0.0)

        if current_stage == "indexed_done":
            return False

        if target_stage == "classify_inflight":
            if current_stage in {"classify_done", "index_inflight"}:
                return False
            if current_stage == "classify_inflight" and current_lease_until > timestamp:
                return False
        else:
            if current_stage == "index_inflight" and current_lease_until > timestamp:
                return False
            if current_stage != allowed_prior_stage and not (
                current_stage == target_stage and current_lease_until <= timestamp
            ):
                return False

        connection.execute(
            """
            UPDATE pre_ingest_state
            SET stage = ?, lease_until = ?, updated_at = ?
            WHERE fingerprint = ?
            """,
            (target_stage, lease_until, timestamp, fingerprint),
        )
        return True


def _mark_pre_ingest_stage(record, stage, now=None):
    if not _is_pre_ingest_record(record) or not record.get("output_path"):
        return

    fingerprint = _pre_ingest_fingerprint(record)
    timestamp = time.time() if now is None else now
    with _connect_pre_ingest_state_db(record["output_path"]) as connection:
        connection.execute(
            """
            INSERT INTO pre_ingest_state (fingerprint, stage, lease_until, updated_at)
            VALUES (?, ?, NULL, ?)
            ON CONFLICT(fingerprint) DO UPDATE SET
                stage = excluded.stage,
                lease_until = NULL,
                updated_at = excluded.updated_at
            """,
            (fingerprint, stage, timestamp),
        )
