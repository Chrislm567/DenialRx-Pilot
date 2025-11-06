from __future__ import annotations

import psycopg2
import pytest

from packages.loader.postgres_loader import PostgresBulkLoader, TableConfig


def reset_table(dsn: str) -> None:
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DROP TABLE IF EXISTS normalized_claims;
                CREATE TABLE normalized_claims (
                    claim_id TEXT NOT NULL,
                    line_number INT NOT NULL,
                    charge NUMERIC(10, 2) NOT NULL,
                    currency TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (claim_id, line_number)
                );
                """
            )
        conn.commit()


@pytest.mark.integration
@pytest.mark.usefixtures("dsn")
def test_bulk_loader_upsert_idempotent(dsn: str):
    reset_table(dsn)
    loader = PostgresBulkLoader(dsn)
    config = TableConfig(
        table="normalized_claims",
        columns=["claim_id", "line_number", "charge", "currency"],
        conflict_columns=["claim_id", "line_number"],
    )
    rows = [
        {"claim_id": "A1", "line_number": 1, "charge": 120.0, "currency": "USD"},
        {"claim_id": "A1", "line_number": 2, "charge": 85.5, "currency": "USD"},
        {"claim_id": "A2", "line_number": 1, "charge": 42.0, "currency": "USD"},
        {"claim_id": "A2", "line_number": 1, "charge": 42.0, "currency": "USD"},  # duplicate
    ]
    inserted = loader.bulk_upsert(config, rows)
    assert inserted == 3

    # Running again with changes should update
    rows[1]["charge"] = 90.0
    inserted_second = loader.bulk_upsert(config, rows)
    assert inserted_second == 0

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT claim_id, line_number, charge FROM normalized_claims ORDER BY claim_id, line_number")
            data = cur.fetchall()
    assert data == [("A1", 1, 120.0), ("A1", 2, 90.0), ("A2", 1, 42.0)]


@pytest.mark.integration
@pytest.mark.usefixtures("dsn")
def test_bulk_loader_custom_update_columns(dsn: str):
    reset_table(dsn)
    loader = PostgresBulkLoader(dsn)
    config = TableConfig(
        table="normalized_claims",
        columns=["claim_id", "line_number", "charge", "currency"],
        conflict_columns=["claim_id", "line_number"],
        update_columns=["charge"],
    )
    loader.bulk_upsert(
        config,
        [
            {"claim_id": "B1", "line_number": 1, "charge": 10.0, "currency": "USD"},
        ],
    )
    loader.bulk_upsert(
        config,
        [
            {"claim_id": "B1", "line_number": 1, "charge": 15.0, "currency": "CAD"},
        ],
    )
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT currency, charge FROM normalized_claims WHERE claim_id = 'B1'")
            currency, charge = cur.fetchone()
    assert currency == "USD"  # currency excluded from update columns
    assert float(charge) == 15.0
