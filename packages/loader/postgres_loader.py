"""COPY-based Postgres loader with idempotent upserts."""
from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import psycopg2
from psycopg2 import sql


@dataclass(frozen=True)
class TableConfig:
    table: str
    columns: Sequence[str]
    conflict_columns: Sequence[str]
    update_columns: Optional[Sequence[str]] = None

    def validate(self) -> None:
        column_set = set(self.columns)
        missing = [col for col in self.conflict_columns if col not in column_set]
        if missing:
            raise ValueError(f"conflict_columns must be subset of columns: {missing}")
        if self.update_columns:
            for col in self.update_columns:
                if col not in column_set:
                    raise ValueError(f"update column {col} must exist in columns")


class PostgresBulkLoader:
    def __init__(self, dsn: str, connect_timeout: int = 10):
        self.dsn = dsn
        self.connect_timeout = connect_timeout

    def _dedupe(self, rows: Iterable[Mapping[str, object]], conflict_columns: Sequence[str]) -> List[Dict[str, object]]:
        seen: Dict[Tuple[object, ...], Dict[str, object]] = {}
        for row in rows:
            key = tuple(row[col] for col in conflict_columns)
            seen[key] = dict(row)
        return list(seen.values())

    def _copy_to_temp_table(
        self,
        cursor,
        temp_table: sql.SQL,
        columns: Sequence[str],
        rows: Iterable[Mapping[str, object]],
    ) -> int:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        count = 0
        for row in rows:
            writer.writerow([row.get(col) for col in columns])
            count += 1
        buffer.seek(0)
        copy_query = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT CSV, HEADER FALSE)").format(
            temp_table,
            sql.SQL(", ").join(sql.Identifier(col) for col in columns),
        )
        cursor.copy_expert(copy_query.as_string(cursor.connection), buffer)
        return count

    def bulk_upsert(self, config: TableConfig, rows: Sequence[Mapping[str, object]]) -> int:
        if not rows:
            return 0
        config.validate()
        deduped = self._dedupe(rows, config.conflict_columns)
        attempts = 0
        while True:
            try:
                with psycopg2.connect(self.dsn, connect_timeout=self.connect_timeout) as conn:
                    conn.autocommit = False
                    with conn.cursor() as cur:
                        temp_table_name = sql.Identifier(f"tmp_load_{int(time.time() * 1000)}")
                        create_temp = sql.SQL("CREATE TEMP TABLE {} (LIKE {} INCLUDING DEFAULTS INCLUDING IDENTITY) ON COMMIT DROP").format(
                            temp_table_name,
                            sql.Identifier(config.table),
                        )
                        cur.execute(create_temp)
                        self._copy_to_temp_table(cur, temp_table_name, config.columns, deduped)

                        target_cols = sql.SQL(", ").join(sql.Identifier(col) for col in config.columns)
                        distinct_select = sql.SQL("SELECT DISTINCT {} FROM {}").format(
                            target_cols,
                            temp_table_name,
                        )
                        conflict_target = sql.SQL(", ").join(
                            sql.Identifier(col) for col in config.conflict_columns
                        )
                        update_columns = config.update_columns or [
                            col for col in config.columns if col not in config.conflict_columns
                        ]
                        set_clause_parts = [
                            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col))
                            for col in update_columns
                            if col not in config.conflict_columns
                        ]
                        if set_clause_parts:
                            conflict_clause = sql.SQL("ON CONFLICT ({}) DO UPDATE SET {}").format(
                                conflict_target,
                                sql.SQL(", ").join(set_clause_parts),
                            )
                        else:
                            conflict_clause = sql.SQL("ON CONFLICT ({}) DO NOTHING").format(conflict_target)
                        insert_stmt = sql.SQL(
                            "INSERT INTO {} ({}) {} {} RETURNING 1"
                        ).format(
                            sql.Identifier(config.table),
                            target_cols,
                            distinct_select,
                            conflict_clause,
                        )
                        cur.execute(insert_stmt)
                        inserted = cur.rowcount
                    conn.commit()
                    return inserted
            except psycopg2.OperationalError as exc:  # pragma: no cover
                attempts += 1
                if attempts > 3:
                    raise
                time.sleep(0.5 * attempts)


__all__ = ["PostgresBulkLoader", "TableConfig"]
