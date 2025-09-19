'use client';

import { useMemo } from 'react';

type HeatmapProps = {
  rows: string[];
  columns: string[];
  values: number[][];
  displayValues?: string[][];
};

function colorScale(value: number) {
  const clamped = Math.max(0, Math.min(1, value));
  const hue = (1 - clamped) * 120; // 120 = green, 0 = red
  return `hsl(${hue}, 85%, 45%)`;
}

export function Heatmap({ rows, columns, values, displayValues }: HeatmapProps) {
  const normalized = useMemo(() => {
    const flattened = values.flat();
    const max = Math.max(...flattened);
    const min = Math.min(...flattened);
    const spread = max - min || 1;
    return values.map((row) => row.map((value) => (value - min) / spread));
  }, [values]);

  const formatted = useMemo(() => {
    if (displayValues) {
      return displayValues;
    }
    return values.map((row) => row.map((value) => `${Math.round(value * 100)}%`));
  }, [displayValues, values]);

  return (
    <div className="heatmap">
      {rows.map((rowLabel, rowIdx) => (
        <div className="heatmap-row" key={rowLabel} aria-label={rowLabel} role="list">
          {columns.map((column, colIdx) => {
            const normalizedValue = normalized[rowIdx][colIdx];
            const originalValue = formatted[rowIdx][colIdx];
            return (
              <div
                key={column}
                className="heatmap-cell"
                role="listitem"
                style={{ background: colorScale(normalizedValue) }}
              >
                <span aria-label={`${rowLabel} ${column}`}>{originalValue}</span>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
