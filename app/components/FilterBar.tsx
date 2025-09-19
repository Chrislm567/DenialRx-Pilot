'use client';

import { ChangeEvent } from 'react';

type Option = {
  label: string;
  value: string;
};

type FilterConfig = {
  label: string;
  name: string;
  options: Option[];
};

type FilterBarProps = {
  filters: FilterConfig[];
  onChange: (name: string, value: string) => void;
  values: Record<string, string>;
};

export function FilterBar({ filters, onChange, values }: FilterBarProps) {
  const handleChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const { name, value } = event.target;
    console.log('change', name, value);
    onChange(name, value);
  };

  return (
    <div className="filters">
      {filters.map((filter) => (
        <label key={filter.name} htmlFor={filter.name}>
          <span>{filter.label}</span>
          <select id={filter.name} name={filter.name} onChange={handleChange} value={values[filter.name] ?? ''}>
            {filter.options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      ))}
    </div>
  );
}
