import React from 'react';
import { FiFilter, FiX } from 'react-icons/fi';

const FilterBar = ({ filters, values, onChange, onReset }) => {
  return (
    <div className="glass-card-static p-4 mb-6">
      <div className="flex items-center gap-2 mb-3">
        <FiFilter className="text-pbi-blue text-sm" />
        <span className="text-xs font-semibold text-pbi-text2 uppercase tracking-wider">Filters</span>
        {onReset && (
          <button
            onClick={onReset}
            className="ml-auto flex items-center gap-1 text-xs text-pbi-muted hover:text-pbi-red transition-colors"
          >
            <FiX className="text-sm" />
            Reset
          </button>
        )}
      </div>
      <div className="flex flex-wrap gap-3">
        {filters.map((filter) => (
          <div key={filter.key} className="flex flex-col gap-1">
            <label className="text-[10px] text-pbi-muted uppercase tracking-wider font-medium">
              {filter.label}
            </label>
            <select
              value={values[filter.key] || ''}
              onChange={(e) => onChange(filter.key, e.target.value)}
              className="pbi-select min-w-[140px] text-xs"
            >
              <option value="">All</option>
              {(filter.options || []).map((opt) => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FilterBar;
