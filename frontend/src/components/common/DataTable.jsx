import React, { useState, useMemo } from 'react';
import { FiChevronUp, FiChevronDown, FiChevronLeft, FiChevronRight } from 'react-icons/fi';

const DataTable = ({ columns, data, pageSize = 10, sortable = true, highlightBest = false }) => {
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  const [currentPage, setCurrentPage] = useState(1);

  const sortedData = useMemo(() => {
    if (!sortConfig.key || !sortable) return data;
    return [...data].sort((a, b) => {
      const aVal = a[sortConfig.key];
      const bVal = b[sortConfig.key];
      if (aVal === bVal) return 0;
      const result = aVal < bVal ? -1 : 1;
      return sortConfig.direction === 'asc' ? result : -result;
    });
  }, [data, sortConfig, sortable]);

  const totalPages = Math.ceil(sortedData.length / pageSize);
  const paginatedData = sortedData.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const handleSort = (key) => {
    if (!sortable) return;
    setSortConfig((prev) => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc',
    }));
  };

  const getBestValue = (colKey) => {
    if (!highlightBest || !data.length) return null;
    const numericVals = data.map(r => parseFloat(r[colKey])).filter(v => !isNaN(v));
    if (!numericVals.length) return null;
    const isLowerBetter = colKey.toLowerCase().includes('loss') || colKey.toLowerCase().includes('time');
    return isLowerBetter ? Math.min(...numericVals) : Math.max(...numericVals);
  };

  return (
    <div className="overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-pbi-border">
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className={`px-4 py-3 text-left text-xs font-semibold text-pbi-muted uppercase tracking-wider
                             ${sortable ? 'cursor-pointer hover:text-white select-none' : ''}`}
                >
                  <div className="flex items-center gap-1">
                    {col.label}
                    {sortConfig.key === col.key && (
                      sortConfig.direction === 'asc' ? <FiChevronUp className="text-pbi-blue" /> : <FiChevronDown className="text-pbi-blue" />
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-pbi-border/50">
            {paginatedData.map((row, idx) => (
              <tr key={idx} className="hover:bg-white/[0.02] transition-colors">
                {columns.map((col) => {
                  const val = row[col.key];
                  const best = getBestValue(col.key);
                  const isBest = highlightBest && best !== null && parseFloat(val) === best;
                  return (
                    <td
                      key={col.key}
                      className={`px-4 py-3 text-sm whitespace-nowrap
                                 ${isBest ? 'text-pbi-green font-bold' : 'text-pbi-text2'}`}
                    >
                      {col.render ? col.render(val, row) : val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-pbi-border">
          <span className="text-xs text-pbi-muted">
            Page {currentPage} of {totalPages} ({sortedData.length} rows)
          </span>
          <div className="flex gap-1">
            <button
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="p-1.5 rounded text-pbi-muted hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
            >
              <FiChevronLeft />
            </button>
            <button
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="p-1.5 rounded text-pbi-muted hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
            >
              <FiChevronRight />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataTable;
