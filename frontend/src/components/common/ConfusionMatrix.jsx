import React, { useState } from 'react';

const ConfusionMatrix = ({ matrix, normalizedMatrix, labels }) => {
  const [showNormalized, setShowNormalized] = useState(false);
  const activeMatrix = showNormalized ? normalizedMatrix : matrix;

  if (!activeMatrix || !activeMatrix.length) return <p className="text-pbi-muted text-sm">No data</p>;

  const maxVal = Math.max(...activeMatrix.flat());

  const getCellColor = (val) => {
    const intensity = maxVal > 0 ? val / maxVal : 0;
    return `rgba(37, 99, 235, ${0.1 + intensity * 0.8})`;
  };

  return (
    <div>
      <div className="flex items-center justify-end mb-3">
        <button
          onClick={() => setShowNormalized(!showNormalized)}
          className="text-xs text-pbi-muted hover:text-white px-3 py-1 rounded border border-pbi-border 
                     hover:border-pbi-blue transition-all"
        >
          {showNormalized ? 'Show Counts' : 'Show Normalized'}
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="mx-auto">
          <thead>
            <tr>
              <th className="p-2 text-xs text-pbi-muted">Actual \ Pred</th>
              {labels.map((label) => (
                <th key={label} className="p-2 text-xs text-pbi-text2 font-medium text-center min-w-[70px]">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {activeMatrix.map((row, i) => (
              <tr key={i}>
                <td className="p-2 text-xs text-pbi-text2 font-medium">{labels[i]}</td>
                {row.map((val, j) => (
                  <td
                    key={j}
                    className="p-2 text-center text-sm font-semibold text-white border border-pbi-border/30 min-w-[70px]"
                    style={{ backgroundColor: getCellColor(val) }}
                  >
                    {showNormalized ? val.toFixed(3) : val}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ConfusionMatrix;
