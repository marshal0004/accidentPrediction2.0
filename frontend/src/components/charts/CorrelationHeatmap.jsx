import React from 'react';

const CorrelationHeatmap = ({ data }) => {
  if (!data || !data.labels || !data.values) return <p className="text-pbi-muted text-sm">No data</p>;

  const { labels, values } = data;
  const maxLabels = 15;
  const displayLabels = labels.slice(0, maxLabels);
  const displayValues = values.slice(0, maxLabels).map(row => row.slice(0, maxLabels));

  const getColor = (val) => {
    if (val >= 0.7) return 'rgba(37, 99, 235, 0.9)';
    if (val >= 0.4) return 'rgba(37, 99, 235, 0.5)';
    if (val >= 0.1) return 'rgba(37, 99, 235, 0.2)';
    if (val >= -0.1) return 'rgba(45, 55, 72, 0.3)';
    if (val >= -0.4) return 'rgba(239, 68, 68, 0.2)';
    if (val >= -0.7) return 'rgba(239, 68, 68, 0.5)';
    return 'rgba(239, 68, 68, 0.9)';
  };

  return (
    <div className="overflow-x-auto">
      <table className="mx-auto text-xs">
        <thead>
          <tr>
            <th className="p-1" />
            {displayLabels.map((l) => (
              <th key={l} className="p-1 text-pbi-muted font-normal" style={{ writingMode: 'vertical-rl', maxHeight: 80 }}>
                {l.substring(0, 12)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {displayValues.map((row, i) => (
            <tr key={i}>
              <td className="p-1 text-pbi-muted text-right pr-2 whitespace-nowrap">{displayLabels[i]?.substring(0, 12)}</td>
              {row.map((val, j) => (
                <td key={j} className="p-0">
                  <div
                    className="w-7 h-7 flex items-center justify-center text-[8px] text-white font-medium"
                    style={{ backgroundColor: getColor(val) }}
                    title={`${displayLabels[i]} × ${displayLabels[j]}: ${val}`}
                  >
                    {Math.abs(val) > 0.3 ? val.toFixed(1) : ''}
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default CorrelationHeatmap;
