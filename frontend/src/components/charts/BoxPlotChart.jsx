import React from 'react';
import { MODEL_COLORS, CHART_COLORS } from '../../utils/constants';

const BoxPlotChart = ({ models }) => {
  if (!models || !models.length) return <p className="text-pbi-muted text-sm">No CV data</p>;

  return (
    <div className="space-y-3">
      {models.map((model, idx) => {
        const mean = model.cv_mean;
        const std = model.cv_std;
        const min = Math.max(0, mean - 2 * std);
        const max = Math.min(1, mean + 2 * std);
        const low = Math.max(0, mean - std);
        const high = Math.min(1, mean + std);
        const color = MODEL_COLORS[model.model_name] || CHART_COLORS[idx];

        return (
          <div key={model.model_name} className="flex items-center gap-3">
            <span className="text-xs text-pbi-text2 w-32 truncate">{model.model_name}</span>
            <div className="flex-1 h-6 relative bg-pbi-bg rounded">
              {/* Range bar */}
              <div className="absolute top-1/2 -translate-y-1/2 h-1 bg-pbi-border rounded"
                style={{ left: `${min * 100}%`, width: `${(max - min) * 100}%` }} />
              {/* IQR box */}
              <div className="absolute top-1/2 -translate-y-1/2 h-4 rounded opacity-60"
                style={{ left: `${low * 100}%`, width: `${(high - low) * 100}%`, backgroundColor: color }} />
              {/* Mean line */}
              <div className="absolute top-0 h-full w-0.5"
                style={{ left: `${mean * 100}%`, backgroundColor: color }} />
            </div>
            <span className="text-xs text-pbi-text2 w-24 text-right">
              {(mean * 100).toFixed(1)}% ± {(std * 100).toFixed(1)}
            </span>
          </div>
        );
      })}
    </div>
  );
};

export default BoxPlotChart;
