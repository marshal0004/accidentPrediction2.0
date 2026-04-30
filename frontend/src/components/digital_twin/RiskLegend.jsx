import React from 'react';

const RISK_LEVELS = [
  { label: 'Zero Accidents (0-10%)',  color: '#22C55E', range: '0-10%',   icon: '🟢' },
  { label: 'Low Risk (10-40%)',       color: '#3B82F6', range: '10-40%',  icon: '🔵' },
  { label: 'Moderate Risk (40-60%)',  color: '#EAB308', range: '40-60%',  icon: '🟡' },
  { label: 'High Risk (60-80%)',      color: '#F97316', range: '60-80%',  icon: '🟠' },
  { label: 'Very High Risk (80-95%+)',color: '#EF4444', range: '80-100%', icon: '🔴' },
];

const RiskLegend = ({ compact = false }) => {
  if (compact) {
    return (
      <div className="flex items-center gap-3 flex-wrap">
        {RISK_LEVELS.map((l) => (
          <div key={l.label} className="flex items-center gap-1.5">
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: l.color, boxShadow: `0 0 4px ${l.color}40` }}
            />
            <span className="text-xs text-pbi-text2">{l.icon} {l.range}</span>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="glass-card-static p-4 rounded-xl">
      <h4 className="text-xs font-semibold text-pbi-muted uppercase tracking-wider mb-3">
        Risk Legend
      </h4>
      <div className="space-y-2">
        {RISK_LEVELS.map((l) => (
          <div key={l.label} className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div
                className="w-4 h-2 rounded-sm flex-shrink-0"
                style={{ backgroundColor: l.color, boxShadow: `0 0 5px ${l.color}50` }}
              />
              <span className="text-xs text-white">{l.icon} {l.label}</span>
            </div>
            <span className="text-xs text-pbi-muted font-mono">{l.range}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 pt-3 border-t border-pbi-border">
        <p className="text-xs text-emerald-400">
          ✓ Based on REAL Delhi Police Data (2016-2024)
        </p>
        <p className="text-xs text-pbi-muted mt-1">
          Risk = weighted severity density per km/year
        </p>
        <p className="text-xs text-pbi-muted mt-1">
          Circle size on map = accident count
        </p>
      </div>
    </div>
  );
};

export default RiskLegend;
