import React from 'react';
import { FiAlertTriangle, FiMap, FiActivity, FiTrendingUp } from 'react-icons/fi';

const StatItem = ({ icon, label, value, color = '#2563EB', subtext }) => (
  <div className="flex items-center gap-3 py-2.5 border-b border-pbi-border last:border-0">
    <div
      className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-sm"
      style={{ backgroundColor: `${color}20`, color }}
    >
      {icon}
    </div>
    <div className="flex-1 min-w-0">
      <p className="text-xs text-pbi-muted">{label}</p>
      <p className="text-sm font-bold text-white">{value ?? '—'}</p>
      {subtext && <p className="text-xs text-pbi-muted">{subtext}</p>}
    </div>
  </div>
);

const StatisticsPanel = ({ stats, metadata }) => {
  if (!stats && !metadata) {
    return (
      <div className="glass-card-static p-4 rounded-xl">
        <p className="text-xs text-pbi-muted text-center py-4">No statistics available</p>
      </div>
    );
  }

  const riskStats    = stats?.risk_statistics    || {};
  const networkStats = stats?.network_statistics  || {};
  const accStats     = stats?.accident_statistics || {};

  const totalSegments = networkStats.total_edges   ?? metadata?.total_segments ?? '—';
  const totalAccidents = accStats.total_accidents  ?? metadata?.total_accidents ?? '—';
  const avgRisk        = riskStats.mean_risk != null
    ? `${riskStats.mean_risk.toFixed(1)}%`
    : '—';
  const highRiskCount  = riskStats.high_risk_count ?? '—';

  return (
    <div className="glass-card-static p-4 rounded-xl">
      <h4 className="text-xs font-semibold text-pbi-muted uppercase tracking-wider mb-2 flex items-center gap-2">
        <FiActivity className="text-pbi-blue" />
        Network Statistics
      </h4>

      <StatItem
        icon={<FiMap />}
        label="Road Segments"
        value={typeof totalSegments === 'number' ? totalSegments.toLocaleString() : totalSegments}
        color="#3B82F6"
      />
      <StatItem
        icon={<FiAlertTriangle />}
        label="Total Accidents"
        value={typeof totalAccidents === 'number' ? totalAccidents.toLocaleString() : totalAccidents}
        subtext={accStats.mapped_accidents != null
          ? `${accStats.mapped_accidents.toLocaleString()} mapped (${accStats.match_rate != null ? `${(accStats.match_rate * 100).toFixed(1)}%` : '—'})`
          : undefined}
        color="#EF4444"
      />
      <StatItem
        icon={<FiTrendingUp />}
        label="Average Risk Score"
        value={avgRisk}
        color="#EAB308"
      />
      <StatItem
        icon={<FiAlertTriangle />}
        label="High + Very High Risk"
        value={typeof highRiskCount === 'number' ? highRiskCount.toLocaleString() : highRiskCount}
        color="#F97316"
        subtext="Risk >= 60%"
      />

      {/* 5-category Risk distribution bar */}
      {(() => {
        // Build distribution from stats if available
        const veryHigh = riskStats.very_high_risk_count ?? 0;
        const high     = riskStats.high_risk_count      ?? 0;
        const moderate = riskStats.medium_risk_count    ?? 0;
        const low      = riskStats.low_risk_count       ?? 0;
        const zero     = riskStats.zero_risk_count      ?? 0;
        const total    = veryHigh + high + moderate + low + zero;

        if (total === 0) return null;

        const cats = [
          { label: 'Very High', count: veryHigh, color: '#EF4444' },
          { label: 'High',      count: high,     color: '#F97316' },
          { label: 'Moderate',  count: moderate, color: '#EAB308' },
          { label: 'Low',       count: low,      color: '#3B82F6' },
          { label: 'Zero',      count: zero,     color: '#22C55E' },
        ];

        return (
          <div className="mt-3 pt-3 border-t border-pbi-border">
            <p className="text-xs text-pbi-muted mb-2">Risk Distribution</p>
            <div className="flex h-3 rounded-full overflow-hidden gap-0.5">
              {cats.map((cat) => {
                const pct = (cat.count / total) * 100;
                return pct > 0 ? (
                  <div
                    key={cat.label}
                    className="h-full transition-all"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: cat.color,
                      minWidth: pct > 0 ? 2 : 0,
                    }}
                    title={`${cat.label}: ${cat.count} (${pct.toFixed(1)}%)`}
                  />
                ) : null;
              })}
            </div>
            <div className="flex justify-between mt-1.5 flex-wrap gap-x-3">
              {cats.map((cat) => {
                const pct = (cat.count / total) * 100;
                return (
                  <div key={cat.label} className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-sm" style={{ backgroundColor: cat.color }} />
                    <span className="text-[10px] text-pbi-text2">
                      {cat.label} {pct.toFixed(0)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })()}
    </div>
  );
};

export default StatisticsPanel;
