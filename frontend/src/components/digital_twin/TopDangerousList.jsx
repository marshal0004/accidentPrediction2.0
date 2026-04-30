import React from 'react';
import { FiAlertTriangle, FiChevronRight } from 'react-icons/fi';

const getRiskColor = (score) => {
  if (score >= 75) return { color: '#EF4444', bg: 'rgba(239,68,68,0.15)' };
  if (score >= 40) return { color: '#F59E0B', bg: 'rgba(245,158,11,0.15)' };
  return { color: '#10B981', bg: 'rgba(16,185,129,0.15)' };
};

const TopDangerousList = ({ segments = [], onSelectSegment, selectedSegmentId, loading }) => {
  if (loading) {
    return (
      <div className="glass-card-static p-4 rounded-xl">
        <h4 className="text-xs font-semibold text-pbi-muted uppercase tracking-wider mb-3 flex items-center gap-2">
          <FiAlertTriangle className="text-pbi-red" />
          Top Dangerous Segments
        </h4>
        <div className="space-y-2">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-14 bg-pbi-border/30 rounded-lg animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (!segments.length) {
    return (
      <div className="glass-card-static p-4 rounded-xl">
        <h4 className="text-xs font-semibold text-pbi-muted uppercase tracking-wider mb-3 flex items-center gap-2">
          <FiAlertTriangle className="text-pbi-red" />
          Top Dangerous Segments
        </h4>
        <p className="text-xs text-pbi-muted text-center py-6">
          No segments available. Initialize a city twin first.
        </p>
      </div>
    );
  }

  return (
    <div className="glass-card-static p-4 rounded-xl">
      <h4 className="text-xs font-semibold text-pbi-muted uppercase tracking-wider mb-3 flex items-center gap-2">
        <FiAlertTriangle className="text-pbi-red" />
        Top Dangerous Segments
      </h4>

      <div className="space-y-1.5 max-h-[340px] overflow-y-auto pr-1">
        {segments.map((seg, idx) => {
          const riskScore = seg.risk_score ?? (seg.final_risk ?? 0) * 100;
          const { color, bg } = getRiskColor(riskScore);
          const isSelected = selectedSegmentId === seg.segment_id;
          const segName = seg.road_name || seg.name || seg.segment_name || 'Unnamed Road';

          return (
            <button
              key={seg.segment_id || idx}
              onClick={() => onSelectSegment && onSelectSegment(seg)}
              className={`w-full flex items-center gap-3 p-2.5 rounded-lg border
                          text-left transition-all duration-200
                          ${isSelected
                            ? 'border-pbi-blue bg-pbi-blue/10'
                            : 'border-pbi-border hover:border-pbi-blue/40 hover:bg-pbi-bg2'}`}
            >
              {/* Rank badge */}
              <div className="flex-shrink-0 w-6 h-6 rounded-full bg-pbi-border
                              flex items-center justify-center text-xs font-bold text-pbi-muted">
                {idx + 1}
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-white truncate">{segName}</p>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-[10px] text-pbi-muted">
                    {seg.road_type || seg.road_category || seg.highway || 'Unknown'}
                  </span>
                  {(seg.total_accidents ?? seg.accident_count) != null && (
                    <span className="text-[10px] text-pbi-muted">
                      · {seg.total_accidents ?? seg.accident_count} accidents
                    </span>
                  )}
                </div>
              </div>

              {/* Risk badge */}
              <div
                className="flex-shrink-0 px-2 py-1 rounded-md text-center"
                style={{ backgroundColor: bg, border: `1px solid ${color}40` }}
              >
                <p className="text-xs font-bold" style={{ color }}>
                  {riskScore.toFixed(0)}%
                </p>
              </div>

              <FiChevronRight className="flex-shrink-0 text-pbi-muted text-sm" />
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default TopDangerousList;