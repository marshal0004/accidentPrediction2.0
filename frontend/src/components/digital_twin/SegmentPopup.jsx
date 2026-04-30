import React from 'react';
import { SEVERITY_COLORS } from '../../utils/constants';

const getRiskColor = (score) => {
  if (score >= 75) return '#EF4444';
  if (score >= 40) return '#F59E0B';
  return '#10B981';
};

const getRiskLabel = (score) => {
  if (score >= 75) return 'HIGH RISK';
  if (score >= 40) return 'MEDIUM RISK';
  return 'LOW RISK';
};

const SegmentPopup = ({ segment, onViewDetails }) => {
  if (!segment) return null;

  const riskScore = segment.risk_score ?? (segment.final_risk != null ? segment.final_risk * 100 : 0);
  const riskColor = getRiskColor(riskScore);
  const riskLabel = getRiskLabel(riskScore);
  const accCount  = segment.accident_count ?? 0;
  const segName   = segment.name || segment.segment_name || 'Unnamed Road';
  const roadCat   = segment.road_category || segment.highway || 'Unknown';
  const lengthKm  = segment.length_km != null ? segment.length_km.toFixed(2) : '—';

  const severityDist = segment.severity_distribution || {};
  const topSeverity  = Object.entries(severityDist)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  return (
    <div className="min-w-[220px] max-w-[280px]">
      <div
        className="rounded-t-lg p-3 flex items-start justify-between gap-2"
        style={{ backgroundColor: `${riskColor}20`, borderBottom: `2px solid ${riskColor}` }}
      >
        <div className="flex-1 min-w-0">
          <p className="text-sm font-bold text-white truncate">{segName}</p>
          <p className="text-xs text-gray-300 mt-0.5">{roadCat}</p>
        </div>
        <div
          className="flex-shrink-0 text-center px-2 py-1 rounded-md"
          style={{ backgroundColor: riskColor }}
        >
          <p className="text-white text-sm font-bold leading-none">{riskScore.toFixed(0)}%</p>
          <p className="text-white text-[9px] leading-none mt-0.5">{riskLabel}</p>
        </div>
      </div>

      <div className="p-3 bg-gray-900 space-y-2 rounded-b-lg">
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-gray-800 rounded-md p-2 text-center">
            <p className="text-white font-bold text-base">{accCount}</p>
            <p className="text-gray-400 text-[10px]">Accidents</p>
          </div>
          <div className="bg-gray-800 rounded-md p-2 text-center">
            <p className="text-white font-bold text-base">{lengthKm}</p>
            <p className="text-gray-400 text-[10px]">km length</p>
          </div>
        </div>

        {topSeverity.length > 0 && (
          <div>
            <p className="text-gray-400 text-[10px] uppercase tracking-wider mb-1.5">
              Severity Breakdown
            </p>
            <div className="space-y-1">
              {topSeverity.map(([sev, count]) => (
                <div key={sev} className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5">
                    <div
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ backgroundColor: SEVERITY_COLORS[sev] || '#6B7280' }}
                    />
                    <span className="text-gray-300 text-xs">{sev}</span>
                  </div>
                  <span className="text-white text-xs font-medium">{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {onViewDetails && (
          <button
            onClick={() => onViewDetails(segment)}
            className="w-full py-1.5 bg-blue-600 hover:bg-blue-500 text-white
                       text-xs font-medium rounded-md transition-colors duration-200 mt-1"
          >
            View Full Details →
          </button>
        )}
      </div>
    </div>
  );
};

export default SegmentPopup;
