import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiX, FiAlertTriangle, FiMap, FiActivity } from 'react-icons/fi';
import { SEVERITY_COLORS } from '../../utils/constants';

const getRiskColor = (score) => {
  if (score >= 75) return '#EF4444';
  if (score >= 40) return '#F59E0B';
  return '#10B981';
};

const InfoRow = ({ label, value, color }) => (
  <div className="flex items-center justify-between py-1.5 border-b border-pbi-border last:border-0">
    <span className="text-xs text-pbi-muted">{label}</span>
    <span
      className="text-xs font-semibold"
      style={color ? { color } : { color: 'white' }}
    >
      {value ?? '—'}
    </span>
  </div>
);

const SegmentDetailPanel = ({ segment, onClose, onSimulate }) => {
  if (!segment) return (
    <div className="glass-card-static rounded-xl p-6 text-center">
      <FiMap className="text-3xl text-pbi-muted mx-auto mb-3" />
      <p className="text-sm text-pbi-muted">Click a segment on the map or from the Top Risk list</p>
    </div>
  );

  const riskScore    = segment.risk_score ?? (segment.final_risk != null ? segment.final_risk * 100 : 0);
  const riskColor    = getRiskColor(riskScore);
  const segName      = segment.name || segment.segment_name || 'Unnamed Road';
  const severityDist = segment.severity_distribution || {};
  const totalInDist  = Object.values(severityDist).reduce((a, b) => a + b, 0);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 20 }}
        transition={{ duration: 0.25 }}
        className="glass-card-static rounded-xl overflow-hidden"
      >
        {/* Header */}
        <div
          className="p-4 flex items-start justify-between gap-3"
          style={{ borderBottom: `2px solid ${riskColor}` }}
        >
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-bold text-white leading-tight truncate">{segName}</h3>
            <p className="text-xs text-pbi-muted mt-0.5">
              {segment.road_type || segment.road_category || segment.highway || 'Unknown'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="px-3 py-1.5 rounded-lg text-center"
              style={{
                backgroundColor: `${riskColor}20`,
                border: `1px solid ${riskColor}`,
              }}
            >
              <p className="text-lg font-bold leading-none" style={{ color: riskColor }}>
                {riskScore.toFixed(0)}%
              </p>
              <p className="text-[9px] font-medium mt-0.5" style={{ color: riskColor }}>
                RISK
              </p>
            </div>
            <button
              onClick={onClose}
              className="w-7 h-7 rounded-lg bg-pbi-border/50 hover:bg-pbi-border
                         flex items-center justify-center text-pbi-muted hover:text-white
                         transition-colors duration-200"
            >
              <FiX className="text-sm" />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="p-4 space-y-4 max-h-[400px] overflow-y-auto">
          {/* Basic info */}
          <div>
            <p className="text-[10px] font-semibold text-pbi-muted uppercase tracking-wider mb-2 flex items-center gap-1">
              <FiMap className="text-pbi-blue" /> Road Info
            </p>
            <div className="bg-pbi-bg2 rounded-lg p-3">
              <InfoRow
                label="Length"
                value={segment.length_km != null ? `${segment.length_km.toFixed(2)} km` : '—'}
              />
              <InfoRow label="Category"  value={segment.road_category || '—'} />
              <InfoRow label="Accidents" value={segment.total_accidents ?? segment.accident_count ?? 0} color={riskColor} />
              <InfoRow
                label="Risk Level"
                value={riskScore >= 75 ? 'High' : riskScore >= 40 ? 'Medium' : 'Low'}
                color={riskColor}
              />
            </div>
          </div>

          {/* Severity distribution */}
          {totalInDist > 0 && (
            <div>
              <p className="text-[10px] font-semibold text-pbi-muted uppercase tracking-wider mb-2 flex items-center gap-1">
                <FiAlertTriangle className="text-pbi-red" /> Severity Distribution
              </p>
              <div className="space-y-2">
                {Object.entries(severityDist)
                  .sort((a, b) => b[1] - a[1])
                  .map(([sev, count]) => {
                    const pct = (count / totalInDist) * 100;
                    const col = SEVERITY_COLORS[sev] || '#6B7280';
                    return (
                      <div key={sev}>
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-1.5">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: col }} />
                            <span className="text-xs text-white">{sev}</span>
                          </div>
                          <span className="text-xs text-pbi-muted">
                            {count} ({pct.toFixed(0)}%)
                          </span>
                        </div>
                        <div className="h-1.5 bg-pbi-border rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{ width: `${pct}%`, backgroundColor: col }}
                          />
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Simulate button */}
          {onSimulate && (
            <button
              onClick={() => onSimulate(segment)}
              className="w-full py-2 bg-pbi-blue hover:bg-blue-600 text-white text-xs
                         font-semibold rounded-lg transition-colors duration-200
                         flex items-center justify-center gap-2"
            >
              <FiActivity />
              Simulate Scenarios
            </button>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default SegmentDetailPanel;
