import React from 'react';
import { motion } from 'framer-motion';
import SeverityBadge from '../common/SeverityBadge';
import ConfidenceGauge from '../common/ConfidenceGauge';
import { SEVERITY_COLORS } from '../../utils/constants';

const PredictionResult = ({ result }) => {
  if (!result) return null;

  const probEntries = Object.entries(result.probabilities || {}).sort((a, b) => b[1] - a[1]);
  const maxProb = Math.max(...probEntries.map(([, v]) => v), 0.01);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="glass-card p-6 space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Prediction Result</h3>
        <SeverityBadge severity={result.prediction} size="lg" />
      </div>

      {/* Confidence + Probabilities */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="flex flex-col items-center justify-center">
          <ConfidenceGauge value={result.confidence} size={140} />
        </div>

        <div className="space-y-3">
          <h4 className="text-xs font-semibold text-pbi-muted uppercase tracking-wider">Probability Distribution</h4>
          {probEntries.map(([label, prob]) => (
            <div key={label} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-pbi-text2">{label}</span>
                <span className="text-white font-semibold">{(prob * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 bg-pbi-bg rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(prob / maxProb) * 100}%` }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className="h-full rounded-full"
                  style={{ backgroundColor: SEVERITY_COLORS[label] || '#2563EB' }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Risk Factors */}
      {result.top_risk_factors && result.top_risk_factors.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-pbi-muted uppercase tracking-wider mb-3">Top Risk Factors</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {result.top_risk_factors.map((factor, i) => (
              <div key={i} className="bg-pbi-bg2 rounded-lg p-3 border border-pbi-border">
                <p className="text-sm font-semibold text-white">{factor.feature}</p>
                <p className="text-xs text-pbi-blue mt-1">SHAP: {factor.contribution.toFixed(4)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="pt-4 border-t border-pbi-border text-xs text-pbi-muted text-center">
        Predicted by <span className="text-pbi-blue font-semibold">{result.model_used}</span>
      </div>
    </motion.div>
  );
};

export default PredictionResult;
