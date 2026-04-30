import React from 'react';
import { motion } from 'framer-motion';

const KPICard = ({ icon, value, label, subtext, trend, trendDirection, color = '#2563EB', delay = 0 }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="glass-card p-5 cursor-pointer group"
    >
      <div className="flex items-start justify-between mb-3">
        <div
          className="w-11 h-11 rounded-xl flex items-center justify-center text-xl transition-transform duration-300 group-hover:scale-110"
          style={{ backgroundColor: `${color}20`, color }}
        >
          {icon}
        </div>
        {trend && (
          <span className={`text-xs font-semibold px-2 py-1 rounded-full
            ${trendDirection === 'up' ? 'bg-pbi-red/10 text-pbi-red' : 'bg-pbi-green/10 text-pbi-green'}`}>
            {trend}
          </span>
        )}
      </div>

      <div className="space-y-1">
        <h3 className="text-2xl font-bold text-white tracking-tight">
          {value ?? '—'}
        </h3>
        <p className="text-xs font-medium text-pbi-muted uppercase tracking-wider">{label}</p>
        {subtext && (
          <p className="text-xs text-pbi-text2 mt-1">{subtext}</p>
        )}
      </div>
    </motion.div>
  );
};

export default KPICard;
