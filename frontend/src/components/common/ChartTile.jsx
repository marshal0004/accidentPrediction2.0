import React from 'react';
import { motion } from 'framer-motion';
import LoadingSpinner from './LoadingSpinner';

const ChartTile = ({ title, subtitle, children, loading, error, className = '', delay = 0 }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className={`glass-card-static p-5 flex flex-col ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-white">{title}</h3>
          {subtitle && <p className="text-xs text-pbi-muted mt-0.5">{subtitle}</p>}
        </div>
        <div className="w-1.5 h-1.5 rounded-full bg-pbi-green animate-pulse" />
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0">
        {loading ? (
          <LoadingSpinner size="sm" text="Loading chart..." />
        ) : error ? (
          <div className="flex items-center justify-center h-full text-pbi-red text-sm">
            {error}
          </div>
        ) : (
          children
        )}
      </div>
    </motion.div>
  );
};

export default ChartTile;
