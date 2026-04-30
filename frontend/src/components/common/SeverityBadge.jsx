import React from 'react';
import { SEVERITY_COLORS, SEVERITY_ICONS } from '../../utils/constants';

const SeverityBadge = ({ severity, size = 'md' }) => {
  const color = SEVERITY_COLORS[severity] || '#8899A6';
  const icon = SEVERITY_ICONS[severity] || '❓';

  const sizeClasses = {
    sm: 'text-xs px-2 py-1',
    md: 'text-sm px-3 py-1.5',
    lg: 'text-lg px-5 py-3 font-bold',
    xl: 'text-2xl px-6 py-4 font-bold',
  };

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full font-semibold ${sizeClasses[size]}`}
      style={{
        backgroundColor: `${color}20`,
        color: color,
        border: `1px solid ${color}40`,
      }}
    >
      <span>{icon}</span>
      {severity}
    </span>
  );
};

export default SeverityBadge;
