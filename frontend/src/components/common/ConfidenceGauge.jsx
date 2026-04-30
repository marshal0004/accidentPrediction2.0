import React from 'react';

const ConfidenceGauge = ({ value, size = 120 }) => {
  const percentage = Math.round(value * 100);
  const radius = (size - 12) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (value * circumference);

  const getColor = (val) => {
    if (val >= 0.8) return '#10B981';
    if (val >= 0.6) return '#F59E0B';
    return '#EF4444';
  };

  const color = getColor(value);

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#2D3748"
          strokeWidth="8"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-2xl font-bold text-white">{percentage}%</span>
        <span className="text-[10px] text-pbi-muted uppercase">Confidence</span>
      </div>
    </div>
  );
};

export default ConfidenceGauge;
