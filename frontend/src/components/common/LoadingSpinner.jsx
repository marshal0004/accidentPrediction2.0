import React from 'react';

const LoadingSpinner = ({ size = 'md', text = 'Loading...' }) => {
  const sizeClasses = {
    sm: 'w-5 h-5 border-2',
    md: 'w-8 h-8 border-3',
    lg: 'w-12 h-12 border-4',
  };

  return (
    <div className="flex flex-col items-center justify-center py-12 gap-3">
      <div className={`${sizeClasses[size]} border-pbi-border border-t-pbi-blue rounded-full animate-spin`} />
      <p className="text-sm text-pbi-muted animate-pulse">{text}</p>
    </div>
  );
};

export const SkeletonCard = () => (
  <div className="glass-card-static p-5 animate-pulse">
    <div className="h-3 bg-pbi-border rounded w-1/3 mb-4" />
    <div className="h-8 bg-pbi-border rounded w-2/3 mb-2" />
    <div className="h-3 bg-pbi-border rounded w-1/2" />
  </div>
);

export const SkeletonChart = () => (
  <div className="glass-card-static p-5 animate-pulse">
    <div className="h-4 bg-pbi-border rounded w-1/3 mb-6" />
    <div className="h-48 bg-pbi-border rounded" />
  </div>
);

export default LoadingSpinner;
