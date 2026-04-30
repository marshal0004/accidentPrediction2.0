export const formatNumber = (num) => {
  if (num === null || num === undefined) return '—';
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
  return num.toLocaleString();
};

export const formatPercent = (num, decimals = 1) => {
  if (num === null || num === undefined) return '—';
  return (num * 100).toFixed(decimals) + '%';
};

export const formatMetric = (num, decimals = 4) => {
  if (num === null || num === undefined) return '—';
  return Number(num).toFixed(decimals);
};

export const formatTime = (seconds) => {
  if (seconds === null || seconds === undefined) return '—';
  if (seconds < 1) return (seconds * 1000).toFixed(0) + 'ms';
  if (seconds < 60) return seconds.toFixed(1) + 's';
  return (seconds / 60).toFixed(1) + 'min';
};

export const truncateText = (text, maxLen = 20) => {
  if (!text) return '';
  return text.length > maxLen ? text.substring(0, maxLen) + '...' : text;
};
