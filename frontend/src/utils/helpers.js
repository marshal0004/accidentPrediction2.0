import { SEVERITY_COLORS } from './constants';

export const getSeverityColor = (severity) => {
  return SEVERITY_COLORS[severity] || '#8899A6';
};

export const getGradientId = (id) => `gradient-${id}`;

export const debounce = (func, wait) => {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(null, args), wait);
  };
};

export const classNames = (...classes) => {
  return classes.filter(Boolean).join(' ');
};

export const downloadCSV = (data, filename = 'download.csv') => {
  const blob = new Blob([data], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  window.URL.revokeObjectURL(url);
};
