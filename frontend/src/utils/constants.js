export const SEVERITY_COLORS = {
  Fatal: '#EF4444',
  Grievous: '#F97316',
  Minor: '#F59E0B',
  'No Injury': '#10B981',
  'Slight Injury': '#F59E0B',
  'Serious Injury': '#F97316',
};

export const CHART_COLORS = [
  '#2563EB', '#EF4444', '#10B981', '#F59E0B',
  '#8B5CF6', '#F97316', '#06B6D4', '#EC4899',
  '#84CC16', '#A855F7'
];

export const MODEL_COLORS = {
  RandomForest: '#2563EB',
  XGBoost: '#EF4444',
  GradientBoosting: '#10B981',
  SVM: '#F59E0B',
  LogisticRegression: '#8B5CF6',
};

export const SEVERITY_ICONS = {
  Fatal: '💀',
  Grievous: '🚨',
  Minor: '⚠️',
  'No Injury': '✅',
};

export const NAV_ITEMS = [
  { path: '/', label: 'Overview', icon: 'BarChart2' },
  { path: '/models', label: 'Models', icon: 'Cpu' },
  { path: '/features', label: 'Features', icon: 'Search' },
  { path: '/predict', label: 'Predict', icon: 'Target' },
  { path: '/data', label: 'Data', icon: 'Database' },
  { path: '/digital-twin', label: 'Digital Twin', icon: 'Map' },
  { path: '/simulation', label: 'Simulation', icon: 'Activity' },
];

// ============================================
// DIGITAL TWIN CONSTANTS
// ============================================

export const RISK_COLORS = {
  High:   '#EF4444',
  Medium: '#F59E0B',
  Low:    '#10B981',
};

export const RISK_THRESHOLDS = {
  High:   75,
  Medium: 40,
  Low:    0,
};

export const CITY_CENTERS = {
  delhi:     { lat: 28.6139, lng: 77.2090, zoom: 12 },
  bangalore: { lat: 12.9716, lng: 77.5946, zoom: 12 },
  dehradun:  { lat: 30.3165, lng: 78.0322, zoom: 13 },
};

export const SCENARIO_OPTIONS = {
  weather: [
    { value: 'Clear', label: '☀️ Clear' },
    { value: 'Rain', label: '🌧️ Rain' },
    { value: 'Fog', label: '🌫️ Fog' },
    { value: 'Cloudy', label: '⛅ Cloudy' },
  ],
  time_period: [
    { value: 'Morning', label: '🌅 Morning (6-12)' },
    { value: 'Afternoon', label: '☀️ Afternoon (12-17)' },
    { value: 'Evening', label: '🌇 Evening (17-21)' },
    { value: 'Night', label: '🌙 Night (21-24)' },
    { value: 'Night_Late', label: '🌑 Late Night (0-6)' },
  ],
  traffic_level: [
    { value: 'Low', label: '🟢 Low Traffic' },
    { value: 'Medium', label: '🟡 Medium Traffic' },
    { value: 'High', label: '🔴 High Traffic' },
  ],
};

export const ROAD_CATEGORY_COLORS = {
  'National Highway': '#EF4444',
  'State Highway':    '#F97316',
  'District Road':    '#F59E0B',
  'Local Road':       '#10B981',
  'Unknown':          '#6B7280',
};
