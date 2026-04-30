import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { FiBell, FiRefreshCw, FiClock, FiWifi, FiWifiOff } from 'react-icons/fi';
import { api } from '../../api/apiClient';

const pageTitles = {
  '/': 'Overview Dashboard',
  '/models': 'Model Performance',
  '/features': 'Feature Analysis & SHAP',
  '/predict': 'Predict Severity',
  '/data': 'Dataset Explorer',
};

const Header = () => {
  const location = useLocation();
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [isOnline, setIsOnline] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const title = pageTitles[location.pathname] || 'Dashboard';

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await api.health();
        setIsOnline(true);
      } catch {
        setIsOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await api.health();
      setIsOnline(true);
      setLastUpdated(new Date());
    } catch {
      setIsOnline(false);
    }
    setTimeout(() => setRefreshing(false), 1000);
    window.location.reload();
  };

  return (
    <header className="sticky top-0 z-40 bg-pbi-bg/80 backdrop-blur-xl border-b border-pbi-border px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left — Title */}
        <div>
          <h1 className="text-xl font-semibold text-white">{title}</h1>
          <p className="text-xs text-pbi-muted mt-0.5">
            Indian Road Accident Analysis Platform
          </p>
        </div>

        {/* Right — Actions */}
        <div className="flex items-center gap-4">
          {/* Status Indicator */}
          <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium
                          ${isOnline ? 'bg-pbi-green/10 text-pbi-green' : 'bg-pbi-red/10 text-pbi-red'}`}>
            {isOnline ? <FiWifi className="text-sm" /> : <FiWifiOff className="text-sm" />}
            {isOnline ? 'API Online' : 'API Offline'}
          </div>

          {/* Last Updated */}
          <div className="hidden md:flex items-center gap-1.5 text-xs text-pbi-muted">
            <FiClock className="text-sm" />
            {lastUpdated.toLocaleTimeString()}
          </div>

          {/* Refresh */}
          <button
            onClick={handleRefresh}
            className="p-2 rounded-lg text-pbi-muted hover:text-white hover:bg-white/5 transition-all duration-200"
            title="Refresh Data"
          >
            <FiRefreshCw className={`text-lg ${refreshing ? 'animate-spin' : ''}`} />
          </button>

          {/* Notification */}
          <button className="p-2 rounded-lg text-pbi-muted hover:text-white hover:bg-white/5 transition-all duration-200 relative">
            <FiBell className="text-lg" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-pbi-red rounded-full" />
          </button>

          {/* Avatar */}
          <div className="w-8 h-8 bg-gradient-to-br from-pbi-blue to-pbi-purple rounded-full flex items-center justify-center">
            <span className="text-xs font-bold text-white">N</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
