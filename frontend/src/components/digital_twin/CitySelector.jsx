import React from 'react';
import { FiMapPin, FiLoader, FiCheck, FiAlertCircle } from 'react-icons/fi';

const STATUS_CONFIG = {
  ready:           { icon: <FiCheck />,        color: 'text-pbi-green',  bg: 'bg-pbi-green/10',  label: 'Ready' },
  initializing:    { icon: <FiLoader />,       color: 'text-pbi-yellow', bg: 'bg-pbi-yellow/10', label: 'Building...' },
  not_initialized: { icon: <FiAlertCircle />,  color: 'text-pbi-muted',  bg: 'bg-pbi-border/20', label: 'Not Built' },
  error:           { icon: <FiAlertCircle />,  color: 'text-pbi-red',    bg: 'bg-pbi-red/10',    label: 'Error' },
};

const CitySelector = ({ cities = [], selectedCity, onSelect, onInitialize, initializingCity }) => {
  return (
    <div className="glass-card-static p-4 rounded-xl">
      <h4 className="text-xs font-semibold text-pbi-muted uppercase tracking-wider mb-3 flex items-center gap-2">
        <FiMapPin className="text-pbi-blue" />
        Select City
      </h4>

      <div className="space-y-2">
        {cities.map((city) => {
          const status = city.status || 'not_initialized';
          const cfg    = STATUS_CONFIG[status] || STATUS_CONFIG.not_initialized;
          const isSelected    = selectedCity === city.key;
          const isInitializing = initializingCity === city.key;

          return (
            <div
              key={city.key}
              className={`rounded-lg border transition-all duration-200 overflow-hidden
                ${isSelected
                  ? 'border-pbi-blue bg-pbi-blue/10'
                  : 'border-pbi-border bg-pbi-bg2 hover:border-pbi-blue/50'}`}
            >
              {/* City row */}
              <button
                onClick={() => status === 'ready' && onSelect(city.key)}
                disabled={status !== 'ready'}
                className="w-full flex items-center justify-between p-3 text-left"
              >
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full flex-shrink-0
                    ${status === 'ready'           ? 'bg-pbi-green animate-pulse' :
                      status === 'initializing'    ? 'bg-pbi-yellow animate-spin' :
                      status === 'error'           ? 'bg-pbi-red' : 'bg-pbi-muted'}`}
                  />
                  <div>
                    <p className={`text-sm font-medium ${isSelected ? 'text-pbi-blue' : 'text-white'}`}>
                      {city.display_name || city.name}
                    </p>
                    {city.metadata?.total_segments && (
                      <p className="text-xs text-pbi-muted">
                        {city.metadata.total_segments.toLocaleString()} segments
                      </p>
                    )}
                  </div>
                </div>

                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${cfg.color} ${cfg.bg}`}>
                  {isInitializing ? 'Building...' : cfg.label}
                </span>
              </button>

              {/* Initialize button if not ready */}
              {status !== 'ready' && status !== 'initializing' && !isInitializing && (
                <div className="px-3 pb-3">
                  <button
                    onClick={() => onInitialize(city.key)}
                    className="w-full py-1.5 px-3 bg-pbi-blue/20 hover:bg-pbi-blue/30
                               text-pbi-blue text-xs font-medium rounded-lg
                               transition-all duration-200 border border-pbi-blue/30"
                  >
                    Initialize Twin
                  </button>
                </div>
              )}

              {/* Progress for initializing */}
              {(isInitializing || status === 'initializing') && (
                <div className="px-3 pb-3">
                  <div className="flex items-center gap-2 text-pbi-yellow text-xs">
                    <FiLoader className="animate-spin flex-shrink-0" />
                    <span>Downloading road network... (~2 min)</span>
                  </div>
                  <div className="mt-2 h-1 bg-pbi-border rounded-full overflow-hidden">
                    <div className="h-full bg-pbi-yellow rounded-full animate-pulse w-1/2" />
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default CitySelector;
