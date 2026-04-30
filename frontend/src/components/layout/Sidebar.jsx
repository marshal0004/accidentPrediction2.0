import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  FiBarChart2, FiCpu, FiSearch, FiTarget, FiDatabase,
  FiChevronLeft, FiChevronRight, FiActivity, FiMap, FiGitBranch
} from 'react-icons/fi';

const navItems = [
  { path: '/',             label: 'Overview',      icon: FiBarChart2,  section: null },
  { path: '/models',       label: 'Models',        icon: FiCpu,        section: null },
  { path: '/features',     label: 'Features',      icon: FiSearch,     section: null },
  { path: '/predict',      label: 'Predict',       icon: FiTarget,     section: null },
  { path: '/data',         label: 'Data',          icon: FiDatabase,   section: null },
  { path: '/digital-twin', label: 'Digital Twin',  icon: FiMap,        section: 'Digital Twin' },
  { path: '/simulation',   label: 'Simulation',    icon: FiGitBranch,  section: 'Digital Twin' },
];

const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  let lastSection = null;

  return (
    <aside
      className={`fixed top-0 left-0 h-screen bg-pbi-sidebar border-r border-pbi-border z-50
                   flex flex-col transition-all duration-300 ease-in-out
                   ${collapsed ? 'w-[70px]' : 'w-[260px]'}`}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-6 border-b border-pbi-border min-h-[80px]">
        <div className="w-9 h-9 bg-gradient-to-br from-pbi-blue to-pbi-cyan rounded-lg flex items-center justify-center flex-shrink-0">
          <FiActivity className="text-white text-lg" />
        </div>
        {!collapsed && (
          <div className="overflow-hidden">
            <h1 className="text-lg font-bold text-white tracking-tight">AccidentAI</h1>
            <p className="text-[10px] text-pbi-muted tracking-wider uppercase">Severity Predictor</p>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 px-3 space-y-0.5 overflow-y-auto">
        {navItems.map((item) => {
          const Icon     = item.icon;
          const isActive = location.pathname === item.path;
          const showSectionHeader = !collapsed && item.section && item.section !== lastSection;
          if (item.section) lastSection = item.section;

          return (
            <React.Fragment key={item.path}>
              {showSectionHeader && (
                <div className="pt-4 pb-1.5 px-3">
                  <p className="text-[10px] font-semibold text-pbi-muted uppercase tracking-widest">
                    {item.section}
                  </p>
                </div>
              )}
              <NavLink
                to={item.path}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group
                           ${isActive
                             ? 'bg-pbi-blue/15 text-pbi-blue border-l-[3px] border-pbi-blue'
                             : 'text-pbi-text2 hover:bg-white/5 hover:text-white border-l-[3px] border-transparent'}`}
              >
                <Icon className={`text-xl flex-shrink-0 ${isActive ? 'text-pbi-blue' : 'text-pbi-muted group-hover:text-white'}`} />
                {!collapsed && (
                  <span className="text-sm font-medium truncate">{item.label}</span>
                )}
                {isActive && !collapsed && (
                  <div className="ml-auto w-1.5 h-1.5 rounded-full bg-pbi-blue animate-pulse" />
                )}
              </NavLink>
            </React.Fragment>
          );
        })}
      </nav>

      {/* Collapse Toggle */}
      <div className="p-3 border-t border-pbi-border">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg
                     text-pbi-muted hover:text-white hover:bg-white/5 transition-all duration-200"
        >
          {collapsed ? <FiChevronRight className="text-lg" /> : <FiChevronLeft className="text-lg" />}
          {!collapsed && <span className="text-xs">Collapse</span>}
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
