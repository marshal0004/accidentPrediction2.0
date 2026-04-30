import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  FiActivity, FiPlay, FiAlertTriangle,
  FiCloud, FiClock, FiTrendingDown, FiInfo
} from 'react-icons/fi';
import toast from 'react-hot-toast';

import { api } from '../api/apiClient';
import { SCENARIO_OPTIONS } from '../utils/constants';
import ChartTile from '../components/common/ChartTile';
import LoadingSpinner from '../components/common/LoadingSpinner';
import KPICard from '../components/common/KPICard';
import TopDangerousList from '../components/digital_twin/TopDangerousList';

const getRiskColor = (score) => {
  if (score >= 75) return '#EF4444';
  if (score >= 40) return '#F59E0B';
  return '#10B981';
};

const ScenarioSelect = ({ label, icon, options, value, onChange }) => (
  <div>
    <label className="flex items-center gap-1.5 text-xs text-pbi-muted mb-1.5">
      {icon} {label}
    </label>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="pbi-select w-full text-sm"
    >
      <option value="">— Select —</option>
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  </div>
);

const SimulationPage = () => {
  const [cities,         setCities]         = useState([]);
  const [selectedCity,   setSelectedCity]   = useState('');
  const [segments,       setSegments]       = useState([]);
  const [selectedSeg,    setSelectedSeg]    = useState(null);
  const [scenarioType,   setScenarioType]   = useState('weather');
  const [weather,        setWeather]        = useState('');
  const [timePeriod,     setTimePeriod]     = useState('');
  const [trafficLevel,   setTrafficLevel]   = useState('');
  const [result,         setResult]         = useState(null);
  const [loading,        setLoading]        = useState(false);
  const [loadingSegs,    setLoadingSegs]    = useState(false);

  // load cities
  useEffect(() => {
    const fetchCities = async () => {
      try {
        const res = await api.twinCities();
        const ready = (res.data?.cities || []).filter((c) => c.status === 'ready');
        setCities(ready);
        if (ready.length > 0) {
          setSelectedCity(ready[0].key);
        }
      } catch (err) {
        console.error(err);
      }
    };
    fetchCities();
  }, []);

  // load top dangerous segments for selected city
  useEffect(() => {
    if (!selectedCity) return;
    const fetchSegments = async () => {
      setLoadingSegs(true);
      setSelectedSeg(null);
      setResult(null);
      try {
        const res = await api.twinTopDangerous(selectedCity, 20, 0);
        setSegments(res.data?.segments || []);
      } catch (err) {
        console.error(err);
      } finally {
        setLoadingSegs(false);
      }
    };
    fetchSegments();
  }, [selectedCity]);

  const handleRunSimulation = async () => {
    if (!selectedCity) return toast.error('Select a city first');
    if (!selectedSeg)  return toast.error('Select a road segment');

    const params = {};
    if (scenarioType === 'weather' && !weather)       return toast.error('Select a weather condition');
    if (scenarioType === 'time'    && !timePeriod)    return toast.error('Select a time period');
    if (scenarioType === 'traffic' && !trafficLevel)  return toast.error('Select a traffic level');

    if (weather)      params.weather       = weather;
    if (timePeriod)   params.time_period   = timePeriod;
    if (trafficLevel) params.traffic_level = trafficLevel;

    setLoading(true);
    setResult(null);
    try {
      const res = await api.twinSimulate(selectedCity, selectedSeg.segment_id, scenarioType, params);
      setResult(res.data?.result || res.data);
      toast.success('Simulation complete!');
    } catch (err) {
      const detail = err.response?.data?.detail || 'Simulation failed';
      if (detail.includes('Scenario simulator not available')) {
        toast.error('Simulation requires ML models to be trained. Run the pipeline first.');
      } else {
        toast.error(detail);
      }
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const baselineRisk  = result?.baseline_risk  ?? null;
  const simulatedRisk = result?.simulated_risk ?? null;
  const riskChange    = baselineRisk != null && simulatedRisk != null
    ? simulatedRisk - baselineRisk : null;

  return (
    <div className="space-y-5">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-xl font-bold text-white flex items-center gap-2">
          <FiActivity className="text-pbi-purple" />
          Scenario Simulator
        </h1>
        <p className="text-sm text-pbi-muted mt-0.5">
          Simulate "What happens if it rains on this road at night?"
        </p>
      </motion.div>

      {/* Results KPIs */}
      {result && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <KPICard
            icon={<FiAlertTriangle />}
            value={baselineRisk != null ? `${baselineRisk.toFixed(1)}%` : '—'}
            label="Baseline Risk"
            color={getRiskColor(baselineRisk ?? 0)}
            delay={0}
          />
          <KPICard
            icon={<FiActivity />}
            value={simulatedRisk != null ? `${simulatedRisk.toFixed(1)}%` : '—'}
            label="Simulated Risk"
            color={getRiskColor(simulatedRisk ?? 0)}
            delay={0.05}
          />
          <KPICard
            icon={<FiTrendingDown />}
            value={riskChange != null
              ? `${riskChange >= 0 ? '+' : ''}${riskChange.toFixed(1)}%`
              : '—'}
            label="Risk Change"
            color={riskChange == null ? '#6B7280' : riskChange < 0 ? '#10B981' : '#EF4444'}
            delay={0.1}
          />
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-[340px_1fr] gap-5">
        {/* ── LEFT: Controls ──────────────────────────────────────────────── */}
        <div className="space-y-4">
          {/* City selector */}
          <ChartTile title="Configuration" subtitle="Set up your simulation">
            <div className="space-y-4">
              {/* City */}
              <div>
                <label className="text-xs text-pbi-muted mb-1.5 block">City</label>
                <select
                  value={selectedCity}
                  onChange={(e) => setSelectedCity(e.target.value)}
                  className="pbi-select w-full text-sm"
                >
                  <option value="">— Select City —</option>
                  {cities.map((c) => (
                    <option key={c.key} value={c.key}>{c.display_name || c.name}</option>
                  ))}
                </select>
                {cities.length === 0 && (
                  <p className="text-xs text-pbi-yellow mt-1 flex items-center gap-1">
                    <FiInfo /> No cities initialized. Go to Digital Twin page first.
                  </p>
                )}
              </div>

              {/* Scenario type */}
              <div>
                <label className="text-xs text-pbi-muted mb-1.5 block">Scenario Type</label>
                <div className="grid grid-cols-3 gap-1">
                  {[
                    { id: 'weather', label: '🌦 Weather',  icon: <FiCloud /> },
                    { id: 'time',    label: '🕐 Time',     icon: <FiClock /> },
                    { id: 'traffic', label: '🚗 Traffic',  icon: <FiActivity /> },
                  ].map((s) => (
                    <button
                      key={s.id}
                      onClick={() => { setScenarioType(s.id); setResult(null); }}
                      className={`py-2 px-2 rounded-lg text-xs font-medium transition-all duration-200
                                  ${scenarioType === s.id
                                    ? 'bg-pbi-purple text-white shadow'
                                    : 'bg-pbi-bg2 border border-pbi-border text-pbi-muted hover:text-white'}`}
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Scenario parameters */}
              {scenarioType === 'weather' && (
                <ScenarioSelect
                  label="Weather Condition"
                  icon={<FiCloud />}
                  options={SCENARIO_OPTIONS.weather}
                  value={weather}
                  onChange={(v) => { setWeather(v); setResult(null); }}
                />
              )}
              {scenarioType === 'time' && (
                <ScenarioSelect
                  label="Time Period"
                  icon={<FiClock />}
                  options={SCENARIO_OPTIONS.time_period}
                  value={timePeriod}
                  onChange={(v) => { setTimePeriod(v); setResult(null); }}
                />
              )}
              {scenarioType === 'traffic' && (
                <ScenarioSelect
                  label="Traffic Level"
                  icon={<FiActivity />}
                  options={SCENARIO_OPTIONS.traffic_level}
                  value={trafficLevel}
                  onChange={(v) => { setTrafficLevel(v); setResult(null); }}
                />
              )}

              {/* Run button */}
              <button
                onClick={handleRunSimulation}
                disabled={loading || !selectedSeg}
                className="pbi-button w-full flex items-center justify-center gap-2"
              >
                {loading ? (
                  <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />Running...</>
                ) : (
                  <><FiPlay /> Run Simulation</>
                )}
              </button>

              {!selectedSeg && (
                <p className="text-xs text-pbi-yellow flex items-center gap-1">
                  <FiInfo /> Select a segment from the list on the right
                </p>
              )}
            </div>
          </ChartTile>

          {/* Result detail */}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card-static rounded-xl p-4 space-y-3"
            >
              <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                <FiActivity className="text-pbi-purple" /> Simulation Results
              </h4>

              {/* Risk gauge bar */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-pbi-muted">Baseline</span>
                  <span className="text-pbi-muted">Simulated</span>
                </div>
                <div className="relative h-4 bg-pbi-border rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{
                      width: `${Math.min(baselineRisk ?? 0, 100)}%`,
                      backgroundColor: getRiskColor(baselineRisk ?? 0),
                      opacity: 0.4,
                    }}
                  />
                  <div
                    className="absolute top-0 left-0 h-full rounded-full transition-all duration-700"
                    style={{
                      width: `${Math.min(simulatedRisk ?? 0, 100)}%`,
                      backgroundColor: getRiskColor(simulatedRisk ?? 0),
                    }}
                  />
                </div>
                <div className="flex justify-between text-xs mt-1">
                  <span style={{ color: getRiskColor(baselineRisk ?? 0) }}>
                    {baselineRisk?.toFixed(1)}%
                  </span>
                  <span style={{ color: getRiskColor(simulatedRisk ?? 0) }}>
                    {simulatedRisk?.toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Details */}
              {result.description && (
                <p className="text-xs text-pbi-text2 bg-pbi-bg2 rounded-lg p-3">
                  {result.description}
                </p>
              )}

              {result.severity_probabilities && (
                <div>
                  <p className="text-xs text-pbi-muted mb-2">Severity Probabilities</p>
                  {Object.entries(result.severity_probabilities).map(([sev, prob]) => (
                    <div key={sev} className="flex items-center justify-between mb-1">
                      <span className="text-xs text-white">{sev}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-1.5 bg-pbi-border rounded-full overflow-hidden">
                          <div
                            className="h-full bg-pbi-blue rounded-full"
                            style={{ width: `${(prob * 100).toFixed(0)}%` }}
                          />
                        </div>
                        <span className="text-xs text-pbi-muted w-10 text-right">
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </div>

        {/* ── RIGHT: Segment selector ──────────────────────────────────────── */}
        <ChartTile
          title="Select Road Segment"
          subtitle={selectedSeg
            ? `Selected: ${selectedSeg.name || selectedSeg.segment_name || selectedSeg.segment_id}`
            : 'Click a segment to simulate'}
        >
          <TopDangerousList
            segments={segments}
            onSelectSegment={(seg) => { setSelectedSeg(seg); setResult(null); }}
            selectedSegmentId={selectedSeg?.segment_id}
            loading={loadingSegs}
          />
        </ChartTile>
      </div>
    </div>
  );
};

export default SimulationPage;
