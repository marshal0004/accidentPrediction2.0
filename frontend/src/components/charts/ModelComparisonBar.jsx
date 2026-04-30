import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, LabelList } from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-pbi-card border border-pbi-border rounded-lg p-3 shadow-xl">
      <p className="text-sm font-semibold text-white mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} className="text-xs" style={{ color: p.color }}>{p.name}: {(p.value * 100).toFixed(1)}%</p>
      ))}
    </div>
  );
};

const ModelComparisonBar = ({ models }) => {
  if (!models || !models.length) return null;

  const chartData = models.map((m) => ({
    name: m.model_name,
    Accuracy: m.accuracy,
    'F1 Weighted': m.f1_weighted,
    'F1 Macro': m.f1_macro,
  }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={chartData} margin={{ top: 20, right: 20, left: 0, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2D374830" />
        <XAxis dataKey="name" tick={{ fill: '#8899A6', fontSize: 10 }} angle={-20} textAnchor="end" tickLine={false} />
        <YAxis tick={{ fill: '#8899A6', fontSize: 11 }} domain={[0, 1]} tickLine={false} axisLine={false} />
        <Tooltip content={<CustomTooltip />} />
        <Legend formatter={(v) => <span className="text-xs text-pbi-text2">{v}</span>} />
        <Bar dataKey="Accuracy" fill="#2563EB" radius={[4, 4, 0, 0]} animationDuration={1200}>
          <LabelList dataKey="Accuracy" position="top" formatter={(v) => (v * 100).toFixed(0) + '%'} fill="#8899A6" fontSize={9} />
        </Bar>
        <Bar dataKey="F1 Weighted" fill="#EF4444" radius={[4, 4, 0, 0]} animationDuration={1200} />
        <Bar dataKey="F1 Macro" fill="#10B981" radius={[4, 4, 0, 0]} animationDuration={1200} />
      </BarChart>
    </ResponsiveContainer>
  );
};

export default ModelComparisonBar;
