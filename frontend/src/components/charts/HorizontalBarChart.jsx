import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import { CHART_COLORS } from '../../utils/constants';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-pbi-card border border-pbi-border rounded-lg p-3 shadow-xl">
      <p className="text-sm font-semibold text-white mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} className="text-xs" style={{ color: p.color }}>{p.name}: {p.value}</p>
      ))}
    </div>
  );
};

const HorizontalBarChart = ({ data }) => {
  if (!data || !data.labels) return null;

  if (data.datasets) {
    const chartData = data.labels.map((label, i) => {
      const point = { name: label };
      data.datasets.forEach((ds) => {
        point[ds.label] = ds.data[i] || 0;
      });
      return point;
    });

    return (
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 20, left: 80, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2D374830" horizontal={false} />
          <XAxis type="number" tick={{ fill: '#8899A6', fontSize: 11 }} />
          <YAxis type="category" dataKey="name" tick={{ fill: '#8899A6', fontSize: 10 }} width={75} />
          <Tooltip content={<CustomTooltip />} />
          <Legend formatter={(v) => <span className="text-xs text-pbi-text2">{v}</span>} />
          {data.datasets.map((ds, i) => (
            <Bar key={ds.label} dataKey={ds.label} stackId="a" fill={ds.color || CHART_COLORS[i]} radius={i === data.datasets.length - 1 ? [0, 4, 4, 0] : [0, 0, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    );
  }

  const chartData = data.labels.map((label, i) => ({ name: label, value: data.values[i] }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 20, left: 80, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2D374830" horizontal={false} />
        <XAxis type="number" tick={{ fill: '#8899A6', fontSize: 11 }} />
        <YAxis type="category" dataKey="name" tick={{ fill: '#8899A6', fontSize: 10 }} width={75} />
        <Tooltip content={<CustomTooltip />} />
        <Bar dataKey="value" fill="#2563EB" radius={[0, 6, 6, 0]} animationDuration={1200} />
      </BarChart>
    </ResponsiveContainer>
  );
};

export default HorizontalBarChart;
