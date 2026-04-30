import React from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-pbi-card border border-pbi-border rounded-lg p-3 shadow-xl">
      <p className="text-sm font-semibold text-white">Hour: {label}:00</p>
      <p className="text-xs text-pbi-blue">{payload[0].value} accidents</p>
    </div>
  );
};

const AreaTimeChart = ({ data }) => {
  if (!data || !data.labels) return null;

  const chartData = data.labels.map((label, i) => ({
    hour: label,
    accidents: data.values[i],
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <defs>
          <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#2563EB" stopOpacity={0.4} />
            <stop offset="95%" stopColor="#2563EB" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#2D374830" />
        <XAxis dataKey="hour" tick={{ fill: '#8899A6', fontSize: 11 }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fill: '#8899A6', fontSize: 11 }} tickLine={false} axisLine={false} />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine x={22} stroke="#EF4444" strokeDasharray="5 5" label={{ value: "Danger Zone", fill: "#EF4444", fontSize: 10 }} />
        <Area type="monotone" dataKey="accidents" stroke="#2563EB" strokeWidth={2.5} fill="url(#areaGradient)" animationDuration={1500} />
      </AreaChart>
    </ResponsiveContainer>
  );
};

export default AreaTimeChart;
