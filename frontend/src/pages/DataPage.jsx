import React, { useState, useEffect } from 'react';
import { FiDatabase, FiExternalLink, FiChevronLeft, FiChevronRight } from 'react-icons/fi';
import { api } from '../api/apiClient';
import ChartTile from '../components/common/ChartTile';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { formatNumber } from '../utils/formatters';

const DataPage = () => {
  const [datasetsInfo, setDatasetsInfo] = useState([]);
  const [loading, setLoading] = useState(true);
  const [previewData, setPreviewData] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState('primary');
  const [page, setPage] = useState(1);
  const perPage = 15;

  useEffect(() => {
    const fetchInfo = async () => {
      try {
        const res = await api.datasetsInfo();
        setDatasetsInfo(res.data.datasets || []);
      } catch (err) {
        console.error('Failed to fetch dataset info:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchInfo();
  }, []);

  useEffect(() => {
    const fetchPreview = async () => {
      setPreviewLoading(true);
      try {
        const res = await api.dataPreview(selectedDataset, page, perPage);
        setPreviewData(res.data);
      } catch (err) {
        console.error('Failed to fetch data preview:', err);
      } finally {
        setPreviewLoading(false);
      }
    };
    fetchPreview();
  }, [selectedDataset, page]);

  if (loading) return <LoadingSpinner text="Loading dataset info..." />;

  return (
    <div className="space-y-6">
      {/* Dataset Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {datasetsInfo.map((ds, i) => (
          <div key={i} className={`glass-card p-6 cursor-pointer ${selectedDataset === (i === 0 ? 'primary' : 'secondary') ? 'ring-2 ring-pbi-blue' : ''}`}
            onClick={() => { setSelectedDataset(i === 0 ? 'primary' : 'secondary'); setPage(1); }}>
            <div className="flex items-start justify-between mb-4">
              <div className="w-10 h-10 bg-pbi-blue/20 rounded-lg flex items-center justify-center">
                <FiDatabase className="text-pbi-blue text-lg" />
              </div>
              <span className={`text-xs font-semibold px-2 py-1 rounded-full
                               ${ds.status === 'loaded' ? 'bg-pbi-green/10 text-pbi-green' : 'bg-pbi-red/10 text-pbi-red'}`}>
                {ds.status === 'loaded' ? '✓ Loaded' : '✗ Not Found'}
              </span>
            </div>
            <h3 className="text-base font-semibold text-white mb-2">{ds.name}</h3>
            <div className="space-y-1.5 text-xs text-pbi-text2">
              <p><span className="text-pbi-muted">Records:</span> {formatNumber(ds.records)}</p>
              <p><span className="text-pbi-muted">Features:</span> {ds.features}</p>
              <p><span className="text-pbi-muted">Severity Classes:</span> {ds.severity_classes}</p>
              <p><span className="text-pbi-muted">File:</span> {ds.filename}</p>
            </div>
            {ds.class_distribution && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {Object.entries(ds.class_distribution).map(([label, count]) => (
                  <span key={label} className="text-[10px] bg-pbi-bg2 px-2 py-1 rounded border border-pbi-border text-pbi-text2">
                    {label}: {formatNumber(count)}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Data Preview */}
      <ChartTile title="Data Preview" subtitle={`${selectedDataset} dataset — Page ${page}`}>
        {previewLoading ? (
          <LoadingSpinner size="sm" text="Loading data..." />
        ) : previewData ? (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-pbi-border">
                    {previewData.columns?.slice(0, 12).map((col) => (
                      <th key={col} className="px-3 py-2 text-left text-pbi-muted font-medium whitespace-nowrap">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-pbi-border/30">
                  {previewData.records?.map((row, i) => (
                    <tr key={i} className="hover:bg-white/[0.02]">
                      {previewData.columns?.slice(0, 12).map((col) => (
                        <td key={col} className="px-3 py-2 text-pbi-text2 whitespace-nowrap max-w-[150px] truncate">
                          {String(row[col] ?? '')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between mt-4 pt-3 border-t border-pbi-border">
              <span className="text-xs text-pbi-muted">
                {formatNumber(previewData.total_records)} total records
              </span>
              <div className="flex items-center gap-2">
                <button onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page === 1}
                  className="p-1.5 rounded text-pbi-muted hover:text-white disabled:opacity-30">
                  <FiChevronLeft />
                </button>
                <span className="text-xs text-pbi-text2">{page} / {previewData.total_pages}</span>
                <button onClick={() => setPage((p) => Math.min(previewData.total_pages, p + 1))} disabled={page === previewData.total_pages}
                  className="p-1.5 rounded text-pbi-muted hover:text-white disabled:opacity-30">
                  <FiChevronRight />
                </button>
              </div>
            </div>
          </>
        ) : (
          <p className="text-pbi-muted text-sm">No preview available</p>
        )}
      </ChartTile>
    </div>
  );
};

export default DataPage;
