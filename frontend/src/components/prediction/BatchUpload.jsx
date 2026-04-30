import React, { useState } from 'react';
import { FiUpload, FiDownload, FiFile } from 'react-icons/fi';
import { api } from '../../api/apiClient';
import { downloadCSV } from '../../utils/helpers';
import toast from 'react-hot-toast';

const BatchUpload = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile?.name.endsWith('.csv')) setFile(droppedFile);
    else toast.error('Please upload a CSV file');
  };

  const handleUpload = async () => {
    if (!file) return toast.error('Select a CSV file first');
    setLoading(true);
    try {
      const res = await api.predictBatch(file);
      setResults(res.data);
      toast.success(`Predicted ${res.data.total_records} records`);
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!results?.predictions) return;
    const headers = 'Row,Prediction,Confidence\n';
    const rows = results.predictions.map(p => `${p.row},${p.prediction},${p.confidence}`).join('\n');
    downloadCSV(headers + rows, 'batch_predictions.csv');
  };

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer
                    ${dragOver ? 'border-pbi-blue bg-pbi-blue/5' : 'border-pbi-border hover:border-pbi-muted'}`}
        onClick={() => document.getElementById('batch-file-input').click()}
      >
        <input id="batch-file-input" type="file" accept=".csv" className="hidden" onChange={(e) => setFile(e.target.files[0])} />
        <FiUpload className="text-3xl text-pbi-muted mx-auto mb-3" />
        <p className="text-sm text-pbi-text2">Drag & drop CSV or click to browse</p>
        {file && (
          <div className="flex items-center justify-center gap-2 mt-3 text-pbi-blue">
            <FiFile /> <span className="text-sm">{file.name}</span>
          </div>
        )}
      </div>

      <button onClick={handleUpload} disabled={!file || loading} className="pbi-button w-full">
        {loading ? 'Processing...' : 'Upload & Predict'}
      </button>

      {/* Results */}
      {results && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-white">{results.total_records} predictions</h4>
            <button onClick={handleDownload} className="pbi-button-outline text-xs flex items-center gap-1 py-1.5 px-3">
              <FiDownload /> Download CSV
            </button>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {Object.entries(results.summary || {}).map(([label, count]) => (
              <div key={label} className="bg-pbi-bg2 rounded-lg p-3 text-center border border-pbi-border">
                <p className="text-lg font-bold text-white">{count}</p>
                <p className="text-xs text-pbi-muted">{label}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default BatchUpload;
