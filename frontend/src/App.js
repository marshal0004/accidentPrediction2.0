import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Layout from './components/layout/Layout';
import OverviewPage    from './pages/OverviewPage';
import ModelsPage      from './pages/ModelsPage';
import FeaturesPage    from './pages/FeaturesPage';
import PredictPage     from './pages/PredictPage';
import DataPage        from './pages/DataPage';
import DigitalTwinPage from './pages/DigitalTwinPage';
import SimulationPage  from './pages/SimulationPage';

function App() {
  return (
    <Router>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#1F2940',
            color: '#FFFFFF',
            border: '1px solid #2D3748',
            borderRadius: '12px',
            fontSize: '14px',
          },
          success: { iconTheme: { primary: '#10B981', secondary: '#1F2940' } },
          error:   { iconTheme: { primary: '#EF4444', secondary: '#1F2940' } },
        }}
      />
      <Layout>
        <Routes>
          <Route path="/"              element={<OverviewPage />}    />
          <Route path="/models"        element={<ModelsPage />}      />
          <Route path="/features"      element={<FeaturesPage />}    />
          <Route path="/predict"       element={<PredictPage />}     />
          <Route path="/data"          element={<DataPage />}        />
          <Route path="/digital-twin"  element={<DigitalTwinPage />} />
          <Route path="/simulation"    element={<SimulationPage />}  />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
