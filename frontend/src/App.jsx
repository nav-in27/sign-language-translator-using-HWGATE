import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import LiveTranslator from './pages/LiveTranslator';
import Benchmark from './pages/Benchmark';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <div className="app-wrapper">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-logo">SIGNAI</div>
          <div className="sidebar-icon active">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
          </div>
          <div className="sidebar-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>
          </div>
          <div className="sidebar-icon" style={{marginTop: 'auto', border: '1px solid var(--accent-neon)', borderRadius: '40px', width: '60px', height: '30px', transform: 'rotate(-90deg)', fontSize: '10px', color: 'var(--accent-neon)'}}>
            UPGRADE
          </div>
        </aside>
        
        {/* Main Content Area */}
        <main className="main-area">
          <nav className="top-nav">
            <div className="nav-pill">
              <span className="nav-brand">SIGNAI</span>
              <div className="nav-links">
                <Link to="/" className="nav-link">Live Translator</Link>
                <Link to="/benchmark" className="nav-link">Live Benchmark</Link>
              </div>
            </div>
          </nav>
          <Routes>
            <Route path="/" element={<LiveTranslator />} />
            <Route path="/benchmark" element={<Benchmark />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
