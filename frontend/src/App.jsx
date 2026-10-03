import React from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import LiveTranslator from './pages/LiveTranslator';
import Benchmark from './pages/Benchmark';
import './index.css';

function NavLinks() {
  const location = useLocation();
  return (
    <div className="nav-links">
      <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>Live Translator</Link>
      <Link to="/benchmark" className={`nav-link ${location.pathname === '/benchmark' ? 'active' : ''}`}>Live Benchmark</Link>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="app-wrapper">
        {/* Main Content Area */}
        <main className="main-area">
          <nav className="top-nav">
            <div className="nav-pill">
              <span className="nav-brand">ISL TRANSLATOR</span>
              <NavLinks />
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

