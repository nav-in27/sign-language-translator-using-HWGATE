import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import LiveTranslator from './pages/LiveTranslator';
import Benchmark from './pages/Benchmark';
import './index.css';

function NavLinks({ theme, toggleTheme }) {
  const location = useLocation();
  return (
    <div className="nav-links">
      <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>Live Translator</Link>
      <Link to="/benchmark" className={`nav-link ${location.pathname === '/benchmark' ? 'active' : ''}`}>Model Benchmark</Link>
      <button 
        className="theme-toggle-btn"
        onClick={toggleTheme}
        title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
        aria-label="Toggle theme"
      >
        {theme === 'dark' ? '☀️ Light' : '🌙 Dark'}
      </button>
    </div>
  );
}

function App() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('isl_theme') || 'dark';
    } catch {
      return 'dark';
    }
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('isl_theme', theme);
    } catch {}
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  return (
    <BrowserRouter>
      <div className="app-wrapper">
        <main className="main-area">
          <nav className="top-nav">
            <div className="nav-pill">
              <span className="nav-brand">ISL TRANSLATOR</span>
              <NavLinks theme={theme} toggleTheme={toggleTheme} />
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

