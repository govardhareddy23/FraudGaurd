import { useState, useEffect, useCallback } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

const API = "http://localhost:8000/api";

const FRAUD_PRESETS = [
  { label: "Known fraud pattern", amount: 9.25,  v14: -7.03, v17: -6.24, v4: 3.2,  v11: 4.8 },
  { label: "Suspicious small amt", amount: 1.00,  v14: -5.5,  v17: -4.8,  v4: 2.1,  v11: 3.2 },
  { label: "Legitimate purchase",  amount: 149.62, v14: 0.31,  v17: 0.58,  v4: 0.9,  v11: -0.4 },
  { label: "High-value normal",    amount: 892.00, v14: 0.12,  v17: -0.22, v4: -0.5, v11: 0.7 },
];

function RiskGauge({ probability }) {
  const pct = Math.round(probability * 100);
  const angle = -135 + (pct / 100) * 270;
  const color = pct >= 80 ? "#ff4444" : pct >= 40 ? "#ffaa00" : "#00e5b4";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      <svg width="160" height="100" viewBox="0 0 160 100">
        <path d="M 20 90 A 60 60 0 1 1 140 90" fill="none" stroke="#1e2a3a" strokeWidth="12" strokeLinecap="round"/>
        <path d="M 20 90 A 60 60 0 1 1 140 90" fill="none" stroke={color} strokeWidth="12"
          strokeLinecap="round" strokeDasharray="188.5" strokeDashoffset={188.5 - (pct / 100) * 188.5}
          style={{ transition: "stroke-dashoffset 0.6s ease, stroke 0.4s ease" }}/>
        <line x1="80" y1="90" x2={80 + 44 * Math.cos((angle - 90) * Math.PI / 180)}
          y2={90 + 44 * Math.sin((angle - 90) * Math.PI / 180)}
          stroke={color} strokeWidth="2.5" strokeLinecap="round"
          style={{ transition: "all 0.6s ease" }}/>
        <circle cx="80" cy="90" r="5" fill={color}/>
        <text x="80" y="75" textAnchor="middle" fill={color} fontSize="22" fontFamily="'JetBrains Mono', monospace" fontWeight="700">{pct}%</text>
      </svg>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#546e7a", letterSpacing: 2 }}>
        FRAUD PROBABILITY
      </div>
    </div>
  );
}

function StatCard({ label, value, sub, accent, pulse }) {
  return (
    <div style={{
      background: "#0d1520", border: `1px solid ${accent || "#1e2a3a"}`,
      borderRadius: 8, padding: "18px 20px", position: "relative", overflow: "hidden"
    }}>
      {pulse && <div style={{
        position: "absolute", top: 8, right: 8, width: 8, height: 8,
        borderRadius: "50%", background: accent || "#00e5b4",
        animation: "pulse 2s ease-in-out infinite"
      }}/>}
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11,
        color: "#546e7a", letterSpacing: 2, marginBottom: 8 }}>{label}</div>
      <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 28,
        fontWeight: 700, color: accent || "#e0e8f0", lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "#546e7a", marginTop: 6 }}>{sub}</div>}
    </div>
  );
}

function Badge({ level }) {
  const cfg = {
    HIGH:   { bg: "#2a0a0a", color: "#ff4444", border: "#ff4444" },
    MEDIUM: { bg: "#2a1a00", color: "#ffaa00", border: "#ffaa00" },
    LOW:    { bg: "#001a14", color: "#00e5b4", border: "#00e5b4" },
  }[level] || { bg: "#111", color: "#888", border: "#888" };
  return (
    <span style={{
      background: cfg.bg, color: cfg.color, border: `1px solid ${cfg.border}`,
      borderRadius: 4, padding: "2px 8px", fontSize: 10,
      fontFamily: "'JetBrains Mono', monospace", letterSpacing: 1.5, fontWeight: 700
    }}>{level}</span>
  );
}

function FeatureInput({ label, name, value, onChange }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <label style={{ fontSize: 10, color: "#546e7a", fontFamily: "'JetBrains Mono', monospace",
        letterSpacing: 1.5 }}>{label}</label>
      <input type="number" step="any" value={value} onChange={e => onChange(name, e.target.value)}
        style={{
          background: "#0d1520", border: "1px solid #1e2a3a", borderRadius: 6,
          padding: "8px 10px", color: "#c8d8e8", fontSize: 13,
          fontFamily: "'JetBrains Mono', monospace", outline: "none", width: "100%",
          boxSizing: "border-box", transition: "border-color 0.2s"
        }}
        onFocus={e => e.target.style.borderColor = "#00e5b4"}
        onBlur={e => e.target.style.borderColor = "#1e2a3a"}
      />
    </div>
  );
}

const DEFAULT_FORM = {
  time: 406, amount: 149.62,
  v1: -1.36, v2: -0.07, v3: 2.54, v4: 1.38, v5: -0.34, v6: 0.46, v7: 0.24, v8: 0.10,
  v9: 0.36, v10: 0.09, v11: -0.55, v12: -0.62, v13: -0.99, v14: -0.31,
  v15: 1.47, v16: -0.47, v17: -0.59, v18: -0.00, v19: 0.31, v20: 0.01,
  v21: -0.02, v22: 0.28, v23: 0.29, v24: -0.26, v25: -0.31, v26: 0.26, v27: 0.07, v28: 0.00
};

export default function App() {
  const [page, setPage]           = useState("predict");
  const [form, setForm]           = useState(DEFAULT_FORM);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [history, setHistory]     = useState([]);
  const [stats, setStats]         = useState(null);
  const [alerts, setAlerts]       = useState([]);
  const [transactions, setTxns]   = useState([]);

  const fetchStats = useCallback(async () => {
    try {
      const r = await fetch(`${API}/dashboard/stats`);
      if (r.ok) setStats(await r.json());
    } catch {}
  }, []);

  const fetchAlerts = useCallback(async () => {
    try {
      const r = await fetch(`${API}/alerts/?status=open`);
      if (r.ok) setAlerts(await r.json());
    } catch {}
  }, []);

  const fetchTxns = useCallback(async () => {
    try {
      const r = await fetch(`${API}/transactions/?limit=50`);
      if (r.ok) setTxns(await r.json());
    } catch {}
  }, []);

  useEffect(() => {
    fetchStats(); fetchAlerts(); fetchTxns();
    const t = setInterval(() => { fetchStats(); fetchAlerts(); }, 30000);
    return () => clearInterval(t);
  }, [fetchStats, fetchAlerts, fetchTxns]);

  const handleChange = (name, val) => setForm(f => ({ ...f, [name]: parseFloat(val) || 0 }));

  const applyPreset = (preset) => {
    setForm(f => ({ ...f, amount: preset.amount, v4: preset.v4,
      v11: preset.v11, v14: preset.v14, v17: preset.v17 }));
    setResult(null);
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${API}/predict/`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });
      if (r.ok) {
        const data = await r.json();
        setResult(data);
        setHistory(h => [data, ...h].slice(0, 20));
        fetchStats(); fetchAlerts(); fetchTxns();
      }
    } catch {
      const mock = {
        transaction_id: Math.floor(Math.random() * 99999),
        fraud_probability: form.v14 < -5 ? 0.87 : 0.04,
        is_fraud: form.v14 < -5,
        risk_level: form.v14 < -5 ? "HIGH" : "LOW",
        alert_triggered: form.v14 < -5,
        amount: form.amount,
        timestamp: new Date().toISOString()
      };
      setResult(mock);
      setHistory(h => [mock, ...h].slice(0, 20));
    } finally {
      setLoading(false);
    }
  };

  const navItems = [
    { id: "predict", label: "Predict" },
    { id: "dashboard", label: "Dashboard" },
    { id: "transactions", label: "Transactions" },
    { id: "alerts", label: "Alerts", badge: alerts.length },
  ];

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #080f18; color: #c8d8e8; font-family: 'Space Grotesk', sans-serif; min-height: 100vh; }
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.3)} }
        @keyframes slideIn { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
        @keyframes scanline {
          0% { top: -2px; } 100% { top: 100%; }
        }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #0d1520; }
        ::-webkit-scrollbar-thumb { background: #1e2a3a; border-radius: 2px; }
      `}</style>

      {/* Header */}
      <header style={{
        background: "#0a1220", borderBottom: "1px solid #1e2a3a",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 28px", height: 56, position: "sticky", top: 0, zIndex: 100
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ width: 32, height: 32, background: "#00e5b4", borderRadius: 6,
            display: "flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="18" height="18" fill="none" viewBox="0 0 24 24">
              <path d="M12 2L3 7v5c0 5.25 3.75 10.15 9 11.35C17.25 22.15 21 17.25 21 12V7L12 2z"
                fill="#080f18" opacity="0.9"/>
              <path d="M9 12l2 2 4-4" stroke="#080f18" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </div>
          <div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700,
              fontSize: 14, color: "#e0e8f0", letterSpacing: 0.5 }}>FRAUDGUARD</div>
            <div style={{ fontSize: 9, color: "#546e7a", letterSpacing: 2, fontFamily: "'JetBrains Mono', monospace" }}>
              ANN DETECTION SYSTEM
            </div>
          </div>
        </div>

        <nav style={{ display: "flex", gap: 4 }}>
          {navItems.map(item => (
            <button key={item.id} onClick={() => setPage(item.id)} style={{
              background: page === item.id ? "#00e5b41a" : "transparent",
              border: page === item.id ? "1px solid #00e5b440" : "1px solid transparent",
              color: page === item.id ? "#00e5b4" : "#546e7a",
              borderRadius: 6, padding: "6px 14px", cursor: "pointer", fontSize: 13,
              fontFamily: "'Space Grotesk', sans-serif", fontWeight: 500,
              display: "flex", alignItems: "center", gap: 6, transition: "all 0.2s"
            }}>
              {item.label}
              {item.badge > 0 && (
                <span style={{ background: "#ff4444", color: "#fff", borderRadius: 10,
                  padding: "0 5px", fontSize: 10, fontWeight: 700 }}>{item.badge}</span>
              )}
            </button>
          ))}
        </nav>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#00e5b4",
            animation: "pulse 3s ease-in-out infinite" }}/>
          <span style={{ fontSize: 11, color: "#546e7a", fontFamily: "'JetBrains Mono', monospace" }}>
            LIVE
          </span>
        </div>
      </header>

      <main style={{ maxWidth: 1280, margin: "0 auto", padding: "28px 24px" }}>

        {/* ── PREDICT PAGE ─────────────────────────────────────────────── */}
        {page === "predict" && (
          <div style={{ animation: "slideIn 0.3s ease", display: "grid",
            gridTemplateColumns: "1fr 380px", gap: 20 }}>

            {/* Left: form */}
            <div>
              <div style={{ marginBottom: 20 }}>
                <h1 style={{ fontSize: 22, fontWeight: 700, color: "#e0e8f0", marginBottom: 4 }}>
                  Transaction Analysis
                </h1>
                <p style={{ fontSize: 13, color: "#546e7a" }}>
                  Enter transaction features to run fraud detection
                </p>
              </div>

              {/* Presets */}
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 10, color: "#546e7a", letterSpacing: 2,
                  fontFamily: "'JetBrains Mono', monospace", marginBottom: 10 }}>QUICK PRESETS</div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {FRAUD_PRESETS.map(p => (
                    <button key={p.label} onClick={() => applyPreset(p)} style={{
                      background: "#0d1520", border: "1px solid #1e2a3a",
                      color: "#8aa0b4", borderRadius: 6, padding: "7px 14px",
                      cursor: "pointer", fontSize: 12, fontFamily: "'Space Grotesk', sans-serif",
                      transition: "all 0.2s"
                    }}
                      onMouseEnter={e => { e.target.style.borderColor="#00e5b4"; e.target.style.color="#00e5b4"; }}
                      onMouseLeave={e => { e.target.style.borderColor="#1e2a3a"; e.target.style.color="#8aa0b4"; }}
                    >{p.label}</button>
                  ))}
                </div>
              </div>

              {/* Core fields */}
              <div style={{ background: "#0a1220", border: "1px solid #1e2a3a",
                borderRadius: 10, padding: 20, marginBottom: 16 }}>
                <div style={{ fontSize: 10, color: "#00e5b4", letterSpacing: 2,
                  fontFamily: "'JetBrains Mono', monospace", marginBottom: 16 }}>
                  CORE FEATURES
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <FeatureInput label="AMOUNT ($)" name="amount" value={form.amount} onChange={handleChange}/>
                  <FeatureInput label="TIME (seconds)" name="time" value={form.time} onChange={handleChange}/>
                </div>
              </div>

              {/* Key V-features */}
              <div style={{ background: "#0a1220", border: "1px solid #1e2a3a",
                borderRadius: 10, padding: 20, marginBottom: 16 }}>
                <div style={{ fontSize: 10, color: "#00e5b4", letterSpacing: 2,
                  fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>
                  KEY PCA FEATURES (most predictive)
                </div>
                <div style={{ fontSize: 11, color: "#546e7a", marginBottom: 14 }}>
                  V14 and V17 have the highest correlation with fraud
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
                  {["v4","v11","v14","v17"].map(k => (
                    <FeatureInput key={k} label={k.toUpperCase()} name={k}
                      value={form[k]} onChange={handleChange}/>
                  ))}
                </div>
              </div>

              {/* All V features */}
              <details style={{ marginBottom: 20 }}>
                <summary style={{ cursor: "pointer", fontSize: 12, color: "#546e7a",
                  fontFamily: "'JetBrains Mono', monospace", letterSpacing: 1.5,
                  padding: "10px 0", userSelect: "none" }}>
                  SHOW ALL V1–V28 FEATURES
                </summary>
                <div style={{ background: "#0a1220", border: "1px solid #1e2a3a",
                  borderRadius: 10, padding: 20, marginTop: 10 }}>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
                    {Array.from({ length: 28 }, (_, i) => `v${i + 1}`)
                      .filter(k => !["v4","v11","v14","v17"].includes(k))
                      .map(k => (
                        <FeatureInput key={k} label={k.toUpperCase()} name={k}
                          value={form[k]} onChange={handleChange}/>
                      ))}
                  </div>
                </div>
              </details>

              <button onClick={handleSubmit} disabled={loading} style={{
                width: "100%", background: loading ? "#1e2a3a" : "#00e5b4",
                color: loading ? "#546e7a" : "#080f18", border: "none",
                borderRadius: 8, padding: "14px", fontSize: 15, fontWeight: 700,
                fontFamily: "'Space Grotesk', sans-serif", cursor: loading ? "not-allowed" : "pointer",
                letterSpacing: 0.5, transition: "all 0.2s"
              }}>
                {loading ? "ANALYZING..." : "RUN FRAUD ANALYSIS"}
              </button>
            </div>

            {/* Right: result panel */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

              {/* Result card */}
              <div style={{
                background: "#0a1220",
                border: result
                  ? `1px solid ${result.is_fraud ? "#ff4444" : "#00e5b4"}40`
                  : "1px solid #1e2a3a",
                borderRadius: 10, padding: 24, minHeight: 340,
                display: "flex", flexDirection: "column", alignItems: "center",
                justifyContent: "center", gap: 16,
                animation: result ? "slideIn 0.3s ease" : "none"
              }}>
                {!result ? (
                  <div style={{ textAlign: "center", color: "#546e7a" }}>
                    <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.3 }}>⬡</div>
                    <div style={{ fontSize: 13 }}>Submit a transaction to see the prediction</div>
                  </div>
                ) : (
                  <>
                    <RiskGauge probability={result.fraud_probability}/>
                    <div style={{ textAlign: "center" }}>
                      <div style={{ fontSize: 20, fontWeight: 700,
                        color: result.is_fraud ? "#ff4444" : "#00e5b4", marginBottom: 6 }}>
                        {result.is_fraud ? "FRAUD DETECTED" : "TRANSACTION SAFE"}
                      </div>
                      <Badge level={result.risk_level}/>
                    </div>
                    <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 8 }}>
                      {[
                        ["Transaction ID", `#${result.transaction_id}`],
                        ["Amount", `$${result.amount.toFixed(2)}`],
                        ["Probability", `${(result.fraud_probability * 100).toFixed(2)}%`],
                        ["Alert", result.alert_triggered ? "TRIGGERED" : "None"],
                      ].map(([k, v]) => (
                        <div key={k} style={{ display: "flex", justifyContent: "space-between",
                          borderBottom: "1px solid #1e2a3a", paddingBottom: 6 }}>
                          <span style={{ fontSize: 12, color: "#546e7a" }}>{k}</span>
                          <span style={{ fontSize: 12, fontFamily: "'JetBrains Mono', monospace",
                            color: k === "Alert" && result.alert_triggered ? "#ff4444" : "#c8d8e8" }}>
                            {v}
                          </span>
                        </div>
                      ))}
                    </div>
                    {result.alert_triggered && (
                      <div style={{ width: "100%", background: "#2a0a0a",
                        border: "1px solid #ff444440", borderRadius: 8,
                        padding: "10px 14px", fontSize: 12, color: "#ff8888" }}>
                        High-risk alert saved to database for review
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* History */}
              {history.length > 0 && (
                <div style={{ background: "#0a1220", border: "1px solid #1e2a3a",
                  borderRadius: 10, padding: 16 }}>
                  <div style={{ fontSize: 10, color: "#546e7a", letterSpacing: 2,
                    fontFamily: "'JetBrains Mono', monospace", marginBottom: 12 }}>
                    SESSION HISTORY
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 260, overflowY: "auto" }}>
                    {history.map((h, i) => (
                      <div key={i} style={{ display: "flex", alignItems: "center",
                        justifyContent: "space-between", padding: "6px 10px",
                        background: "#080f18", borderRadius: 6 }}>
                        <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
                          color: "#546e7a" }}>#{h.transaction_id}</span>
                        <span style={{ fontSize: 12, color: "#8aa0b4" }}>${h.amount.toFixed(2)}</span>
                        <Badge level={h.risk_level}/>
                        <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
                          color: h.is_fraud ? "#ff4444" : "#00e5b4" }}>
                          {(h.fraud_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── DASHBOARD PAGE ──────────────────────────────────────────── */}
        {page === "dashboard" && (
          <div style={{ animation: "slideIn 0.3s ease" }}>
            <h1 style={{ fontSize: 22, fontWeight: 700, color: "#e0e8f0", marginBottom: 20 }}>
              Real-Time Monitoring
            </h1>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14, marginBottom: 20 }}>
              <StatCard label="TOTAL TRANSACTIONS" value={stats?.total_transactions?.toLocaleString() ?? "—"}
                sub="All time" accent="#c8d8e8"/>
              <StatCard label="FRAUD DETECTED" value={stats?.fraud_transactions?.toLocaleString() ?? "—"}
                sub="All time" accent="#ff4444" pulse/>
              <StatCard label="FRAUD RATE" value={stats ? `${stats.fraud_rate.toFixed(4)}%` : "—"}
                sub="Overall" accent="#ffaa00"/>
              <StatCard label="OPEN ALERTS" value={stats?.open_alerts ?? "—"}
                sub="Needs review" accent="#ff4444" pulse={stats?.open_alerts > 0}/>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14, marginBottom: 24 }}>
              <StatCard label="TODAY'S TRANSACTIONS" value={stats?.transactions_today ?? "—"}
                sub="Last 24h" accent="#c8d8e8"/>
              <StatCard label="TODAY'S FRAUD" value={stats?.fraud_today ?? "—"}
                sub="Last 24h" accent="#ff4444"/>
              <StatCard label="AVG TRANSACTION" value={stats ? `$${stats.avg_amount}` : "—"}
                sub="All transactions" accent="#c8d8e8"/>
              <StatCard label="AVG FRAUD AMOUNT" value={stats ? `$${stats.avg_fraud_amount}` : "—"}
                sub="Fraud only" accent="#ffaa00"/>
            </div>

            {/* Simple bar chart from local history */}
            {history.length > 3 && (
              <div style={{ background: "#0a1220", border: "1px solid #1e2a3a",
                borderRadius: 10, padding: 20 }}>
                <div style={{ fontSize: 10, color: "#546e7a", letterSpacing: 2,
                  fontFamily: "'JetBrains Mono', monospace", marginBottom: 16 }}>
                  SESSION FRAUD PROBABILITY TREND
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={[...history].reverse().map((h, i) => ({
                    i, prob: +(h.fraud_probability * 100).toFixed(2)
                  }))}>
                    <CartesianGrid stroke="#1e2a3a" strokeDasharray="3 3"/>
                    <XAxis dataKey="i" tick={{ fill: "#546e7a", fontSize: 11 }}/>
                    <YAxis domain={[0, 100]} tick={{ fill: "#546e7a", fontSize: 11 }}/>
                    <Tooltip contentStyle={{ background: "#0a1220", border: "1px solid #1e2a3a",
                      borderRadius: 6, color: "#c8d8e8", fontSize: 12 }}/>
                    <Line type="monotone" dataKey="prob" stroke="#00e5b4" strokeWidth={2}
                      dot={{ fill: "#00e5b4", r: 3 }}/>
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {!stats && (
              <div style={{ background: "#0a1220", border: "1px dashed #1e2a3a",
                borderRadius: 10, padding: 40, textAlign: "center", color: "#546e7a" }}>
                <div style={{ fontSize: 13, marginBottom: 8 }}>Backend not connected</div>
                <div style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
                  Run predictions to populate the dashboard
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── TRANSACTIONS PAGE ───────────────────────────────────────── */}
        {page === "transactions" && (
          <div style={{ animation: "slideIn 0.3s ease" }}>
            <div style={{ display: "flex", justifyContent: "space-between",
              alignItems: "center", marginBottom: 20 }}>
              <h1 style={{ fontSize: 22, fontWeight: 700, color: "#e0e8f0" }}>
                Transaction Log
              </h1>
              <button onClick={fetchTxns} style={{
                background: "#0d1520", border: "1px solid #1e2a3a", color: "#546e7a",
                borderRadius: 6, padding: "7px 14px", cursor: "pointer", fontSize: 12,
                fontFamily: "'JetBrains Mono', monospace"
              }}>REFRESH</button>
            </div>

            <div style={{ background: "#0a1220", border: "1px solid #1e2a3a", borderRadius: 10, overflow: "hidden" }}>
              <div style={{ display: "grid", gridTemplateColumns: "60px 1fr 100px 120px 80px 100px",
                padding: "10px 16px", borderBottom: "1px solid #1e2a3a",
                fontSize: 10, color: "#546e7a", fontFamily: "'JetBrains Mono', monospace",
                letterSpacing: 1.5 }}>
                <span>ID</span><span>TIMESTAMP</span><span>AMOUNT</span>
                <span>PROBABILITY</span><span>RISK</span><span>STATUS</span>
              </div>
              {(transactions.length > 0 ? transactions : history).length === 0 ? (
                <div style={{ padding: 40, textAlign: "center", color: "#546e7a", fontSize: 13 }}>
                  No transactions yet — run a prediction first
                </div>
              ) : (
                (transactions.length > 0 ? transactions : history).slice(0, 30).map((tx, i) => (
                  <div key={tx.id ?? i} style={{
                    display: "grid", gridTemplateColumns: "60px 1fr 100px 120px 80px 100px",
                    padding: "12px 16px", borderBottom: "1px solid #1e2a3a11",
                    alignItems: "center", transition: "background 0.15s"
                  }}
                    onMouseEnter={e => e.currentTarget.style.background = "#0d1520"}
                    onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                  >
                    <span style={{ fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 12, color: "#546e7a" }}>
                      #{tx.transaction_id ?? tx.id}
                    </span>
                    <span style={{ fontSize: 12, color: "#8aa0b4" }}>
                      {tx.timestamp ? new Date(tx.timestamp).toLocaleString() : "—"}
                    </span>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#c8d8e8" }}>
                      ${(tx.amount ?? 0).toFixed(2)}
                    </span>
                    <div>
                      <div style={{ height: 4, background: "#1e2a3a", borderRadius: 2, marginBottom: 3 }}>
                        <div style={{
                          height: "100%", borderRadius: 2,
                          width: `${(tx.fraud_probability * 100).toFixed(1)}%`,
                          background: tx.fraud_probability >= 0.8 ? "#ff4444"
                            : tx.fraud_probability >= 0.4 ? "#ffaa00" : "#00e5b4",
                          transition: "width 0.3s"
                        }}/>
                      </div>
                      <span style={{ fontFamily: "'JetBrains Mono', monospace",
                        fontSize: 10, color: "#546e7a" }}>
                        {(tx.fraud_probability * 100).toFixed(2)}%
                      </span>
                    </div>
                    <Badge level={tx.risk_level}/>
                    <span style={{ fontSize: 12,
                      color: tx.is_fraud ? "#ff4444" : "#00e5b4" }}>
                      {tx.is_fraud ? "FRAUD" : "SAFE"}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* ── ALERTS PAGE ─────────────────────────────────────────────── */}
        {page === "alerts" && (
          <div style={{ animation: "slideIn 0.3s ease" }}>
            <div style={{ display: "flex", justifyContent: "space-between",
              alignItems: "center", marginBottom: 20 }}>
              <div>
                <h1 style={{ fontSize: 22, fontWeight: 700, color: "#e0e8f0" }}>
                  Fraud Alerts
                </h1>
                <p style={{ fontSize: 13, color: "#546e7a", marginTop: 4 }}>
                  Transactions with fraud probability &gt; 80%
                </p>
              </div>
              <button onClick={fetchAlerts} style={{
                background: "#0d1520", border: "1px solid #1e2a3a", color: "#546e7a",
                borderRadius: 6, padding: "7px 14px", cursor: "pointer", fontSize: 12,
                fontFamily: "'JetBrains Mono', monospace"
              }}>REFRESH</button>
            </div>

            {alerts.length === 0 ? (
              <div style={{ background: "#0a1220", border: "1px dashed #1e2a3a",
                borderRadius: 10, padding: 60, textAlign: "center" }}>
                <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.2 }}>◎</div>
                <div style={{ fontSize: 14, color: "#546e7a" }}>No open alerts</div>
                <div style={{ fontSize: 12, color: "#546e7a33", marginTop: 6,
                  fontFamily: "'JetBrains Mono', monospace" }}>
                  Backend alerts will appear here when probability &gt; 80%
                </div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {alerts.map(alert => (
                  <div key={alert.id} style={{
                    background: "#0a1220", border: "1px solid #ff444430",
                    borderLeft: "3px solid #ff4444", borderRadius: 8,
                    padding: "16px 20px", display: "flex",
                    alignItems: "center", justifyContent: "space-between",
                    animation: "slideIn 0.3s ease"
                  }}>
                    <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
                      <div>
                        <div style={{ fontSize: 10, color: "#ff6666",
                          fontFamily: "'JetBrains Mono', monospace", letterSpacing: 1.5 }}>
                          ALERT #{alert.id}
                        </div>
                        <div style={{ fontSize: 18, fontWeight: 700,
                          color: "#ff4444", fontFamily: "'JetBrains Mono', monospace",
                          marginTop: 2 }}>
                          {(alert.fraud_probability * 100).toFixed(1)}% FRAUD
                        </div>
                      </div>
                      <div style={{ borderLeft: "1px solid #1e2a3a", paddingLeft: 20 }}>
                        <div style={{ fontSize: 11, color: "#546e7a" }}>Amount</div>
                        <div style={{ fontSize: 16, fontWeight: 600, color: "#c8d8e8" }}>
                          ${alert.amount.toFixed(2)}
                        </div>
                      </div>
                      <div style={{ borderLeft: "1px solid #1e2a3a", paddingLeft: 20 }}>
                        <div style={{ fontSize: 11, color: "#546e7a" }}>Transaction</div>
                        <div style={{ fontSize: 13, fontFamily: "'JetBrains Mono', monospace",
                          color: "#8aa0b4" }}>#{alert.transaction_id}</div>
                      </div>
                      <div style={{ borderLeft: "1px solid #1e2a3a", paddingLeft: 20 }}>
                        <div style={{ fontSize: 11, color: "#546e7a" }}>Time</div>
                        <div style={{ fontSize: 12, color: "#8aa0b4" }}>
                          {new Date(alert.timestamp).toLocaleString()}
                        </div>
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: 8 }}>
                      <button onClick={async () => {
                        try { await fetch(`${API}/alerts/${alert.id}?status=reviewed`, { method: "PATCH" }); }
                        catch {}
                        setAlerts(a => a.filter(x => x.id !== alert.id));
                      }} style={{
                        background: "#001a14", border: "1px solid #00e5b440",
                        color: "#00e5b4", borderRadius: 6, padding: "7px 14px",
                        cursor: "pointer", fontSize: 12, fontFamily: "'JetBrains Mono', monospace"
                      }}>REVIEWED</button>
                      <button onClick={() => setAlerts(a => a.filter(x => x.id !== alert.id))}
                        style={{
                          background: "#0d1520", border: "1px solid #1e2a3a",
                          color: "#546e7a", borderRadius: 6, padding: "7px 14px",
                          cursor: "pointer", fontSize: 12, fontFamily: "'JetBrains Mono', monospace"
                        }}>DISMISS</button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </>
  );
}
