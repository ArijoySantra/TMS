import { useState } from "react";

import {
  Activity,
  AlertTriangle,
  BarChart3,
  CarFront,
  Clock,
  Gauge,
  LayoutDashboard,
  Map,
  Settings,
  ShieldCheck,
  TrafficCone,
  TrendingUp,
  Video,
  Zap,
} from "lucide-react";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import "./App.css";

// DEMONSTRATION DATA
// These values are not real-time traffic measurements.

const trafficData = [
  { time: "08:00", vehicles: 120 },
  { time: "09:00", vehicles: 185 },
  { time: "10:00", vehicles: 145 },
  { time: "11:00", vehicles: 220 },
  { time: "12:00", vehicles: 175 },
  { time: "13:00", vehicles: 260 },
  { time: "14:00", vehicles: 210 },
  { time: "15:00", vehicles: 290 },
  { time: "16:00", vehicles: 240 },
  { time: "17:00", vehicles: 330 },
];

const incidents = [
  {
    id: "INC-1024",
    location: "NH-16, Main Junction",
    type: "High congestion",
    time: "10:42 AM",
    severity: "High",
  },
  {
    id: "INC-1023",
    location: "College Square",
    type: "Accident risk elevated",
    time: "10:35 AM",
    severity: "Medium",
  },
  {
    id: "INC-1022",
    location: "Station Road",
    type: "Traffic signal delay",
    time: "10:21 AM",
    severity: "Low",
  },
];

const navigation = [
  { label: "Overview", icon: LayoutDashboard },
  { label: "Live Monitoring", icon: Video },
  { label: "Traffic Prediction", icon: TrendingUp },
  { label: "Accident Detection", icon: ShieldCheck },
  { label: "Signal Control", icon: TrafficCone },
  { label: "Analytics", icon: BarChart3 },
  { label: "Settings", icon: Settings },
];

function App() {
  const [activePage, setActivePage] = useState("Overview");

  const metrics = [
    {
      title: "Vehicles Detected",
      value: "1,284",
      subtitle: "Sample vehicle count",
      icon: CarFront,
      color: "blue",
    },
    {
      title: "Traffic Congestion",
      value: "Moderate",
      subtitle: "Sample traffic status",
      icon: Gauge,
      color: "orange",
    },
    {
      title: "Average Speed",
      value: "38 km/h",
      subtitle: "Sample average speed",
      icon: Activity,
      color: "green",
    },
    {
      title: "Accident Alerts",
      value: "03",
      subtitle: "Sample alerts",
      icon: AlertTriangle,
      color: "red",
    },
  ];

  return (
    <div className="app-shell">

      {/* SIDEBAR */}

      <aside className="sidebar">

        <div className="brand">
          <div className="brand-icon">
            <TrafficCone size={25} />
          </div>

          <div>
            <h2>TrafficAI</h2>
            <p>SMART CITY SYSTEM</p>
          </div>
        </div>

        <div className="sidebar-label">MAIN MENU</div>

        <nav className="navigation">

          {navigation.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.label}
                className={`nav-item ${
                  activePage === item.label ? "active" : ""
                }`}
                onClick={() => setActivePage(item.label)}
              >
                <Icon size={19} />
                <span>{item.label}</span>
              </button>
            );
          })}

        </nav>

        <div className="sidebar-bottom">

          <div className="system-status">
            <span className="status-dot" />

            <div>
              <strong>Demo Mode</strong>
              <p>Sample data active</p>
            </div>
          </div>

          <p className="sidebar-footer">
            Smart Traffic Management
          </p>

        </div>

      </aside>

      {/* MAIN CONTENT */}

      <main className="main-content">

        {/* TOP BAR */}

        <header className="topbar">

          <div>
            <p className="breadcrumb">
              TrafficAI / Dashboard
            </p>

            <h1>{activePage}</h1>
          </div>

          <div className="topbar-right">

            <span className="demo-badge">
              <span className="status-dot" />
              DEMO ENVIRONMENT
            </span>

            <div className="profile-avatar">AI</div>

          </div>

        </header>

        {/* WELCOME SECTION */}

        <section className="welcome-section">

          <div>
            <p className="eyebrow">
              INTELLIGENT TRAFFIC MANAGEMENT
            </p>

            <h2>Traffic Overview</h2>

            <p>
              Monitor traffic conditions, predict congestion,
              and review traffic safety indicators.
            </p>
          </div>

          <div className="date-card">
            <Clock size={17} />
            <span>Sample monitoring session</span>
          </div>

        </section>

        {/* DEMO NOTICE */}

        <div className="demo-notice">
          <Zap size={17} />

          <span>
            Demonstration mode: all displayed traffic values
            and incident records are sample data, not live measurements.
          </span>
        </div>

        {/* STATISTICS CARDS */}

        <section className="metrics-grid">

          {metrics.map((metric) => {
            const Icon = metric.icon;

            return (
              <article
                className="metric-card"
                key={metric.title}
              >

                <div className={`metric-icon ${metric.color}`}>
                  <Icon size={22} />
                </div>

                <p className="metric-title">
                  {metric.title}
                </p>

                <h3>{metric.value}</h3>

                <p className="metric-change">
                  {metric.subtitle}
                </p>

              </article>
            );
          })}

        </section>

        {/* CHART AND AI MODULES */}

        <section className="dashboard-grid">

          {/* TRAFFIC CHART */}

          <article className="panel traffic-panel">

            <div className="panel-heading">

              <div>
                <h3>Traffic Volume</h3>

                <p>
                  Illustrative vehicle counts by hour
                </p>
              </div>

              <span className="chart-tag">
                SAMPLE DATA
              </span>

            </div>

            <div className="chart-container">

              <ResponsiveContainer width="100%" height="100%">

                <AreaChart
                  data={trafficData}
                  margin={{
                    top: 15,
                    right: 10,
                    left: -15,
                    bottom: 0,
                  }}
                >

                  <defs>
                    <linearGradient
                      id="trafficGradient"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop
                        offset="0%"
                        stopColor="#38bdf8"
                        stopOpacity={0.3}
                      />

                      <stop
                        offset="95%"
                        stopColor="#38bdf8"
                        stopOpacity={0}
                      />
                    </linearGradient>
                  </defs>

                  <CartesianGrid
                    stroke="#243247"
                    strokeDasharray="4 4"
                    vertical={false}
                  />

                  <XAxis
                    dataKey="time"
                    stroke="#8091a8"
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 12 }}
                  />

                  <YAxis
                    stroke="#8091a8"
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 12 }}
                  />

                  <Tooltip
                    contentStyle={{
                      background: "#111d2d",
                      border: "1px solid #2c4058",
                      borderRadius: "10px",
                      color: "#ffffff",
                    }}
                  />

                  <Area
                    type="monotone"
                    dataKey="vehicles"
                    name="Vehicles"
                    stroke="#38bdf8"
                    strokeWidth={3}
                    fill="url(#trafficGradient)"
                  />

                </AreaChart>

              </ResponsiveContainer>

            </div>

          </article>

          {/* AI MODULE STATUS */}

          <article className="panel system-panel">

            <div className="panel-heading">

              <div>
                <h3>System Modules</h3>
                <p>AI integration overview</p>
              </div>

              <Activity
                size={20}
                className="muted-icon"
              />

            </div>

            <div className="module-list">

              <Module
                name="YOLO Vehicle Detection"
                detail="Vehicle detection"
              />

              <Module
                name="ByteTrack"
                detail="Vehicle tracking"
              />

              <Module
                name="LSTM"
                detail="Traffic prediction"
              />

              <Module
                name="XGBoost"
                detail="Accident-risk estimation"
              />

              <Module
                name="PPO + SUMO"
                detail="Adaptive signal simulation"
              />

            </div>

          </article>

        </section>

        {/* RECENT EVENTS */}

        <section className="panel incidents-panel">

          <div className="panel-heading">

            <div>
              <h3>Recent Traffic Events</h3>

              <p>
                Illustrative incident records for the dashboard preview
              </p>
            </div>

            <button
              className="text-button"
              onClick={() => setActivePage("Analytics")}
            >
              View analytics
            </button>

          </div>

          <div className="table-wrapper">

            <table>

              <thead>
                <tr>
                  <th>EVENT ID</th>
                  <th>LOCATION</th>
                  <th>EVENT TYPE</th>
                  <th>TIME</th>
                  <th>SEVERITY</th>
                </tr>
              </thead>

              <tbody>

                {incidents.map((incident) => (
                  <tr key={incident.id}>

                    <td className="event-id">
                      {incident.id}
                    </td>

                    <td>{incident.location}</td>

                    <td>{incident.type}</td>

                    <td>{incident.time}</td>

                    <td>
                      <span
                        className={`severity ${incident.severity.toLowerCase()}`}
                      >
                        {incident.severity}
                      </span>
                    </td>

                  </tr>
                ))}

              </tbody>

            </table>

          </div>

        </section>

        {/* FOOTER */}

        <footer className="page-footer">

          <span>
            TrafficAI · Smart Traffic Management System
          </span>

          <span>
            <Map size={14} />
            AI-powered traffic analytics interface
          </span>

        </footer>

      </main>

    </div>
  );
}

// REUSABLE AI MODULE COMPONENT

function Module({ name, detail }) {
  return (
    <div className="module-item">

      <div className="module-symbol">
        <Activity size={17} />
      </div>

      <div className="module-description">
        <strong>{name}</strong>
        <span>{detail}</span>
      </div>

      <span className="module-status">
        Not connected
      </span>

    </div>
  );
}

export default App;