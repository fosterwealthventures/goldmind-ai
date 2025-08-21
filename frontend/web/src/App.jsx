import { useEffect, useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { getHealth, getMarketStatus, getRecommendation, getBiasAlert, getPortfolio } from './api'

export default function App() {
  const [count, setCount] = useState(0)
  const [health, setHealth] = useState(null)
  const [data, setData] = useState({})
  const [err, setErr] = useState(null)

  useEffect(() => {
    getHealth().then(setHealth).catch(e => setErr(String(e)))
  }, [])

  async function loadDashboard() {
    const results = {}
    const tasks = [
      ['market',        getMarketStatus],
      ['recommendation',() => getRecommendation('XAUUSD')],
      ['bias',          getBiasAlert],
      ['portfolio',     getPortfolio],
    ]
    for (const [key, fn] of tasks) {
      try { results[key] = await fn() } 
      catch (e) { results[key] = { ok:false, status:'ERR', body:String(e) } }
    }
    setData(results)
  }

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank"><img src={viteLogo} className="logo" alt="Vite logo" /></a>
        <a href="https://react.dev" target="_blank"><img src={reactLogo} className="logo react" alt="React logo" /></a>
      </div>

      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount(c => c + 1)}>count is {count}</button>
        <p>Edit <code>src/App.jsx</code> and save to test HMR</p>
      </div>

      <div className="card" style={{marginTop:16}}>
        <h2>API Connectivity</h2>
        <p><strong>API Base:</strong> {import.meta.env.VITE_API_BASE || '/api'}</p>
        <pre><strong>/health →</strong> {health ? JSON.stringify(health, null, 2) : '…'}</pre>
        {err && <pre style={{color:'red'}}>{err}</pre>}
        <button onClick={loadDashboard} style={{marginTop:8}}>Load dashboard data</button>
        {Object.entries(data).map(([k,v]) => (
          <pre key={k} style={{whiteSpace:'pre-wrap'}}><strong>{k}:</strong> {JSON.stringify(v,null,2)}</pre>
        ))}
      </div>
    </>
  )
}
