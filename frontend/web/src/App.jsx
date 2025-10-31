// frontend/web/src/App.jsx
import { useEffect, useState } from 'react';
import BiasCard from './panels/BiasCard';
import PredictCard from './panels/PredictCard';
import InsightsCard from './panels/InsightsCard';
import { combinedHealth } from './api'; // make sure src/api.js exports combinedHealth()

const dot = (ok) =>
  <span className={`inline-block h-2 w-2 rounded-full ${ok ? 'bg-green-500' : 'bg-red-500'}`} />;

export default function App() {
  const [apiUp, setApiUp] = useState(null);
  const [computeUp, setComputeUp] = useState(null);
  const [timeframe, setTimeframe] = useState('1d');
  const [insightRange, setInsightRange] = useState('6m');

  useEffect(() => {
    let mounted = true;
    combinedHealth().then(({ api, compute }) => {
      if (!mounted) return;
      setApiUp(api.status === 'fulfilled' && api.value?.ok);
      setComputeUp(compute.status === 'fulfilled' && compute.value?.ok);
    }).catch(() => {
      if (!mounted) return;
      setApiUp(false);
      setComputeUp(false);
    });
    return () => { mounted = false; };
  }, []);

  const symbol = 'XAU';      // use XAU as your canonical symbol
  const researchSymbol = 'XAUUSD';

  return (
    <div className="p-4 space-y-4">
      {/* Health banner */}
      <div className="rounded-xl p-3 bg-black/10 flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          {dot(apiUp === null ? false : apiUp)} <span>API</span>
        </div>
        <div className="flex items-center gap-2">
          {dot(computeUp === null ? false : computeUp)} <span>Compute</span>
        </div>
        <div className="opacity-60 ml-auto">
          Primary: /api • Fallback: /compute
        </div>
      </div>

      <div className="flex flex-wrap gap-3 text-sm">
        <label className="flex items-center gap-2">
          <span className="opacity-70">Timeframe</span>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="bg-black/10 border border-white/10 rounded px-2 py-1"
          >
            <option value="1d">1D</option>
            <option value="4h">4H</option>
            <option value="1h">1H</option>
            <option value="1wk">1W</option>
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="opacity-70">Range</span>
          <select
            value={insightRange}
            onChange={(e) => setInsightRange(e.target.value)}
            className="bg-black/10 border border-white/10 rounded px-2 py-1"
          >
            <option value="3m">3M</option>
            <option value="6m">6M</option>
            <option value="1y">1Y</option>
            <option value="2y">2Y</option>
          </select>
        </label>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="rounded-xl p-4 bg-black/10">
          <BiasCard symbol={symbol} timeframe={timeframe} />
        </div>
        <div className="rounded-xl p-4 bg-black/10">
          {/* rename prop from style -> view to avoid React's reserved "style" */}
          <PredictCard symbol={symbol} timeframe={timeframe} view="day" />
        </div>
        <div className="rounded-xl p-4 bg-black/10 md:col-span-2">
          <InsightsCard symbol={researchSymbol} timeframe={timeframe} range={insightRange} />
        </div>
      </div>
    </div>
  );
}
