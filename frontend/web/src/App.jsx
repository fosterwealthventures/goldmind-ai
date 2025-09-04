// frontend/web/src/App.jsx
import { useEffect, useState } from 'react';
import BiasCard from './panels/BiasCard';
import PredictCard from './panels/PredictCard';
import { combinedHealth } from './api'; // make sure src/api.js exports combinedHealth()

const dot = (ok) =>
  <span className={`inline-block h-2 w-2 rounded-full ${ok ? 'bg-green-500' : 'bg-red-500'}`} />;

export default function App() {
  const [apiUp, setApiUp] = useState(null);
  const [computeUp, setComputeUp] = useState(null);

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
  const timeframe = '1d';

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

      <div className="grid md:grid-cols-2 gap-4">
        <div className="rounded-xl p-4 bg-black/10">
          <BiasCard symbol={symbol} timeframe={timeframe} />
        </div>
        <div className="rounded-xl p-4 bg-black/10">
          {/* rename prop from style -> view to avoid React's reserved "style" */}
          <PredictCard symbol={symbol} timeframe={timeframe} view="day" />
        </div>
      </div>
    </div>
  );
}
