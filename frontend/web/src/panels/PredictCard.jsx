import { useEffect, useState } from 'react';
import { fetchPredictSMART as apiPredict } from '../apiClient';
import { fetchComputePredictSMART as computePredict } from '../computeClient';

export default function PredictCard({
  symbol = 'XAU',
  timeframe = '1d',
  view = 'day',
}) {
  const [state, setState] = useState({
    loading: true,
    error: null,
    data: null,
    source: null, // 'api' | 'compute'
  });

  useEffect(() => {
    let cancelled = false;
    setState({ loading: true, error: null, data: null, source: null });

    (async () => {
      try {
        const data = await apiPredict({ symbol, timeframe, view });
        if (!cancelled) setState({ loading: false, error: null, data, source: 'api' });
      } catch (e) {
        const status = e?.status ?? 0;
        // Fall back on 404/405 *and* 5xx/network
        const shouldFallback =
          !status || status >= 500 || status === 404 || status === 405;

        if (!shouldFallback) {
          if (!cancelled) setState({ loading: false, error: `API ${status}`, data: null, source: null });
          return;
        }

        try {
          const data = await computePredict({ symbol, timeframe, view });
          if (!cancelled) setState({ loading: false, error: null, data, source: 'compute' });
        } catch (e2) {
          if (!cancelled) setState({ loading: false, error: `Compute ${e2?.status ?? 0}`, data: null, source: null });
        }
      }
    })();

    return () => { cancelled = true; };
  }, [symbol, timeframe, view]);

  if (state.loading) return <div>Loading prediction…</div>;
  if (state.error)   return <div className="text-red-500 text-sm">Prediction unavailable ({state.error}).</div>;

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="font-semibold">Prediction</div>
        <span className="text-xs opacity-60">source: {state.source}</span>
      </div>
      <pre className="text-xs bg-black/20 p-2 rounded">{JSON.stringify(state.data, null, 2)}</pre>
    </div>
  );
}
