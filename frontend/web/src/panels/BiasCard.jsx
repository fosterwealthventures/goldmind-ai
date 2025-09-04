import { useEffect, useState } from 'react';
import { fetchBias as apiFetchBias } from '../apiClient';
import { fetchComputeBias } from '../computeClient';

function pickLabel(bias) {
  if (bias == null) return '';
  if (typeof bias === 'string') return bias;
  if (typeof bias === 'number') return String(bias);
  if (typeof bias === 'object') {
    return bias.label ?? bias.direction ?? bias.name ?? '';
  }
  return '';
}

function pickScore(bias) {
  if (bias && typeof bias === 'object') {
    return bias.score ?? bias.value ?? null;
  }
  return null;
}

export default function BiasCard({ symbol = 'XAU', timeframe = '1d' }) {
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
        const data = await apiFetchBias({ symbol, timeframe });
        if (!cancelled) setState({ loading: false, error: null, data, source: 'api' });
      } catch (e) {
        const status = e?.status ?? 0;
        // Only fall back on network/5xx
        if (status && status < 500 && status !== 0) {
          if (!cancelled) setState({ loading: false, error: `API ${status}`, data: null, source: null });
          return;
        }
        try {
          const data = await fetchComputeBias({ symbol, timeframe });
          if (!cancelled) setState({ loading: false, error: null, data, source: 'compute' });
        } catch (e2) {
          if (!cancelled) setState({ loading: false, error: `Compute ${e2?.status ?? 0}`, data: null, source: null });
        }
      }
    })();

    return () => { cancelled = true; };
  }, [symbol, timeframe]);

  if (state.loading) return <div>Loading bias…</div>;
  if (state.error)   return <div className="text-red-500 text-sm">Bias unavailable ({state.error}).</div>;

  // Accept { bias: ... } or the bias payload directly
  const payload = state.data;
  const biasObj = (payload && Object.prototype.hasOwnProperty.call(payload, 'bias')) ? payload.bias : payload;
  const label = pickLabel(biasObj);
  const score = pickScore(biasObj);

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="font-semibold">Bias Alert</div>
        <span className="text-xs opacity-60">
          {label ? `${label}${score != null ? ` (${score})` : ''}` : '—'} • source: {state.source}
        </span>
      </div>
      <pre className="text-xs bg-black/20 p-2 rounded">{JSON.stringify(payload, null, 2)}</pre>
    </div>
  );
}
