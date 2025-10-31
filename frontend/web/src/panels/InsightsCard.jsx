import { useEffect, useState } from 'react';
import { fetchResearchHighlights } from '../apiClient';

export default function InsightsCard({ symbol = 'XAUUSD', timeframe = '1d', range = '6m' }) {
  const [state, setState] = useState({ loading: true, error: null, data: null });

  useEffect(() => {
    let cancelled = false;
    setState({ loading: true, error: null, data: null });

    fetchResearchHighlights({ symbol, timeframe, range })
      .then((data) => {
        if (!cancelled) setState({ loading: false, error: null, data });
      })
      .catch((err) => {
        if (!cancelled) setState({ loading: false, error: err?.message || 'Error loading insights', data: null });
      });

    return () => {
      cancelled = true;
    };
  }, [symbol, timeframe, range]);

  if (state.loading) return <div>Generating insights…</div>;
  if (state.error) return <div className="text-red-500 text-sm">Insights unavailable ({state.error}).</div>;

  const { narrative, highlights = [], sources = [], generated_at: generatedAt } = state.data ?? {};

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm opacity-70">
        <span className="font-semibold">Research Highlights</span>
        {generatedAt ? <span>• {new Date(generatedAt).toLocaleString()}</span> : null}
      </div>
      {narrative ? <p className="text-sm leading-relaxed">{narrative}</p> : null}
      {highlights.length > 0 ? (
        <ul className="list-disc list-inside text-sm space-y-1">
          {highlights.map((item, idx) => (
            <li key={idx}>{item}</li>
          ))}
        </ul>
      ) : null}
      {sources.length > 0 ? (
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="opacity-60">Sources:</span>
          {sources.map((src) => (
            <span key={src} className="px-2 py-0.5 rounded-full bg-black/20 border border-white/10">
              {src}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
