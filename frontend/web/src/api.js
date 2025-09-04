// src/api.js
export async function combinedHealth() {
  const [api, compute] = await Promise.allSettled([
    fetchJSON(`${API_BASE}/health`),    // <-- backticks
    fetchJSON(`${COMPUTE_BASE}/health`) // <-- backticks
  ]);
  return { api, compute };
}

export async function predict(payload) {
  const init = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify(payload),
  };

  // API first
  const r = await fetchJSON(`${API_BASE}/predict`, init);      // <-- backticks
  if (r.ok) return r.body;
  if (r.status && r.status < 500 && r.status !== 0) {
    throw new Error(`API ${r.status}: ${JSON.stringify(r.body)}`);
  }

  // Fallback to Compute
  const f = await fetchJSON(`${COMPUTE_BASE}/predict`, init);  // <-- backticks
  if (f.ok) return f.body;

  throw new Error(
    `Both API and Compute failed. API ${r.status}: ${JSON.stringify(r.body)} | ` +
    `Compute ${f.status}: ${JSON.stringify(f.body)}`
  );
}
