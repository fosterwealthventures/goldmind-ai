import { fetchHealth as fetchApiHealth, post as apiPost } from './apiClient';
import { fetchComputeHealth, post as computePost } from './computeClient';

export { API_BASE } from './apiClient';
export { COMPUTE_BASE } from './computeClient';

const stringifyBody = (value) => {
  if (value === undefined || value === null) return 'null';
  try {
    return typeof value === 'string' ? value : JSON.stringify(value);
  } catch {
    return '[unserializable]';
  }
};

export async function combinedHealth() {
  const [api, compute] = await Promise.allSettled([
    fetchApiHealth(),
    fetchComputeHealth(),
  ]);
  return { api, compute };
}

export async function predict(payload) {
  try {
    return await apiPost('/predict', payload);
  } catch (err) {
    const status = err?.status ?? 0;
    if (status && status < 500 && status !== 0) {
      const formatted = stringifyBody(err?.body);
      const apiErr = new Error(`API ${status}: ${formatted}`);
      apiErr.status = status;
      apiErr.body = err?.body;
      throw apiErr;
    }

    try {
      return await computePost('/predict', payload);
    } catch (computeErr) {
      const computeStatus = computeErr?.status ?? 0;
      const apiBody = stringifyBody(err?.body);
      const computeBody = stringifyBody(computeErr?.body);
      const merged = new Error(
        `Predict failed. API ${status || 'ERR'}: ${apiBody} | Compute ${computeStatus || 'ERR'}: ${computeBody}`
      );
      merged.status = computeStatus || status || 0;
      merged.apiError = err;
      merged.computeError = computeErr;
      throw merged;
    }
  }
}
