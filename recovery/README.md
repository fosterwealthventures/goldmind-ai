# GoldMIND v11.2 Recovery

This branch preserves and integrates the recovered GoldMIND v11.2 files with the existing GoldMIND application.

## Source artifacts

- `goldmind-ai-main.zip` — existing application foundation
  - SHA-256: `debf65661938e6e504afb1c4a539140222e0d5710b63d5ff1139fd4e8d23b00a`
- `GoldMIND-v11.2-engine.jsx` — advanced scalp/day/swing engine and permission system
  - SHA-256: `fcf64d55854d7840d74e625087973bfc346d7761413fb2ab03e26b22b460e518`
- `GoldMIND-v11.2.jsx` — production decision-first presentation layer
  - SHA-256: `264088d268846065d8124446ac2318d36b04acc3b8bb3b74996379506bd029d2`
- `GoldMIND-v11_2-preview.jsx` — development preview with mock scenarios
  - SHA-256: `2f0f6ba91a4fcf281a2e230aae2d992568e832016514bc62e25678707a5dd55e`
- `GoldMIND-Complete.jsx` — chart, MT5 account, journal, and analytics reference
  - SHA-256: `d77c62dc4aa6279a66b52425ef8979e176fbe4abbb9e72f635fea0511b88c85d`

## Integration rules

1. Keep `main` unchanged until the recovery build passes review.
2. Use the existing repository for deployment, API routing, and infrastructure.
3. Use the v11.2 engine as the authoritative trade-intelligence and permission layer.
4. Use the v11.2 presentation layer for the Setup/Home experience.
5. Reuse the chart, MT5 account, journal, and analytics components from `GoldMIND-Complete.jsx` where compatible.
6. Never silently generate mock candles in production.
7. The preview route may use mock scenarios only when visibly labeled as demo data.
8. When live market data is unavailable, pause guidance and show an explicit unavailable state.

## Planned structure

```text
frontend/web/src/
├── recovery/                 # untouched recovered originals
├── engine/                   # extracted v11.2 engine modules
├── components/goldmind/      # v11.2 presentation and supporting panels
├── hooks/useGoldMINDEngine.js
└── services/mt5Api.js
```

## Initial build sequence

1. Preserve original recovered files.
2. Audit the ZIP and GitHub tree for differences.
3. Compile the recovered engine unchanged and document missing dependencies.
4. Add a development-only preview route.
5. Connect real candle/tick data to one normalized engine hook.
6. Replace the old prediction authority with `calcPermission()`.
7. Add chart, account, and journal tabs.
8. Test every capital, timing, spread, structure, and reversal gate.
