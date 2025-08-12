import requests
from typing import Dict, Any

def forward_to_compute(compute_url: str, json_payload: Dict[str, Any],
                       shared_secret: str, timeout_sec: int = 55) -> Dict[str, Any]:
    headers = {"X-Internal-Secret": shared_secret} if shared_secret else {}
    resp = requests.post(compute_url, json=json_payload, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    return resp.json()
