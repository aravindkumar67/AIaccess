import logging
import requests
log = logging.getLogger(__name__)

class EventApi:
    """Configurable HTTP adapter for the external event API."""
    def __init__(self, enabled=False, url="", method="POST", token="", timeout=4.0):
        self.enabled, self.url, self.method, self.token = bool(enabled), url, method.upper(), token
        self.timeout = float(timeout)
    def emit(self, payload):
        if not self.enabled:
            log.info("API disabled; retained locally: %s", payload.get("event_id"))
            return {"sent": False, "reason": "disabled"}
        if not self.url: return {"sent": False, "reason": "missing_url"}
        if self.method not in {"POST", "PUT", "PATCH"}: return {"sent": False, "reason": f"unsupported_method:{self.method}"}
        headers = {"Content-Type": "application/json"}
        if self.token: headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = requests.request(self.method, self.url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return {"sent": True, "status_code": response.status_code}
        except requests.RequestException as exc:
            log.exception("Event API failed for %s", payload.get("event_id"))
            return {"sent": False, "reason": str(exc)}
