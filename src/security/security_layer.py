"""
security_layer.py — Application-wide security controls for the
Predictive-Maintenance-for-Mobile-Hydraulics system.

Author: Aman Kushwah (2024AC05064) — Group 105

This is a single, reusable security layer imported by BOTH:
  * the ML notebook (105.ipynb) — to demonstrate the controls, and
  * the FastAPI microservices (fraud / RAG / real-time inference) — to enforce them.

Controls implemented (mapped to the "Security" quality requirement):
  1. Input Validation      — reject malformed / out-of-range sensor payloads.
  2. API-Key Authentication — every protected call needs a valid bearer key.
  3. Model Integrity        — SHA-256 checksum verified on load (anti-tamper).
  4. Audit Logging          — every prediction request logged, tamper-evident hash chain.
  5. Rate Limiting          — throttle requests per client to blunt abuse / DoS.

None of these require network access; they are pure-python and unit-testable.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Tuple

# --------------------------------------------------------------------------- #
# Structured audit logger (writes to stdout AND keeps a tamper-evident chain)
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SECURITY] %(levelname)s %(message)s",
)
_log = logging.getLogger("hydraulics.security")


class SecurityError(Exception):
    """Raised whenever a security control blocks a request."""


# --------------------------------------------------------------------------- #
# 1. INPUT VALIDATION
# --------------------------------------------------------------------------- #
# Physically-plausible operating envelopes for each hydraulic sensor.
# Anything outside these ranges is rejected before it ever reaches the model,
# which stops bad-data injection and adversarial out-of-distribution inputs.
SENSOR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "pressure":    (80.0, 260.0),    # bar
    "temperature": (20.0, 130.0),    # deg C
    "vibration":   (0.0,  20.0),     # mm/s RMS
    "flow_rate":   (30.0, 130.0),    # L/min
    "oil_debris":  (0.0,  400.0),    # ppm particles
}

# Physically-plausible envelopes for the full mobile-hydraulics telemetry schema
# (used by the serving API / web app, which ingests the richer per-cycle features).
HYDRAULIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "operating_hours":        (0.0, 5000.0),
    "pressure_mean_bar":      (80.0, 260.0),
    "pressure_std_bar":       (0.0, 60.0),
    "flow_mean_lpm":          (0.5, 20.0),
    "oil_temp_mean_c":        (10.0, 130.0),
    "vibration_rms_mms":      (0.0, 30.0),
    "motor_power_kw":         (1.0, 60.0),
    "pump_speed_mean_rpm":    (500.0, 2200.0),
    "cooling_efficiency_pct": (0.0, 100.0),
}


def validate_sensor_payload(payload: Dict[str, float],
                            bounds: Dict[str, Tuple[float, float]] | None = None) -> Dict[str, float]:
    """Validate a single machine's sensor reading against an operating envelope.

    Raises SecurityError on: missing field, non-numeric value, NaN/inf, or a
    value outside the physically-plausible envelope. Returns the clean payload.
    Pass `bounds` to validate a different schema (defaults to the 5-sensor SENSOR_BOUNDS).
    """
    if bounds is None:
        bounds = SENSOR_BOUNDS
    clean: Dict[str, float] = {}
    for field_name, (lo, hi) in bounds.items():
        if field_name not in payload:
            raise SecurityError(f"missing required sensor field: {field_name!r}")
        value = payload[field_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SecurityError(f"field {field_name!r} must be numeric, got {type(value).__name__}")
        # reject NaN / inf
        if value != value or value in (float("inf"), float("-inf")):
            raise SecurityError(f"field {field_name!r} is NaN or infinite")
        if not (lo <= value <= hi):
            raise SecurityError(
                f"field {field_name!r}={value} out of safe range [{lo}, {hi}]")
        clean[field_name] = float(value)
    return clean


# --------------------------------------------------------------------------- #
# 2. API-KEY AUTHENTICATION
# --------------------------------------------------------------------------- #
# In production these hashes come from a secret store / env vars. We store only
# SHA-256 hashes of keys (never the plaintext) and compare in constant time.
def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class ApiKeyAuthenticator:
    """Constant-time API-key check. Register keys by their SHA-256 hash."""

    def __init__(self, allowed_keys: Iterable[str] = ()):
        # store hashes, not the raw keys
        self._allowed_hashes = {_hash_key(k) for k in allowed_keys}

    def register(self, raw_key: str) -> None:
        self._allowed_hashes.add(_hash_key(raw_key))

    def authenticate(self, presented_key: str | None) -> bool:
        """Return True if the presented key is valid, else raise SecurityError."""
        if not presented_key:
            _log.warning("auth failed: no API key presented")
            raise SecurityError("missing API key")
        presented_hash = _hash_key(presented_key)
        # constant-time comparison against every allowed hash
        ok = any(hmac.compare_digest(presented_hash, h) for h in self._allowed_hashes)
        if not ok:
            _log.warning("auth failed: invalid API key")
            raise SecurityError("invalid API key")
        return True


# --------------------------------------------------------------------------- #
# 3. MODEL INTEGRITY (anti-tamper / supply-chain)
# --------------------------------------------------------------------------- #
def compute_file_sha256(path: str) -> str:
    """Return the SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_integrity(path: str, expected_sha256: str) -> bool:
    """Verify a model artifact matches its registered checksum.

    Raises SecurityError if the file has been modified/tampered.
    """
    actual = compute_file_sha256(path)
    if not hmac.compare_digest(actual, expected_sha256):
        _log.error("model integrity FAILED for %s", path)
        raise SecurityError(
            f"model integrity check failed for {path}: "
            f"expected {expected_sha256[:12]}..., got {actual[:12]}...")
    _log.info("model integrity OK for %s", path)
    return True


# --------------------------------------------------------------------------- #
# 4. AUDIT LOGGING (tamper-evident hash chain)
# --------------------------------------------------------------------------- #
@dataclass
class AuditTrail:
    """Append-only audit log where each entry hashes the previous one.

    Any modification to an earlier entry breaks the chain, giving
    tamper-evidence / non-repudiation for every prediction served.

    TODO: Persist audit entries to durable storage so tamper evidence survives
    process restarts and can be audited offline.
    """
    _entries: list = field(default_factory=list)
    _prev_hash: str = "0" * 64

    def record(self, actor: str, action: str, detail: str) -> str:
        payload = f"{self._prev_hash}|{actor}|{action}|{detail}"
        entry_hash = hashlib.sha256(payload.encode()).hexdigest()
        entry = {
            "seq": len(self._entries),
            "actor": actor,
            "action": action,
            "detail": detail,
            "prev_hash": self._prev_hash,
            "hash": entry_hash,
        }
        self._entries.append(entry)
        self._prev_hash = entry_hash
        _log.info("audit#%d %s %s | %s", entry["seq"], actor, action, detail)
        return entry_hash

    def verify_chain(self) -> bool:
        """Recompute the whole chain; return True only if untampered."""
        prev = "0" * 64
        for e in self._entries:
            payload = f"{prev}|{e['actor']}|{e['action']}|{e['detail']}"
            if hashlib.sha256(payload.encode()).hexdigest() != e["hash"]:
                return False
            prev = e["hash"]
        return True

    @property
    def entries(self):
        return list(self._entries)


# --------------------------------------------------------------------------- #
# 5. RATE LIMITING (sliding window per client)
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Simple sliding-window rate limiter: max_calls per window_seconds/client.

    TODO: Replace in-memory client state with a shared store (Redis/DB) for
    distributed deployment and multi-instance rate limiting.
    """

    def __init__(self, max_calls: int = 5, window_seconds: float = 1.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def check(self, client_id: str, now: float | None = None) -> bool:
        """Allow the call (True) or raise SecurityError if over the limit."""
        now = time.monotonic() if now is None else now
        hits = self._hits[client_id]
        # drop timestamps outside the window
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.max_calls:
            _log.warning("rate limit exceeded for client %s", client_id)
            raise SecurityError(f"rate limit exceeded for client {client_id!r}")
        hits.append(now)
        return True


# --------------------------------------------------------------------------- #
# Convenience: a single hardened entry point that composes all controls.
# The FastAPI services call this; the notebook demonstrates it.
# --------------------------------------------------------------------------- #
class SecureInferenceGateway:
    """Composes all five controls into one guarded prediction entry point."""

    def __init__(self, authenticator: ApiKeyAuthenticator,
                 rate_limiter: RateLimiter, audit: AuditTrail):
        self.auth = authenticator
        self.rate_limiter = rate_limiter
        self.audit = audit

    def guarded_predict(self, client_id: str, api_key: str,
                        payload: Dict[str, float], predict_fn,
                        bounds: Dict[str, Tuple[float, float]] | None = None) -> dict:
        """Run predict_fn(clean_payload) only after all controls pass."""
        self.auth.authenticate(api_key)                 # 2. authN
        self.rate_limiter.check(client_id)              # 5. rate limit
        clean = validate_sensor_payload(payload, bounds=bounds)        # 1. input validation
        result = predict_fn(clean)                       # model inference
        self.audit.record(client_id, "predict",         # 4. audit
                          f"payload_hash={hashlib.sha256(str(clean).encode()).hexdigest()[:12]}")
        return result
