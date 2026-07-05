"""Security facade for the application.

This module re-exports the concrete security primitives implemented in
`src/security/security_layer.py`. Keep this thin wrapper so application code
imports from `src.security_layer` and the implementation can evolve without
changing callers. Key responsibilities:

- Authentication & authorization helpers (ApiKeyAuthenticator)
- Input validation and sensor bounds checks (`validate_sensor_payload`)
- Model integrity verification (`verify_model_integrity`) using checksums
- Audit trail and rate limiting to meet operational security requirements

Design patterns: Facade (thin re-export) + Single Responsibility
"""

from .security.security_layer import (
    AuditTrail,
    ApiKeyAuthenticator,
    RateLimiter,
    SecurityError,
    SecureInferenceGateway,
    compute_file_sha256,
    validate_sensor_payload,
    verify_model_integrity,
    SENSOR_BOUNDS,
    HYDRAULIC_BOUNDS,
)

__all__ = [
    "AuditTrail",
    "ApiKeyAuthenticator",
    "RateLimiter",
    "SecurityError",
    "SecureInferenceGateway",
    "compute_file_sha256",
    "validate_sensor_payload",
    "verify_model_integrity",
    "SENSOR_BOUNDS",
    "HYDRAULIC_BOUNDS",
]
