"""Delay-tolerant sync: get evidence from the aircraft to the operator over
whatever link happens to exist, without ever blocking the perception pipeline.
"""
from saresq.sync.agent import SyncAgent, SyncReport
from saresq.sync.alerts import Alert, ALERT_BYTES, alert_from_target, pack_alert, unpack_alert
from saresq.sync.link import Link, Receiver, SimulatedLink, TelemetryLink, WifiLink

__all__ = [
    "SyncAgent", "SyncReport",
    "Alert", "ALERT_BYTES", "alert_from_target", "pack_alert", "unpack_alert",
    "Link", "Receiver", "SimulatedLink", "TelemetryLink", "WifiLink",
]
