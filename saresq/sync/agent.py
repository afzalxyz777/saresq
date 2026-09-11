"""Store-and-forward sync agent.

The contract, in one line: **the pipeline never blocks on the link.** Evidence
is written to SQLite and the blob store first; this agent ships whatever it can,
whenever it can, in the order that maximises value per byte.

Ordering, highest value first:

1. **Alerts** -- 28 bytes each. A map pin on the operator's screen is worth
   more than any image, and costs a thousandth as much.
2. **Thumbnails** -- ~3 KB. Enough to reject a sun-heated corrugated sheet.
3. **Full RGB crops**, then **thermal patches**, then **clips**.

Within a kind, higher machine confidence goes first.

Every partial transfer persists its offset in `media.sent_bytes`, so a link that
drops halfway through a clip resumes at that byte rather than restarting. That
is the difference between "resumable" and "resumable in principle".
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from saresq.store.db import Store
from saresq.store.media import MediaStore
from saresq.sync.link import Link, frame_alert, frame_media


@dataclass
class SyncReport:
    alerts_sent: int = 0
    media_completed: int = 0
    media_partial: int = 0
    bytes_sent: int = 0
    seconds: float = 0.0
    stopped: str = "empty"       # empty | budget | link_down | send_failed

    @property
    def did_work(self) -> bool:
        return bool(self.alerts_sent or self.bytes_sent)


class SyncAgent:
    def __init__(self, store: Store, media: MediaStore, link: Link):
        self.store = store
        self.media = media
        self.link = link

    def pump(self, budget_s: float = 1.0, now_ns: int | None = None) -> SyncReport:
        """Ship as much as fits in `budget_s` seconds of link time."""
        rep = SyncReport()
        now_ns = now_ns if now_ns is not None else time.time_ns()

        if not self.link.is_up():
            rep.stopped = "link_down"
            return rep

        # ---- 1. alerts ------------------------------------------------
        for a in self.store.pending_alerts():
            frame = frame_alert(a["payload"])
            cost = self.link.seconds_for(len(frame))
            if rep.seconds + cost > budget_s:
                rep.stopped = "budget"
                return rep
            if not self.link.send(frame):
                rep.stopped = "link_down" if not self.link.is_up() else "send_failed"
                return rep
            self.store.mark_alert_sent(a["alert_id"], now_ns)
            rep.alerts_sent += 1
            rep.bytes_sent += len(frame)
            rep.seconds += cost

        # ---- 2. media, cheapest-and-most-confident first ---------------
        for row in self.store.pending_media():
            path = self.media.root / row["rel_path"]
            if not path.exists():
                # The row outlived its blob. Skip rather than crash the agent;
                # a missing artefact must not stall the whole queue.
                continue
            data = path.read_bytes()
            offset = int(row["sent_bytes"])
            mtu_body = max(1, self.link.mtu - 13)  # leave room for the frame header

            while offset < len(data):
                chunk = data[offset:offset + mtu_body]
                frame = frame_media(row["media_id"], offset, chunk)
                cost = self.link.seconds_for(len(frame))
                if rep.seconds + cost > budget_s:
                    self.store.set_media_progress(row["media_id"], offset)
                    if offset > int(row["sent_bytes"]):
                        rep.media_partial += 1
                    rep.stopped = "budget"
                    return rep
                if not self.link.send(frame):
                    self.store.set_media_progress(row["media_id"], offset)
                    if offset > int(row["sent_bytes"]):
                        rep.media_partial += 1
                    rep.stopped = "link_down" if not self.link.is_up() else "send_failed"
                    return rep
                offset += len(chunk)
                rep.bytes_sent += len(frame)
                rep.seconds += cost

            self.store.mark_media_synced(row["media_id"], now_ns)
            rep.media_completed += 1

        return rep

    # ------------------------------------------------------------------
    def backlog(self) -> dict:
        """What is still on the aircraft. Useful on the operator's screen:
        'everything you have not seen yet' is a real operational question."""
        pending = self.store.pending_media()
        by_kind: dict[str, int] = {}
        remaining = 0
        for r in pending:
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
            remaining += int(r["bytes"]) - int(r["sent_bytes"])
        return {
            "alerts": len(self.store.pending_alerts()),
            "media_items": len(pending),
            "by_kind": by_kind,
            "bytes_remaining": remaining,
            "seconds_at_link_rate": self.link.seconds_for(remaining),
        }
