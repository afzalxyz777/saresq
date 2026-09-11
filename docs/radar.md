# The radar scope

`/radar` is a plan-position display: the aircraft, the survivors it has
reported, and the state of the radio between them, over the real basemap.

It exists because the old map page was a database view. It drew whatever rows
were in `targets` at the moment you hit refresh, which cannot express the three
things that matter most during a sortie: *how old is this?*, *is this where the
aircraft is or where we think it is?*, and *is anything queued that we have not
seen yet?*

## What was copied from air traffic radar, and why

| Borrowed | From | What it buys the operator |
|---|---|---|
| α-β smoothing | terminal radar trackers | a position that does not jitter with GPS noise, and a velocity estimate for free |
| Validation gate | plot-to-track association | a detection 800 m away starts a new track instead of teleporting an old one |
| M-of-N initiation | track-initiation logic (3 of 4) | one noisy frame does not become a track |
| **Coasting** | ASTERIX Cat 062 `CST` | the aircraft does not vanish when the link drops |
| History trail | ARTS/STARS, tar1090 | speed and turn rate read at a glance |
| Data block on a leader | ARTS onward | callsign, altitude, groundspeed without clicking |
| Range rings + bearing rose | every PPI | "bearing 070, 240 m", not "over there" |

Track status flags follow **EUROCONTROL ASTERIX Category 062** (SDPS Track
Messages) item I062/080, so the vocabulary is one a surveillance engineer
recognises: `CNF` tentative/confirmed, `CST` coasting, `TSB`/`TSE` first/last
report, `SIM` simulated. Mode of movement (`I062/200`) reports constant course /
left / right, accelerating / decelerating, level / climb / descent.

### The one deliberate departure

A controller's tracker **drops** a track after a few coasts, because an aircraft
that stops replying has almost certainly left the coverage volume. A survivor
has not left anything. So the tracker runs two profiles:

* **Platform** (`SQ-01`) — coasts, then terminates, like ATC.
* **Contact** (survivors) — coasts, then goes to **HELD** and stays on the
  display until a human clears it.

Losing the radio must never delete a person from the map.

Contacts also use a **static filter**: velocity is never estimated and the
position uncertainty does not grow. A constant-velocity dead-reckoning model
draws a 300-metre uncertainty circle around a casualty who has not moved a
metre. A fix on something stationary does not become less certain with time; it
becomes *older*, and age is displayed separately.

## The link model, and why outages are injected

`saresq/surveillance/flight.py` computes received signal strength honestly: free-space
path loss at 433 MHz, plus 5 dB per metre of line-of-sight that passes through a
building, plus ITU-R P.526 knife-edge diffraction where the path grazes a roof.

What that model honestly says is that **a SiK telemetry link at 400 m has about
60 dB of margin**. Over a full lap of Segment A it never falls below −98 dBm
against a −105 dBm floor. It *degrades* behind the north-east block. It does not
die.

Rather than tune the constants until a dropout appeared — which would make every
other number in the model a lie — outages are **injected explicitly** and
labelled `INJECTED FAULT` everywhere they are shown. Two buttons on the scope:

* **Cut link** — the radio goes. The aircraft keeps flying and keeps finding
  people, but nothing reaches the ground: the track coasts and alerts queue.
* **Deny GPS** — the radio stays. Telemetry still arrives, but with no position
  in it: the track coasts while the link panel still reads `UP`.

Two genuinely different failures, and a display that conflates them is worse
than no display.

## Alerts are the real packets

When the aircraft detects someone, the service builds an actual
`saresq.sync.alerts` Tier-1 packet — 28 bytes, position as int32 degrees×1e7,
CRC16 — and either transmits it or queues it. The scope decodes what was
"received". So the backlog byte counts on the panel are measured, not asserted,
and if the wire format ever lost a digit of precision the map is where it would
show up.

Tier-1 alerts still flow in `DEGRADED`: 28 bytes survive a channel that stalls a
JPEG. That is the whole argument for the tiered sync design.

## Detections use the same sweep width as the analytics page

Detection probability follows a Gaussian lateral range curve
`P(y) = p_max · exp(−(y/σ)²)`. Sweep width is *by definition* the integral of
that curve, so fixing `W = 14 m` and `p_max = 0.6` fixes `σ = 13.2 m`. There is
a test asserting the curve integrates back to `W`.

This matters because the Koopman coverage figure on the scope
(`C = W·L/A`, `POD = 1 − e^(−C)`) and the simulated detections would otherwise
be describing two different sensors.

**`W = 14 m` is still an assumption.** `eval_pixels_on_target.py` calibrates it
from real footage, and that is blocked on the Colab run.

## One clock, always

`RadarService` catches up at most `MAX_CATCHUP_SCANS` (400) scans per request, so
a tab left open overnight does not try to replay eleven hours of flight. When it
hits that limit it re-bases its clock — and shifts **every** track, contact
timestamp, link-state timer and pending fault by the same amount, so the gap
reads as "the sim was paused".

Shifting only `t0` is a bug worth naming, because the symptom is confusing: the
platform gets re-plotted on the very next scan and looks fine, while the
contacts do not, so they render as `HELD 847:37` — fourteen hours stale on a
picture ten minutes old. Two objects on the same scope on two different clocks.
`test_a_long_gap_leaves_the_whole_picture_on_one_clock` pins it.

## Where the contacts come from

If `targets` has rows, the radar rediscovers **those** — the store is the source
of truth. Only when the store is empty does it fall back to six synthetic
casualties on a fixed lattice. Either way every track carries the ASTERIX `SIM`
bit and the page says `simulated: true`, so a screen grab can never be mistaken
for a real sortie.

## Files

| Path | What |
|---|---|
| `saresq/surveillance/filters.py` | α-β filter, Kalata gains, variance reduction, gate |
| `saresq/surveillance/tracker.py` | track lifecycle, association, ASTERIX status |
| `saresq/surveillance/geo.py` | ellipsoidal local tangent plane |
| `saresq/surveillance/flight.py` | survey pattern, link budget, sensor model |
| `saresq/surveillance/service.py` | the live picture behind `/api/radar` |
| `saresq/dashboard/static/mapcore.js` | shared projection + basemap (also used by `/map`) |
| `saresq/dashboard/static/radar.js` | the scope |
| `tests/test_surveillance.py` | filters, geometry, lifecycle, flight, link, service |
| `tests/test_dashboard.py` | the `/api/radar` routes and the PWA assets |
