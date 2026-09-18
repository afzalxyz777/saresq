/* Hazards page.
 *
 * Two sources, deliberately kept apart:
 *
 *   /api/live     the payload's CURRENT scene call, five-way. Present whether
 *                 or not there is a GPS fix, which is the whole reason this
 *                 page works on a bench indoors.
 *   /api/hazards  durable, GEOLOCATED hazard events. A row lands here only
 *                 when a non-normal class scored >= 0.5 AND the capture had a
 *                 position, because a hazard pin without a position is a lie.
 *
 * So an empty log with a live "NORMAL 0.99" is the correct and expected state
 * in a room, and the page says so in words rather than looking broken.
 */
"use strict";

var CLASSES = [
  ["collapsed_building", "Collapsed building"],
  ["fire", "Fire"],
  ["flooded_areas", "Flooded area"],
  ["traffic_incident", "Traffic incident"],
  ["normal", "Normal"]
];
var DANGER = { collapsed_building: 1, fire: 1, flooded_areas: 1, traffic_incident: 1 };
var LABEL = {};
CLASSES.forEach(function (c) { LABEL[c[0]] = c[1]; });

var HIST = [];          // {p, danger} rolling, for the sparkline
var HIST_MAX = 240;

function el(id) { return document.getElementById(id); }
function pct(x) { return (100 * (x || 0)).toFixed(1) + "%"; }
function degc(x) { return (x === null || x === undefined) ? "—" : x.toFixed(1) + " °C"; }

/* ---- the five bars ------------------------------------------------------ */
function buildBars() {
  var host = el("bars");
  host.innerHTML = "";
  CLASSES.forEach(function (c) {
    var row = document.createElement("div");
    row.className = "bar";
    row.id = "bar-" + c[0];
    row.innerHTML = '<div class="nm"></div>' +
                    '<div class="track"><div class="fill"></div></div>' +
                    '<div class="pc">—</div>';
    row.querySelector(".nm").textContent = c[1];
    host.appendChild(row);
  });
}

function paintBars(probs, top) {
  CLASSES.forEach(function (c) {
    var key = c[0], row = el("bar-" + key);
    if (!row) return;
    var v = probs && probs[key] !== undefined ? probs[key] : null;
    row.querySelector(".fill").style.width = v === null ? "0%" : pct(v);
    row.querySelector(".pc").textContent = v === null ? "—" : pct(v);
    row.classList.toggle("top", key === top);
    row.classList.toggle("danger", key === top && !!DANGER[key]);
  });
}

/* ---- headline assessment ------------------------------------------------ */
function paintAssess(d) {
  var box = el("assess"), t = el("assessTitle"), s = el("assessSub");
  box.className = "";

  if (!d.configured) {
    box.className = "down";
    t.textContent = "NO PAYLOAD";
    s.textContent = "Start the ground station with --payload <host> to classify the scene.";
    return;
  }
  if (!d.connected) {
    box.className = "down";
    t.textContent = "PAYLOAD OFFLINE";
    s.textContent = "No link. The last classification is not shown, because a stale "
                  + "hazard call is worse than none.";
    return;
  }
  if (!d.scene) {
    box.className = "down";
    t.textContent = "CLASSIFIER NOT RUNNING";
    s.textContent = d.scene_why ? String(d.scene_why)
                                : "The payload is connected but reported no scene.";
    return;
  }

  var p = d.scene_p || 0;
  if (d.scene === "normal") {
    box.className = "clear";
    t.textContent = "NO HAZARD DETECTED";
    s.textContent = "Scene classified normal at " + pct(p)
                  + " — all four hazard classes evaluated and scored low.";
  } else {
    box.className = "haz";
    t.textContent = (LABEL[d.scene] || d.scene).toUpperCase();
    s.textContent = "Classified at " + pct(p) + "."
                  + (d.fix ? "" : " No GPS fix, so this is not placed on the map.");
  }
}

/* ---- sparkline ---------------------------------------------------------- */
function paintSpark() {
  var c = el("spark"), g = c.getContext("2d");
  var W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  if (!HIST.length) return;

  // 0.5 is the threshold at which a non-normal call becomes a logged hazard.
  g.strokeStyle = "#2b3a42"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, H * 0.5); g.lineTo(W, H * 0.5); g.stroke();

  var n = Math.min(HIST.length, HIST_MAX);
  var step = W / HIST_MAX;
  for (var i = 0; i < n; i++) {
    var h = HIST[HIST.length - n + i];
    var y = H - (h.p * H);
    g.fillStyle = h.danger ? "#E0533D" : "#4FC489";
    g.fillRect(i * step, y, Math.max(step - 0.6, 1), H - y);
  }
}

/* ---- durable, geolocated log -------------------------------------------- */
function paintLog(rows) {
  var host = el("logBody");
  el("logCount").textContent = rows.length;
  if (!rows.length) {
    host.innerHTML =
      '<div class="empty"><b>No geolocated hazards this session.</b>' +
      'A row is written here only when all three hold:' +
      '<ol><li>the classifier called something other than <i>normal</i></li>' +
      '<li>at confidence 0.5 or above</li>' +
      '<li>and the capture carried a real position</li></ol>' +
      'Indoors the receiver never fixes, so bench runs legitimately produce ' +
      'none — the live call above is the reading that matters there.</div>';
    return;
  }
  var h = '<table class="log"><thead><tr><th>Time</th><th>Class</th>' +
          '<th>Conf.</th><th>Position</th></tr></thead><tbody>';
  rows.forEach(function (r) {
    var when = r.t_ns ? new Date(r.t_ns / 1e6).toLocaleTimeString() : "—";
    var cls = r["class"] || r.cls || "—";
    h += "<tr><td>" + when + '</td><td><span class="tag">'
       + (LABEL[cls] || cls) + "</span></td><td>" + pct(r.p) + "</td><td>"
       + (r.lat != null ? r.lat.toFixed(5) + ", " + r.lon.toFixed(5) : "—")
       + "</td></tr>";
  });
  host.innerHTML = h + "</tbody></table>";
}

/* ---- polling ------------------------------------------------------------ */
function tickLive() {
  fetch("/api/live").then(function (r) { return r.json(); }).then(function (d) {
    paintAssess(d);
    paintBars(d.scene_probs, d.scene);

    el("clsAge").textContent = d.age_s == null ? "—" : d.age_s.toFixed(1) + "s ago";
    el("clsFoot").textContent = "MobileNetV2 / AIDER · five classes · 1 Hz"
      + (d.scene_ms ? " · " + Number(d.scene_ms).toFixed(0) + " ms on the payload" : "");

    el("envAmb").textContent    = degc(d.t_min);
    el("envMax").textContent    = degc(d.t_max);
    el("envSpread").textContent = d.t_spread == null ? "—" : d.t_spread.toFixed(2) + " K";
    el("envFps").textContent    = d.t_fps == null ? "—" : d.t_fps.toFixed(1) + " Hz";
    el("envSoc").textContent    = d.temp == null ? "—" : Number(d.temp).toFixed(1) + " °C";

    if (d.connected && d.scene) {
      HIST.push({ p: d.scene_p || 0, danger: !!DANGER[d.scene] });
      if (HIST.length > HIST_MAX) HIST.shift();
      paintSpark();
    }
  }).catch(function () {});
}

function tickLog() {
  fetch("/api/hazards").then(function (r) { return r.json(); })
    .then(paintLog).catch(function () {});
}

buildBars();
paintLog([]);
tickLive(); tickLog();
setInterval(tickLive, 1000);
setInterval(tickLog, 5000);
