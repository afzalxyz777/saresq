/* Live payload video in the ground station.
 *
 * WHY CANVAS AND DISCRETE JPEGS, NOT <img src=".mjpg">
 * The payload serves both. A browser repaints a multipart/x-mixed-replace
 * stream while a frame is still arriving, so the top strip of the incoming
 * image appears over grey and the panel visibly tears. Fetching one complete
 * JPEG, waiting for decode, and only then drawing it cannot tear. That was
 * diagnosed on the payload's own page and the same fix applies here.
 *
 * Each panel also fetches INDEPENDENTLY and at its own rate: the thermal array
 * refreshes at 4 Hz, the detector produces a frame every ~600 ms on the Pi, and
 * coupling them would drag every panel to the slowest one.
 */
(function () {
  "use strict";

  function mmss(sec) {
    if (sec == null) return "\u2014";
    var m = Math.floor(sec / 60), s = Math.floor(sec % 60);
    return m >= 60 ? Math.floor(m / 60) + "h" + (m % 60) + "m"
                   : m + "m" + (s < 10 ? "0" : "") + s + "s";
  }
  function f(v, n, dash) { return (v == null || isNaN(v)) ? (dash || "—") : v.toFixed(n); }

  /* One panel. Holds its own in-flight flag so a slow link cannot queue up a
     backlog of requests that all land at once and then draw out of order. */
  function Panel(canvasId, url, periodMs) {
    var cv = document.getElementById(canvasId);
    if (!cv) return null;
    var ctx = cv.getContext("2d");
    var busy = false, alive = false;
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cv.width, cv.height);

    function tick() {
      if (busy) return;
      busy = true;
      var img = new Image();
      img.onload = function () {
        // Size the canvas to the frame once, so a 640x480 detector view and a
        // 512x384 thermal view each keep their own aspect ratio.
        if (cv.width !== img.naturalWidth || cv.height !== img.naturalHeight) {
          cv.width = img.naturalWidth; cv.height = img.naturalHeight;
        }
        ctx.drawImage(img, 0, 0, cv.width, cv.height);
        alive = true; busy = false;
      };
      img.onerror = function () {
        busy = false;
        if (alive) {
          // Link just dropped: dim the last good frame instead of blanking it.
          // The last thing the payload saw is still the most useful thing on
          // screen, and a black rectangle throws it away.
          ctx.fillStyle = "rgba(11,20,26,.55)";
          ctx.fillRect(0, 0, cv.width, cv.height);
          alive = false;
        }
      };
      img.src = url + "?t=" + Date.now();      // defeat any intermediate cache
    }
    tick();
    setInterval(tick, periodMs);
    return { live: function () { return alive; } };
  }

  Panel("c-detect", "/api/feed/detect", 700);
  Panel("c-thermal", "/api/feed/thermal", 260);
  Panel("c-rgb", "/api/feed/rgb", 900);

  // ---- crops ----
  var cropsEl = document.getElementById("crops");
  function crops(list) {
    document.getElementById("p-crops").textContent = list.length ? list.length : "none";
    if (!list.length) {
      cropsEl.innerHTML = "<div class='ps' style='color:var(--muted)'>gate quiet — no regions handed over</div>";
      return;
    }
    // Rebuild only when the count changes; otherwise just refresh the images,
    // so a crop does not flicker white every poll while it reloads.
    if (cropsEl.children.length !== list.length) {
      cropsEl.innerHTML = list.map(function (c, i) {
        return "<figure><img alt='gate crop " + (i + 1) + "'>"
             + "<figcaption id='cc" + i + "'></figcaption></figure>";
      }).join("");
    }
    list.forEach(function (c, i) {
      var img = cropsEl.children[i].querySelector("img");
      img.src = "/api/feed/crop" + i + "?t=" + Date.now();
      var cap = document.getElementById("cc" + i);
      if (cap) cap.innerHTML = "+" + f(c.z, 1) + "σ<br>" + f(c.T, 1) + "°C";
    });
  }

  // ---- stats ----
  function stats() {
    fetch("/api/live").then(function (r) { return r.json(); }).then(function (L) {
      var ver = document.getElementById("verdict");
      var down = !L.configured || !L.connected;
      // BODY_HEAT is its own state. A blob at body temperature with a camera
      // that had no light is the night case this payload is built for, and it
      // must not be reported with the same words as a candidate the detector
      // examined and rejected.
      ver.className = down ? "down"
        : (L.verdict === "PERSON" ? "found"
        : (L.verdict === "BODY_HEAT" ? "found"
        : (L.verdict === "HEAT" ? "heat" : "")));
      document.getElementById("v-main").textContent =
        !L.configured ? "NO PAYLOAD"
        : !L.connected ? "LINK DOWN"
        : (L.n ? L.n + " PERSON" + (L.n > 1 ? "S" : "")
              : (L.verdict === "BODY_HEAT" ? "BODY HEAT"
              : (L.fired ? "HEAT SOURCE" : "CLEAR")));
      document.getElementById("v-sub").textContent =
        !L.configured ? "start the dashboard with --payload <pi-ip>"
        : !L.connected ? (L.host + " — " + (L.why || "unreachable"))
        : (L.n ? "detector confirmed · " + f(L.det_ms, 0) + " ms"
        : (L.verdict === "BODY_HEAT"
             ? "thermal signature at +" + f(L.z_max, 1) + "σ · camera blind ("
               + f(L.crop_lum, 0) + "/255) — visible branch cannot corroborate"
             : (L.fired ? "gate fired at +" + f(L.z_max, 1) + "σ · no person confirmed"
                        : "peak +" + f(L.z_max, 1) + "σ of " + f(L.z_t, 1) + " needed")));

      document.getElementById("f-detect").innerHTML = L.connected
        ? "<b>" + (L.n || 0) + "</b> detection(s) · " + f(L.det_ms, 0) + " ms · " + (L.model || "")
        : "no frames — link down";
      document.getElementById("p-det").textContent = L.connected ? (L.n ? L.n + " found" : "none") : "down";
      document.getElementById("p-det").className = "pill" + (L.n ? " on" : "");

      document.getElementById("f-thermal").innerHTML = L.connected
        ? "<b>" + f(L.t_min, 1) + "–" + f(L.t_max, 1) + "°C</b> · spread "
          + f(L.t_spread, 1) + "°C · peak <b>+" + f(L.z_max, 1) + "σ</b> of "
          + f(L.z_t, 1) + " · " + f(L.t_fps, 1) + " fps"
        : "—";
      var gp = document.getElementById("p-gate");
      gp.textContent = L.connected ? (L.fired ? "fired · " + L.blobs : "quiet") : "down";
      gp.className = "pill" + (L.fired ? " on" : "");
      document.getElementById("f-rgb").textContent = L.connected ? "1640×1232 · rotated 180°" : "—";

      document.getElementById("kv").innerHTML =
        row("Link", L.configured ? (L.connected ? "up · " + L.host : "down · " + L.host) : "not configured")
        + row("Position", L.lat != null ? f(L.lat, 5) + "°N " + f(L.lon, 5) + "°E"
                                        : "no GPS fix")
        + row("Satellites", L.fix ? (L.sats || 0) + " used · HDOP " + f(L.hdop, 1)
                                  : (L.in_view || 0) + " in view · " + (L.gps_state || "no fix"))
        + row("Scene", L.scene ? L.scene.replace(/_/g, " ") + " · "
                                 + Math.round((L.scene_p || 0) * 100) + "%" : "—")
        + row("Mission", L.session
                ? L.session + " \u00b7 " + mmss(L.session_age_s)
                : "\u2014")
        + row("Events", (L.ingested || 0) + " this mission")
        + row("CPU", L.temp || "—");

      crops(L.connected ? (L.crops || []) : []);
    }).catch(function () {});
  }
  function row(k, v) { return "<dt>" + k + "</dt><dd>" + v + "</dd>"; }

  stats();
  setInterval(stats, 1000);

})();
