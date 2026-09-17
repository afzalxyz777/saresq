/* SaResQ ground-station map: the flood/trafficability picture.
 *
 * The projection, the OSM geometry, the SRTM terrain and the road classifier
 * all live in mapcore.js and are shared with the radar scope. What is left
 * here is what makes this page different from that one: the flood level
 * control, the vehicle profile, and the store's targets and hazards.
 */
(function () {
  "use strict";
  var M = window.MapCore, sve = M.sve;

  var S = {
    show: { roads: true, bldg: true, d3: false },
    targets: [], hazards: [], sel: null, fitted: false,
    // Tiles on by default: if the network is there the coordinator gets the
    // whole city, and mapcore falls back to the baked geometry by itself when
    // no tile arrives, so switching this on cannot leave a blank map.
    tiles: { on: true, source: "dark" },
    live: null,
    // Breadcrumb of where the aircraft has been. Capped because a long flight
    // at 1 Hz would otherwise grow without bound in a page nobody reloads.
    trail: [], TRAIL_MAX: 600,
    followed: false
  };

  var host = document.getElementById("maparea");
  var ov = document.getElementById("ov");
  var view = M.create({
    host: host,
    canvas: document.getElementById("cv"),
    svg: ov,
    cam: { lat: 22.57323, lon: 88.36497, z: 17.2 },
    onChange: draw,
    onHover: function (ll) {
      document.getElementById("readout").textContent =
        ll[0].toFixed(5) + "\u00b0N " + ll[1].toFixed(5) + "\u00b0E";
    }
  });

  // ---- overlay: live targets from the store ----
  var CLS_COL = { HIGH: "#F46454", MEDIUM: "#F3A83C", LOW: "#6D838D" };
  function label(x, y, s, at) {
    var o = { x: x, y: y, "font-family": "ui-monospace, monospace", "font-size": 11, fill: "#DCE7EC",
              "paint-order": "stroke fill", stroke: "#0B141A", "stroke-width": 4,
              "stroke-linejoin": "round" };
    for (var k in at || {}) o[k] = at[k];
    var e = sve("text", o); e.textContent = s; return e;
  }
  /* The aircraft itself. Drawn last so it is never hidden by a target pin,
     and drawn differently depending on where the position CAME from: a GPS fix
     is a solid ring, an operator-set datum is dashed and labelled MANUAL. The
     two must never look alike -- a rescuer reading a manual datum as a measured
     fix is the worst failure this page could cause. */
  function drawPayload() {
    var L = S.live;
    if (!L || !L.configured) return;
    var lat = L.lat, lon = L.lon, src = "gps";
    if (lat == null || lon == null) {
      if (!L.origin) return;                 // nothing to draw, and nothing invented
      lat = L.origin.lat; lon = L.origin.lon; src = "manual";
    }
    var s = view.P(lat, lon);
    var stale = L.age_s != null && L.age_s > 5;
    var col = !L.connected || stale ? "#6D838D"
            : (L.verdict === "PERSON" ? "#F46454"
            : (L.verdict === "HEAT" ? "#F3A83C" : "#3FCDEC"));

    // breadcrumb
    if (S.trail.length > 1) {
      var d = S.trail.map(function (p, i) {
        var q = view.P(p[0], p[1]);
        return (i ? "L " : "M ") + q[0].toFixed(1) + " " + q[1].toFixed(1);
      }).join(" ");
      ov.appendChild(sve("path", { d: d, fill: "none", stroke: col,
        "stroke-opacity": .38, "stroke-width": 1.6, "stroke-linejoin": "round" }));
    }

    var g = sve("g", {});
    // Position uncertainty, to scale. HDOP is unitless, so it is turned into
    // metres with the receiver's nominal ~2.5 m UERE; a manual datum gets a
    // deliberately large 25 m ring because that is honestly what it is worth.
    var errM = src === "manual" ? 25 : Math.max(2.5, (L.hdop || 1) * 2.5);
    g.appendChild(sve("circle", { cx: s[0], cy: s[1], r: Math.max(6, errM / view.mpp()),
      fill: col, "fill-opacity": .07, stroke: col, "stroke-opacity": .5,
      "stroke-width": 1.1, "stroke-dasharray": src === "manual" ? "3 4" : null }));
    g.appendChild(sve("circle", { cx: s[0], cy: s[1], r: 13, fill: "#091015",
      "fill-opacity": .92, stroke: col, "stroke-width": 2.4,
      "stroke-dasharray": src === "manual" ? "4 3" : null }));
    // A quadcopter glyph: four arms, so it reads as the aircraft and not as
    // another target pin.
    ["M -7 -7 L 7 7", "M 7 -7 L -7 7"].forEach(function (d) {
      g.appendChild(sve("path", { d: d, transform: "translate(" + s[0] + "," + s[1] + ")",
        stroke: col, "stroke-width": 2, "stroke-linecap": "round" }));
    });
    [[-7,-7],[7,-7],[-7,7],[7,7]].forEach(function (o) {
      g.appendChild(sve("circle", { cx: s[0] + o[0], cy: s[1] + o[1], r: 3.2,
        fill: "none", stroke: col, "stroke-width": 1.5 }));
    });
    g.appendChild(label(s[0] + 18, s[1] - 2, "PAYLOAD",
      { "font-weight": 700, "font-size": 11, fill: col }));
    g.appendChild(label(s[0] + 18, s[1] + 10,
      src === "manual" ? "MANUAL DATUM \u00b7 not a GPS fix"
                       : (L.sats || 0) + " sats \u00b7 HDOP " + (L.hdop == null ? "\u2014" : L.hdop.toFixed(1)),
      { "font-size": 9.5, fill: src === "manual" ? "#F3A83C" : "#6D838D" }));
    ov.appendChild(g);
  }

  function drawOverlay() {
    ov.innerHTML = "";
    S.hazards.forEach(function (h) {
      if (h.lat == null) return;
      var s = view.P(h.lat, h.lon), r = 9, g = sve("g", {});
      g.appendChild(sve("path", { d: "M 0 " + (-r) + " L " + r + " 0 L 0 " + r + " L " + (-r) + " 0 Z",
        transform: "translate(" + s[0] + "," + s[1] + ")", fill: "#091015", "fill-opacity": .85,
        stroke: "#A98BFF", "stroke-width": 1.8 }));
      g.appendChild(label(s[0] + r + 5, s[1] + 4, (h.class || "hazard").toUpperCase(),
        { "font-size": 9.5, fill: "#A98BFF", "font-weight": 600 }));
      ov.appendChild(g);
    });
    S.targets.forEach(function (t) {
      if (t.lat == null) return;
      var s = view.P(t.lat, t.lon), col = CLS_COL[t.class] || "#3FCDEC", r = 11;
      var g = sve("g", { style: "cursor:pointer", class: "hit" });
      if (t.pos_err_m)
        g.appendChild(sve("circle", { cx: s[0], cy: s[1], r: Math.max(3, t.pos_err_m / view.mpp()),
          fill: col, "fill-opacity": .10, stroke: col, "stroke-opacity": .35,
          "stroke-width": .9, "stroke-dasharray": "2 3" }));
      if (S.sel === t.target_id)
        g.appendChild(sve("circle", { cx: s[0], cy: s[1], r: r + 9, fill: "none",
          stroke: col, "stroke-width": 1.4 }));
      g.appendChild(sve("circle", { cx: s[0], cy: s[1], r: r, fill: "#091015",
        "fill-opacity": .9, stroke: col, "stroke-width": 2.2 }));
      g.appendChild(sve("circle", { cx: s[0], cy: s[1] - 2.6, r: 2, fill: col }));
      g.appendChild(sve("path", { d: "M " + (s[0] - 3) + " " + (s[1] + 4.4) + " a 3 3.6 0 0 1 6 0 Z", fill: col }));
      g.appendChild(label(s[0] + r + 6, s[1] - 1, "T-" + String(t.target_id).padStart(3, "0"),
        { "font-weight": 600, "font-size": 11.5 }));
      g.appendChild(label(s[0] + r + 6, s[1] + 11,
        (t.class || "?") + " P" + (t.p_final == null ? "\u2014" : t.p_final.toFixed(2))
        + " \u00b7 " + (t.n_passes || 1) + " pass"
        + ((t.n_passes || 1) > 1 ? "es" : ""),
        { "font-size": 9.5, fill: "#6D838D" }));
      g.addEventListener("click", function () {
        if (view.dragged()) return;
        S.sel = t.target_id; showInfo(t); draw();
      });
      ov.appendChild(g);
    });
    drawPayload();
  }

  var raf = false;
  function draw() {
    if (raf) return;
    raf = true;
    requestAnimationFrame(function () {
      raf = false;
      view.drawBase({
        show: S.show,
        tiles: { on: S.tiles.on, source: S.tiles.source, onTile: draw }
      });
      drawOverlay();
      chrome();
    });
  }
  function chrome() {
    var sc = M.niceScale(view.mpp());
    var bar = document.getElementById("scaleBar"), lab = document.getElementById("scaleLab");
    bar.style.width = sc.px.toFixed(1) + "px";
    lab.style.width = sc.px.toFixed(1) + "px";
    lab.textContent = sc.label;

  }
  function showInfo(t) {
    document.getElementById("info").innerHTML =
      "<div class='lbl'>Target T-" + String(t.target_id).padStart(3, "0") + "</div>"
      + "<div class='ps' style='margin-top:4px;line-height:1.8'>"
      + (t.class || "?") + " · P " + (t.p_final == null ? "—" : t.p_final.toFixed(2))
      + " · " + (t.decision || "—") + "<br>"
      + t.lat.toFixed(5) + "°N " + t.lon.toFixed(5) + "°E<br>"
      + (t.pos_err_m >= 20 ? "<span style='color:#F3A83C'>manual datum \u00b7 \u00b1"
                              + (t.pos_err_m || 0).toFixed(0) + " m</span>"
                           : "CEP " + (t.pos_err_m || 0).toFixed(1) + " m") + "<br>"
      + "<a href='/review'>open in review queue</a></div>";
  }

  // ---- controls ----
  document.getElementById("zin").addEventListener("click", function () { view.zoom(view.cam.z + 1); });
  document.getElementById("zout").addEventListener("click", function () { view.zoom(view.cam.z - 1); });
  document.querySelectorAll("[data-layer]").forEach(function (b) {
    b.addEventListener("click", function () {
      var k = b.getAttribute("data-layer");
      S.show[k] = !S.show[k];
      b.setAttribute("aria-pressed", String(!!S.show[k]));
      draw();
    });
  });
  document.querySelectorAll("[data-prof]").forEach(function (b) {
    b.addEventListener("click", function () {
      S.profile = b.getAttribute("data-prof");
      document.querySelectorAll("[data-prof]").forEach(function (c) {
        c.setAttribute("aria-pressed", String(c.getAttribute("data-prof") === S.profile));
      });
      draw();
    });
  });

  // ---- live data ----
  function refresh() {
    Promise.all([
      fetch("/api/targets").then(function (r) { return r.json(); }),
      fetch("/api/hazards").then(function (r) { return r.json(); })
    ]).then(function (res) {
      S.targets = res[0]; S.hazards = res[1];
      var pts = S.targets.filter(function (t) { return t.lat != null; });
      if (!S.fitted && pts.length) {
        view.panTo(pts.reduce(function (a, t) { return a + t.lat; }, 0) / pts.length,
                   pts.reduce(function (a, t) { return a + t.lon; }, 0) / pts.length);
        S.fitted = true;
      }
      // An empty store is the correct state before a flight, and it must say so
      // plainly. Shipping a database of invented targets so the map "looks
      // populated" is how a demo starts asserting people who do not exist.
      document.getElementById("status").textContent = pts.length
        ? pts.length + " target(s) reported by the payload \u00b7 "
          + new Date().toLocaleTimeString()
        : "no targets \u2014 nothing reported yet";
      draw();
    }).catch(function () {
      document.getElementById("status").textContent = "offline / no data";
      draw();
    });
  }

  // ---- basemap source ----
  document.querySelectorAll("[data-base]").forEach(function (b) {
    b.addEventListener("click", function () {
      var v = b.getAttribute("data-base");
      S.tiles.on = v !== "baked";
      if (S.tiles.on) S.tiles.source = v;
      document.querySelectorAll("[data-base]").forEach(function (c) {
        var cv = c.getAttribute("data-base");
        c.setAttribute("aria-pressed", String(S.tiles.on ? cv === S.tiles.source : cv === "baked"));
      });
      draw();
      setTimeout(baseStatus, 700);
    });
  });
  function baseStatus() {
    var el = document.getElementById("baseStat");
    if (!el) return;
    if (!S.tiles.on) { el.textContent = "offline geometry"; return; }
    var h = window.TileLayer ? window.TileLayer.health() : null;
    var t = view.tileInfo();
    var at = document.getElementById("attr");
    if (at) at.textContent = S.tiles.on && window.TileLayer
      ? window.TileLayer.attribution(S.tiles.source)
      : "Baked OSM geometry \u00a9 OpenStreetMap contributors (ODbL)";
    el.textContent = !h || h.ok === 0
      ? (h && h.fail ? "no network \u2014 offline geometry" : "loading\u2026")
      : (t && t.drawn ? "z" + t.z + " \u00b7 " + h.ok + " tiles" : "offline geometry");
  }

  /* Shift-click sets the datum used for captures with no GPS fix. Deliberately
     a modifier: a bare click selects a target, and an operator who nudges the
     map must not silently relocate every unfixed find in the mission. */
  document.getElementById("cv").addEventListener("click", function (ev) {
    if (!ev.shiftKey || view.dragged()) return;
    var r = host.getBoundingClientRect();
    var ll = view.unP(ev.clientX - r.left, ev.clientY - r.top);
    fetch("/api/origin", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat: ll[0], lon: ll[1] })
    }).then(function (r) { return r.json(); })
      .then(function () { pollLive(); })
      .catch(function () {});
  });

  // ---- live payload ----
  function fmt(v, n, unit) {
    return v == null ? "\u2014" : v.toFixed(n) + (unit || "");
  }
  function pollLive() {
    fetch("/api/live").then(function (r) { return r.json(); }).then(function (L) {
      S.live = L;
      if (L.configured && L.connected && L.lat != null && L.lon != null) {
        var last = S.trail[S.trail.length - 1];
        // Only extend the breadcrumb on real movement: a stationary payload
        // would otherwise pile thousands of identical points into the path.
        if (!last || Math.abs(last[0] - L.lat) > 1e-6 || Math.abs(last[1] - L.lon) > 1e-6) {
          S.trail.push([L.lat, L.lon]);
          if (S.trail.length > S.TRAIL_MAX) S.trail.shift();
        }
        if (!S.followed) { view.panTo(L.lat, L.lon); S.followed = true; }
      }
      livePanel(L); hazPanel(L);
      draw();
    }).catch(function () {
      S.live = { configured: true, connected: false, why: "dashboard unreachable" };
      livePanel(S.live); hazPanel(S.live); draw();
    });
  }
  /* The observed counterpart to what used to be a modelled flood layer. This
     says what the payload's camera actually saw, with the classifier's own
     confidence, or says nothing at all. */
  function hazPanel(L) {
    var box = document.getElementById("hazbox");
    if (!box) return;
    if (!L || !L.configured || !L.scene) {
      box.innerHTML = "<div class='ps'>no imagery classified yet</div>";
      return;
    }
    var bad = L.scene !== "normal";
    box.innerHTML =
      "<b style='font-family:var(--mono);font-size:13px;color:"
      + (bad ? "#F3A83C" : "#4FC489") + "'>"
      + L.scene.replace(/_/g, " ").toUpperCase() + "</b>"
      + "<div class='ps' style='margin-top:5px;line-height:1.7'>"
      + Math.round((L.scene_p || 0) * 100) + "% confidence"
      + "<br>MobileNetV2 / AIDER"
      + "<br><span style='color:var(--muted)'>whole frame, 1 Hz</span></div>";
  }

  function livePanel(L) {
    var box = document.getElementById("livebox");
    if (!box) return;
    if (!L || !L.configured) {
      box.innerHTML = "<div class='ps'>No payload configured."
        + "<br><span style='color:var(--muted)'>start with <code>--payload &lt;pi-ip&gt;</code></span></div>";
      return;
    }
    var stale = L.age_s != null && L.age_s > 5;
    var dot = !L.connected || stale ? "#6D838D"
            : (L.verdict === "PERSON" ? "#F46454"
            : (L.verdict === "HEAT" ? "#F3A83C" : "#4FC489"));
    var pos = L.lat != null
      ? L.lat.toFixed(5) + "\u00b0N " + L.lon.toFixed(5) + "\u00b0E"
      : (L.origin ? "<span style='color:#F3A83C'>manual datum \u00b7 no GPS fix</span>"
                  : "<span style='color:#F46454'>no position \u2014 shift-click to set datum</span>");
    box.innerHTML =
      "<div style='display:flex;align-items:center;gap:7px'>"
      + "<span style='width:8px;height:8px;border-radius:50%;background:" + dot + "'></span>"
      + "<b style='font-family:var(--mono);font-size:13px;color:" + dot + "'>"
      + (!L.connected ? "LINK DOWN" : (L.verdict || "\u2014")) + "</b>"
      + "<span class='ps' style='margin-left:auto'>" + (L.host || "") + "</span></div>"
      + "<div class='ps' style='margin-top:6px;line-height:1.75'>" + pos
      + "<br>" + (L.fix ? (L.sats || 0) + " sats \u00b7 HDOP " + fmt(L.hdop, 1)
                        : "GPS: " + (L.gps_state || "no fix")
                          + " \u00b7 " + (L.in_view || 0) + " in view")
      + "<br>thermal " + fmt(L.t_min, 1) + "\u2013" + fmt(L.t_max, 1) + "\u00b0C"
      + " \u00b7 peak +" + fmt(L.z_max, 1) + "\u03c3"
      + (L.blobs ? " \u00b7 " + L.blobs + " blob(s)" : "")
      + (L.scene ? "<br><span class='obs'>OBSERVED</span> scene <b style='color:"
                   + (L.scene === "normal" ? "#4FC489" : "#F3A83C") + "'>"
                   + L.scene.replace(/_/g, " ") + "</b> "
                   + Math.round((L.scene_p || 0) * 100) + "%" : "")
      + "<br>" + (L.ingested || 0) + " event(s) stored"
      + (L.age_s != null ? " \u00b7 " + L.age_s.toFixed(1) + "s ago" : "")
      + (L.temp ? " \u00b7 CPU " + L.temp : "")
      + "</div>";
  }

  refresh();
  setInterval(refresh, 4000);
  pollLive();
  setInterval(pollLive, 1000);
  setTimeout(baseStatus, 900);
  setInterval(baseStatus, 3000);
})();
