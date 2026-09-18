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
    // Keep the aircraft centred. On by default, because a map that opens
    // somewhere other than the payload is a map the operator has to go and
    // find the payload on -- and indoors, where there is never a GPS fix, the
    // old one-shot pan was keyed on L.lat and so never fired at all: the
    // camera just sat on its hardcoded Kolkata default while the drone sat in
    // the room. Any pan, pinch or wheel hands control back to the operator,
    // which is what every map does and what they will expect.
    follow: true,
    // Frame the contacts ONCE, when the first ones arrive. Not on every poll:
    // the gate fires and clears several times a minute and a map that
    // re-zoomed each time would be unusable. After that the operator owns the
    // camera and gets a button.
    contactsFitted: false
  };

  var host = document.getElementById("maparea");
  var ov = document.getElementById("ov");
  var view = M.create({
    host: host,
    canvas: document.getElementById("cv"),
    svg: ov,
    cam: { lat: 22.57323, lon: 88.36497, z: 17.2 },
    // Above the 19.5 the shared default allows. At 20 m survey altitude
    // contacts sit metres apart and 19.5 frames them fine, but on a bench the
    // payload is three metres up and its contacts are DECIMETRES apart -- at
    // 19.5 they are half a pixel apart and pile onto the aircraft glyph.
    // tiles.js already clamps the tile request to each source's own max and
    // upscales beyond it, so this costs sharpness in the basemap and nothing
    // else.
    maxZoom: 22.5,
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
      lat = L.origin.lat; lon = L.origin.lon;
      src = L.origin.source === "browser" ? "browser" : "manual";
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
    // A browser fix reports its own accuracy, and Wi-Fi trilateration is
    // usually tens of metres. Drawing the circle it actually claims is the
    // honest thing; a fixed small ring would overstate it.
    var errM = src === "gps" ? Math.max(2.5, (L.hdop || 1) * 2.5)
             : (src === "browser" ? Math.max(15, (L.origin.accuracy_m || 40)) : 25);
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
      src === "gps" ? (L.sats || 0) + " sats \u00b7 HDOP " + (L.hdop == null ? "\u2014" : L.hdop.toFixed(1))
      : src === "browser" ? "DEVICE LOCATION \u00b7 ground station, \u00b1"
                            + Math.round(L.origin.accuracy_m || 40) + " m"
      : "MANUAL DATUM \u00b7 not a GPS fix",
      { "font-size": 9.5, fill: src === "gps" ? "#6D838D" : "#F3A83C" }));
    ov.appendChild(g);
  }

  /* LIVE CONTACTS -- where the gate's blobs are on the ground, right now.
     Deliberately drawn unlike a stored target: a target has been through the
     ledger and carries a fused probability over several passes, a contact is
     one frame old and vanishes with the blob. Same picture, different claim.

     A dashed pin with a range spoke back to the aircraft, because RANGE is
     measured (one angle, one height) while BEARING is not -- this payload has
     no magnetometer. The spoke says "this far from the drone, direction
     unsurveyed", which is the true statement. */
  function drawContacts() {
    var L = S.live;
    if (!L || !L.connected) return;
    var cs = L.contacts || [];
    if (!cs.length) return;
    // Anchor: wherever the aircraft glyph itself was drawn.
    var alat = L.lat, alon = L.lon;
    if (alat == null && L.origin) { alat = L.origin.lat; alon = L.origin.lon; }
    if (alat == null) return;
    var a = view.P(alat, alon);

    // Screen positions first, so labels can be stacked when two contacts land
    // within a few pixels of each other -- which is the NORMAL case indoors,
    // where two people a metre apart are a metre apart on the ground too.
    var pts = cs.map(function (c) { return view.P(c.lat, c.lon); });
    var lanes = pts.map(function (q, i) {
      var n = 0;
      for (var j = 0; j < i; j++)
        if (Math.hypot(q[0] - pts[j][0], q[1] - pts[j][1]) < 46) n++;
      return n;
    });

    cs.forEach(function (c, ci) {
      var s2 = pts[ci], lane = lanes[ci] * 32;
      var brgGuess = (c.assumed || []).indexOf("heading") >= 0;
      // Colour by what the payload concluded, not by the blob alone: a
      // contact the whole stack calls LIVE is not the same find as a warm
      // patch the gate has not corroborated.
      var col = (L.verdict === "LIVE_PERSON" || L.verdict === "LIVE_BODY") ? "#F46454"
              : (L.verdict === "PERSON" || L.verdict === "BODY_HEAT") ? "#F3A83C"
              : "#3FCDEC";
      var g = sve("g", {});

      // With no compass the contact is not AT a bearing -- it is somewhere on
      // a circle of this radius about the aircraft. Drawing that locus is the
      // true statement; drawing only a pin would assert a direction nothing
      // measured. The pin still goes on the circle so the operator has
      // something to click and read, but the circle is what carries the claim.
      if (brgGuess) {
        var lr = (c.range_m || 0) / view.mpp();
        if (lr > 3) g.appendChild(sve("circle", { cx: a[0], cy: a[1], r: lr,
          fill: "none", stroke: col, "stroke-opacity": .30,
          "stroke-width": 1.1, "stroke-dasharray": "5 6" }));
      }

      // range spoke
      g.appendChild(sve("path", {
        d: "M " + a[0].toFixed(1) + " " + a[1].toFixed(1)
         + " L " + s2[0].toFixed(1) + " " + s2[1].toFixed(1),
        stroke: col, "stroke-width": 1.1, "stroke-opacity": .45,
        "stroke-dasharray": "2 4", fill: "none" }));

      // How well the contact is known RELATIVE to the aircraft. Not the
      // absolute figure: the aircraft's own ring already draws that, and a
      // datum error moves the drone and the contact together, so drawing it
      // twice would show a 25 m circle around something sitting 0.2 m away.
      var rr = Math.max(4, (c.rel_err_m || 1.5) / view.mpp());
      g.appendChild(sve("circle", { cx: s2[0], cy: s2[1], r: rr,
        fill: col, "fill-opacity": .08, stroke: col, "stroke-opacity": .40,
        "stroke-width": 1, "stroke-dasharray": "3 4" }));

      // person glyph -- head and shoulders, the same mark the target pin
      // uses, so the two read as the same KIND of thing at a glance...
      g.appendChild(sve("circle", { cx: s2[0], cy: s2[1], r: 9.5,
        fill: "#091015", "fill-opacity": .9, stroke: col,
        "stroke-width": 1.8,
        // ...but dashed, because this one has not been through the ledger.
        "stroke-dasharray": "3.5 2.5" }));
      g.appendChild(sve("circle", { cx: s2[0], cy: s2[1] - 2.3, r: 1.8, fill: col }));
      g.appendChild(sve("path", {
        d: "M " + (s2[0] - 2.7) + " " + (s2[1] + 3.9)
         + " a 2.7 3.2 0 0 1 5.4 0 Z", fill: col }));

      if (lane)
        g.appendChild(sve("path", {
          d: "M " + (s2[0] + 10) + " " + s2[1]
           + " L " + (s2[0] + 11.5) + " " + (s2[1] + lane - 4),
          stroke: col, "stroke-opacity": .5, "stroke-width": .9, fill: "none" }));
      g.appendChild(label(s2[0] + 13, s2[1] - 1 + lane,
        "LIVE \u00b7 +" + (c.z == null ? "?" : c.z.toFixed(1)) + "\u03c3",
        { "font-weight": 700, "font-size": 10.5, fill: col }));
      g.appendChild(label(s2[0] + 13, s2[1] + 10 + lane,
        (c.T == null ? "\u2014" : c.T.toFixed(1)) + "\u00b0C \u00b7 "
        + (c.range_m == null ? "\u2014" : c.range_m.toFixed(1)) + " m"
        + (brgGuess ? "" : " \u00b7 " + Math.round(c.bearing_deg) + "\u00b0"),
        { "font-size": 9.5, fill: "#6D838D" }));
      if (brgGuess)
        g.appendChild(label(s2[0] + 13, s2[1] + 20 + lane,
          "ON THIS RING \u00b7 bearing unmeasured",
          { "font-size": 8.5, fill: "#F3A83C", "font-weight": 600 }));
      ov.appendChild(g);
    });
  }

  /* Where the aircraft glyph is actually drawn -- a GPS fix if there is one,
     otherwise the operator's datum. The map follows the DRAWN position rather
     than the fix, because on a bench there is no fix and the drawn position is
     the only one that exists. */
  function payloadLL() {
    var L = S.live;
    if (!L || !L.configured) return null;
    if (L.lat != null && L.lon != null) return [L.lat, L.lon];
    if (L.origin) return [L.origin.lat, L.origin.lon];
    return null;
  }
  function centreOnPayload() {
    var ll = payloadLL();
    if (!ll) return false;
    // Only actually move when it differs. A stationary payload would
    // otherwise re-pan and redraw the whole basemap once a second for the
    // length of the mission.
    if (Math.abs(view.cam.lat - ll[0]) < 1e-7 &&
        Math.abs(view.cam.lon - ll[1]) < 1e-7) return true;
    view.panTo(ll[0], ll[1]);
    return true;
  }

  /* Any deliberate camera gesture hands control back. Attached here rather
     than in mapcore because following is this page's idea, not the shared
     view's -- the radar scope has its own. These listeners run after
     mapcore's, so drag.moved is already set by the time dragged() is read. */
  (function () {
    var cvEl = document.getElementById("cv");
    if (!cvEl) return;
    function release() {
      if (!S.follow) return;
      S.follow = false;
      syncFollowBtn();
    }
    cvEl.addEventListener("pointermove", function () {
      if (view.dragged()) release();
    });
    cvEl.addEventListener("wheel", release, { passive: true });
  })();

  function syncFollowBtn() {
    if (!fitBtn) return;
    fitBtn.setAttribute("aria-pressed", String(!!S.follow));
    fitBtn.title = S.follow
      ? "Following the payload \u2014 drag the map to take over"
      : "Recentre on the payload and follow it";
  }

  /* Frame the aircraft and everything the gate is currently looking at.
     Padded to a floor of ~8 m across so a single contact 20 cm from the
     payload does not zoom the map to a scale where the basemap is one
     upscaled pixel and nothing around it is recognisable. */
  function fitContacts() {
    var L = S.live;
    if (!L) return false;
    var alat = L.lat, alon = L.lon;
    if (alat == null && L.origin) { alat = L.origin.lat; alon = L.origin.lon; }
    if (alat == null) return false;
    var cs = L.contacts || [];
    var la = [alat], lo = [alon];
    cs.forEach(function (c) { la.push(c.lat); lo.push(c.lon); });
    var mLat = 8 / 111320, mLon = mLat / Math.cos(alat * Math.PI / 180);
    var lat0 = Math.min.apply(null, la), lat1 = Math.max.apply(null, la);
    var lon0 = Math.min.apply(null, lo), lon1 = Math.max.apply(null, lo);
    if (lat1 - lat0 < mLat) { var cy2 = (lat0 + lat1) / 2; lat0 = cy2 - mLat / 2; lat1 = cy2 + mLat / 2; }
    if (lon1 - lon0 < mLon) { var cx2 = (lon0 + lon1) / 2; lon0 = cx2 - mLon / 2; lon1 = cx2 + mLon / 2; }
    view.fitBounds(lat0, lon0, lat1, lon1, 0.7);
    return true;
  }
  var fitBtn = document.getElementById("fitBtn");
  if (fitBtn) fitBtn.addEventListener("click", function () {
    // One button, one meaning: "put me back on the payload". It re-engages
    // following as well as reframing, because an operator who pressed it and
    // then watched the aircraft drift back out of view would reasonably call
    // that broken.
    S.follow = true;
    syncFollowBtn();
    if (!fitContacts()) centreOnPayload();
  });

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
    drawContacts();
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

  /* Height above ground. Debounced: the slider fires on every pixel of drag
     and the projection is server-side, so posting each one would put a few
     hundred requests behind one gesture. */
  var aglR = document.getElementById("aglR"),
      aglVal = document.getElementById("aglVal"),
      aglNote = document.getElementById("aglNote"),
      aglT = null;
  function aglLabel(v, assumed) {
    if (aglVal) aglVal.textContent = v + " m";
    if (!aglNote) return;
    aglNote.innerHTML = assumed
      ? "<b style='color:#F3A83C'>assumed</b> \u2014 survey altitude. Drag to the "
        + "real height and every live contact tightens up."
      : "<b style='color:#4FC489'>operator set</b> \u2014 contacts are projected "
        + "from this height.";
  }
  if (aglR) {
    aglR.addEventListener("input", function () {
      var v = parseFloat(aglR.value);
      aglLabel(v, false);
      clearTimeout(aglT);
      aglT = setTimeout(function () {
        // Cleared BEFORE the request, not after: pollLive() checks this flag
        // to decide whether the operator is mid-gesture, and a handle that is
        // never cleared would freeze the slider at its first value for the
        // rest of the session.
        aglT = null;
        fetch("/api/agl", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ agl_m: v })
        }).then(function () { pollLive(); }).catch(function () {});
      }, 180);
    });
  }

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
      : (t && t.drawn
          ? (t.up && t.up >= 4
              // Past the source's own maximum zoom the basemap is one tile
              // magnified, which reads as a blank map unless it says so.
              ? "z" + t.z + " \u00b7 upscaled " + Math.round(t.up) + "\u00d7"
              : "z" + t.z + " \u00b7 " + h.ok + " tiles")
          : "offline geometry");
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
      .then(function () {
        // The operator just said where the payload is. Go there -- but only
        // recentre, never re-zoom: they placed that datum at the scale they
        // were looking at, and yanking the zoom out from under them would
        // lose the spot they just picked.
        S.follow = true;
        syncFollowBtn();
        pollLive();
      })
      .catch(function () {});
  });

  /* Browser geolocation as a third position source.
     Chromium and Safari resolve this from surrounding Wi-Fi networks, which
     indoors is typically good to tens of metres where a NEO-6M gets nothing at
     all. It answers "roughly where is this operation happening" so a demo has
     a real place on a real map -- but it is the LAPTOP's position, so it is
     posted with source=browser and never allowed to render like a GPS fix. */
  var geoAsked = false;
  function useBrowserLocation(manual) {
    if (!navigator.geolocation) return;
    var btn = document.getElementById("geoBtn");
    if (btn) { btn.disabled = true; btn.textContent = "locating\u2026"; }
    navigator.geolocation.getCurrentPosition(function (pos) {
      fetch("/api/origin", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat: pos.coords.latitude, lon: pos.coords.longitude,
          source: "browser", accuracy_m: pos.coords.accuracy
        })
      }).then(function () {
        S.follow = true;                    // recentre on the new datum
        syncFollowBtn();
        pollLive();
        if (btn) { btn.disabled = false; btn.textContent = "Use this device"; }
      });
    }, function (err) {
      if (btn) {
        btn.disabled = false;
        // Permission is the usual cause, and over plain http on a LAN address
        // the API is not offered at all -- say which rather than "failed".
        btn.textContent = err.code === 1 ? "permission denied"
                        : (window.isSecureContext ? "unavailable" : "needs https");
      }
      if (manual) console.warn("geolocation:", err.message);
    }, { enableHighAccuracy: false, timeout: 8000, maximumAge: 60000 });
  }
  var gb = document.getElementById("geoBtn");
  if (gb) gb.addEventListener("click", function () { useBrowserLocation(true); });

  // ---- live payload ----
  /* Mission clock. The mission starts when the PAYLOAD starts, not when the
     dashboard did, so this is the age of the aircraft's own session. */
  function mmss(sec) {
    if (sec == null) return "\u2014";
    var m = Math.floor(sec / 60), s2 = Math.floor(sec % 60);
    return m >= 60 ? Math.floor(m / 60) + "h" + (m % 60) + "m"
                   : m + "m" + (s2 < 10 ? "0" : "") + s2 + "s";
  }
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
      }
      // Centre on the payload wherever it is drawn from -- fix or datum --
      // rather than only when a fix exists.
      if (S.follow) centreOnPayload();
      // One automatic attempt, only when there is genuinely nothing else: a
      // real GPS fix always wins, and an operator-set datum is not overridden.
      if (!geoAsked && L.configured && L.connected && !L.fix && !L.origin) {
        geoAsked = true;
        useBrowserLocation(false);
      }
      // Only when the operator is not mid-drag, or the poll would fight them.
      if (aglR && document.activeElement !== aglR && aglT === null) {
        var srv = L.agl_m == null ? 20 : L.agl_m;
        if (parseFloat(aglR.value) !== srv) aglR.value = srv;
        aglLabel(srv, L.agl_assumed !== false);
      }
      if (!S.contactsFitted && (L.contacts || []).length) {
        S.contactsFitted = fitContacts();
      }
      // Enabled whenever there is anything to centre ON, not only when the
      // gate is firing -- its job is the payload first and the contacts second.
      if (fitBtn) fitBtn.disabled = !payloadLL();
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
      : (L.origin
          ? "<span style='color:#F3A83C'>"
            + (L.origin.source === "browser"
                ? "device location \u00b7 \u00b1" + Math.round(L.origin.accuracy_m || 40) + " m"
                : "manual datum")
            + " \u00b7 no GPS fix</span>"
          : "<span style='color:#F46454'>no position \u2014 shift-click, or use this device</span>");
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
      // The gate is plainly firing but there is nowhere to put the result. Say
      // that, rather than showing an empty map next to a panel reporting four
      // sigma -- an operator reading those two together concludes the map is
      // broken, and the next thing they stop trusting is the gate.
      + (L.fired && !(L.contacts && L.contacts.length) && L.lat == null && !L.origin
          ? "<br><span style='color:#F3A83C'><b>" + (L.blobs || 0)
            + " blob(s) held</b> \u00b7 no datum, so nothing can be placed"
            + "<br>shift-click the map or use this device</span>"
          : "")
      + (L.contacts && L.contacts.length
          ? "<br><b style='color:#F46454'>" + L.contacts.length
            + " live contact(s)</b> \u00b7 "
            + L.contacts.map(function (c) { return c.range_m.toFixed(1) + " m"; }).join(", ")
            + " from the aircraft"
            + "<br><span style='color:var(--muted)'>range measured \u00b7 "
            + "bearing assumed (no compass) \u00b7 AGL "
            + (L.agl_m == null ? "20 m assumed" : L.agl_m.toFixed(0) + " m set")
            + "</span>"
          : "")
      + "<br>" + (L.ingested || 0) + " event(s) this mission"
      + (L.session ? " \u00b7 " + mmss(L.session_age_s) : "")
      + (L.age_s != null ? " \u00b7 " + L.age_s.toFixed(1) + "s ago" : "")
      + (L.temp ? " \u00b7 CPU " + L.temp : "")
      + "</div>";
  }

  syncFollowBtn();
  refresh();
  setInterval(refresh, 4000);
  pollLive();
  setInterval(pollLive, 1000);
  setTimeout(baseStatus, 900);
  setInterval(baseStatus, 3000);
})();
