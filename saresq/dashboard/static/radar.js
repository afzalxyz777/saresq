/* SaResQ radar scope.
 *
 * A plan-position display over the real basemap. What it borrows from terminal
 * air-traffic radar, and why:
 *
 *   history trail   the last N scan positions as fading dots, so you can see
 *                   where the aircraft came from and how fast, at a glance
 *   coast rendering when a plot is missed the symbol goes hollow, the trail
 *                   goes dashed, the data block reads CST, and an uncertainty
 *                   circle grows at the rate dead reckoning actually degrades
 *   data block      callsign / altitude / groundspeed on a leader line, the
 *                   ARTS-and-later convention
 *   range rings     centred on the ground station, because comms range and
 *                   dispatch distance are the two numbers a coordinator needs
 *
 * The one thing it does NOT borrow is a rotating antenna sweep for its own
 * sake. Our sensor does not rotate -- it is a camera sweeping along a flight
 * line -- so the sweep here is tied to the actual scan period and the swath
 * drawn on the map is the real thermal footprint at the real altitude.
 *
 * Depends on: basemap.js, mapcore.js. Data: GET /api/radar.
 */
(function () {
  "use strict";
  var M = window.MapCore, sve = M.sve, NS = M.NS;

  var COL = {
    plat: "#3FCDEC", coast: "#F3A83C", gcs: "#A98BFF", cover: "#4FC489",
    HIGH: "#F46454", MEDIUM: "#F3A83C", LOW: "#6D838D",
    UP: "#4FC489", DEGRADED: "#F3A83C", DOWN: "#F46454"
  };
  var S = {
    snap: null, sel: null, err: null,
    show: { rings: true, plan: true, coverage: true, swath: true, trail: true, rf: true },
    lastFetch: 0, paused: false
  };
  var reduceMotion = window.matchMedia
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var host = document.getElementById("scope");
  var ov = document.getElementById("ov");
  var view = M.create({
    host: host, canvas: document.getElementById("cv"), svg: ov,
    // Opens framed on Segment A, not on the whole AOI: the operator's first
    // look should be the ground being searched right now.
    cam: { lat: 22.57320, lon: 88.36500, z: 18.4 },
    minZoom: 14, maxZoom: 20.5,
    onChange: draw,
    onHover: function (ll) {
      var el = null;
      document.getElementById("cursor").textContent =
        ll[0].toFixed(5) + "°N " + ll[1].toFixed(5) + "°E"
        + (el == null ? "" : "  ·  " + el.toFixed(1) + " m");
    }
  });

  // ---------------------------------------------------------------- helpers
  function esc(s) {
    return String(s == null ? "" : s).replace(/[<>&"]/g, function (c) {
      return { "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" }[c];
    });
  }
  function clock(s) {
    s = Math.max(0, Math.round(s));
    var m = Math.floor(s / 60);
    return m + ":" + String(s % 60).padStart(2, "0");
  }
  function bearing(a, b, c, d) {
    var y = (d - b) * Math.cos((a + c) / 2 * M.D2R), x = c - a;
    return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
  }
  function metresPerDegLat() { return 110574; }
  function metresPerDegLon(lat) { return 111320 * Math.cos(lat * M.D2R); }

  // ------------------------------------------------------- canvas: the scope
  var covCv = document.createElement("canvas"), covKey = null;
  function coverageRaster(sea) {
    var key = sea.grid.length + ":" + sea.cells_seen + ":" + sea.grid.charCodeAt(0);
    if (covKey === key) return covCv;
    covCv.width = sea.nx; covCv.height = sea.ny;
    var c = covCv.getContext("2d"), img = c.createImageData(sea.nx, sea.ny), d = img.data;
    for (var j = 0; j < sea.ny; j++) for (var i = 0; i < sea.nx; i++) {
      var v = sea.grid.charCodeAt(j * sea.nx + i) - 48;
      // Rows are stored south-to-north; canvas draws top-down.
      var o = ((sea.ny - 1 - j) * sea.nx + i) * 4;
      if (v <= 0) { d[o + 3] = 0; continue; }
      d[o] = 79; d[o + 1] = 196; d[o + 2] = 137;
      // Kept deliberately faint. Once the segment is fully swept every cell is
      // painted, and at the old 0.42 ceiling that was a solid green sheet over
      // the map the operator still needs to read.
      d[o + 3] = Math.round(Math.min(0.22, 0.07 + 0.05 * v) * 255);
    }
    c.putImageData(img, 0, 0); covKey = key;
    return covCv;
  }

  function drawRings(ctx, snap) {
    var g = view.P(snap.gcs.lat, snap.gcs.lon), mpp = view.mpp();
    var step = M.niceScale(mpp, 70, 240).m;
    ctx.save();
    ctx.strokeStyle = "rgba(169,139,255,.30)";
    ctx.fillStyle = "rgba(169,139,255,.75)";
    ctx.font = "600 9px ui-monospace, monospace";
    ctx.lineWidth = 1;
    for (var k = 1; k <= 6; k++) {
      var r = k * step / mpp;
      if (r > Math.hypot(view.vp.w, view.vp.h)) break;
      ctx.beginPath(); ctx.arc(g[0], g[1], r, 0, Math.PI * 2); ctx.stroke();
      var lbl = (k * step) >= 1000 ? (k * step / 1000).toFixed(1) + " km" : (k * step) + " m";
      ctx.fillText(lbl, g[0] + 4, g[1] - r - 4);
    }
    // Bearing rose: ticks every 10 degrees, labels every 30 -- the convention
    // on every PPI, and what turns "over there" into "bearing 070, 240 metres".
    var R = 6 * step / mpp;
    for (var b = 0; b < 360; b += 10) {
      var rad = (90 - b) * M.D2R, inner = R * (b % 30 === 0 ? 0.955 : 0.978);
      ctx.beginPath();
      ctx.moveTo(g[0] + Math.cos(rad) * inner, g[1] - Math.sin(rad) * inner);
      ctx.lineTo(g[0] + Math.cos(rad) * R, g[1] - Math.sin(rad) * R);
      ctx.stroke();
      if (b % 30 === 0) {
        ctx.save(); ctx.translate(g[0] + Math.cos(rad) * R * 0.92, g[1] - Math.sin(rad) * R * 0.92);
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(b).padStart(3, "0"), 0, 0); ctx.restore();
      }
    }
    ctx.restore();
  }

  /* The scan pulse. Not a rotating antenna -- we do not have one -- but a ring
   * that expands once per scan period, so the display has a heartbeat that
   * means "the picture just refreshed" rather than one that means nothing. */
  function drawScanPulse(ctx, snap) {
    if (reduceMotion || S.paused) return;
    var age = (Date.now() / 1000 - S.lastFetch);
    var phase = (age % snap.scan_period_s) / snap.scan_period_s;
    if (!snap.platform) return;
    var p = view.P(snap.platform.lat, snap.platform.lon);
    var maxR = Math.max(40, 90 / view.mpp());
    ctx.save();
    ctx.globalAlpha = (1 - phase) * 0.5;
    ctx.strokeStyle = snap.platform.status.CST ? COL.coast : COL.plat;
    ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(p[0], p[1], 8 + phase * maxR, 0, Math.PI * 2); ctx.stroke();
    ctx.restore();
  }

  function drawScope() {
    var ctx = view.ctx, snap = S.snap;
    view.drawBase({
      show: { roads: true, bldg: view.cam.z >= 16.4, d3: false },
      aoiBox: false, dimRoads: true
    });
    // Knock the map back so the surveillance symbology reads on top of it. A
    // scope where the basemap competes with the tracks is a pretty map, not a
    // usable display -- but knock it back too far and you have thrown away the
    // context that makes a position mean anything.
    ctx.save();
    ctx.fillStyle = "rgba(4,8,11,.24)";
    ctx.fillRect(0, 0, view.vp.w, view.vp.h);
    ctx.restore();
    if (!snap) return;

    var a = snap.aoi, nw = view.P(a.lat1, a.lon0), se = view.P(a.lat0, a.lon1);

    if (S.show.coverage && snap.search.grid) {
      ctx.save();
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(coverageRaster(snap.search), nw[0], nw[1], se[0] - nw[0], se[1] - nw[1]);
      ctx.restore();
    }

    // RF shadow footprints -- why the link drops where it drops.
    if (S.show.rf) {
      ctx.save();
      ctx.strokeStyle = "rgba(244,100,84,.55)";
      ctx.fillStyle = "rgba(244,100,84,.10)";
      ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
      ctx.font = "600 8.5px ui-monospace, monospace";
      snap.obstructions.forEach(function (o) {
        var p = view.P(o.lat1, o.lon0), q = view.P(o.lat0, o.lon1);
        ctx.fillRect(p[0], p[1], q[0] - p[0], q[1] - p[1]);
        ctx.strokeRect(p[0], p[1], q[0] - p[0], q[1] - p[1]);
        ctx.fillStyle = "rgba(244,100,84,.8)";
        ctx.fillText(o.top_m.toFixed(0) + " m", p[0] + 3, p[1] - 3);
        ctx.fillStyle = "rgba(244,100,84,.10)";
      });
      ctx.restore();
    }

    // Planned pattern, and the segment boundary.
    if (S.show.plan) {
      ctx.save();
      ctx.strokeStyle = "rgba(63,205,236,.26)";
      ctx.setLineDash([3, 4]); ctx.lineWidth = 1;
      ctx.beginPath();
      snap.waypoints.forEach(function (w, i) {
        var p = view.P(w[0], w[1]);
        i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
      });
      ctx.stroke();
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = "rgba(63,205,236,.45)";
      ctx.strokeRect(nw[0], nw[1], se[0] - nw[0], se[1] - nw[1]);
      ctx.font = "600 9.5px ui-monospace, monospace";
      ctx.fillStyle = "rgba(63,205,236,.7)";
      ctx.fillText("SEGMENT A", nw[0] + 5, nw[1] + 13);
      ctx.restore();
    }

    // Line of sight to the ground station, coloured by what the radio is doing.
    var pl = snap.platform;
    if (pl) {
      var g = view.P(snap.gcs.lat, snap.gcs.lon), p = view.P(pl.lat, pl.lon);
      ctx.save();
      ctx.strokeStyle = COL[snap.link.state] || COL.DOWN;
      ctx.globalAlpha = snap.link.state === "UP" ? .30 : .65;
      ctx.lineWidth = snap.link.state === "UP" ? 1 : 1.8;
      if (snap.link.state !== "UP") ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(g[0], g[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
      ctx.restore();

      // The thermal swath at the current altitude: the actual ground the
      // sensor sees this instant, not a decorative cone.
      if (S.show.swath && !pl.status.CST) {
        var halfW = (snap.search.swath_m / 2) / view.mpp();
        var halfL = (snap.search.swath_m * 0.32) / view.mpp();
        ctx.save();
        ctx.translate(p[0], p[1]);
        ctx.rotate(pl.course_deg * M.D2R);
        ctx.fillStyle = "rgba(63,205,236,.13)";
        ctx.strokeStyle = "rgba(63,205,236,.42)";
        ctx.lineWidth = 1;
        ctx.fillRect(-halfW, -halfL, halfW * 2, halfL * 2);
        ctx.strokeRect(-halfW, -halfL, halfW * 2, halfL * 2);
        ctx.restore();
      }
    }

    if (S.show.rings) drawRings(ctx, snap);
    drawScanPulse(ctx, snap);
  }

  // ------------------------------------------------------- svg: the symbology
  function text(x, y, s, at) {
    var o = { x: x, y: y, "font-family": "ui-monospace, monospace", "font-size": 11,
              fill: "#DCE7EC", "paint-order": "stroke fill", stroke: "#04080B",
              "stroke-width": 4, "stroke-linejoin": "round" };
    for (var k in at || {}) o[k] = at[k];
    var e = sve("text", o); e.textContent = s; return e;
  }

  /* Data block: callsign, altitude and groundspeed, on a leader line. The
   * third line only appears when the track is not healthy, so a clean picture
   * stays clean and anything abnormal is the only thing shouting.
   *
   * Decluttering follows the same rules a terminal scope uses, because the
   * problem is identical: on a phone, or with contacts a few metres apart,
   * every block lands on top of every other one and the display becomes
   * unreadable. So the leader line tries each of the four diagonals for a free
   * slot, and if none is free the block degrades to a LIMITED data block --
   * callsign only -- exactly as a controller's display does when tracks
   * converge. The symbol itself is never hidden; only its text.
   */
  var placed = [];
  function clears(r) {
    for (var i = 0; i < placed.length; i++) {
      var p = placed[i];
      if (r.x < p.x + p.w && r.x + r.w > p.x && r.y < p.y + p.h && r.y + r.h > p.y) return false;
    }
    return r.x > -4 && r.y > -4 && r.x + r.w < view.vp.w + 4 && r.y + r.h < view.vp.h + 4;
  }
  function scale() { return view.vp.w < 620 ? 0.84 : 1; }

  function dataBlock(g, x, y, lines, col) {
    var k = scale(), lead = 26 * k, f0 = 12 * k, f1 = 10.5 * k, lh = 12 * k;
    var rows = lines.filter(Boolean);
    function box(dx, dy, n) {
      var bx = x + lead * dx, by = y + lead * dy;
      var w = 0;
      for (var i = 0; i < n; i++) w = Math.max(w, rows[i].t.length * (i ? f1 : f0) * 0.62);
      // Left-going leaders need the block right-aligned or it covers the symbol.
      return { bx: dx < 0 ? bx - w : bx, by: by, w: w + 6, h: n * lh + 4,
               x: dx < 0 ? bx - w : bx, y: by - lh + 2 };
    }
    var dirs = [[1, -1], [1, 1], [-1, -1], [-1, 1]], pick = null;
    for (var d = 0; d < dirs.length && !pick; d++) {
      var b = box(dirs[d][0], dirs[d][1], rows.length);
      if (clears(b)) pick = { b: b, d: dirs[d], n: rows.length };
    }
    if (!pick) {                                  // fall back to a limited block
      for (var e = 0; e < dirs.length && !pick; e++) {
        var lb = box(dirs[e][0], dirs[e][1], 1);
        if (clears(lb)) pick = { b: lb, d: dirs[e], n: 1 };
      }
    }
    if (!pick) return;                            // symbol stands alone
    placed.push(pick.b);
    g.appendChild(sve("line", {
      x1: x, y1: y, x2: x + lead * pick.d[0], y2: y + lead * pick.d[1],
      stroke: col, "stroke-width": 1, "stroke-opacity": .7 }));
    for (var i = 0; i < pick.n; i++) {
      g.appendChild(text(pick.b.bx + 3, pick.b.by + i * lh - 1, rows[i].t, {
        "font-size": i === 0 ? f0 : f1,
        "font-weight": i === 0 ? 700 : 500,
        fill: rows[i].c || (i === 0 ? col : "#A6B9C2")
      }));
    }
  }

  function platformSymbol(g, x, y, course, col, coasting) {
    // MIL-STD-2525 friendly air: a dome, flat side down, frame solid when the
    // track is being updated and dashed when it is anticipated -- exactly the
    // present/anticipated distinction the standard uses.
    var k = scale();
    g.appendChild(sve("path", {
      d: "M -13 8 L -13 0 A 13 13 0 0 1 13 0 L 13 8 Z",
      transform: "translate(" + x + "," + y + ") scale(" + k + ")",
      fill: "#04080B", "fill-opacity": coasting ? .40 : .90,
      stroke: col, "stroke-width": 2.4 / k,
      "stroke-dasharray": coasting ? "4 3" : "none"
    }));
    var r = (course - 90) * M.D2R;
    g.appendChild(sve("line", {
      x1: x, y1: y, x2: x + Math.cos(r) * 11 * k, y2: y + Math.sin(r) * 11 * k,
      stroke: col, "stroke-width": 2, "stroke-opacity": .95
    }));
  }

  function contactSymbol(g, x, y, col, stale) {
    var k = scale();
    g.appendChild(sve("circle", { cx: x, cy: y, r: 10 * k, fill: "#04080B",
      "fill-opacity": stale ? .4 : .9, stroke: col, "stroke-width": 2,
      "stroke-dasharray": stale ? "3.5 3" : "none" }));
    g.appendChild(sve("g", { transform: "translate(" + x + "," + y + ") scale(" + k + ")" }, [
      sve("circle", { cx: 0, cy: -2.6, r: 1.9, fill: col }),
      sve("path", { d: "M -2.9 4.2 a 2.9 3.5 0 0 1 5.8 0 Z", fill: col })
    ]));
  }

  function drawTracks() {
    ov.innerHTML = "";
    var snap = S.snap;
    if (!snap) return;
    placed = [];

    // The aircraft's block is built first so it gets first claim on screen
    // space -- it is the one track that must never degrade to a limited block
    // -- but appended last so it draws on top of everything else.
    var pg = snap.platform ? platformGroup(snap.platform) : null;

    // ground station
    var g0 = view.P(snap.gcs.lat, snap.gcs.lon), gg = sve("g", {});
    gg.appendChild(sve("path", {
      d: "M -8 6 L 0 -8 L 8 6 Z", transform: "translate(" + g0[0] + "," + g0[1] + ")",
      fill: "#04080B", stroke: COL.gcs, "stroke-width": 1.8 }));
    gg.appendChild(text(g0[0] + 12, g0[1] + 4, "GCS",
      { "font-size": 9.5, "font-weight": 700, fill: COL.gcs }));
    ov.appendChild(gg);

    // contacts
    (snap.contacts || []).forEach(function (c) {
      var pay = c.payload || {}, cls = pay.class || "LOW", col = COL[cls] || COL.LOW;
      var p = view.P(c.lat, c.lon), stale = c.age_s > 20;
      var g = sve("g", { class: "hit", style: "cursor:pointer" });
      var rr = Math.max(4, c.sigma_m / view.mpp());
      g.appendChild(sve("circle", { cx: p[0], cy: p[1], r: rr, fill: col, "fill-opacity": .09,
        stroke: col, "stroke-opacity": .3, "stroke-width": .9, "stroke-dasharray": "2 3" }));
      if (S.sel === "C" + c.number)
        g.appendChild(sve("circle", { cx: p[0], cy: p[1], r: 18, fill: "none",
          stroke: col, "stroke-width": 1.3 }));
      contactSymbol(g, p[0], p[1], col, stale);
      dataBlock(g, p[0], p[1], [
        { t: pay.label || c.label },
        { t: cls + " " + (pay.p == null ? "—" : pay.p.toFixed(2))
             + " ×" + (pay.n_looks || 1), c: col },
        { t: c.state === "HELD" ? "HELD " + clock(c.age_s) : c.state,
          c: c.state === "HELD" ? "#6D838D" : "#A6B9C2" }
      ], col);
      g.addEventListener("click", function () {
        if (view.dragged()) return;
        S.sel = "C" + c.number; showDetail("contact", c); draw();
      });
      ov.appendChild(g);
    });

    // platform: the trail goes on before the symbol group, so the aircraft
    // sits on top of its own history rather than under it.
    var pl = snap.platform;
    if (!pl) return;
    var k = scale();

    if (S.show.trail && pl.history.length > 1) {
      var h = pl.history, n = h.length;
      for (var i = 1; i < n; i++) {
        var A = view.P(h[i - 1].lat, h[i - 1].lon), B = view.P(h[i].lat, h[i].lon);
        var age = (n - i) / n;
        ov.appendChild(sve("line", {
          x1: A[0], y1: A[1], x2: B[0], y2: B[1],
          stroke: h[i].coast ? COL.coast : COL.plat,
          "stroke-width": h[i].coast ? 2.2 : 1.9,
          "stroke-opacity": (0.20 + 0.75 * (1 - age)).toFixed(3),
          "stroke-dasharray": h[i].coast ? "4 3" : "none"
        }));
      }
      // History dots at every third scan, fading with age -- the trail an ATC
      // controller reads speed and turn rate off without thinking about it.
      for (var j = n - 1; j >= 0; j -= 3) {
        var q = view.P(h[j].lat, h[j].lon), f = (n - j) / n;
        ov.appendChild(sve("circle", {
          cx: q[0], cy: q[1], r: (h[j].coast ? 2.6 : 2.0) * k,
          fill: h[j].coast ? "none" : COL.plat,
          stroke: h[j].coast ? COL.coast : "none", "stroke-width": 1.1,
          "fill-opacity": (0.1 + 0.7 * (1 - f)).toFixed(3),
          "stroke-opacity": (0.2 + 0.7 * (1 - f)).toFixed(3)
        }));
      }
    }

    if (pg) ov.appendChild(pg);
  }

  function platformGroup(pl) {
    var coasting = !!pl.status.CST;
    var col = coasting ? COL.coast : COL.plat;
    var p = view.P(pl.lat, pl.lon);
    var pg = sve("g", { class: "hit", style: "cursor:pointer" });

    // Uncertainty circle. Only drawn while coasting, because that is the only
    // time it is telling you something you did not already know.
    if (coasting) {
      var r = Math.max(6, pl.sigma_m / view.mpp());
      pg.appendChild(sve("circle", { cx: p[0], cy: p[1], r: r, fill: COL.coast,
        "fill-opacity": .07, stroke: COL.coast, "stroke-opacity": .55,
        "stroke-width": 1.2, "stroke-dasharray": "5 4" }));
      pg.appendChild(text(p[0] + r + 4, p[1] + r - 2, "±" + pl.sigma_m.toFixed(0) + " m",
        { "font-size": 9.5 * scale(), fill: COL.coast }));
    }

    // Velocity leader: where the aircraft will be in 60 seconds. ATC draws
    // exactly this and calls it the vector line.
    if (pl.speed_ms > 0.4) {
      var lead = 60 * pl.speed_ms / view.mpp();
      var rad = (pl.course_deg - 90) * M.D2R;
      pg.appendChild(sve("line", {
        x1: p[0], y1: p[1],
        x2: p[0] + Math.cos(rad) * lead, y2: p[1] + Math.sin(rad) * lead,
        stroke: col, "stroke-width": 1.3, "stroke-opacity": .55,
        "stroke-dasharray": coasting ? "3 3" : "none" }));
    }

    platformSymbol(pg, p[0], p[1], pl.course_deg, col, coasting);
    dataBlock(pg, p[0], p[1], [
      { t: pl.label },
      { t: String(Math.round(pl.alt_m || 0)).padStart(3, "0") + "  "
           + pl.speed_ms.toFixed(1) + " m/s" },
      coasting ? { t: "CST " + clock(pl.age_s), c: COL.coast }
               : (pl.state === "TENTATIVE" ? { t: "TENT", c: COL.coast } : null)
    ], col);
    pg.addEventListener("click", function () {
      if (view.dragged()) return;
      S.sel = "P"; showDetail("platform", pl); draw();
    });
    return pg;
  }

  // ------------------------------------------------------------- html panels
  function renderPanels() {
    var snap = S.snap;
    var lk = document.getElementById("linkbox");
    if (!snap) {
      lk.innerHTML = '<div class="ps">' + esc(S.err || "connecting…") + "</div>";
      return;
    }
    var L = snap.link, st = L.state, col = COL[st];
    var span = Math.max(0, Math.min(1, (L.rssi_dbm + 120) / 80));
    lk.innerHTML =
      '<div class="lk-row"><span class="lk-dot" style="background:' + col + '"></span>'
      + '<b style="color:' + col + '">' + st + "</b>"
      + '<span class="ps">' + L.rssi_dbm.toFixed(0) + " dBm · " + clock(L.since_s) + "</span></div>"
      + '<div class="lk-bar"><i style="width:' + (span * 100).toFixed(0) + "%;background:" + col + '"></i>'
      + '<u style="left:' + ((L.down_dbm + 120) / 80 * 100).toFixed(0) + '%"></u>'
      + '<u style="left:' + ((L.up_dbm + 120) / 80 * 100).toFixed(0) + '%"></u></div>'
      + '<div class="lk-grid">'
      + kv("Range", L.range_m == null ? "\u2014" : L.range_m.toFixed(0) + " m")
      + kv("Free space", "−" + (L.fspl_db == null ? "\u2014" : L.fspl_db.toFixed(0)) + " dB")
      + kv("Structure", L.structure_db > 0 ? "−" + (L.structure_db == null ? "\u2014" : L.structure_db.toFixed(0)) + " dB" : "—")
      + kv("Diffraction", L.diffraction_db > 0 ? "−" + (L.diffraction_db == null ? "\u2014" : L.diffraction_db.toFixed(0)) + " dB" : "—")
      + kv("Blocker", L.blocker || "clear")
      + kv("GPS", L.gps_denied ? "<b style='color:" + COL.DOWN + "'>DENIED</b>" : "nominal")
      + "</div>"
      + '<div class="lk-q ' + (L.queued ? "hot" : "") + '">'
      + "<b>" + L.queued + "</b> alert(s) held on aircraft"
      + (L.queued ? " · " + L.queued_bytes + " B" : "")
      + "<span class='ps'>" + L.delivered + " delivered · " + L.delivered_bytes + " B</span></div>";

    // track list -- a controller's track table
    var rows = [];
    if (snap.platform) rows.push(trackRow(snap.platform, "platform"));
    (snap.contacts || []).slice().sort(function (a, b) {
      return (b.payload.p || 0) - (a.payload.p || 0);
    }).forEach(function (c) { rows.push(trackRow(c, "contact")); });
    document.getElementById("tracks").innerHTML = rows.join("")
      || '<div class="empty"><div class="why">no tracks</div></div>';
    Array.prototype.forEach.call(document.querySelectorAll("[data-trk]"), function (el) {
      el.addEventListener("click", function () {
        var id = el.getAttribute("data-trk");
        S.sel = id;
        var t = id === "P" ? snap.platform
              : snap.contacts.filter(function (c) { return "C" + c.number === id; })[0];
        if (t) {
          view.panTo(t.lat, t.lon);
          showDetail(id === "P" ? "platform" : "contact", t);
        }
        draw();
      });
    });

    var se = snap.search;
    document.getElementById("searchbox").innerHTML =
      '<div class="lk-grid">'
      + kv("Sweeps", se.sweeps.toFixed(2))
      + kv("Coverage C", se.coverage_c.toFixed(2))
      + kv("Cum. POD", (se.pod * 100).toFixed(1) + "%")
      + kv("Area seen", (se.fraction * 100).toFixed(0) + "%")
      + kv("Swath", se.swath_m.toFixed(1) + " m")
      + kv("Leg spacing", se.leg_spacing_m.toFixed(1) + " m")
      + "</div>"
      + '<div class="pod"><i style="width:' + (se.pod * 100).toFixed(1) + '%"></i></div>'
      + '<div class="ps" style="margin-top:6px">POD = 1 − e<sup>−C</sup>, C = W·L/A with '
      + "W assumed " + se.sweep_width_m.toFixed(0) + " m until calibration.</div>";

    document.getElementById("events").innerHTML = (snap.events || []).map(function (e) {
      return '<div class="ev ev-' + e.sev + '"><span class="ev-t">T+' + clock(e.t) + "</span>"
        + '<span class="ev-k">' + esc(e.kind) + "</span>"
        + '<span class="ev-x">' + esc(e.text) + "</span></div>";
    }).join("") || '<div class="ps" style="padding:10px 12px">no events yet</div>';

    document.getElementById("scanstat").textContent =
      "SCAN " + snap.scan + " · " + snap.scan_period_s.toFixed(1) + " s · T+" + clock(snap.t);
    var fl = snap.faults || {};
    Array.prototype.forEach.call(document.querySelectorAll("[data-fault]"), function (b) {
      var k = b.getAttribute("data-fault"), on = fl[k] != null;
      b.setAttribute("aria-pressed", String(on));
      b.textContent = on ? b.dataset.on + " " + Math.round(fl[k]) + "s" : b.dataset.off;
    });
  }

  function kv(k, v) {
    return '<div class="kv"><span class="lbl">' + esc(k) + "</span><span>" + v + "</span></div>";
  }
  function trackRow(t, role) {
    var pay = t.payload || {};
    var col = role === "platform"
      ? (t.status.CST ? COL.coast : COL.plat)
      : (COL[pay.class] || COL.LOW);
    var id = role === "platform" ? "P" : "C" + t.number;
    var right = role === "platform"
      ? t.speed_ms.toFixed(1) + " m/s"
      : (pay.p == null ? "—" : pay.p.toFixed(2));
    return '<div class="trk' + (S.sel === id ? " on" : "") + '" data-trk="' + id + '">'
      + '<span class="trk-n" style="color:' + col + '">' + esc(t.label) + "</span>"
      + '<span class="trk-s st-' + t.state + '">' + t.state + "</span>"
      + '<span class="trk-a">' + (t.age_s > 3 ? clock(t.age_s) : "live") + "</span>"
      + '<span class="trk-v">' + right + "</span></div>";
  }

  function showDetail(kind, t) {
    var box = document.getElementById("detail"), pay = t.payload || {};
    var b = bearing(S.snap.gcs.lat, S.snap.gcs.lon, t.lat, t.lon);
    var dx = (t.lon - S.snap.gcs.lon) * metresPerDegLon(t.lat);
    var dy = (t.lat - S.snap.gcs.lat) * metresPerDegLat();
    var rng = Math.hypot(dx, dy);
    var rows = [
      ["Track number", t.number],
      ["State", t.state + (t.status.CST ? "  (CST)" : "")],
      ["Position", t.lat.toFixed(5) + "°N " + t.lon.toFixed(5) + "°E"],
      ["From GCS", "bearing " + String(Math.round(b)).padStart(3, "0")
                   + "° · " + rng.toFixed(0) + " m"],
      ["Uncertainty", "±" + t.sigma_m.toFixed(1) + " m (1σ)"],
      ["Last plot", t.age_s.toFixed(1) + " s ago"],
      ["Updates", t.hits + " plot(s) · quality " + (t.quality * 100).toFixed(0) + "%"]
    ];
    if (kind === "platform") {
      rows.push(["Altitude", (t.alt_m == null ? "—" : t.alt_m.toFixed(1) + " m AGL")]);
      rows.push(["Velocity", t.speed_ms.toFixed(2) + " m/s · course "
                             + String(Math.round(t.course_deg)).padStart(3, "0") + "°"]);
      rows.push(["Movement", t.trans + " · " + t.long + " · " + t.vert]);
    } else {
      rows.push(["Likelihood", (pay.p == null ? "—" : pay.p.toFixed(3))
                               + " — " + (pay.likelihood || "")]);
      rows.push(["Class", (pay.class || "—") + " · machine says " + (pay.decision || "—")]);
      rows.push(["Looks", (pay.n_looks || 0) + " independent pass(es)"]);
      rows.push(["Reported over", (pay.wire_bytes || 0) + " B Tier-1 packet"]);
    }
    var bits = t.status;
    rows.push(["ASTERIX I062/080", Object.keys(bits).filter(function (k) { return bits[k]; })
      .join(" ") || "—"]);
    box.innerHTML = '<div class="ph"><span class="pt">' + esc(t.label) + "</span>"
      + '<button class="x" id="dclose" aria-label="Close">×</button></div>'
      + '<div class="pb"><table class="dt"><tbody>'
      + rows.map(function (r) {
          return "<tr><td class='k'>" + esc(r[0]) + "</td><td>" + esc(r[1]) + "</td></tr>";
        }).join("")
      + "</tbody></table>"
      + (kind === "contact" && pay.target_id != null
          ? '<a class="go" href="/review">open T-' + pay.target_id + " in review queue →</a>"
          : "")
      + "</div>";
    box.hidden = false;
    document.getElementById("dclose").addEventListener("click", function () {
      box.hidden = true; S.sel = null; draw();
    });
  }

  // ------------------------------------------------------------------- loop
  var raf = false;
  function draw() {
    if (raf) return;
    raf = true;
    requestAnimationFrame(function () {
      raf = false;
      drawScope(); drawTracks();
      var sc = M.niceScale(view.mpp());
      document.getElementById("scaleBar").style.width = sc.px.toFixed(1) + "px";
      document.getElementById("scaleLab").textContent = sc.label;
    });
  }

  function poll() {
    if (S.paused) return;
    fetch("/api/radar").then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (d) {
      S.snap = d; S.err = null; S.lastFetch = Date.now() / 1000;
      if (!S.framed) {
        // Frame the segment plus the ground station, once. A fixed zoom that
        // looks right on a laptop shows a corner of the search area on a phone.
        var a = d.aoi;
        S.framed = true;
        // 0.6 rather than filling the frame: the segment is only 240 m across,
        // and a scope cropped to it loses the streets the dispatcher routes
        // down. Leave a ring of city around the search area.
        view.fitBounds(Math.min(a.lat0, d.gcs.lat), Math.min(a.lon0, d.gcs.lon),
                       Math.max(a.lat1, d.gcs.lat), Math.max(a.lon1, d.gcs.lon), 0.60);
      }
      renderPanels(); draw();
    }).catch(function (e) {
      S.err = "ground station unreachable — " + e.message;
      renderPanels();
    });
  }

  // Animate between polls so the scan pulse and the coast clock move smoothly
  // rather than stepping once every two seconds.
  function tick() {
    if (!reduceMotion && !S.paused && S.snap) draw();
    requestAnimationFrame(tick);
  }

  // ----------------------------------------------------------- controls
  document.getElementById("zin").addEventListener("click", function () { view.zoom(view.cam.z + 1); });
  document.getElementById("zout").addEventListener("click", function () { view.zoom(view.cam.z - 1); });
  document.getElementById("centre").addEventListener("click", function () {
    if (S.snap && S.snap.platform) view.panTo(S.snap.platform.lat, S.snap.platform.lon);
  });
  document.querySelectorAll("[data-show]").forEach(function (b) {
    b.addEventListener("click", function () {
      var k = b.getAttribute("data-show");
      S.show[k] = !S.show[k];
      b.setAttribute("aria-pressed", String(S.show[k]));
      layerCount();
      draw();
    });
  });

  // ---- layers panel ----
  // On a wide screen the chips sit inline and the button is hidden by CSS; on
  // a phone the same buttons become a proper 44 px-row menu behind it.
  var lyrBtn = document.getElementById("lyrBtn"), lyrPanel = document.getElementById("lyrPanel");
  function narrow() { return window.matchMedia("(max-width:900px)").matches; }
  function layerCount() {
    var on = 0;
    for (var k in S.show) if (S.show[k]) on++;
    document.getElementById("lyrCount").textContent = narrow() ? "(" + on + ")" : "";
  }
  function setPanel(open) {
    lyrPanel.hidden = narrow() ? !open : false;
    lyrBtn.setAttribute("aria-expanded", String(!!open && narrow()));
  }
  lyrBtn.addEventListener("click", function (e) {
    e.stopPropagation();
    setPanel(lyrPanel.hidden);
  });
  // Tapping the map should dismiss the menu, but dragging the map should not
  // count as a tap.
  document.getElementById("scope").addEventListener("pointerup", function (e) {
    if (!lyrPanel.contains(e.target) && e.target !== lyrBtn && !view.dragged()) setPanel(false);
  });
  window.addEventListener("resize", function () { setPanel(false); layerCount(); });
  setPanel(false);
  layerCount();
  document.querySelectorAll("[data-fault]").forEach(function (b) {
    b.addEventListener("click", function () {
      fetch("/api/radar/fault", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fault: b.getAttribute("data-fault"), seconds: 30 })
      }).then(poll);
    });
  });
  document.addEventListener("visibilitychange", function () {
    S.paused = document.hidden;
    if (!S.paused) poll();
  });

  poll();
  setInterval(poll, 2000);
  tick();
})();
