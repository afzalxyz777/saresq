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
    level: 12.0,
    profile: "emrg",
    show: { flood: true, roads: true, bldg: true, d3: false },
    targets: [], hazards: [], sel: null, fitted: false
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
      var el = M.demAt(ll[0], ll[1]);
      var d = el == null ? null : Math.max(0, S.level - el);
      document.getElementById("readout").innerHTML =
        ll[0].toFixed(5) + "°N " + ll[1].toFixed(5) + "°E"
        + (el == null ? " · outside AOI"
                      : " · elev " + el.toFixed(1) + " m"
                        + (d > 0 ? " · depth " + d.toFixed(1) + " m" : " · dry"));
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
      var e = M.demAt(t.lat, t.lon), depth = e == null ? null : Math.max(0, S.level - e);
      var g = sve("g", { style: "cursor:pointer", class: "hit" });
      if (t.pos_err_m)
        g.appendChild(sve("circle", { cx: s[0], cy: s[1], r: Math.max(3, t.pos_err_m / view.mpp()),
          fill: col, "fill-opacity": .10, stroke: col, "stroke-opacity": .35,
          "stroke-width": .9, "stroke-dasharray": "2 3" }));
      if (depth > 0)
        g.appendChild(sve("circle", { cx: s[0], cy: s[1], r: r + 5, fill: "none",
          stroke: "#F46454", "stroke-width": 1.6 }));
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
        (t.class || "?") + " P" + (t.p_final == null ? "—" : t.p_final.toFixed(2))
        + (depth > 0 ? "  water " + depth.toFixed(1) + "m" : ""),
        { "font-size": 9.5, fill: depth > 0 ? "#F46454" : "#6D838D" }));
      g.addEventListener("click", function () {
        if (view.dragged()) return;
        S.sel = t.target_id; showInfo(t, depth); draw();
      });
      ov.appendChild(g);
    });
  }

  var raf = false;
  function draw() {
    if (raf) return;
    raf = true;
    requestAnimationFrame(function () {
      raf = false;
      view.drawBase({ level: S.level, show: S.show, profile: S.profile });
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
    var R = M.classifyRoads(S.level, S.profile);
    document.getElementById("cutStat").textContent =
      Math.round(R.cutFrac * 100) + "% of " + (R.tot / 1000).toFixed(1) + " km cut";
  }
  function showInfo(t, depth) {
    document.getElementById("info").innerHTML =
      "<div class='lbl'>Target T-" + String(t.target_id).padStart(3, "0") + "</div>"
      + "<div class='ps' style='margin-top:4px;line-height:1.8'>"
      + (t.class || "?") + " · P " + (t.p_final == null ? "—" : t.p_final.toFixed(2))
      + " · " + (t.decision || "—") + "<br>"
      + t.lat.toFixed(5) + "°N " + t.lon.toFixed(5) + "°E<br>"
      + (depth > 0 ? "<span style='color:#F46454'>in " + depth.toFixed(2) + " m of water</span>"
                   : "dry ground")
      + " · CEP " + (t.pos_err_m || 0).toFixed(1) + " m<br>"
      + "<a href='/review'>open in review queue</a></div>";
  }

  // ---- controls ----
  document.getElementById("zin").addEventListener("click", function () { view.zoom(view.cam.z + 1); });
  document.getElementById("zout").addEventListener("click", function () { view.zoom(view.cam.z - 1); });
  var lvl = document.getElementById("lvl");
  lvl.addEventListener("input", function () {
    S.level = parseFloat(lvl.value);
    document.getElementById("lvlVal").textContent = S.level.toFixed(1);
    draw();
  });
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
      document.getElementById("status").textContent = pts.length
        ? pts.length + " target(s), " + S.hazards.length + " hazard(s) — "
          + new Date().toLocaleTimeString()
        : "store is empty — run the pipeline or replay a recording";
      draw();
    }).catch(function () {
      document.getElementById("status").textContent = "offline / no data";
      draw();
    });
  }

  refresh();
  setInterval(refresh, 4000);
})();
