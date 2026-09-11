/* Shared map engine: Web Mercator projection, the baked-in OSM geometry, the
 * SRTM terrain, the flood raster and road trafficability.
 *
 * Extracted from mapview.js when the radar scope needed the same basemap. Two
 * copies of a projection is how you end up with two pages that disagree about
 * where a survivor is, so there is exactly one.
 *
 * No tile server and no map library: static/basemap.js carries real OSM
 * geometry delta-encoded to ~95 KB and an SRTM grid, which is what makes this
 * work in a field tent with no internet.
 *
 * Depends on: basemap.js (global BASEMAP). Exposes: global MapCore.
 */
window.MapCore = (function () {
  "use strict";
  var D2R = Math.PI / 180, TILE = 256, NS = "http://www.w3.org/2000/svg";

  // ---- projection ---------------------------------------------------------
  function wx(lon, z) { return (lon + 180) / 360 * TILE * Math.pow(2, z); }
  function wy(lat, z) {
    var s = Math.sin(lat * D2R);
    return (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * TILE * Math.pow(2, z);
  }
  function lonAt(x, z) { return x / (TILE * Math.pow(2, z)) * 360 - 180; }
  function latAt(y, z) {
    var n = Math.PI - 2 * Math.PI * y / (TILE * Math.pow(2, z));
    return Math.atan(Math.sinh(n)) / D2R;
  }

  // ---- decode the delta-encoded geometry once, for every page -------------
  var GEO = {};
  (function () {
    var o = BASEMAP.origin, q = BASEMAP.q;
    for (var name in BASEMAP.layers) {
      GEO[name] = BASEMAP.layers[name].map(function (enc) {
        var n = enc.length / 2, p = new Float64Array(enc.length), cx = 0, cy = 0;
        var mla = 90, Mla = -90, mlo = 180, Mlo = -180;
        for (var i = 0; i < n; i++) {
          cx += enc[i * 2]; cy += enc[i * 2 + 1];
          var la = o[0] + cy * q, lo = o[1] + cx * q;
          p[i * 2] = la; p[i * 2 + 1] = lo;
          if (la < mla) mla = la; if (la > Mla) Mla = la;
          if (lo < mlo) mlo = lo; if (lo > Mlo) Mlo = lo;
        }
        return { p: p, b: [mla, mlo, Mla, Mlo] };
      });
    }
  })();

  var DEM = BASEMAP.dem;
  function demAt(lat, lon) {
    var fx = (lon - DEM.lon0) / (DEM.lon1 - DEM.lon0) * (DEM.nx - 1);
    var fy = (lat - DEM.lat0) / (DEM.lat1 - DEM.lat0) * (DEM.ny - 1);
    if (!(fx >= 0 && fy >= 0 && fx <= DEM.nx - 1 && fy <= DEM.ny - 1)) return null;
    var i = Math.floor(fx), j = Math.floor(fy), tx = fx - i, ty = fy - j;
    var i1 = Math.min(DEM.nx - 1, i + 1), j1 = Math.min(DEM.ny - 1, j + 1), z = DEM.z;
    return (z[j * DEM.nx + i] * (1 - tx) + z[j * DEM.nx + i1] * tx) * (1 - ty)
         + (z[j1 * DEM.nx + i] * (1 - tx) + z[j1 * DEM.nx + i1] * tx) * ty;
  }

  // ---- road trafficability (published vehicle-stability depth bands) ------
  var TRAFFIC = [
    { k: "PASSABLE",   max: 0.15, col: "#4FC489" },
    { k: "CAUTION",    max: 0.30, col: "#F3A83C" },
    { k: "RESTRICTED", max: 0.60, col: "#E8792B" },
    { k: "IMPASSABLE", max: 1e9,  col: "#F46454" }
  ];
  var PROFILES = { car: 0.30, emrg: 0.60 };
  var RG = null, RC = null;

  function roadGraph() {
    if (RG) return RG;
    var segs = [], adj = new Map();
    function nid(la, lo) {
      var k = la.toFixed(5) + "," + lo.toFixed(5);
      if (!adj.has(k)) adj.set(k, []);
      return k;
    }
    [["major", 2.4], ["road", 1.7], ["minor", 1.1]].forEach(function (pair) {
      (GEO[pair[0]] || []).forEach(function (f) {
        var p = f.p;
        for (var i = 0; i + 3 < p.length; i += 2) {
          var a = nid(p[i], p[i + 1]), b = nid(p[i + 2], p[i + 3]);
          if (a === b) continue;
          var idx = segs.length;
          segs.push({ a: a, b: b, la0: p[i], lo0: p[i + 1], la1: p[i + 2], lo1: p[i + 3], w: pair[1] });
          adj.get(a).push(idx); adj.get(b).push(idx);
        }
      });
    });
    RG = { segs: segs, adj: adj };
    return RG;
  }

  /* Depth-classify every road segment, then flood-fill outward from the
   * staging point to find what an ambulance can actually reach. A road can be
   * dry and still be useless if every route to it is under water, and that
   * distinction is the one the dispatcher needs. */
  function classifyRoads(level, profile, origin) {
    var key = level.toFixed(2) + "|" + profile;
    if (RC && RC.key === key) return RC;
    var G = roadGraph();
    G.segs.forEach(function (s) {
      var worst = 0, seen = 0;
      for (var i = 0; i <= 3; i++) {
        var t = i / 3;
        var e = demAt(s.la0 + (s.la1 - s.la0) * t, s.lo0 + (s.lo1 - s.lo0) * t);
        if (e == null) continue;
        seen++; worst = Math.max(worst, level - e);
      }
      if (!seen) { s.cls = null; s.depth = null; return; }
      s.depth = Math.max(0, worst);
      for (var c = 0; c < TRAFFIC.length; c++) if (s.depth <= TRAFFIC[c].max) { s.cls = c; break; }
    });
    var ford = PROFILES[profile], start = null, bd = Infinity;
    var org = origin || [22.5726, 88.3639];
    G.adj.forEach(function (_, k) {
      var p = k.split(","), d = Math.hypot(+p[0] - org[0], +p[1] - org[1]);
      if (d < bd) { bd = d; start = k; }
    });
    var reach = new Set();
    if (start) {
      var q = [start]; reach.add(start);
      while (q.length) {
        var cur = q.pop();
        (G.adj.get(cur) || []).forEach(function (si) {
          var s = G.segs[si];
          if (s.depth == null || s.depth > ford) return;   // unassessed is never assumed safe
          var o = s.a === cur ? s.b : s.a;
          if (!reach.has(o)) { reach.add(o); q.push(o); }
        });
      }
    }
    var cut = 0, tot = 0;
    G.segs.forEach(function (s) {
      s.reach = reach.has(s.a) && reach.has(s.b);
      if (s.depth != null) {
        var L = Math.hypot((s.la1 - s.la0) * 110574, (s.lo1 - s.lo0) * 102796);
        tot += L; if (!s.reach) cut += L;
      }
    });
    RC = { key: key, cutFrac: tot ? cut / tot : 0, tot: tot };
    return RC;
  }

  // ---- flood raster, rebuilt only when the level moves --------------------
  var fcv = document.createElement("canvas"), fkey = null;
  fcv.width = DEM.nx; fcv.height = DEM.ny;
  function floodRaster(level) {
    var k = level.toFixed(3);
    if (fkey === k) return fcv;
    var c = fcv.getContext("2d"), img = c.createImageData(DEM.nx, DEM.ny), d = img.data, z = DEM.z;
    for (var j = 0; j < DEM.ny; j++) for (var i = 0; i < DEM.nx; i++) {
      var dep = level - z[j * DEM.nx + i], o = ((DEM.ny - 1 - j) * DEM.nx + i) * 4;
      if (dep <= 0) { d[o + 3] = 0; continue; }
      var t = Math.min(1, dep / 5);
      d[o] = Math.round(120 - 88 * t); d[o + 1] = Math.round(200 - 122 * t);
      d[o + 2] = Math.round(240 - 54 * t); d[o + 3] = Math.round((0.20 + 0.32 * t) * 255);
    }
    c.putImageData(img, 0, 0); fkey = k;
    return fcv;
  }

  function css(n) {
    return getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  }
  function sve(tag, at, kids) {
    var e = document.createElementNS(NS, tag);
    for (var k in at || {}) e.setAttribute(k, at[k]);
    (kids || []).forEach(function (c) { e.appendChild(c); });
    return e;
  }

  // ---- a view: one canvas, one camera, one basemap -------------------------
  function create(opts) {
    var host = opts.host, cv = opts.canvas, ctx = cv.getContext("2d"), ov = opts.svg || null;
    var cam = opts.cam || { lat: 22.57323, lon: 88.36497, z: 17.2 };
    var vp = { w: 0, h: 0, dpr: 1 };
    var minZ = opts.minZoom == null ? 12 : opts.minZoom;
    var maxZ = opts.maxZoom == null ? 19.5 : opts.maxZoom;

    function mpp() { return 156543.03392 * Math.cos(cam.lat * D2R) / Math.pow(2, cam.z); }
    function P(lat, lon) {
      return [wx(lon, cam.z) - wx(cam.lon, cam.z) + vp.w / 2,
              wy(lat, cam.z) - wy(cam.lat, cam.z) + vp.h / 2];
    }
    function unP(x, y) {
      return [latAt(wy(cam.lat, cam.z) + y - vp.h / 2, cam.z),
              lonAt(wx(cam.lon, cam.z) + x - vp.w / 2, cam.z)];
    }
    function size() {
      var r = host.getBoundingClientRect();
      vp.w = r.width; vp.h = r.height;
      vp.dpr = Math.min(2, window.devicePixelRatio || 1);
      cv.width = Math.round(vp.w * vp.dpr); cv.height = Math.round(vp.h * vp.dpr);
      ctx.setTransform(vp.dpr, 0, 0, vp.dpr, 0, 0);
      if (ov) ov.setAttribute("viewBox", "0 0 " + vp.w + " " + vp.h);
    }

    function visible(b) {
      var a = P(b[2], b[1]), c = P(b[0], b[3]);
      return !(c[0] < -40 || a[0] > vp.w + 40 || c[1] < -40 || a[1] > vp.h + 40);
    }
    function trace(f) {
      ctx.beginPath();
      var n = 0;
      for (var i = 0; i < f.length; i++) {
        if (!visible(f[i].b)) continue;
        var p = f[i].p;
        for (var j = 0; j < p.length; j += 2) {
          var s = P(p[j], p[j + 1]);
          if (j === 0) ctx.moveTo(s[0], s[1]); else ctx.lineTo(s[0], s[1]);
        }
        n++;
      }
      return n;
    }
    function fillL(f, c) { if (trace(f)) { ctx.fillStyle = c; ctx.fill("evenodd"); } }
    function strokeL(f, c, w, dash) {
      if (!trace(f)) return;
      ctx.save();
      if (dash) ctx.setLineDash(dash);
      ctx.strokeStyle = c; ctx.lineWidth = w; ctx.lineCap = "round"; ctx.lineJoin = "round";
      ctx.stroke(); ctx.restore();
    }
    function drawExtruded(level, floodOn) {
      var lift = Math.max(2, 9 / mpp()), list = [];
      GEO.bldg.forEach(function (f) {
        if (!visible(f.b)) return;
        var la = (f.b[0] + f.b[2]) / 2, lo = (f.b[1] + f.b[3]) / 2, e = demAt(la, lo);
        list.push({ f: f, y: P(la, lo)[1], d: floodOn && e != null ? Math.max(0, level - e) : 0 });
      });
      list.sort(function (a, b) { return a.y - b.y; });
      var wall = css("--rule"), roof = "#18272F";
      list.forEach(function (v) {
        var p = v.f.p, pts = [];
        for (var i = 0; i < p.length; i += 2) pts.push(P(p[i], p[i + 1]));
        if (pts.length < 3) return;
        ctx.beginPath();
        pts.forEach(function (q, i) { i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
        ctx.closePath(); ctx.fillStyle = wall; ctx.fill();
        ctx.beginPath();
        pts.forEach(function (q, i) { i ? ctx.lineTo(q[0], q[1] - lift) : ctx.moveTo(q[0], q[1] - lift); });
        ctx.closePath();
        if (v.d > 0) {
          var t = Math.min(1, v.d / 4);
          ctx.fillStyle = "rgba(" + Math.round(96 - 60 * t) + "," + Math.round(170 - 90 * t)
                        + "," + Math.round(215 - 40 * t) + ",.92)";
        } else ctx.fillStyle = roof;
        ctx.fill(); ctx.strokeStyle = wall; ctx.lineWidth = .6; ctx.stroke();
      });
    }
    function drawRoads(level, profile, origin, dim) {
      classifyRoads(level, profile, origin);
      var G = roadGraph();
      // Road width tracks zoom so a lane stays roughly a lane -- but only up to
      // a point. Uncapped, `4/mpp` reaches 7.5x past z18 and paints 18-pixel
      // arteries that swamp everything drawn on top of them.
      var k = Math.min(3.0, Math.max(1, 4 / mpp()));
      ctx.save(); ctx.lineCap = "round";
      G.segs.forEach(function (s) {
        var a = P(s.la0, s.lo0), b = P(s.la1, s.lo1);
        if (Math.max(a[0], b[0]) < -30 || Math.min(a[0], b[0]) > vp.w + 30) return;
        if (Math.max(a[1], b[1]) < -30 || Math.min(a[1], b[1]) > vp.h + 30) return;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
        ctx.lineWidth = Math.max(1.1, s.w * k * (dim ? 0.55 : 1));
        ctx.setLineDash(s.reach ? [] : [5, 4]);
        ctx.globalAlpha = (s.cls == null ? .3 : (s.reach ? .95 : .5)) * (dim ? 0.5 : 1);
        ctx.strokeStyle = s.cls == null ? css("--muted") : TRAFFIC[s.cls].col;
        ctx.stroke();
      });
      ctx.restore();
    }

    function drawBase(o) {
      o = o || {};
      var level = o.level == null ? 12.0 : o.level;
      var show = o.show || { flood: true, roads: true, bldg: true, d3: false };
      ctx.fillStyle = o.ground || "#0B141A";
      ctx.fillRect(0, 0, vp.w, vp.h);
      fillL(GEO.park, "#102A20");
      fillL(GEO.water, "#123544");
      strokeL(GEO.river, "#123544", Math.max(2, 14 / mpp()));
      if (show.flood) {
        var nw = P(DEM.lat1, DEM.lon0), se = P(DEM.lat0, DEM.lon1);
        ctx.save();
        ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
        ctx.globalAlpha = o.floodAlpha == null ? 1 : o.floodAlpha;
        ctx.drawImage(floodRaster(level), nw[0], nw[1], se[0] - nw[0], se[1] - nw[1]);
        ctx.restore();
      }
      if (show.bldg && cam.z >= 15.6) {
        if (show.d3) drawExtruded(level, show.flood);
        else { fillL(GEO.bldg, "#18272F"); strokeL(GEO.bldg, "#253945", .7); }
      }
      if (show.roads) {
        drawRoads(level, o.profile || "emrg", o.origin, o.dimRoads);
        strokeL(GEO.rail, "#33454F", 1.3, [7, 5]);
      }
      if (o.aoiBox !== false) {
        var a = P(DEM.lat1, DEM.lon0), b = P(DEM.lat0, DEM.lon1);
        ctx.save(); ctx.setLineDash([6, 5]); ctx.strokeStyle = css("--muted");
        ctx.lineWidth = 1; ctx.globalAlpha = .7;
        ctx.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
        ctx.font = "600 10px ui-monospace, monospace"; ctx.fillStyle = css("--muted");
        ctx.fillText("SRTM ANALYSIS AOI", a[0] + 6, a[1] + 14);
        ctx.restore();
      }
    }

    // ---- interaction ------------------------------------------------------
    var onChange = opts.onChange || function () {};
    var drag = null;
    function zoom(nz, ax, ay) {
      nz = Math.max(minZ, Math.min(maxZ, nz));
      if (ax == null) { ax = vp.w / 2; ay = vp.h / 2; }
      var b = unP(ax, ay);
      cam.z = nz;
      var a = unP(ax, ay);
      cam.lat += b[0] - a[0]; cam.lon += b[1] - a[1];
      onChange();
    }
    function panTo(lat, lon) { cam.lat = lat; cam.lon = lon; onChange(); }

    /* Frame a lat/lon box. Needed because a fixed starting zoom that looks
     * right on a 1600 px desktop shows a quarter of the search area on a
     * phone -- and the phone is where this gets installed. */
    function fitBounds(lat0, lon0, lat1, lon1, padFrac) {
      var pad = padFrac == null ? 0.86 : padFrac;
      var dx = Math.abs(wx(lon1, 0) - wx(lon0, 0));
      var dy = Math.abs(wy(lat1, 0) - wy(lat0, 0));
      var zx = dx > 0 ? Math.log2(vp.w * pad / dx) : maxZ;
      var zy = dy > 0 ? Math.log2(vp.h * pad / dy) : maxZ;
      cam.z = Math.max(minZ, Math.min(maxZ, Math.min(zx, zy)));
      cam.lat = (lat0 + lat1) / 2;
      cam.lon = (lon0 + lon1) / 2;
      onChange();
    }

    if (opts.interactive !== false) {
      cv.addEventListener("pointerdown", function (e) {
        cv.setPointerCapture(e.pointerId);
        drag = { x: e.clientX, y: e.clientY, lat: cam.lat, lon: cam.lon, moved: false };
        cv.style.cursor = "grabbing";
      });
      cv.addEventListener("pointermove", function (e) {
        if (drag) {
          drag.moved = drag.moved || Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > 3;
          cam.lat = latAt(wy(drag.lat, cam.z) - (e.clientY - drag.y), cam.z);
          cam.lon = lonAt(wx(drag.lon, cam.z) - (e.clientX - drag.x), cam.z);
          onChange();
        }
        if (opts.onHover) {
          var r = cv.getBoundingClientRect();
          opts.onHover(unP(e.clientX - r.left, e.clientY - r.top));
        }
      });
      function endDrag() { drag = null; cv.style.cursor = "grab"; }
      cv.addEventListener("pointerup", endDrag);
      cv.addEventListener("pointercancel", endDrag);
      cv.addEventListener("wheel", function (e) {
        e.preventDefault();
        var r = cv.getBoundingClientRect();
        zoom(cam.z - e.deltaY * 0.0022, e.clientX - r.left, e.clientY - r.top);
      }, { passive: false });

      // Pinch to zoom. Without this the page is unusable on the phone it is
      // about to be installed on.
      var pts = new Map(), pinch = null;
      cv.addEventListener("pointerdown", function (e) { pts.set(e.pointerId, e); });
      cv.addEventListener("pointermove", function (e) {
        if (!pts.has(e.pointerId)) return;
        pts.set(e.pointerId, e);
        if (pts.size !== 2) return;
        drag = null;
        var it = Array.from(pts.values()), d = Math.hypot(
          it[0].clientX - it[1].clientX, it[0].clientY - it[1].clientY);
        var r = cv.getBoundingClientRect();
        var mx = (it[0].clientX + it[1].clientX) / 2 - r.left;
        var my = (it[0].clientY + it[1].clientY) / 2 - r.top;
        if (pinch) zoom(cam.z + Math.log2(Math.max(d, 1) / Math.max(pinch, 1)), mx, my);
        pinch = d;
      });
      function drop(e) { pts.delete(e.pointerId); if (pts.size < 2) pinch = null; }
      cv.addEventListener("pointerup", drop);
      cv.addEventListener("pointercancel", drop);
    }

    var view = {
      cam: cam, vp: vp, ctx: ctx, canvas: cv, svg: ov,
      P: P, unP: unP, mpp: mpp, size: size, drawBase: drawBase,
      zoom: zoom, panTo: panTo, fitBounds: fitBounds,
      classifyRoads: classifyRoads, demAt: demAt,
      dragged: function () { return !!(drag && drag.moved); }
    };
    size();
    if (window.ResizeObserver) {
      new ResizeObserver(function () { size(); onChange(); }).observe(host);
    } else {
      window.addEventListener("resize", function () { size(); onChange(); });
    }
    return view;
  }

  // Nice round numbers for a scale bar, in metres.
  var NICE = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
  function niceScale(mpp, minPx, maxPx) {
    var best = NICE[0];
    for (var i = 0; i < NICE.length; i++) {
      var w = NICE[i] / mpp;
      if (w >= (minPx || 55) && w <= (maxPx || 165)) best = NICE[i];
    }
    return { m: best, px: best / mpp, label: best >= 1000 ? (best / 1000) + " km" : best + " m" };
  }

  return {
    D2R: D2R, NS: NS, GEO: GEO, DEM: DEM, demAt: demAt,
    TRAFFIC: TRAFFIC, PROFILES: PROFILES,
    roadGraph: roadGraph, classifyRoads: classifyRoads, floodRaster: floodRaster,
    css: css, sve: sve, niceScale: niceScale, create: create,
    wx: wx, wy: wy, lonAt: lonAt, latAt: latAt
  };
})();
