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

  /* The 48x48 DEM, the flood raster and the road passability classifier used
   * to live here and have been removed. The DEM had no generating script in
   * the repo, no verifiable provenance, 19.6 m cells resampled from SRTM's
   * native 30 m, and -- being a SURFACE model -- put rooftops above the
   * streets beside them in a delta city that is genuinely flat at ~9 m.
   * floodRaster() thresholded it with no hydraulic connectivity at all, so an
   * isolated depression "flooded" with no path to the river.
   *
   * None of that is a flood model, and a road coloured "impassable" is a
   * dispatch instruction. Observed flooding now comes from the AIDER scene
   * classifier, which looks at the imagery the payload actually captured.
   */
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
    function drawExtruded() {
      var lift = Math.max(2, 9 / mpp()), list = [];
      GEO.bldg.forEach(function (f) {
        if (!visible(f.b)) return;
        var la = (f.b[0] + f.b[2]) / 2, lo = (f.b[1] + f.b[3]) / 2;
        list.push({ f: f, y: P(la, lo)[1], d: 0 });
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
    /* Roads, coloured by their OWN class -- major / road / minor -- which is
     * the one thing the baked geometry actually knows about them.
     *
     * This used to colour them passable/caution/impassable from a flood model.
     * That model was removed: it thresholded an unsourced 48x48 DEM with no
     * hydraulic connectivity, resampled 1.5x finer than SRTM's native 30 m, on
     * a surface model that puts rooftops above streets. Colouring a road
     * "impassable" is a dispatch instruction, and nothing here could support
     * one. Road importance can be drawn honestly; road passability cannot.
     */
    var ROAD_STYLE = { major: [2.4, "#5A7180"], road: [1.7, "#465A67"], minor: [1.1, "#38494F"] };

    function drawRoads(dim) {
      var k = Math.min(3.0, Math.max(1, 4 / mpp()));
      ctx.save(); ctx.lineCap = "round";
      ["minor", "road", "major"].forEach(function (cls) {
        var st = ROAD_STYLE[cls];
        ctx.strokeStyle = st[1];
        ctx.globalAlpha = dim ? 0.45 : 0.9;
        (GEO[cls] || []).forEach(function (f) {
          if (!visible(f.b)) return;
          ctx.lineWidth = Math.max(1.1, st[0] * k * (dim ? 0.55 : 1));
          if (trace(f)) ctx.stroke();
        });
      });
      ctx.restore();
    }

    var lastTiles = null;

    function drawBase(o) {
      o = o || {};
      var show = o.show || { roads: true, bldg: true, d3: false };
      ctx.fillStyle = o.ground || "#0B141A";
      ctx.fillRect(0, 0, vp.w, vp.h);

      /* Raster tiles, when a network is there. They REPLACE the baked base
       * geometry rather than sitting under it -- the baked water and building
       * fills are opaque, so drawing both would simply hide the tiles and cost
       * the download for nothing.
       *
       * What is never replaced is the flood surface and the road passability
       * classification below: those are this project's own model output and
       * exist at no zoom level on anyone's tile server. Tiles change what the
       * city looks like, not what we know about it. */
      var tiles = null;
      if (o.tiles && o.tiles.on && window.TileLayer) {
        tiles = window.TileLayer.draw(ctx, cam, vp, wx, wy, {
          source: o.tiles.source, onTile: o.tiles.onTile, alpha: o.tiles.alpha
        });
      }
      // Only treat the tile base as painted once tiles have actually arrived.
      // Mid-load we keep drawing the baked layers, so a slow network degrades
      // to the offline map instead of to an empty rectangle.
      var tiled = !!(tiles && tiles.drawn > 0);
      lastTiles = tiles;

      if (!tiled) {
        fillL(GEO.park, "#102A20");
        fillL(GEO.water, "#123544");
        strokeL(GEO.river, "#123544", Math.max(2, 14 / mpp()));
      }
      if (show.bldg && cam.z >= 15.6) {
        // 3D extrusion is ours and stays over tiles; the flat footprint fill is
        // only a stand-in for what the tiles already draw better.
        if (show.d3) drawExtruded();
        else if (!tiled) { fillL(GEO.bldg, "#18272F"); strokeL(GEO.bldg, "#253945", .7); }
      }
      if (show.roads) {
        drawRoads(o.dimRoads);
        if (!tiled) strokeL(GEO.rail, "#33454F", 1.3, [7, 5]);
      }
      // The dashed "SRTM ANALYSIS AOI" rectangle was the flood model's own
      // analysis extent. With that model gone the box outlined nothing, so it
      // is gone too rather than left as a border with no meaning.
    }

    // ---- interaction ------------------------------------------------------
    var onChange = opts.onChange || function () {};
    var drag = null, dragEndedAt = 0;
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
      /* PAN. Registered on the overlay too, for the same reason the wheel is:
         a target pin is pointer-events:auto so it can be clicked, which means
         a press that lands on one never reaches the canvas and the map will
         not drag from there. */
      function onDown(e) {
        e.currentTarget.setPointerCapture(e.pointerId);
        drag = { x: e.clientX, y: e.clientY, lat: cam.lat, lon: cam.lon, moved: false };
        cv.style.cursor = "grabbing";
      }
      function onMove(e) {
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
      }
      function endDrag() {
        // Remember that a real drag just finished. `click` fires AFTER
        // pointerup, by which point `drag` is already null, so a caller
        // asking dragged() from a click handler would be told "no" and would
        // select whatever the pan happened to finish on top of.
        if (drag && drag.moved) dragEndedAt = Date.now();
        drag = null;
        cv.style.cursor = "grab";
      }
      [cv, ov].forEach(function (el) {
        if (!el) return;
        el.addEventListener("pointerdown", onDown);
        el.addEventListener("pointermove", onMove);
        el.addEventListener("pointerup", endDrag);
        el.addEventListener("pointercancel", endDrag);
      });
      /* WHEEL ZOOM.
       *
       * Two things here, both of which made zooming work only SOMETIMES.
       *
       * 1. The listener has to sit on the OVERLAY as well as the canvas. The
       *    SVG is pointer-events:none so events fall through to the canvas --
       *    except over a target pin, which sets pointer-events:auto so it can
       *    be clicked. A wheel over a pin therefore bubbles up the SVG and
       *    never reaches the canvas at all, and the map simply ignores it.
       *    Pins are small, so this reads as random.
       *
       * 2. deltaY is not comparable across devices. A mouse wheel reports
       *    pixels (~100 per notch); Firefox reports LINES (deltaMode 1, ~3 per
       *    notch) and some setups report PAGES. Multiplying all three by the
       *    same constant makes a wheel work and a trackpad do nothing
       *    perceptible. Normalise to pixels first.
       */
      function onWheel(e) {
        e.preventDefault();
        var r = cv.getBoundingClientRect();
        var dy = e.deltaY;
        if (e.deltaMode === 1) dy *= 16;           // lines  -> px
        else if (e.deltaMode === 2) dy *= vp.h;    // pages  -> px
        // A trackpad pinch arrives as a wheel with ctrlKey set, and its deltas
        // are far smaller than a scroll's. Without this a pinch on a laptop
        // barely moves the zoom.
        var k = e.ctrlKey ? 0.012 : 0.0022;
        // One gesture should never teleport the camera: a momentum flick can
        // deliver a single event of several hundred pixels.
        var dz = Math.max(-1.5, Math.min(1.5, -dy * k));
        zoom(cam.z + dz, e.clientX - r.left, e.clientY - r.top);
      }
      cv.addEventListener("wheel", onWheel, { passive: false });
      if (ov) ov.addEventListener("wheel", onWheel, { passive: false });

      // Pinch to zoom. Without this the page is unusable on the phone it is
      // about to be installed on.
      var pts = new Map(), pinch = null;
      function pinchDown(e) { pts.set(e.pointerId, e); }
      function pinchMove(e) {
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
      }
      function drop(e) { pts.delete(e.pointerId); if (pts.size < 2) pinch = null; }
      // Overlay too, or a pinch that starts with a finger on a pin is lost --
      // and on a phone, where the pins are the thing you are reaching for,
      // that is most of them.
      [cv, ov].forEach(function (el) {
        if (!el) return;
        el.addEventListener("pointerdown", pinchDown);
        el.addEventListener("pointermove", pinchMove);
        el.addEventListener("pointerup", drop);
        el.addEventListener("pointercancel", drop);
      });
    }

    var view = {
      cam: cam, vp: vp, ctx: ctx, canvas: cv, svg: ov,
      P: P, unP: unP, mpp: mpp, size: size, drawBase: drawBase,
      tileInfo: function () { return lastTiles; },
      zoom: zoom, panTo: panTo, fitBounds: fitBounds,
      // True during a drag that has actually moved, and for a moment after it
      // ends -- see endDrag. Callers use this to tell a click from the end of
      // a pan.
      dragged: function () {
        return !!(drag && drag.moved) || (Date.now() - dragEndedAt) < 250;
      }
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
    D2R: D2R, NS: NS, GEO: GEO,
    css: css, sve: sve, niceScale: niceScale, create: create,
    wx: wx, wy: wy, lonAt: lonAt, latAt: latAt
  };
})();
