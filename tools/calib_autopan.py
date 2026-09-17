"""Find the saucepan in each RGB frame automatically, constrained by thermal."""
import sys, pathlib, importlib.util, numpy as np, cv2, json
sys.argv=['x','d']; sys.path.insert(0,'.')
spec=importlib.util.spec_from_file_location("cp","tools/calib_pick.py")
cp=importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)

# Thermal sees 55x35 deg, RGB 62.2x48.8 (configs/pipeline.yaml). So the thermal
# frame covers only the CENTRAL 88.4% x 71.7% of the RGB frame. live_pipeline's
# naive map assumes they coincide, which is why its crops sit off-target; this
# prior is merely good enough to place a search window.
FX, FY = 55.0/62.2, 35.0/48.8

rows=[]
for p in sorted(pathlib.Path("results/calib").glob("live_thermal_*.npy")):
    rgb_p = p.with_name(p.name.replace("thermal","rgb")).with_suffix(".jpg")
    if not rgb_p.exists(): continue
    t = np.load(p); tg = cp.find_targets(t,"hot",3.0,3,1)
    if not tg: continue
    b = tg[0]
    img = cv2.imread(str(rgb_p)); H,W = img.shape[:2]
    px = W/2 + ((b["col"]+0.5)/32 - 0.5)*W*FX
    py = H/2 + ((b["row"]+0.5)/24 - 0.5)*H*FY

    R = 320
    x0,y0 = int(max(0,px-R)), int(max(0,py-R))
    x1,y1 = int(min(W,px+R)), int(min(H,py+R))
    win = img[y0:y1, x0:x1]
    if win.size == 0: continue
    g = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(g,5)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
                               param1=110, param2=38, minRadius=30, maxRadius=115)
    best=None
    if circles is not None:
        for c in np.round(circles[0]).astype(int):
            cx,cy,r = c
            # prefer bright (steel pan on floor) and close to the prediction
            mask=np.zeros(g.shape,np.uint8); cv2.circle(mask,(cx,cy),max(r-6,4),255,-1)
            bright=cv2.mean(g,mask=mask)[0]
            d=np.hypot(cx-(px-x0), cy-(py-y0))
            score = bright - 0.25*d
            if best is None or score>best[0]: best=(score,cx+x0,cy+y0,r,bright,d)
    rows.append({"frame":p.name,"th":[b["col"],b["row"]],"pred":[px,py],
                 "found": best is not None,
                 "rgb":[int(best[1]),int(best[2])] if best else None,
                 "r": int(best[3]) if best else None,
                 "bright": round(best[4],1) if best else None,
                 "dist": round(best[5],1) if best else None})
json.dump(rows, open("/tmp/autopan.json","w"), indent=1)
ok=[r for r in rows if r["found"]]
print(f"{len(ok)}/{len(rows)} frames got a circle")
for r in rows[:40]:
    print(f"  {r['frame'][-10:-4]} th=({r['th'][0]:5.2f},{r['th'][1]:5.2f}) "
          f"pred=({r['pred'][0]:6.0f},{r['pred'][1]:6.0f}) "
          + (f"found=({r['rgb'][0]:5d},{r['rgb'][1]:5d}) r={r['r']:3d} bright={r['bright']:5.1f} d={r['dist']:5.1f}" if r['found'] else "MISS"))
