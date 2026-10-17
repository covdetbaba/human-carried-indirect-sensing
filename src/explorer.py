import base64
import hashlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODE = "standing" if "--standing" in sys.argv else "walking"
DATA = os.path.join(ROOT, "data", f"explorer_{MODE}.json")
PAGE = os.path.join(ROOT, "docs", f"explorer-{MODE}.html")


def _arg(flag, default, cast=str):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


WORKERS = int(_arg("--workers", max(1, (os.cpu_count() or 4) - 2), int))
PASSES = int(_arg("--passes", 20, int))
TIER = int(_arg("--tier", 2, int))
PORT = int(_arg("--port", 8771 if MODE == "standing" else 8770, int))
NF = 750

PARAMS_ALL = ["deck_E", "deck_rho", "sec_I_set", "deck_zeta", "deck_L",
              "deck_W", "deck_t", "bench_force_amps", "walk_speed",
              "sens_noise"]
PARAMS = [p for p in PARAMS_ALL
          if not (MODE == "standing" and p == "walk_speed")]

NAME = {
    "deck_E": "Elastic modulus",
    "deck_rho": "Density",
    "sec_I_set": "Second moment of area",
    "deck_zeta": "Damping ratio",
    "deck_L": "Span",
    "deck_W": "Width",
    "deck_t": "Thickness",
    "bench_force_amps": "Shaker force",
    "walk_speed": "Walking speed",
    "sens_noise": "Sensor noise",
}

NOTE = {
    "deck_E": "Concrete grades C20 to C70. Frequencies follow the square root of E.",
    "deck_rho": "Heavier decks have lower frequencies.",
    "sec_I_set": "Bending stiffness of the section. Mass stays the same.",
    "deck_zeta": "Changes peak height and width. Frequencies stay put.",
    "deck_L": "Longer spans lower every mode. Bending drops fastest.",
    "deck_W": "Width splits the families. Torsion drops, lateral modes rise.",
    "deck_t": "Thicker slabs raise vertical bending. Lateral modes stay put.",
    "bench_force_amps": "Ambient force at each of the three shaker points, in x, y and z.",
    "walk_speed": "Cadence and step force follow the speed.",
    "sens_noise": "The same noise density on both devices and all axes.",
}

LEVELS = {
    "deck_E": [21.02e9, 23.5e9, 25.74e9, 27.81e9, 29.73e9, None, 33.23e9,
               34.68e9],
    "deck_rho": [1800.0, 2300.0, 2400.0, 2500.0, 7850.0],
    "sec_I_set": [0.0007192479034367999, 0.0024274616740991993,
                  0.0064732311309311985, 0.011238248491199998,
                  0.019419693392793595],
    "deck_zeta": [0.002, 0.01, 0.02, 0.03, 0.06],
    "deck_L": [8.0, 10.5, 14.0, 18.0, 24.0],
    "deck_W": [2.0, 2.5, 3.0, 3.5, 4.0],
    "deck_t": [0.12, 0.15, 0.2, 0.24, 0.3],
    "bench_force_amps": [(a, a, a) for a in (10.0, 30.0, 60.0, 90.0, 120.0,
                                            150.0, 300.0, 600.0, 800.0,
                                            900.0)],
    "walk_speed": [0.5, 1.0, 1.5, 2.0, 2.5],
    "sens_noise": [20.0, None, 80.0, 320.0, 1280.0, 5120.0, 20480.0,
                   81920.0],
}

LABELS = {
    "deck_E": ["21.0 GPa  C20", "23.5 GPa  C25", "25.7 GPa  C30",
               "27.8 GPa  C35", "29.7 GPa  C40", "30.0 GPa", "33.2 GPa  C50",
               "34.7 GPa  C70"],
    "deck_rho": ["1800 kg/m³", "2300 kg/m³", "2400 kg/m³", "2500 kg/m³",
                 "7850 kg/m³"],
    "sec_I_set": ["0.72e-3 m⁴", "2.43e-3 m⁴", "6.47e-3 m⁴", "11.24e-3 m⁴",
                  "19.42e-3 m⁴"],
    "deck_zeta": ["0.2 %", "1 %", "2 %", "3 %", "6 %"],
    "deck_L": ["8 m", "10.5 m", "14 m", "18 m", "24 m"],
    "deck_W": ["2 m", "2.5 m", "3 m", "3.5 m", "4 m"],
    "deck_t": ["0.12 m", "0.15 m", "0.20 m", "0.24 m", "0.30 m"],
    "bench_force_amps": ["10 N", "30 N", "60 N", "90 N", "120 N", "150 N",
                         "300 N", "600 N", "800 N", "900 N"],
    "walk_speed": ["0.5 m/s", "1.0 m/s", "1.5 m/s", "2.0 m/s", "2.5 m/s"],
    "sens_noise": ["20 µg/√Hz", "75 µg/√Hz", "80 µg/√Hz", "320 µg/√Hz",
                   "1280 µg/√Hz", "5120 µg/√Hz", "20480 µg/√Hz",
                   "81920 µg/√Hz"],
}

BASE = {"deck_E": 5, "deck_rho": 2, "sec_I_set": 1, "deck_zeta": 2,
        "deck_L": 2, "deck_W": 2, "deck_t": 2, "bench_force_amps": 8,
        "walk_speed": 2, "sens_noise": 1}


def base_cfg(mode=None):
    import human_di as hv
    mode = MODE if mode is None else mode
    c = dict(hv.CONFIG)
    c.update(n_passes=PASSES, run_ssi=False, run_efdd=False,
             run_update=False, interactive_pick=False,
             deck_section="rectangle", deck_material="concrete",
             sec_catalog=None, sec_A_set=None)
    if mode == "standing":
        c.update(human_posture="stand", stand_x_frac=0.5)
    return c


def store_sig():
    import human_di as hv
    c = hv.CONFIG
    h = hashlib.sha1()
    h.update(repr((MODE, PARAMS, [LEVELS[p] for p in PARAMS],
                   [BASE[p] for p in PARAMS], NF, PASSES, c["psd_fmin"],
                   c["psd_fmax"], c["psd_smooth_hz"], c["deck_n_modes"],
                   hv.PHYSICS_REV, hv.OBS_T1, hv.OBS_T2, hv.OBS_SIGMA)).encode())
    return h.hexdigest()[:16]


def key_of(sel):
    if not sel:
        return "base"
    return "+".join(f"{p}={sel[p]}" for p in PARAMS if p in sel)


_W = {}


def _init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import human_di as hv
    import deck_bench as db
    db._init()
    _W.update(hv=hv, db=db, bg={})


def background_of(cfg):
    hv = _W["hv"]
    k = hv.background_key(cfg)
    if k not in _W["bg"]:
        _W["bg"][k] = hv.run_ensemble(hv.rigid_twin_cfg(cfg),
                                      list(cfg["sensors"]))
    return _W["bg"][k]


def _enc(y):
    y = np.asarray(y, float)
    ok = np.isfinite(y) & (y > 0)
    if not ok.any():
        return "", 0.0
    pk = float(np.max(y[ok]))
    q = np.full(len(y), 65535, dtype=np.uint16)
    d = 10.0 * np.log10(np.maximum(y[ok], pk * 1e-12) / pk)
    q[ok] = np.round(-np.clip(d, -110.0, 0.0) * 10.0).astype(np.uint16)
    return base64.b64encode(q.astype("<u2").tobytes()).decode("ascii"), pk


def record_window(cfg, stand):
    import human_di as hv
    if stand:
        Trec = float(cfg["stand_T"])
    else:
        Trec = float(cfg["deck_L"]) / float(cfg["walk_speed"]) - hv.walk_trim_s(cfg)
    seg = cfg.get("psd_seg_s") or Trec
    return Trec, min(Trec, float(seg))


PMODE_KEYS = ("oi", "f", "band", "fh", "ph", "G", "F", "Po", "Pb")


def run_one(task):
    key, sel, levels = task[:3]
    mode = task[3] if len(task) > 3 else MODE
    hv, db = _W["hv"], _W["db"]
    cfg = base_cfg(mode)
    for p, v in levels.items():
        cfg[p] = v
        db.apply_depends(cfg, p)
    t0 = time.time()
    try:
        bench = db.get_bench(cfg)
        ens = hv.run_ensemble(cfg, list(cfg["sensors"]))
    except Exception as ex:
        return dict(key=key, sel=sel, failed=f"{type(ex).__name__}: {ex}")
    stand = hv.is_standing(cfg)
    fmin, fmax = cfg["psd_fmin"], cfg["psd_fmax"]
    g = np.linspace(fmin, fmax, NF)
    deck = hv.smooth_psd(bench["sv1"], bench["df"], cfg["psd_smooth_hz"])
    cb, pkb = _enc(np.interp(g, bench["f"], deck, left=np.nan, right=np.nan))
    br = hv.benchmark_references(bench, cfg)
    _Trec, Tres = record_window(cfg, stand)
    out = dict(key=key, sel=sel, failed="", cb=cb, pkb=pkb, fmin=fmin,
               fmax=fmax,
               refs=[[str(l), float(f)] for l, f in br["refs"]],
               bands=[[float(a), float(b)] for a, b in br["bands"]],
               Tres=float(Tres), L=float(cfg["deck_L"]),
               W=float(hv.deck_width(cfg)),
               fp=(None if stand else round(float(ens["cfg"]["pacing_freq"]), 3)),
               hum=[[round(float(f), 4), fam]
                    for f, fam, _lab in hv.peak_sources(cfg, ens)
                    if fam in ("gait", "arm", "body", "drive")
                    and fmin <= f <= fmax])
    rigid = background_of(cfg)
    for part, dev in (("wrist", "watch"), ("hip", "phone")):
        f, P = hv.ens_part_sv1(ens, part, comb_twin=rigid)
        Pd = hv.smooth_psd(P, f[1] - f[0], cfg["psd_smooth_hz"])
        fr, Pr = hv.ens_part_sv1(rigid, part, comb_from=ens)
        Prd = hv.smooth_psd(Pr, fr[1] - fr[0], cfg["psd_smooth_hz"])
        c, pk = _enc(np.interp(g, f, Pd, left=np.nan, right=np.nan))
        rc, rpk = _enc(np.interp(g, fr, Prd, left=np.nan, right=np.nan))
        out[dev] = dict(c=c, pk=pk, rc=rc, rpk=rpk)
    hv.curve_index_run(out, NF)
    slim_scores(out)
    out["secs"] = round(time.time() - t0, 1)
    return out


def slim_scores(r):
    for k in ("n_peak", "pobs"):
        r.pop(k, None)
    for dev in ("watch", "phone"):
        d = r[dev]
        r[dev] = dict(c=d["c"], pk=d["pk"], rc=d["rc"], rpk=d["rpk"],
                      oi=d["oi"], band=d["band"], caps=d.get("caps", []),
                      pmode={m: {k: v[k] for k in PMODE_KEYS if k in v}
                             for m, v in d["pmode"].items()})
    return r


def task_of(sel):
    sel = {p: i for p, i in sel.items() if i != BASE[p]}
    lv = {p: LEVELS[p][BASE[p]] for p in PARAMS}
    for p, i in sel.items():
        lv[p] = LEVELS[p][i]
    return (key_of(sel), sel, lv, MODE)


def plan():
    seen, tasks = set(), []

    def add(sel):
        t = task_of(sel)
        if t[0] not in seen:
            seen.add(t[0])
            tasks.append(t)

    add({})
    for p in PARAMS:
        for i in range(len(LEVELS[p])):
            add({p: i})
    if TIER >= 2:
        for a in range(len(PARAMS)):
            for b in range(a + 1, len(PARAMS)):
                pa, pb = PARAMS[a], PARAMS[b]
                for i in range(len(LEVELS[pa])):
                    for j in range(len(LEVELS[pb])):
                        add({pa: i, pb: j})
    return tasks


def load_store():
    try:
        store = json.load(open(DATA, encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"no data yet: run  python3 explorer.py"
                         f"{' --standing' if MODE == 'standing' else ''}")
    if store.get("sig") != store_sig():
        raise SystemExit("the stored runs were made with other levels or "
                         "settings. Run the study again.")
    return store


def save_store(store):
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w", encoding="utf-8") as fh:
        json.dump(store, fh, separators=(",", ":"))


def page_meta(store):
    import human_di as hv
    return dict(mode=MODE, params=PARAMS, name=NAME, note=NOTE,
                labels={p: LABELS[p] for p in PARAMS},
                base={p: BASE[p] for p in PARAMS}, passes=store["passes"],
                fmin=store["fmin"], fmax=store["fmax"], nf=store["NF"],
                smooth=hv.OBS_SIGMA, t1=hv.OBS_T1, t2=hv.OBS_T2, port=PORT,
                cmd="python3 explorer.py"
                    + (" --standing" if MODE == "standing" else "")
                    + " --serve",
                stand_T=float(hv.CONFIG["stand_T"]))


def rebuild(store=None):
    from explorer_page import write_page
    store = store or load_store()
    write_page(PAGE, page_meta(store), store["runs"])
    print(f"  wrote {PAGE}")


def serve(port):
    import http.server
    import socketserver
    store = load_store()
    if not os.path.exists(PAGE):
        rebuild(store)
    print(f"  explorer ({MODE}) at http://localhost:{port}/   Ctrl-C to stop")

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except ConnectionError:
                pass

        def do_GET(self):
            if self.path.split("?")[0] not in ("/", "/index.html",
                                               "/" + os.path.basename(PAGE)):
                self._send(b"not found", "text/plain", 404)
                return
            with open(PAGE, "rb") as fh:
                self._send(fh.read(), "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                sweep = req["sweep"]
                if sweep not in PARAMS:
                    raise ValueError(f"unknown parameter {sweep!r}")
                sel = {k: int(v) for k, v in (req.get("sel") or {}).items()
                       if k in PARAMS}
                tasks = []
                for i in range(len(LEVELS[sweep])):
                    s = dict(sel)
                    s[sweep] = i
                    tasks.append(task_of(s))
                t0 = time.time()
                print(f"  computing {len(tasks)} configurations ...",
                      flush=True)
                with ProcessPoolExecutor(max_workers=WORKERS,
                                         initializer=_init) as ex:
                    runs = [f.result() for f in
                            [ex.submit(run_one, t) for t in tasks]]
                ok = [r for r in runs if not r.get("failed")]
                print(f"    {len(ok)}/{len(runs)} in {time.time() - t0:.0f} s",
                      flush=True)
                have = {r["key"] for r in store["runs"]}
                store["runs"] += [r for r in ok if r["key"] not in have]
                save_store(store)
                rebuild(store)
                body = json.dumps(dict(runs=ok, failed=[
                    dict(key=r["key"], failed=r["failed"])
                    for r in runs if r.get("failed")]))
            except Exception as ex:
                body = json.dumps(dict(error=f"{type(ex).__name__}: {ex}",
                                       runs=[]))
            self._send(body.encode(), "application/json")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), H) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n  stopped")


def main():
    if "--serve" in sys.argv:
        serve(PORT)
        return
    if "--rebuild" in sys.argv:
        rebuild()
        return
    import human_di as hv
    tasks = plan()
    print(f"  explorer ({MODE}): {len(tasks)} configurations, {PASSES} "
          f"records each, {WORKERS} workers")
    t0, runs = time.time(), []
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for n, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            runs.append(r)
            if r.get("failed"):
                print(f"    {r['key']} failed: {r['failed'][:60]}", flush=True)
            elif n % 25 == 0 or n == len(tasks):
                print(f"    {n}/{len(tasks)}  {time.time() - t0:.0f} s",
                      flush=True)
    runs.sort(key=lambda r: r["key"])
    store = dict(version=2, mode=MODE, sig=store_sig(), passes=PASSES, NF=NF,
                 fmin=hv.CONFIG["psd_fmin"], fmax=hv.CONFIG["psd_fmax"],
                 built=time.strftime("%Y-%m-%d"), runs=runs)
    save_store(store)
    rebuild(store)


if __name__ == "__main__":
    main()
