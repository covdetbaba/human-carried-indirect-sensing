import hashlib


import os


for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")


import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


BCACHE = os.path.join(os.path.dirname(HERE), "cache", "bench")


DEPENDS = {
    "crowd_type": [("crowd_on", True)],
    "crowd_n": [("crowd_on", True)],
    "crowd_amp": [("crowd_on", True)],
    "crowd_amp_var": [("crowd_on", True)],
    "crowd_seed": [("crowd_on", True)],
    "crowd_freq": [("crowd_on", True), ("crowd_type", "walk")],
    "exc_amp": [("exc_type", "random")],
    "exc_x_frac": [("exc_type", "random")],
    "sens_seed": [("exc_type", "random")],
    "exc_freq": [("exc_type", "sine")],
    "sec_flange_w": [("deck_section", "channel")],
    "sec_flange_t": [("deck_section", "channel")],
    "sec_web_h": [("deck_section", "channel")],
    "sec_web_t": [("deck_section", "channel")],
    "sec_n_web": [("deck_section", "channel")],
    "sec_web_off": [("deck_section", "channel")],
    "sec_box_h": [("deck_section", "box")],
    "sec_box_wall": [("deck_section", "box")],
    "bench_force_freqs": [("bench_force_type", "jump")],
    "bench_force_type": [("bench_force_amps", (800.0, 800.0, 800.0))],
    "pacing_freq": [("auto_gait", False)],
    "duty_factor": [("auto_gait", False)],
}


_W = {}


BENCH_KEYS = ("deck_material", "deck_E", "deck_rho", "deck_L",
              "deck_W", "deck_section", "deck_model", "deck_zeta",
              "deck_n_elem", "deck_n_modes", "psd_fmin", "psd_fmax",
              "bench_nperseg", "bench_seg_s", "bench_noise", "bench_seed",
              "bench_T",
              "run_dt", "bench_force_case", "bench_force_type",
              "bench_force_amps", "bench_force_freqs", "bench_force_amp_var",
              "bench_edge_frac", "bench_exc_amp", "bench_ambient",
              "bench_force_dir")


CROWD_MASS_KEYS = ("crowd_on", "crowd_type", "crowd_amp", "crowd_n",
                   "crowd_seed", "crowd_amp_var")


SEC_ACTIVE = {
    "rectangle": ("deck_W", "deck_t"),
    "channel": ("sec_flange_w", "sec_flange_t", "sec_web_h", "sec_web_t",
                "sec_n_web", "sec_web_off"),
    "tee": ("sec_flange_w", "sec_flange_t", "sec_web_h", "sec_web_t"),
    "box": ("sec_flange_w", "sec_box_h", "sec_box_wall"),
    "custom": ("sec_A", "sec_I", "sec_J", "sec_Ip"),
}


def bench_key(cfg):
    act = SEC_ACTIVE.get(cfg.get("deck_section", "rectangle"), ())
    keys = BENCH_KEYS + act
    if cfg.get("crowd_on") and cfg.get("crowd_type") == "static":
        keys = keys + CROWD_MASS_KEYS
    for k in ("sec_A_set", "sec_I_set", "sec_catalog", "sec_units"):
        if cfg.get(k) is not None:
            keys = keys + (k,)
    rev = getattr(_W.get("hv") or __import__("human_di"),
                  "BENCH_EXC_REV", {})
    tag = rev.get(str(cfg.get("bench_force_type", "random")), "")
    raw = "v4|" + (f"exc={tag}|" if tag else "") + \
          "|".join(f"{k}={cfg.get(k)!r}" for k in keys)
    return hashlib.md5(raw.encode()).hexdigest()[:20]


def bench_sig(cfg):
    hv = _W.get("hv")
    if hv is None:
        import human_di as hv
    E, rho, A, I = hv.deck_props(cfg)
    return np.array([A, I, E, rho, float(cfg["deck_L"]),
                     float(cfg["deck_zeta"])], float)


def _scratch_dir():
    for d in ("/dev/shm", "/tmp", os.path.expanduser("~")):
        try:
            if os.path.isdir(d):
                p = os.path.join(d, "param_explorer_scratch")
                os.makedirs(p, exist_ok=True)
                t = os.path.join(p, f".w{os.getpid()}")
                with open(t, "w") as fh:
                    fh.write("x")
                os.remove(t)
                return p
        except OSError:
            continue
    return HERE


def _init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import human_di as hv
    hv.SAVE_DIR = _scratch_dir()
    _W["hv"] = hv
    _W["bench"] = {}
    try:
        os.makedirs(BCACHE, exist_ok=True)
    except OSError:
        pass


def get_bench(cfg):
    hv = _W["hv"]
    k = bench_key(cfg)
    if k in _W["bench"]:
        return _W["bench"][k]
    path = os.path.join(BCACHE, k + ".npz")
    b = None
    if os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            b = dict(f=z["f"], sv1=z["sv1"], df=float(z["df"]),
                     T=float(z["T"]), n_avg=int(z["n_avg"]),
                     Yrms=float(z["Yrms"]), fe_freqs=list(z["fe"]),
                     sv1_axis={ax: z["sv1_" + ax] for ax in "xyz"
                               if "sv1_" + ax in z.files},
                     rms={ax: float(z["rms_" + ax]) for ax in "xyz"
                          if "rms_" + ax in z.files},
                     mode_labels=[str(x) for x in z["labels"]],
                     more=list(zip([float(x) for x in z["mf"]],
                                   [str(x) for x in z["ml"]]))
                     if "mf" in z.files else [])
            if "sig" in z.files and not np.allclose(
                    z["sig"], bench_sig(cfg), rtol=1e-9, atol=0.0):
                b = None
        except Exception:
            b = None
    if b is None:
        r = hv.run_benchmark_avt(cfg, verbose=False)
        b = dict(f=r["f"], sv1=r["sv1"], df=r["df"], T=r["T"],
                 n_avg=r["n_avg"], Yrms=float(r["rms"]["z"]),
                 sv1_axis=dict(r["sv1_axis"]), rms=dict(r["rms"]),
                 fe_freqs=list(r["fe_freqs"]),
                 mode_labels=list(r["mode_labels"]),
                 more=[(float(f), str(l)) for f, l in r.get("more", [])])
        try:
            tmp = f"{path}.{os.getpid()}.tmp.npz"
            np.savez_compressed(
                tmp, f=b["f"], sv1=b["sv1"], df=b["df"], T=b["T"],
                n_avg=b["n_avg"], Yrms=b["Yrms"],
                fe=np.array(b["fe_freqs"], float),
                labels=np.array(b["mode_labels"], dtype="U8"),
                mf=np.array([f for f, _ in b["more"]], float),
                ml=np.array([l for _, l in b["more"]], dtype="U8"),
                sig=bench_sig(cfg),
                **{"sv1_" + ax: v for ax, v in b["sv1_axis"].items()},
                **{"rms_" + ax: v for ax, v in b["rms"].items()})
            os.replace(tmp, path)
        except Exception:
            pass
    _W["bench"][k] = b
    if len(_W["bench"]) > 8:
        _W["bench"].pop(next(iter(_W["bench"])))
    return b


def apply_depends(cfg, probed):
    for k, v in DEPENDS.get(probed, ()):
        cfg[k] = v
    return cfg
