import os


import sys


import time


import numpy as np


from scipy import signal as sps


from scipy.signal import find_peaks


from scipy.linalg import eigh


from scipy.ndimage import gaussian_filter1d


import plotly.graph_objects as go


import openseespy.opensees as ops


G_ACC = 9.81


CONFIG = dict(
    interactive_pick = False,

    deck_material = "concrete",
    deck_E        = None,
    deck_rho      = None,
    deck_L        = 14.0,
    deck_zeta     = 0.02,
    deck_n_elem   = 40,
    deck_n_modes  = 20,

    deck_section  = "rectangle",
    deck_model    = "beam",
    walk_y_frac     = 0.5,
    step_width      = 0.12,
    stance_width    = 0.17,
    sec_web_off     = 0.50,
    deck_W        = 3.00,
    deck_t        = 0.200,
    sec_flange_w    = 2.44,
    sec_flange_t    = 0.12,
    sec_web_h       = 0.45,
    sec_web_t       = 0.20,
    sec_n_web       = 2,
    sec_box_h       = 0.60,
    sec_box_wall    = 0.15,
    sec_A           = 0.30,
    sec_I           = 3.0e-3,
    sec_J           = None,
    sec_Ip          = None,
    sec_A_set       = None,
    sec_I_set       = 0.0024274616740991993,
    sec_I_ref       = (0.20, 3.00),
    sec_catalog     = None,
    sec_units       = None,
    sec_J_set       = None,
    sec_depth_set   = None,

    body_mass       = 75.0,
    body_height     = 1.75,
    frac_hip        = 0.20,
    frac_torso      = 0.50,
    frac_wrist      = 0.05,

    k_leg           = 2.0e4,
    c_leg           = 1400.0,
    k_spine         = 6.0e4,
    c_spine         = 1600.0,
    k_arm           = 3.79e4,
    c_arm           = 264.0,
    k_leg_x         = 7.4e2,
    c_leg_x         = 140.0,
    k_leg_y         = 1.7e3,
    c_leg_y         = 300.0,
    k_spine_x       = 4.6e3,
    c_spine_x       = 177.0,
    k_spine_y       = 3.0e3,
    c_spine_y       = 141.0,
    k_arm_x         = 148.0,
    c_arm_x         = 9.4,
    k_arm_y         = 148.0,
    c_arm_y         = 9.4,
    k_leg_x_stand   = 15.0,
    c_leg_x_stand   = 14.0,
    k_leg_y_stand   = 370.0,
    c_leg_y_stand   = 94.0,
    arm_swing_on    = True,
    arm_swing_deg   = 20.0,
    arm_swing_auto  = True,
    arm_swing_v_ref = 1.5,
    arm_swing_gain  = 1.0,

    walk_speed      = 1.5,
    auto_gait       = True,
    pacing_freq     = 2.0,
    gait_dlf        = (0.40, 0.10, 0.06, 0.05, 0.03),
    gait_dlf_auto   = True,
    gait_phase      = (0.0, 1.5708, 1.5708, 1.5708, 1.5708),
    gait_hf_on      = True,
    gait_hf_coef    = 42.0,
    gait_hf_fmax    = 75.0,
    gait_dlf_x      = (0.20, 0.08),
    gait_phase_x    = (0.0, 0.0),
    gait_dlf_y      = (0.04, 0.01, 0.03, 0.01, 0.01),
    gait_phase_y    = (0.0, 0.0, 0.0, 0.0, 0.0),
    gait_hf_ratio_x = 0.20,
    gait_hf_ratio_y = 0.04,
    gait_step_cv    = 0.02,
    gait_seed       = 0,
    rep_seed        = 0,
    duty_factor     = 0.62,
    foot_lift       = 0.09,
    walk_force_on   = True,
    human_posture   = "walk",
    stand_x_frac    = 0.5,
    stand_T         = 60.0,
    lead_min_s      = 20.0,
    lead_decays     = 3.0,
    lead_s          = None,

    sensors         = ["hipL", "hipR", "wristL", "wristR"],
    sweep_sensor    = "wristL",
    sens_noise_wrist = 75.0,
    sens_noise_hip   = 75.0,
    sens_noise      = None,
    sens_seed       = 1,
    noise_draw      = 0,
    rigid_base      = False,

    exc_type        = "none",
    exc_amp         = 150.0,
    exc_freq        = 3.0,
    exc_x_frac      = 0.5,
    exc_dir         = (0.0, 0.0, 1.0),

    crowd_on        = False,
    crowd_n         = 6,
    crowd_amp       = 500.0,
    crowd_amp_var   = 0.20,
    crowd_type      = "random",
    crowd_freq      = 2.0,
    crowd_seed      = 7,

    run_dt          = 0.001,
    psd_fmin        = 0.3,
    psd_fmax        = 50.0,
    psd_smooth_hz   = 0.05,
    psd_seg_s       = 30.0,
    psd_normalize   = False,
    psd_offset_dec  = 0.0,
    psd_capture_hz  = 0.35,
    psd_reject_hz   = 0.12,
    contact_rise_s  = 0.10,
    gait_contact    = "ramp",
    comb_detect     = True,
    comb_order_track = True,
    comb_snr_db     = 6.0,
    comb_win_s      = 0.0,
    comb_short_hz   = 8.0,
    notch_kmax      = None,
    notch_gait      = True,
    cp_denoise      = True,
    cp_floor        = 0.05,
    cp_reg_hz       = 4.0,
    cp_reg          = 0.5,
    cp_correct      = False,

    n_passes        = 20,
    pass_speed_jitter = 0.13,
    bench_x_fracs   = (0.20, 0.30, 0.50, 0.70, 0.80),

    bench_T         = None,
    bench_force_case = 1,
    bench_force_type = "random",
    bench_force_amps  = (800.0, 800.0, 800.0),
    bench_force_freqs = (2.0, 2.0, 2.0),
    bench_force_amp_var = 0.0,
    bench_force_dir = (1.0, 1.0, 1.0),
    bench_force_live = True,
    bench_abs_scale = True,
    bench_noise     = 30.0,
    bench_ambient   = 5.0,
    bench_exc_amp   = 120.0,
    bench_edge_frac = 0.90,
    bench_seed      = 11,
    bench_seg_s     = 41.0,
    bench_nperseg   = None,

    run_ssi         = False,
    run_efdd        = False,
    run_update      = False,

    run_label       = "",

    anim_fps        = 25.0,
    anim_max_frames = 450,
    anim_time_scale = 1.0,
    anim_disp_scale = 18.0,
    anim_body_scale = 6.0,
    anim_bob        = 0.028,
    show_joints     = True,
    anim_ymin       = -1.0,
    anim_ymax       = 2.4,
)


GAIT_STAGES = {
    0.5: dict(pacing_freq=1.55, duty_factor=0.68, foot_lift=0.05, bob=0.016),
    1.5: dict(pacing_freq=2.00, duty_factor=0.62, foot_lift=0.09, bob=0.028),
    2.5: dict(pacing_freq=2.45, duty_factor=0.52, foot_lift=0.15, bob=0.045),
}


def dlf_scale(fp):
    return float(np.clip((float(fp) - 0.95) / 1.05, 0.25, 2.0))


def apply_gait_stage(cfg):
    if not cfg.get("auto_gait", True):
        return cfg
    xs = sorted(GAIT_STAGES); v = cfg["walk_speed"]
    for key in ("pacing_freq", "duty_factor", "foot_lift"):
        cfg[key] = float(np.interp(v, xs, [GAIT_STAGES[s][key] for s in xs]))
    cfg["anim_bob"] = float(np.interp(v, xs, [GAIT_STAGES[s]["bob"] for s in xs]))
    if cfg.get("gait_dlf_auto", True):
        base = cfg.setdefault("_gait_dlf_base", tuple(cfg["gait_dlf"]))
        cfg["gait_dlf"] = tuple(
            dlf_scale(cfg["pacing_freq"]) * np.asarray(base, float))
    return cfg


SWEEPS = {
    "walk_speed":  [0.5, 1.5, 2.5],
    "k_leg":       [8.0e3, 2.0e4, 6.0e4],
    "sens_noise":  [80.0, 1280.0, 20480.0],
    "crowd_amp":   [0.0, 500.0, 1000.0],
    "stand_x_frac": [0.25, 0.4, 0.5],
}


SAVE_DIR       = "."


MATERIALS = {
    "concrete": dict(E=30.0e9, rho=2400.0),
    "steel":    dict(E=200.0e9, rho=7850.0),
}


if "--quick" in sys.argv:
    CONFIG.update(run_dt=0.004, deck_n_elem=16, anim_max_frames=80, n_passes=3,
                  bench_T=60.0)
    SWEEPS = {k: v[-1:] for k, v in SWEEPS.items()}


BENCH_EXC_REV = {
    "random": "2026-09-21-ambient-per-point",
    "jump":   "2026-09-21-ambient-per-point",
    "static": "2026-09-21-ramp-per-point",
}


IN2_M2 = 0.0254 ** 2


IN4_M4 = 0.0254 ** 4


def _cat(sec, src, w_in, h_in, a_in2, i_in4, note="", units=None,
         j_in4=None):
    return dict(sec=sec, src=src, w_in=float(w_in), h_in=float(h_in), units=units,
                A_in2=float(a_in2), I_in4=float(i_in4), note=note,
                J_in4=(None if j_in4 is None else float(j_in4)))


_AA = "AASHTO/PCI Bridge Design Manual App. B (Nov 2011)"


_AISC = "AISC Shapes Database v16.0"


_ACI = "48 in solid strip of an ACI 318 size, A = b h, I = b h^3/12"


SECTION_CATALOG = {
    "AASHTO SI-36 solid slab":  _cat("rectangle", _AA, 36, 12, 432.0, 5184.0),
    "AASHTO SI-48 solid slab":  _cat("rectangle", _AA, 48, 12, 576.0, 6912.0),
    "ACI solid slab 6 in":      _cat("rectangle", _ACI, 48, 6, 288.0, 864.0),
    "ACI solid slab 9 in":      _cat("rectangle", _ACI, 48, 9, 432.0, 2916.0),
    "ACI solid slab 15 in":     _cat("rectangle", _ACI, 48, 15, 720.0, 13500.0),
    "ACI solid slab 18 in":     _cat("rectangle", _ACI, 48, 18, 864.0, 23328.0),
    "AASHTO BI-36 box":   _cat("box", _AA, 36, 27, 560.5, 50334.0),
    "AASHTO BI-48 box":   _cat("box", _AA, 48, 27, 692.5, 65941.0),
    "AASHTO BII-36 box":  _cat("box", _AA, 36, 33, 620.5, 85153.0),
    "AASHTO BII-48 box":  _cat("box", _AA, 48, 33, 752.5, 110499.0),
    "AASHTO BIII-36 box": _cat("box", _AA, 36, 39, 680.5, 131145.0),
    "AASHTO BIII-48 box": _cat("box", _AA, 48, 39, 812.5, 168367.0),
    "AASHTO BIV-36 box":  _cat("box", _AA, 36, 42, 710.5, 158644.0),
    "AASHTO BIV-48 box":  _cat("box", _AA, 48, 42, 842.5, 203088.0),
    "AASHTO SII-36 voided":  _cat("box", _AA, 36, 15, 439.0, 9725.0),
    "AASHTO SIII-36 voided": _cat("box", _AA, 36, 18, 491.0, 16514.0),
    "AASHTO SIV-36 voided":  _cat("box", _AA, 36, 21, 530.0, 25747.0),
    "AASHTO SII-48 voided":  _cat("box", _AA, 48, 15, 569.0, 12897.0),
    "AASHTO SIII-48 voided": _cat("box", _AA, 48, 18, 628.0, 21855.0),
    "AASHTO SIV-48 voided":  _cat("box", _AA, 48, 21, 703.0, 34517.0),
    "AISC HSS20x12x1/2 box": _cat("box", _AISC, 12, 20, 28.3, 1550.0,
                                  "steel", units=2, j_in4=1540.0),
    "AISC HSS16x8x1/2 box":  _cat("box", _AISC, 8, 16, 20.9, 679.0,
                                  "steel", units=2, j_in4=563.0),
    "AASHTO DBT35x48 tee": _cat("tee", _AA, 48, 35, 677.0, 101540.0),
    "AASHTO DBT35x72 tee": _cat("tee", _AA, 72, 35, 823.0, 116071.0),
    "AASHTO DBT35x96 tee": _cat("tee", _AA, 96, 35, 967.0, 126353.0),
    "AASHTO DBT53x48 tee": _cat("tee", _AA, 48, 53, 785.0, 294350.0),
    "AASHTO DBT53x72 tee": _cat("tee", _AA, 72, 53, 931.0, 335679.0),
    "AASHTO DBT53x96 tee": _cat("tee", _AA, 96, 53, 1075.0, 365827.0),
    "AASHTO DBT65x48 tee": _cat("tee", _AA, 48, 65, 857.0, 490755.0),
    "AASHTO DBT65x72 tee": _cat("tee", _AA, 72, 65, 1003.0, 559367.0),
    "AASHTO DBT65x96 tee": _cat("tee", _AA, 96, 65, 1147.0, 610435.0),
    "AISC C15x50 channel":  _cat("tee", _AISC, 3.72, 15, 14.7, 404.0,
                                 "steel stringer", units=5, j_in4=2.65),
    "AISC MC18x58 channel": _cat("tee", _AISC, 4.2, 18, 17.1, 675.0,
                                 "steel stringer", units=5, j_in4=2.81),
    "PCI DT 5ftx27 light":  _cat("channel", _AA, 60, 27, 575.0, 33740.0),
    "PCI DT 6ftx23 light":  _cat("channel", _AA, 72, 23, 558.0, 21366.0),
    "PCI DT 6ftx27 light":  _cat("channel", _AA, 72, 27, 635.0, 35758.0),
    "PCI DT 8ftx27 light":  _cat("channel", _AA, 96, 27, 689.0, 32888.0),
    "PCI DT 8ftx35 light":  _cat("channel", _AA, 96, 35, 787.0, 72421.0),
    "PCI DT 6ftx21 heavy":  _cat("channel", _AA, 72, 21, 644.0, 22720.0),
    "PCI DT 8ftx21 heavy":  _cat("channel", _AA, 96, 21, 764.0, 24920.0),
    "PCI DT 6ftx27 heavy":  _cat("channel", _AA, 72, 27, 731.0, 45084.0),
    "PCI DT 8ftx27 heavy":  _cat("channel", _AA, 96, 27, 851.0, 49566.0),
    "PCI DT 5ftx36 heavy":  _cat("channel", _AA, 60, 36, 780.0, 90286.0),
    "PCI DT 6ftx35 heavy":  _cat("channel", _AA, 72, 35, 840.0, 90164.0),
    "PCI DT 8ftx35 heavy":  _cat("channel", _AA, 96, 35, 960.0, 99299.0),
}


def catalog_AI(name, deck_w_m=None, units=None):
    e = SECTION_CATALOG[name]
    if units is None:
        units = e.get("units")
    if units is None:
        w_m = e["w_in"] * 0.0254
        units = max(1, int(round((deck_w_m or 2.44) / w_m))) if w_m > 0 else 1
    return (units * e["A_in2"] * IN2_M2,
            units * e["I_in4"] * IN4_M4, units, e)


def apply_catalog(cfg):
    name = cfg.get("sec_catalog")
    if not name:
        return cfg
    if name not in SECTION_CATALOG:
        import difflib
        near = difflib.get_close_matches(str(name), SECTION_CATALOG, 3, 0.4)
        msg = [f"unknown sec_catalog {name!r}."]
        if near:
            msg.append("Did you mean: " + ", ".join(repr(x) for x in near))
        msg.append("Run human_di.print_catalog() for the full menu, or "
                   "human_di.catalog_names('box') for one family.")
        raise ValueError(" ".join(msg))
    A, I, units, e = catalog_AI(name, deck_w_m=float(cfg.get("deck_W", 2.44)),
                                units=cfg.get("sec_units"))
    out = dict(cfg, deck_section=e["sec"], sec_A_set=A, sec_I_set=I)

    if e.get("J_in4"):
        out["sec_J_set"] = units * float(e["J_in4"]) * IN4_M4
    out["sec_depth_set"] = e["h_in"] * 0.0254
    if e["src"].startswith("AISC") and cfg.get("deck_E") is None \
            and cfg.get("deck_rho") is None:
        out["deck_material"] = "steel"
    return out


SEC_HORIZ = ("deck_W", "sec_flange_w", "sec_web_t", "sec_web_off")


SEC_VERT = ("deck_t", "sec_flange_t", "sec_web_h", "sec_box_h")


def _box_walls(cfg):
    t = float(cfg["sec_box_wall"])
    return (float(cfg.get("_box_wall_w") or t),
            float(cfg.get("_box_wall_f") or t))


def _I_following_geometry(cfg, Ireq):
    ref = cfg.get("sec_I_ref")
    if (Ireq is None or not ref or cfg.get("sec_catalog")
            or cfg.get("deck_section", "rectangle") != "rectangle"):
        return Ireq
    t_ref, w_ref = float(ref[0]), float(ref[1])
    if t_ref <= 0 or w_ref <= 0:
        return Ireq
    return (float(Ireq) * (float(cfg["deck_W"]) / w_ref)
            * (float(cfg["deck_t"]) / t_ref) ** 3)


def sec_geom(cfg):
    cfg = apply_catalog(cfg)
    Areq, Ireq = cfg.get("sec_A_set"), cfg.get("sec_I_set")
    if Areq is None and Ireq is None:
        return cfg
    Ireq = _I_following_geometry(cfg, Ireq)
    A0, I0 = _section_AI_raw(cfg)
    if A0 <= 0 or I0 <= 0:
        return cfg
    fA = 1.0 if Areq is None else float(Areq) / A0
    fI = 1.0 if Ireq is None else float(Ireq) / I0
    if fA <= 0 or fI <= 0:
        return cfg
    beta = float(np.sqrt(fI / fA))
    alpha = fA / beta
    c = dict(cfg)
    for k in SEC_HORIZ:
        if isinstance(c.get(k), (int, float)) and not isinstance(c[k], bool):
            c[k] = float(c[k]) * alpha
    for k in SEC_VERT:
        if isinstance(c.get(k), (int, float)) and not isinstance(c[k], bool):
            c[k] = float(c[k]) * beta
    tw = float(cfg["sec_box_wall"])
    c["_box_wall_w"], c["_box_wall_f"] = tw * alpha, tw * beta
    c["sec_box_wall"] = tw * min(alpha, beta)
    if cfg.get("deck_section") == "custom":
        c["sec_A"] = A0 * alpha * beta
        c["sec_I"] = I0 * alpha * beta ** 3
    c["sec_A_set"] = c["sec_I_set"] = None
    return c


def _section_AI_raw(cfg):
    sec = cfg.get("deck_section", "rectangle")
    if sec == "custom":
        return cfg["sec_A"], cfg["sec_I"]
    if sec == "rectangle":
        W, t = cfg["deck_W"], cfg["deck_t"]
        return W * t, W * t**3 / 12.0
    if sec in ("channel", "tee"):
        bf, tf = cfg["sec_flange_w"], cfg["sec_flange_t"]
        hw, tw = cfg["sec_web_h"], cfg["sec_web_t"]
        nw = 1 if sec == "tee" else cfg["sec_n_web"]
        Af, yf = bf * tf, tf / 2.0
        Aw, yw = tw * hw, tf + hw / 2.0
        A = Af + nw * Aw
        ybar = (Af * yf + nw * Aw * yw) / A
        If = bf * tf**3 / 12.0 + Af * (ybar - yf)**2
        Iw = tw * hw**3 / 12.0 + Aw * (yw - ybar)**2
        return A, If + nw * Iw
    if sec == "box":
        b, h = cfg["sec_flange_w"], cfg["sec_box_h"]
        tw, tf = _box_walls(cfg)
        A = b * h - (b - 2 * tw) * (h - 2 * tf)
        I = (b * h**3 - (b - 2 * tw) * (h - 2 * tf)**3) / 12.0
        return A, I
    raise ValueError(f"unknown deck_section '{sec}'")


def section_AI(cfg):
    return _section_AI_raw(sec_geom(cfg))


def section_depth(cfg):
    cfg = apply_catalog(cfg)
    if cfg.get("sec_depth_set"):
        return float(cfg["sec_depth_set"])
    cfg = sec_geom(cfg)
    sec = cfg.get("deck_section", "rectangle")
    if sec == "rectangle":
        return cfg["deck_t"]
    if sec in ("channel", "tee"):
        return cfg["sec_flange_t"] + cfg["sec_web_h"]
    if sec == "box":
        return cfg["sec_box_h"]
    A, I = section_AI(cfg)
    return float(np.sqrt(12.0 * I / max(A, 1e-9)))


def _web_ys(cfg, sec):
    if sec == "tee":
        return [0.0]
    bf = cfg["sec_flange_w"]
    nw = cfg["sec_n_web"]
    yo = bf / 2.0 - cfg["sec_web_off"]
    if nw == 1:
        return [0.0]
    return list(np.linspace(-yo, +yo, nw))


def section_JIp(cfg):
    cfg = apply_catalog(cfg)
    if cfg.get("sec_J_set"):
        J = float(cfg["sec_J_set"])
        return J, section_AI(cfg) [1] + section_Ilat(cfg)
    cfg = sec_geom(cfg)
    sec = cfg.get("deck_section", "rectangle")
    if sec == "custom":
        if cfg.get("sec_J") and cfg.get("sec_Ip"):
            return float(cfg["sec_J"]), float(cfg["sec_Ip"])
        A, I = section_AI(cfg)
        t = float(np.sqrt(12.0 * I / max(A, 1e-9))); W = A / t
        return section_JIp(dict(cfg, deck_section="rectangle",
                                deck_W=W, deck_t=t))
    if sec == "rectangle":
        W, t = cfg["deck_W"], cfg["deck_t"]
        a, b = max(W, t), min(W, t)
        J = a * b**3 * (1.0/3.0 - 0.21 * (b/a) * (1 - b**4 / (12 * a**4)))
        Ip = W * t * (W**2 + t**2) / 12.0
        return J, Ip
    if sec in ("channel", "tee"):
        bf, tf = cfg["sec_flange_w"], cfg["sec_flange_t"]
        hw, tw = cfg["sec_web_h"], cfg["sec_web_t"]
        ys = _web_ys(cfg, sec)
        J = bf * tf**3 / 3.0 + len(ys) * hw * tw**3 / 3.0
        Af, zf = bf * tf, tf / 2.0
        Aw, zw = tw * hw, tf + hw / 2.0
        A = Af + len(ys) * Aw
        zbar = (Af * zf + len(ys) * Aw * zw) / A
        Ip = Af * (bf**2 + tf**2) / 12.0 + Af * (zbar - zf)**2
        for y in ys:
            Ip += (Aw * (tw**2 + hw**2) / 12.0
                   + Aw * ((zw - zbar)**2 + y**2))
        return J, Ip
    if sec == "box":
        b, h = cfg["sec_flange_w"], cfg["sec_box_h"]
        tw, tf = _box_walls(cfg)
        bm, hm = b - tw, h - tf
        J = 4.0 * (bm * hm)**2 / (2.0 * bm / tf + 2.0 * hm / tw)
        bi, hi = b - 2*tw, h - 2*tf
        Ip = (b*h*(b**2 + h**2) - bi*hi*(bi**2 + hi**2)) / 12.0
        return J, Ip
    raise ValueError(f"unknown deck_section '{sec}'")


def section_Ilat(cfg):
    cfg = sec_geom(cfg)
    sec = cfg.get("deck_section", "rectangle")
    if sec == "custom":
        A, I = section_AI(cfg)
        t = float(np.sqrt(12.0 * I / max(A, 1e-9))); W = A / t
        return t * W**3 / 12.0
    if sec == "rectangle":
        W, t = cfg["deck_W"], cfg["deck_t"]
        return t * W**3 / 12.0
    if sec in ("channel", "tee"):
        bf, tf = cfg["sec_flange_w"], cfg["sec_flange_t"]
        hw, tw = cfg["sec_web_h"], cfg["sec_web_t"]
        I = tf * bf**3 / 12.0
        for y in _web_ys(cfg, sec):
            I += hw * tw**3 / 12.0 + hw * tw * y**2
        return I
    if sec == "box":
        b, h = cfg["sec_flange_w"], cfg["sec_box_h"]
        tw, tf = _box_walls(cfg)
        return (h * b**3 - (h - 2*tf) * (b - 2*tw)**3) / 12.0
    raise ValueError(f"unknown deck_section '{sec}'")


def deck_props(cfg):
    mat = MATERIALS[cfg["deck_material"]]
    E   = cfg["deck_E"]   if cfg["deck_E"]   else mat["E"]
    rho = cfg["deck_rho"] if cfg["deck_rho"] else mat["rho"]
    A, I = section_AI(cfg)
    return E, rho, A, I


def _eigen(n):
    try:
        lam = ops.eigen("-genBandArpack", n)
    except Exception:
        lam = ops.eigen("-fullGenLapack", n)
    return [float(np.sqrt(l) / (2.0 * np.pi)) for l in lam]


def _eigen_to_band(n, cfg, cap=160):
    fmax = float(cfg["psd_fmax"])
    fe = _eigen(n)
    while fe and fe[-1] <= fmax and n < cap:
        n = min(cap, int(np.ceil(1.5 * n)) + 4)
        fe = _eigen(n)
    return fe


N_EXTRA_MODES = 12


def _rayleigh(fe, zeta):
    w1, w2 = 2 * np.pi * fe[0], 2 * np.pi * fe[1]
    aM, bK = zeta * 2 * w1 * w2 / (w1 + w2), zeta * 2 / (w1 + w2)
    ops.rayleigh(aM, bK, 0.0, 0.0)
    return dict(alphaM=float(aM), betaK=float(bK), f1=float(fe[0]))


def build_beam(cfg):
    E, rho, A, Ivert = deck_props(cfg)
    G = E / 2.6
    J, Ip = section_JIp(cfg)
    Ilat = section_Ilat(cfg)
    L, n_el = cfg["deck_L"], cfg["deck_n_elem"]
    yedge = deck_width(cfg) / 2.0
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    xs = np.linspace(0.0, L, n_el + 1)
    for i, x in enumerate(xs):
        ops.node(i + 1, float(x), 0.0, 0.0)
    ops.fix(1,        1, 1, 1, 1, 0, 0)
    ops.fix(n_el + 1, 0, 1, 1, 1, 0, 0)
    ops.geomTransf("Linear", 1, 0, 0, 1)
    for i in range(n_el):
        ops.element("elasticBeamColumn", 100 + i, i + 1, i + 2,
                    A, E, G, J, Ivert, Ilat, 1, "-mass", rho * A)
    le = xs[1] - xs[0]
    nm_ = {i + 1: [0.0, 0.0, 0.0,
                   rho * Ip * (le if 0 < i < n_el else le / 2.0), 0.0, 0.0]
           for i in range(n_el + 1)}
    for mem in make_crowd_members(cfg, L):
        if mem["type"] == "static":
            node = int(np.argmin(np.abs(xs - mem["x0"]))) + 1
            mkg = mem["amp"] / G_ACC
            for d in range(3):
                nm_[node][d] += mkg
            nm_[node][3] += mkg * mem["y"]**2
    for n, v in nm_.items():
        ops.mass(n, *v)
    n_want = cfg["deck_n_modes"]
    fe_all = _eigen_to_band(n_want + N_EXTRA_MODES, cfg)
    ray = _rayleigh(fe_all, cfg["deck_zeta"])

    lab_all = []
    counts = {"B": 0, "T": 0, "L": 0, "A": 0}
    for k in range(1, len(fe_all) + 1):
        uz = np.array([ops.nodeEigenvector(i + 1, k, 3)
                       for i in range(n_el + 1)])
        uy = np.array([ops.nodeEigenvector(i + 1, k, 2)
                       for i in range(n_el + 1)])
        rx = np.array([ops.nodeEigenvector(i + 1, k, 4)
                       for i in range(n_el + 1)])
        ux = np.array([ops.nodeEigenvector(i + 1, k, 1)
                       for i in range(n_el + 1)])
        cand = {"B": float(np.max(np.abs(uz))),
                "L": float(np.max(np.abs(uy))),
                "T": float(np.max(np.abs(rx))) * yedge,
                "A": float(np.max(np.abs(ux)))}
        kind = max(cand, key=cand.get)
        counts[kind] += 1
        lab_all.append(f"{kind}{counts[kind]}")
    keep = list(range(len(lab_all)))
    n_use = max(n_want, sum(1 for k in keep
                            if fe_all[k] <= float(cfg["psd_fmax"])))
    fe = [fe_all[k] for k in keep][:n_use]
    labels = [lab_all[k] for k in keep][:n_use]
    more = [(fe_all[k], lab_all[k]) for k in keep[n_use:]]

    yw = float(np.clip(cfg["walk_y_frac"], -1.0, 1.0)) * max(yedge - 0.35, 0.0)

    def read(x):
        i = min(int(x / le), n_el - 1)
        xi = (x - xs[i]) / le
        n1, n2 = i + 1, i + 2
        u = ((1 - xi) * (ops.nodeDisp(n1, 3) + yw * ops.nodeDisp(n1, 4))
             + xi * (ops.nodeDisp(n2, 3) + yw * ops.nodeDisp(n2, 4)))
        v = ((1 - xi) * (ops.nodeVel(n1, 3) + yw * ops.nodeVel(n1, 4))
             + xi * (ops.nodeVel(n2, 3) + yw * ops.nodeVel(n2, 4)))
        return u, v

    def load(x, Fz, loads):
        i = min(int(x / le), n_el - 1)
        xi = (x - xs[i]) / le
        for n, w in ((i + 1, 1 - xi), (i + 2, xi)):
            e = loads.setdefault(n, [0.0, 0.0])
            e[0] += Fz * w
            e[1] += Fz * yw * w

    def apply(loads):
        for n, v in loads.items():
            ops.load(n, 0.0, 0.0, v[0], v[1], 0.0, 0.0)

    def unit_nodes(x):
        return [(int(np.argmin(np.abs(xs - x))) + 1, 1.0)]

    bench = []
    for fr in cfg["bench_x_fracs"]:
        i = int(np.argmin(np.abs(xs - fr * L))) + 1
        bench += [(i, fr, -1), (i, fr, +1)]
    return dict(kind="beam", torsional=True, yedge=yedge, xs=xs, le=le,
                yw=yw, ztop=section_ztop(cfg),
                n_el=n_el, freqs=fe, labels=labels, more=more,
                read=read, load=load, apply=apply, unit_nodes=unit_nodes,
                bench=bench, acc_dof=3, disp_dof=3,
                new_loads=lambda: {}, rayleigh=ray)


def build_grillage(cfg):
    E, rho, A, I = deck_props(cfg)
    G = E / 2.6
    L, n_el = cfg["deck_L"], cfg["deck_n_elem"]
    bf, tf = cfg["sec_flange_w"], cfg["sec_flange_t"]
    hw, tw = cfg["sec_web_h"], cfg["sec_web_t"]
    yL = bf / 2.0 - cfg["sec_web_off"]
    yL = max(yL, 0.15 * bf)
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    xs = np.linspace(0.0, L, n_el + 1)
    nA = lambda i: 1000 + i
    nB = lambda i: 2000 + i
    for i, x in enumerate(xs):
        ops.node(nA(i), float(x), -yL, 0.0)
        ops.node(nB(i), float(x), +yL, 0.0)
    for n in (nA(0), nB(0), nA(n_el), nB(n_el)):
        ops.fix(n, 1, 1, 1, 0, 0, 0)

    bstrip = bf / 2.0
    A_l, Iy_l = A / 2.0, I / 2.0
    Iz_l = (tf * bstrip**3 / 12.0) + (hw * tw**3 / 12.0)
    J_l  = (bstrip * tf**3 + hw * tw**3) / 3.0
    ops.geomTransf("Linear", 1, 0, 0, 1)
    for i in range(n_el):
        ops.element("elasticBeamColumn", 100 + i, nA(i), nA(i + 1),
                    A_l, E, G, J_l, Iy_l, Iz_l, 1, "-mass", rho * A_l)
        ops.element("elasticBeamColumn", 200 + i, nB(i), nB(i + 1),
                    A_l, E, G, J_l, Iy_l, Iz_l, 1, "-mass", rho * A_l)
    dx = xs[1] - xs[0]
    A_t, Iz_t = dx * tf, dx * tf**3 / 12.0
    Iy_t, J_t = tf * dx**3 / 12.0, dx * tf**3 / 3.0
    ops.geomTransf("Linear", 2, 1, 0, 0)
    for i in range(n_el + 1):
        ops.element("elasticBeamColumn", 300 + i, nA(i), nB(i),
                    A_t, E, G, J_t, Iy_t, Iz_t, 2, "-mass", 0.0)

    for mem in make_crowd_members(cfg, L):
        if mem["type"] == "static":
            i = int(np.argmin(np.abs(xs - mem["x0"])))
            eta = float(np.clip(mem["y"] / max(yL, 1e-9), -1, 1))
            mkg = mem["amp"] / G_ACC
            for tagf, wgt in ((nA, (1 - eta) / 2), (nB, (1 + eta) / 2)):
                ops.mass(tagf(i), mkg * wgt, mkg * wgt, mkg * wgt,
                         0.0, 0.0, 0.0)

    n_want = cfg["deck_n_modes"]
    fe_all = _eigen_to_band(n_want + N_EXTRA_MODES, cfg)
    ray = _rayleigh(fe_all, cfg["deck_zeta"])

    lab_all = []
    counts = {"B": 0, "T": 0, "L": 0}
    for k in range(1, len(fe_all) + 1):
        za = np.array([ops.nodeEigenvector(nA(i), k, 3) for i in range(n_el + 1)])
        zb = np.array([ops.nodeEigenvector(nB(i), k, 3) for i in range(n_el + 1)])
        uy = np.array([ops.nodeEigenvector(nA(i), k, 2) for i in range(n_el + 1)])
        if np.max(np.abs(uy)) > 2.0 * max(np.max(np.abs(za)), 1e-12):
            kind = "L"
        else:
            kind = "B" if float(za @ zb) >= 0 else "T"
        counts[kind] += 1
        lab_all.append(f"{kind}{counts[kind]}")
    keep = list(range(len(lab_all)))
    n_use = max(n_want, sum(1 for k in keep
                            if fe_all[k] <= float(cfg["psd_fmax"])))
    fe = [fe_all[k] for k in keep][:n_use]
    labels = [lab_all[k] for k in keep][:n_use]
    more = [(fe_all[k], lab_all[k]) for k in keep[n_use:]]

    le = xs[1] - xs[0]
    eta = float(np.clip(cfg["walk_y_frac"], -1.0, 1.0))
    wA, wB = (1 - eta) / 2.0, (1 + eta) / 2.0

    def _line_uv(tagf, x):
        i = min(int(x / le), n_el - 1)
        xi = (x - xs[i]) / le
        u = (1 - xi) * ops.nodeDisp(tagf(i), 3) + xi * ops.nodeDisp(tagf(i + 1), 3)
        v = (1 - xi) * ops.nodeVel(tagf(i), 3) + xi * ops.nodeVel(tagf(i + 1), 3)
        return u, v

    def read(x):
        ua, va = _line_uv(nA, x)
        ub, vb = _line_uv(nB, x)
        return wA * ua + wB * ub, wA * va + wB * vb

    def load(x, Fz, loads):
        i = min(int(x / le), n_el - 1)
        xi = (x - xs[i]) / le
        for tagf, w in ((nA, wA), (nB, wB)):
            loads[tagf(i)] = loads.get(tagf(i), 0.0) + Fz * w * (1 - xi)
            loads[tagf(i + 1)] = loads.get(tagf(i + 1), 0.0) + Fz * w * xi

    def apply(loads):
        for n, fz in loads.items():
            ops.load(n, 0.0, 0.0, fz, 0.0, 0.0, 0.0)

    def unit_nodes(x):
        i = int(np.argmin(np.abs(xs - x)))
        return [(nA(i), 0.5), (nB(i), 0.5)]

    bench = []
    for fr in cfg["bench_x_fracs"]:
        i = int(np.argmin(np.abs(xs - fr * L)))
        bench += [(nA(i), fr, -1), (nB(i), fr, +1)]
    return dict(kind="shell", xs=xs, le=le, n_el=n_el,
                yw=eta * yL, ztop=section_ztop(cfg), yedge=yL,
                freqs=fe, labels=labels, more=more,
                read=read, load=load, apply=apply, unit_nodes=unit_nodes,
                bench=bench, acc_dof=3, disp_dof=3,
                new_loads=lambda: {}, nA=nA, nB=nB, yL=yL, rayleigh=ray)


def build_deck(cfg):
    if cfg.get("deck_model", "beam") in ("shell", "grillage"):
        return build_grillage(cfg)
    return build_beam(cfg)


AXES = ("x", "y", "z")


def section_ztop(cfg):
    c = sec_geom(apply_catalog(cfg))
    sec = c.get("deck_section", "rectangle")
    if sec == "rectangle":
        return 0.5 * float(c["deck_t"])
    if sec in ("channel", "tee"):
        bf, tf = c["sec_flange_w"], c["sec_flange_t"]
        hw, tw = c["sec_web_h"], c["sec_web_t"]
        nw = 1 if sec == "tee" else c["sec_n_web"]
        Af, Aw = bf * tf, tw * hw
        return float((Af * tf / 2.0 + nw * Aw * (tf + hw / 2.0))
                     / (Af + nw * Aw))
    if sec == "box":
        return 0.5 * float(c["sec_box_h"])
    return 0.5 * float(section_depth(c))


def deck_rows(B, x, y):
    xs, le, n_el = B["xs"], B["le"], B["n_el"]
    zt = float(B.get("ztop", 0.0))
    x = float(min(max(x, xs[0]), xs[-1]))
    i = min(int(x / le), n_el - 1)
    xi = (x - xs[i]) / le
    rows = {"x": [], "y": [], "z": []}
    if B["kind"] == "shell":
        yL = max(float(B["yL"]), 1e-9)
        eta = float(np.clip(y / yL, -1.0, 1.0))
        for n_of, w_line in ((B["nA"], (1 - eta) / 2.0),
                             (B["nB"], (1 + eta) / 2.0)):
            for ii, w in ((i, 1 - xi), (i + 1, xi)):
                n = n_of(ii); ww = w_line * w
                rows["z"].append((n, 2, ww))
                rows["y"].append((n, 1, ww))
                rows["x"].append((n, 0, ww))
                rows["x"].append((n, 4, zt * ww))
        for ii, w in ((i, 1 - xi), (i + 1, xi)):
            rows["y"].append((B["nA"](ii), 2, +zt * w / (2.0 * yL)))
            rows["y"].append((B["nB"](ii), 2, -zt * w / (2.0 * yL)))
        return rows
    for n, w in ((i + 1, 1 - xi), (i + 2, xi)):
        rows["z"] += [(n, 2, w), (n, 3, y * w)]
        rows["y"] += [(n, 1, w), (n, 3, -zt * w)]
        rows["x"] += [(n, 0, w), (n, 4, zt * w), (n, 5, -y * w)]
    return rows


def deck_mck(B):
    ops.wipeAnalysis()
    ops.logFile(os.devnull, "-noEcho")
    ops.constraints("Transformation"); ops.numberer("Plain")
    ops.system("FullGeneral"); ops.algorithm("Linear")
    ops.integrator("GimmeMCK", 1.0, 0.0, 0.0)
    ops.analysis("Transient")
    mats = []
    for m, c, k in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        ops.integrator("GimmeMCK", m, c, k)
        ops.analyze(1, 0.0)
        n = ops.systemSize()
        mats.append(np.array(ops.printA("-ret"), float).reshape(n, n))
    dof = {int(nd): list(ops.nodeDOFs(int(nd))) for nd in ops.getNodeTags()}
    ops.wipeAnalysis()
    ops.setTime(0.0)
    ops.logFile(os.devnull)
    return mats[0], mats[1], mats[2], dof


class DeckSolver:

    def __init__(self, B, dt):
        self.B = B
        self.dt = float(dt)
        M, C, K, dof = deck_mck(B)
        self.M, self.C, self.K, self.dof = M, C, K, dof
        self.neq = M.shape[0]
        g, b = 0.5, 0.25
        self.a0 = 1.0 / (b * dt * dt); self.a1 = g / (b * dt)
        self.a2 = 1.0 / (b * dt); self.a3 = 1.0 / (2 * b) - 1.0
        self.a4 = g / b - 1.0; self.a5 = dt * (g / (2 * b) - 1.0)
        self.gamma = g
        self.Keff = K + self.a1 * C + self.a0 * M
        self.G = np.linalg.inv(self.Keff)
        md = np.diag(M)
        self.Mdiag = md if np.allclose(M, np.diag(md)) else None
        self._rowcache = {}

    def eq_row(self, node_rows):
        acc = {}
        for n, d, w in node_rows:
            e = self.dof[int(n)][d]
            if e >= 0 and w != 0.0:
                acc[e] = acc.get(e, 0.0) + float(w)
        idx = np.array(sorted(acc), dtype=int)
        return idx, np.array([acc[e] for e in idx], float)

    def point(self, x, y):
        key = (round(float(x), 9), round(float(y), 9))
        hit = self._rowcache.get(key)
        if hit is not None:
            return hit
        rows = deck_rows(self.B, x, y)
        per = [self.eq_row(rows[a]) for a in AXES]
        cols = np.array(sorted(set(int(e) for idx, _ in per for e in idx)),
                        dtype=int)
        pos = {int(e): k for k, e in enumerate(cols)}
        Ns = np.zeros((3, cols.size))
        for a, (idx, w) in enumerate(per):
            for e, ww in zip(idx, w):
                Ns[a, pos[int(e)]] += ww
        if len(self._rowcache) > 4096:
            self._rowcache.clear()
        self._rowcache[key] = (cols, Ns)
        return cols, Ns

    def load_vector(self, x, y, axis_weights):
        cols, Ns = self.point(x, y)
        P = np.zeros(self.neq)
        P[cols] += np.asarray(axis_weights, float) @ Ns
        return P


SENSOR_DOF = {"hipL": 0, "hipR": 1, "torso": 2, "wristL": 3, "wristR": 4}


SENSOR_SEED = {"hipL": 0, "hipR": 1, "torso": 2, "wristL": 3, "wristR": 4}


def _two(val):
    a = np.atleast_1d(np.asarray(val, dtype=float))
    return np.array([a[0], a[0]]) if a.size == 1 else a[:2]


def human_segments(cfg):
    H = cfg["body_height"]
    return dict(
        thigh = 0.245 * H, shank = 0.246 * H,
        pelvis_h = 0.10 * H,
        torso = 0.30 * H,
        head  = 0.13 * H,
        uparm = 0.186 * H, forearm = 0.146 * H,
        hipw  = 0.10 * H,
        shoulderw = 0.11 * H,
        leg_len = 0.49 * H,
    )


LUMPS = ("hipL", "hipR", "torso", "wristL", "wristR")


N_LUMP = len(LUMPS)


N_HDOF = 3 * N_LUMP


def hdof(lump, ax):
    return 3 * int(lump) + int(ax)


def _spring_xyz(cfg, name):
    k = np.zeros((2, 3)); c = np.zeros((2, 3))
    for a, ax in enumerate(AXES):
        kk = cfg.get(f"k_{name}") if ax == "z" else cfg.get(f"k_{name}_{ax}")
        cc = cfg.get(f"c_{name}") if ax == "z" else cfg.get(f"c_{name}_{ax}")
        if kk is None:
            kk = cfg[f"k_{name}"]
        if cc is None:
            cc = cfg[f"c_{name}"]
        k[:, a] = _two(kk); c[:, a] = _two(cc)
    return k, c


def build_human(cfg):
    Mb = cfg["body_mass"]
    m_h = cfg["frac_hip"] * Mb
    m_t = cfg["frac_torso"] * Mb
    m_w = cfg["frac_wrist"] * Mb
    masses = np.array([m_h, m_h, m_t, m_w, m_w])
    Mm = np.diag(np.repeat(masses, 3))
    K = np.zeros((N_HDOF, N_HDOF)); C = np.zeros((N_HDOF, N_HDOF))

    def add(mat, i, j, k):
        mat[i, i] += k; mat[j, j] += k; mat[i, j] -= k; mat[j, i] -= k

    ks, cs = _spring_xyz(cfg, "spine")
    ka, ca = _spring_xyz(cfg, "arm")
    kleg, cleg = _spring_xyz(cfg, "leg")
    if is_standing(cfg):
        for a, ax in ((0, "x"), (1, "y")):
            kk = cfg.get(f"k_leg_{ax}_stand"); cc = cfg.get(f"c_leg_{ax}_stand")
            if kk is not None:
                kleg[:, a] = _two(kk)
            if cc is not None:
                cleg[:, a] = _two(cc)
    for a in range(3):
        for side in (0, 1):
            add(K, hdof(side, a), hdof(2, a), ks[side, a])
            add(C, hdof(side, a), hdof(2, a), cs[side, a])
        for side, w in ((0, 3), (1, 4)):
            add(K, hdof(2, a), hdof(w, a), ka[side, a])
            add(C, hdof(2, a), hdof(w, a), ca[side, a])
    K2, C2 = K.copy(), C.copy()
    for side in (0, 1):
        for a in range(3):
            K2[hdof(side, a), hdof(side, a)] += kleg[side, a]
            C2[hdof(side, a), hdof(side, a)] += cleg[side, a]
    f3 = {}
    for a, ax in enumerate(AXES):
        sel = [hdof(l, a) for l in range(N_LUMP)]
        lam = eigh(K2[np.ix_(sel, sel)], Mm[np.ix_(sel, sel)],
                   eigvals_only=True)
        f3[ax] = np.sort(np.sqrt(np.abs(lam)) / (2 * np.pi))
    return dict(M=Mm, K_free=K, C_free=C, K=K2, C=C2, kleg=kleg, cleg=cleg,
                kl=kleg[0, 2], kr=kleg[1, 2], cl=cleg[0, 2], cr=cleg[1, 2],
                m_h=m_h, m_t=m_t, m_w=m_w, masses=masses,
                f_body=f3["z"], f_body3=f3)


def foot_schedule(side, t, cfg, seg):
    fp, v, beta = cfg["pacing_freq"], cfg["walk_speed"], cfg["duty_factor"]
    sf = 0.5 * fp
    step_len = v / fp
    stride_len = v / sf
    off = 0.5 if side == "R" else 0.0
    ph = sf * t - off
    n = np.floor(ph)
    frac = ph - n
    plant_x = v * ((n + off) / sf) + 0.5 * step_len
    if frac < beta:
        return plant_x, 0.0, True
    s = (frac - beta) / (1.0 - beta)
    s_smooth = 0.5 - 0.5 * np.cos(np.pi * s)
    x = plant_x + stride_len * s_smooth
    y = cfg["foot_lift"] * np.sin(np.pi * s)
    return x, y, False


def stance_weight(side, t, cfg, kind="load"):
    fp, beta = cfg["pacing_freq"], float(cfg["duty_factor"])
    sf = 0.5 * fp
    off = 0.5 if side == "R" else 0.0
    ph = sf * np.asarray(t, dtype=float) - off
    frac = ph - np.floor(ph)
    tw = float(np.clip(beta - 0.5, 1e-3, 0.5 * beta))
    if kind == "contact":
        tw *= 0.25
    rise = float(cfg.get("contact_rise_s", 0.0))
    ds = float(np.clip(beta - 0.5, 1e-3, 0.5 * beta))
    if rise > 0.0 and kind == "contact":
        tw = max(tw, rise * sf)
        tw = min(tw, 0.45 * beta)
        tw = min(tw, ds)
    ramp = np.minimum(frac / tw, (beta - frac) / tw)
    ramp = np.clip(np.where(frac < beta, ramp, 0.0), 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(np.pi * ramp)
    return float(w) if np.ndim(t) == 0 else w


def gait_clock(cfg, t):
    t = np.asarray(t, float)
    cv = float(cfg.get("gait_step_cv", 0.0) or 0.0)
    if cv <= 0.0 or t.size == 0:
        return t.copy()
    fp = float(cfg["pacing_freq"])
    T0 = 1.0 / fp
    n = int(np.ceil(float(t[-1]) * fp * 1.6)) + 8
    rng = np.random.default_rng(900001 + 7919 * int(cfg.get("gait_seed", 0)))
    d = T0 * np.clip(1.0 + cv * rng.standard_normal(n), 0.5, 1.5)
    s = np.concatenate(([0.0], np.cumsum(d)))
    return np.interp(t, s, np.arange(s.size) * T0)


def gait_hf_amplitudes(cfg, fp):
    if not cfg.get("gait_hf_on", True):
        return []
    fp = float(fp)
    if fp <= 0.0:
        return []
    W = float(cfg["body_mass"]) * G_ACC
    C = float(cfg.get("gait_hf_coef", 42.0)) * W / 746.0
    fmax = float(cfg.get("gait_hf_fmax", 75.0))
    h0 = len(cfg["gait_dlf"]) + 1
    out = []
    h = h0
    while h * fp <= fmax:
        out.append((h, 2.0 * fp * C * fp ** 1.43 / (h * fp) ** 1.3))
        h += 1
    return out


def gait_force(cfg, t, tau=None):
    if not cfg["walk_force_on"]:
        return np.zeros_like(t)
    tau = gait_clock(cfg, t) if tau is None else np.asarray(tau, float)
    G = cfg["body_mass"] * G_ACC
    fp = cfg["pacing_freq"]
    th = 2 * np.pi * fp * tau
    out = np.zeros_like(tau)
    for i, (a, phi) in enumerate(zip(cfg["gait_dlf"], cfg["gait_phase"]), start=1):
        out += G * a * np.sin(i * th - phi)
    for h, A in gait_hf_amplitudes(cfg, fp):
        out += A * np.cos(h * th)
    return out


def gait_force3(cfg, t, tau=None):
    t = np.asarray(t, float)
    out = np.zeros((3, t.size))
    if not cfg["walk_force_on"] or is_standing(cfg):
        return out
    tau = gait_clock(cfg, t) if tau is None else np.asarray(tau, float)
    out[2] = gait_force(cfg, t, tau)
    G = cfg["body_mass"] * G_ACC
    fp = float(cfg["pacing_freq"])
    th = 2 * np.pi * fp * tau
    dx = tuple(cfg.get("gait_dlf_x") or ())
    px = tuple(cfg.get("gait_phase_x") or ()) + (0.0,) * len(dx)
    for i, a in enumerate(dx, start=1):
        out[0] += G * a * np.sin(i * th - px[i - 1])
    dy = tuple(cfg.get("gait_dlf_y") or ())
    py = tuple(cfg.get("gait_phase_y") or ()) + (0.0,) * len(dy)
    for k, a in enumerate(dy, start=1):
        out[1] += G * a * np.sin(k * 0.5 * th - py[k - 1])
    rx = float(cfg.get("gait_hf_ratio_x", 0.0) or 0.0)
    ry = float(cfg.get("gait_hf_ratio_y", 0.0) or 0.0)
    if cfg.get("gait_hf_on", True) and (rx > 0 or ry > 0) and fp > 0:
        W = float(cfg["body_mass"]) * G_ACC
        C = float(cfg.get("gait_hf_coef", 42.0)) * W / 746.0
        fmax = float(cfg.get("gait_hf_fmax", 75.0))
        if rx > 0:
            for h in range(len(dx) + 1, int(fmax / fp) + 1):
                A = 2.0 * fp * C * fp ** 1.43 / (h * fp) ** 1.3
                out[0] += rx * A * np.cos(h * th)
        if ry > 0:
            k = len(dy) + 1
            k += (k % 2 == 0)
            while k * 0.5 * fp <= fmax:
                fk = k * 0.5 * fp
                A = 2.0 * fp * C * fp ** 1.43 / fk ** 1.3
                out[1] += ry * A * np.cos(k * 0.5 * th)
                k += 2
    return out


def is_standing(cfg):
    return str(cfg.get("human_posture", "walk")).lower() == "stand"


def gait_kinematics(cfg, t, tau, seg, F3=None):
    t = np.asarray(t, float)
    out = np.zeros((t.size, N_LUMP, 3))
    if is_standing(cfg):
        return out
    F3 = gait_force3(cfg, t, tau) if F3 is None else F3
    Mb = float(cfg["body_mass"])
    m_w = float(cfg["frac_wrist"]) * Mb
    axL, azL = arm_swing_kin(cfg, tau, "L", seg)
    axR, azR = arm_swing_kin(cfg, tau, "R", seg)
    trunk = np.zeros((t.size, 3))
    trunk[:, 0] = (F3[0] - m_w * (axL + axR)) / Mb
    trunk[:, 1] = F3[1] / Mb
    trunk[:, 2] = (F3[2] - m_w * (azL + azR)) / Mb
    for l in range(N_LUMP):
        out[:, l, :] = trunk
    out[:, 3, 0] += axL; out[:, 3, 2] += azL
    out[:, 4, 0] += axR; out[:, 4, 2] += azR
    return out


def deck_width(cfg):
    return (cfg["deck_W"] if cfg.get("deck_section", "rectangle") ==
            "rectangle" else cfg["sec_flange_w"])


def make_crowd_members(cfg, L, rng=None):
    if not (cfg["crowd_on"] and cfg["crowd_amp"] > 0 and cfg["crowd_n"] > 0):
        return []
    rng = rng or np.random.default_rng(cfg["crowd_seed"])
    Wd = deck_width(cfg); margin = min(0.30, 0.3 * Wd)
    out = []
    for i in range(cfg["crowd_n"]):
        typ = cfg["crowd_type"]
        out.append(dict(
            type=typ,
            amp=cfg["crowd_amp"] * (1 + rng.uniform(-cfg["crowd_amp_var"],
                                                    cfg["crowd_amp_var"])),
            x0=float(rng.uniform(0.08 * L, 0.92 * L)),
            y=float(rng.uniform(-(Wd / 2 - margin), Wd / 2 - margin)),
            fp=cfg["crowd_freq"] * (1 + rng.uniform(-0.1, 0.1)),
            ph=float(rng.uniform(0, 2 * np.pi)),
            speed=float(rng.uniform(0.9, 1.4)) if typ == "walk" else 0.0,
            dirn=int(rng.choice([-1, 1])),
            seed=int(cfg["crowd_seed"]) + 31 * i + 7))
    return out


def crowd_pos(mem, t, L):
    t = np.asarray(t, dtype=float)
    if mem["type"] != "walk" or mem["speed"] <= 0:
        return np.full(t.shape, mem["x0"])
    lo, hi = 0.06 * L, 0.94 * L
    span = hi - lo
    s = (mem["x0"] - lo) + mem["dirn"] * mem["speed"] * t
    sm = np.mod(s, 2 * span)
    return lo + np.where(sm <= span, sm, 2 * span - sm)


def _stoch_cycles(mem, T_total, rate, key, seed_off, sigT, sigA):
    cached = mem.get(key)
    if cached is not None and cached[0] >= T_total - 1e-9:
        return cached[1]
    rng = np.random.default_rng(mem["seed"] + seed_off)
    ts = []; Ts = []; As = []
    tc = -float(rng.uniform(0.0, 1.0 / rate))
    while tc < T_total:
        Tj = (1.0 / rate) * max(0.5, 1 + sigT * rng.standard_normal())
        ts.append(tc); Ts.append(Tj)
        As.append(max(0.2, 1 + sigA * rng.standard_normal()))
        tc += Tj
    tab = (np.array(ts), np.array(Ts), np.array(As))
    mem[key] = (float(T_total), tab)
    return tab


def _jump_cycles(mem, T_total):
    return _stoch_cycles(mem, T_total, mem["fp"], "_jc", 11, 0.05, 0.15)


def _impulse_train(t, cycles, amp, dur):
    ts, Ts, As = cycles
    out = np.zeros_like(t)
    dt = float(t[1] - t[0])
    for tc, Tj, Aj in zip(ts, Ts, As):
        i0 = max(0, int(np.ceil((tc - t[0]) / dt)))
        i1 = min(len(t), int((tc + dur - t[0]) / dt) + 1)
        if i1 > i0:
            tau = np.clip(t[i0:i1] - tc, 0.0, dur)
            out[i0:i1] -= amp * Aj * np.sin(np.pi * tau / dur)
    return out


def _jump_phase(mem, tk):
    if "_jc" not in mem:
        return ((mem["fp"] * tk + mem["ph"] / (2 * np.pi)) % 1.0, 1.0,
                1.0 / mem["fp"])
    ts, Ts, As = mem["_jc"][1]
    i = int(np.searchsorted(ts, tk, side="right") - 1)
    i = max(0, min(i, len(ts) - 1))
    ph = (tk - ts[i]) / Ts[i]
    return float(np.clip(ph, 0.0, 0.9999)), float(As[i]), float(Ts[i])


def crowd_member_force(mem, t, cfg):
    A, fp, ph = mem["amp"], mem["fp"], mem["ph"]
    if mem["type"] == "static":
        return np.zeros_like(t)
    if mem["type"] == "jump":
        cycles = _jump_cycles(mem, float(t[-1]))
        ts, Ts, As = cycles
        out = np.zeros_like(t)
        dt = float(t[1] - t[0])
        for tc, Tj, Aj in zip(ts, Ts, As):
            tcont = 0.5 * Tj
            i0 = max(0, int(np.ceil((tc - t[0]) / dt)))
            i1 = min(len(t), int((tc + tcont - t[0]) / dt) + 1)
            if i1 > i0:
                tau = np.clip(t[i0:i1] - tc, 0.0, tcont)
                out[i0:i1] -= A * Aj * np.sin(np.pi * tau / tcont)
        out += _impulse_train(t, cycles, 0.8 * A, 0.030)
        return out
    if mem["type"] == "walk":
        rng = np.random.default_rng(mem["seed"] + 5)
        dt = float(t[1] - t[0])
        drift = sps.sosfilt(sps.butter(2, 0.3, fs=1 / dt, output="sos"),
                            rng.standard_normal(len(t)))
        drift = 0.03 * drift / (np.std(drift) + 1e-12)
        phase = 2 * np.pi * np.cumsum(fp * (1.0 + drift)) * dt + ph
        out = np.zeros_like(t)
        for i, (a, phi) in enumerate(zip(cfg["gait_dlf"],
                                         cfg["gait_phase"]), 1):
            out += A * a * np.sin(i * phase - phi)
        out += _impulse_train(
            t, _stoch_cycles(mem, float(t[-1]), fp, "_wc", 23, 0.03, 0.20),
            0.25 * A, 0.030)
        return out
    rng = np.random.default_rng(mem["seed"])
    w = rng.standard_normal(len(t))
    dt = float(t[1] - t[0])
    fc = min(cfg["psd_fmax"] * 1.5, 0.45 / dt)
    w = sps.sosfilt(sps.butter(4, fc, btype="low", fs=1 / dt, output="sos"), w)
    return A * w / (np.std(w) + 1e-12)


def crowd_member_force_xy(mem, t, cfg):
    z = np.zeros_like(np.asarray(t, float))
    A, fp, ph = mem["amp"], mem["fp"], mem["ph"]
    if mem["type"] == "walk":
        rng = np.random.default_rng(mem["seed"] + 5)
        dt = float(t[1] - t[0])
        drift = sps.sosfilt(sps.butter(2, 0.3, fs=1 / dt, output="sos"),
                            rng.standard_normal(len(t)))
        drift = 0.03 * drift / (np.std(drift) + 1e-12)
        phase = 2 * np.pi * np.cumsum(fp * (1.0 + drift)) * dt + ph
        fx, fy = z.copy(), z.copy()
        dx = tuple(cfg.get("gait_dlf_x") or ())
        px = tuple(cfg.get("gait_phase_x") or ()) + (0.0,) * len(dx)
        for i, a in enumerate(dx, start=1):
            fx += A * a * np.sin(i * phase - px[i - 1])
        dy = tuple(cfg.get("gait_dlf_y") or ())
        py = tuple(cfg.get("gait_phase_y") or ()) + (0.0,) * len(dy)
        for k, a in enumerate(dy, start=1):
            fy += A * a * np.sin(k * 0.5 * phase - py[k - 1])
        return fx, fy
    if mem["type"] == "random":
        d = tuple(cfg.get("bench_force_dir") or (0.0, 0.0, 1.0))
        dt = float(t[1] - t[0])
        fc = min(cfg["psd_fmax"] * 1.5, 0.45 / dt)
        sos = sps.butter(4, fc, btype="low", fs=1 / dt, output="sos")
        out = []
        for a_i, off in ((0, 101), (1, 202)):
            r = float(d[a_i]) if len(d) > a_i else 0.0
            if r <= 0.0:
                out.append(z.copy())
                continue
            w = sps.sosfilt(sos, np.random.default_rng(
                mem["seed"] + off).standard_normal(len(t)))
            out.append(r * A * w / (np.std(w) + 1e-12))
        return out[0], out[1]
    return z, z.copy()


def crowd_jump_pose(mem, tk):
    ph, _, Tj = _jump_phase(mem, tk)
    T_fl = 0.5 * Tj
    hmax = G_ACC * T_fl ** 2 / 8.0
    crouch = 0.11
    if ph < 0.12:
        s = np.sin(np.pi * ph / 0.12)
        return dict(dz=-0.55 * crouch * s, knee=0.55 * s, arm=0.30 * s,
                    air=False)
    if ph < 0.20:
        return dict(dz=0.0, knee=0.0, arm=0.0, air=False)
    if ph < 0.38:
        e = _sstep((ph - 0.20) / 0.18)
        return dict(dz=-crouch * e, knee=e, arm=-0.9 * e, air=False)
    if ph < 0.50:
        e = _sstep((ph - 0.38) / 0.12)
        return dict(dz=-crouch * (1 - e), knee=0.9 * (1 - e),
                    arm=-0.9 + 1.9 * e, air=False)
    s = (ph - 0.50) / 0.50
    return dict(dz=hmax * 4 * s * (1 - s), knee=0.0,
                arm=(1 - s) * 0.8, air=True)


def _sstep(x):
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def crowd_heading(mem, tk, L, T_turn=0.8):
    if mem["type"] != "walk" or mem["speed"] <= 0:
        return 0.0, 0.0
    lo, hi = 0.06 * L, 0.94 * L
    span = hi - lo
    v = mem["speed"]
    toff = mem["ph"] / (2 * np.pi * mem["fp"])
    sm = float(np.mod((mem["x0"] - lo) + mem["dirn"] * v * (tk + toff),
                      2 * span))
    w = max(v * T_turn / 2.0, 1e-6)
    if span - w <= sm <= span + w:
        p = (sm - (span - w)) / (2 * w)
        th = np.pi * _sstep(p); turn = 1.0 - abs(2 * p - 1)
    elif sm >= 2 * span - w:
        q = (sm - (2 * span - w)) / (2 * w)
        th = np.pi * (1 - _sstep(q)); turn = 1.0 - abs(2 * q - 1)
    elif sm < w:
        q = 0.5 + sm / (2 * w)
        th = np.pi * (1 - _sstep(q)); turn = 1.0 - abs(2 * q - 1)
    else:
        th = 0.0 if sm <= span else np.pi; turn = 0.0
    if mem["dirn"] < 0:
        th = np.pi - th
    return float(th), float(turn)


def crowd_pose(mem, tk, L):
    if mem["type"] == "walk" and mem["speed"] > 0:
        mc = dict(pacing_freq=mem["fp"], walk_speed=mem["speed"],
                  duty_factor=0.62, foot_lift=0.05)
        toff = mem["ph"] / (2 * np.pi * mem["fp"])
        s_body = mem["speed"] * (tk + toff)
        sL, liftL, stL = foot_schedule("L", tk + toff, mc, None)
        sR, liftR, stR = foot_schedule("R", tk + toff, mc, None)

        def to_x(s):
            lo, hi = 0.06 * L, 0.94 * L
            span = hi - lo
            sm = float(np.mod((mem["x0"] - lo) + mem["dirn"] * s, 2 * span))
            return lo + (sm if sm <= span else 2 * span - sm)
        bodyx = to_x(s_body)
        th, turn = crowd_heading(mem, tk, L)
        bob = 0.028 * np.cos(2 * np.pi * mem["fp"] * (tk + toff)) * (1 - turn)
        armsw = np.sin(2 * np.pi * (mem["fp"] / 2) * (tk + toff)) * (1 - turn)
        fL = float(np.clip(sL - s_body, -0.6, 0.6)) * (1 - 0.8 * turn)
        fR = float(np.clip(sR - s_body, -0.6, 0.6)) * (1 - 0.8 * turn)
        return dict(x=bodyx, feet=((fL, liftL * (1 - turn), stL),
                                   (fR, liftR * (1 - turn), stR)),
                    bob=bob, armsw=armsw, hj=0.0, theta=th, jump=None)
    if mem["type"] == "jump":
        jp = crowd_jump_pose(mem, tk)
        return dict(x=mem["x0"], feet=((-0.09, 0.0, not jp["air"]),
                                       (+0.09, 0.0, not jp["air"])),
                    bob=0.0, armsw=0.0, hj=max(jp["dz"], 0.0), theta=0.0,
                    jump=jp)
    return dict(x=mem["x0"], feet=((-0.12, 0.0, True), (+0.12, 0.0, True)),
                bob=0.0, armsw=0.0, hj=0.0, theta=0.0, jump=None)


def build_crowd(cfg, t, B, L, dt, sink=None):
    members = make_crowd_members(cfg, L)
    if not members:
        return [], []
    n_steps = len(t)
    xs = B["xs"]; le = B["le"]; nel = B["n_el"]
    lines = (("A", "B") if B["kind"] == "shell" else ("C",))
    acc = {ln: np.zeros((nel + 1, n_steps)) for ln in lines}
    tor = np.zeros((nel + 1, n_steps)) if B.get("torsional") else None
    hx = {ln: np.zeros((nel + 1, n_steps)) for ln in lines}
    hy = {ln: np.zeros((nel + 1, n_steps)) for ln in lines}
    my = np.zeros((nel + 1, n_steps)) if tor is not None else None
    mz = np.zeros((nel + 1, n_steps)) if tor is not None else None
    zt = float(B.get("ztop", 0.0))
    anyh = False
    hists = []
    steps = np.arange(n_steps)
    for mem in members:
        F = crowd_member_force(mem, t, cfg)
        hists.append(F)
        Fx, Fy = crowd_member_force_xy(mem, t, cfg)
        X = crowd_pos(mem, t, L)
        ie = np.minimum((X / le).astype(int), nel - 1)
        xi = (X - xs[ie]) / le
        if B["kind"] == "shell":
            eta = float(np.clip(mem["y"] / max(B["yL"], 1e-9), -1, 1))
            parts = (("A", (1 - eta) / 2), ("B", (1 + eta) / 2))
        else:
            parts = (("C", 1.0),)
        for ln, w in parts:
            np.add.at(acc[ln], (ie, steps), F * w * (1 - xi))
            np.add.at(acc[ln], (ie + 1, steps), F * w * xi)
        if tor is not None:
            np.add.at(tor, (ie, steps), F * mem["y"] * (1 - xi))
            np.add.at(tor, (ie + 1, steps), F * mem["y"] * xi)
        if np.any(Fx) or np.any(Fy):
            anyh = True
            for ln, w in parts:
                for H, Fh in ((hx, Fx), (hy, Fy)):
                    np.add.at(H[ln], (ie, steps), Fh * w * (1 - xi))
                    np.add.at(H[ln], (ie + 1, steps), Fh * w * xi)
            if tor is not None:
                for M_, Fm in ((tor, -zt * Fy), (my, zt * Fx),
                               (mz, -mem["y"] * Fx)):
                    np.add.at(M_, (ie, steps), Fm * (1 - xi))
                    np.add.at(M_, (ie + 1, steps), Fm * xi)
    if sink is None:
        return members, hists
    for ln in lines:
        for i in range(nel + 1):
            h = acc[ln][i]
            if not np.any(h):
                continue
            node = (B["nA"](i) if ln == "A" else B["nB"](i)) \
                if B["kind"] == "shell" else i + 1
            sink(node, 2, h)
    if tor is not None:
        for i in range(nel + 1):
            h = tor[i]
            if np.any(h):
                sink(i + 1, 3, h)
    if anyh:
        for ln in lines:
            for i in range(nel + 1):
                node = (B["nA"](i) if ln == "A" else B["nB"](i)) \
                    if B["kind"] == "shell" else i + 1
                for d0, H in ((0, hx), (1, hy)):
                    if np.any(H[ln][i]):
                        sink(node, d0, H[ln][i])
        if tor is not None:
            for i in range(nel + 1):
                for d0, M_ in ((4, my), (5, mz)):
                    if np.any(M_[i]):
                        sink(i + 1, d0, M_[i])
    return members, hists


def excitation_series(cfg, t):
    A = cfg["exc_amp"]
    if cfg["exc_type"] == "sine" and A > 0:
        return A * np.sin(2 * np.pi * cfg["exc_freq"] * t)
    if cfg["exc_type"] == "random" and A > 0:
        dt = float(t[1] - t[0])
        rng = np.random.default_rng(int(cfg["sens_seed"]) + 777)
        w = rng.standard_normal(len(t))
        fc = min(cfg["psd_fmax"] * 1.5, 0.45 / dt)
        w = sps.sosfilt(sps.butter(4, fc, btype="low", fs=1/dt, output="sos"), w)
        return A * w / (np.std(w) + 1e-12)
    return np.zeros_like(t)


def bench_force_histories3(cfg, t):
    pts, Hz = bench_force_histories(cfg, t)
    H = np.zeros((len(pts), 3, len(t)))
    if not pts:
        return pts, H
    H[:, 2] = Hz
    dirs = tuple(cfg.get("bench_force_dir") or (0.0, 0.0, 1.0))
    dt = float(t[1] - t[0]) if len(t) > 1 else cfg["run_dt"]
    fs = 1.0 / dt
    fc = min(cfg["psd_fmax"] * 1.5, 0.45 * fs)
    sos = sps.butter(4, fc, btype="low", fs=fs, output="sos")
    ftype = cfg.get("bench_force_type", "random")
    amb = float(cfg.get("bench_ambient", 0.0) or 0.0)
    for j, pt in enumerate(pts):
        for a in (0, 1):
            fr = float(dirs[a]) if a < len(dirs) else 0.0
            if fr <= 0.0:
                continue
            level = pt["amp"] if ftype == "random" else amb
            if level <= 0.0:
                continue
            g = np.random.default_rng(int(cfg["bench_seed"]) * 31 + 7001
                                      + 104729 * j + 15485863 * a)
            w = sps.sosfilt(sos, g.standard_normal(len(t)))
            H[j, a] = fr * level * w / (np.std(w) + 1e-12)
    return pts, H


def bench_channels(B):
    out = []
    xs = B["xs"]
    ye = float(B.get("yedge", B.get("yL", 0.0)))
    for bn, fr, sd in B["bench"]:
        x = float(fr) * float(xs[-1])
        for a in range(3):
            out.append((bn, fr, sd, a, x, sd * ye))
    return out


def ambient_lead_s(cfg, B):
    if cfg.get("rigid_base", False):
        return 0.0
    if cfg.get("lead_s") is not None:
        return max(0.0, float(cfg["lead_s"]))
    ray = B.get("rayleigh") or {}
    w1 = 2.0 * np.pi * float(ray.get("f1", 0.0))
    zw = 0.5 * float(ray.get("alphaM", 0.0)) \
        + 0.5 * float(ray.get("betaK", 0.0)) * w1 * w1
    lead = float(cfg.get("lead_min_s", 20.0))
    if zw > 0.0:
        lead = max(lead, float(cfg.get("lead_decays", 3.0)) / zw)
    return lead


def _deck_lead_in(solver, GbR, H):
    neq = solver.neq
    u = np.zeros(neq); ud = np.zeros(neq); ud2 = np.zeros(neq)
    if GbR is None or H is None or H.shape[1] == 0:
        return u, ud, ud2
    s = solver
    GM = (s.G * s.Mdiag[None, :]) if s.Mdiag is not None else s.G @ s.M
    GC = s.G @ s.C
    A = np.hstack([s.a0 * GM + s.a1 * GC, s.a2 * GM + s.a4 * GC,
                   s.a3 * GM + s.a5 * GC])
    g_, dt = s.gamma, s.dt
    x = np.zeros(3 * neq)
    n = H.shape[1]
    for c0 in range(0, n, 4096):
        U = GbR @ H[:, c0:c0 + 4096]
        for k in range(U.shape[1]):
            x[:neq], x[neq:2 * neq], x[2 * neq:] = u, ud, ud2
            un = A @ x + U[:, k]
            ud2n = s.a0 * (un - u) - s.a2 * ud - s.a3 * ud2
            ud = ud + dt * ((1 - g_) * ud2 + g_ * ud2n)
            u, ud2 = un, ud2n
    return u, ud, ud2


def run_case(cfg, record_anim=False):
    cfg = apply_gait_stage(dict(cfg))
    stand = is_standing(cfg)
    B = build_deck(cfg)
    xs, le, n_el = B["xs"], B["le"], B["n_el"]
    L, dt = cfg["deck_L"], cfg["run_dt"]
    v = 0.0 if stand else float(cfg["walk_speed"])
    seg = human_segments(cfg)
    hum = build_human(cfg)
    T_lead = ambient_lead_s(cfg, B)
    n_trim = int(round(T_lead / dt)) if stand else 0
    n_steps = (int(round(float(cfg.get("stand_T", 60.0)) / dt)) + n_trim
               if stand else int(round((L / v) / dt)))
    t = np.arange(1, n_steps + 1) * dt
    n_lead = 0 if stand else int(round(T_lead / dt))
    t_ext = (np.arange(1, n_lead + n_steps + 1) * dt) if n_lead else t
    P_stat = cfg["body_mass"] * G_ACC
    P_w = 0.0 if stand else P_stat
    tau = t.copy() if stand else gait_clock(cfg, t)
    F3 = gait_force3(cfg, t, tau)
    a_gait = gait_kinematics(cfg, t, tau, seg, F3)
    rigid = bool(cfg.get("rigid_base", False))
    x_stand = float(cfg.get("stand_x_frac", 0.5)) * L
    half_w = 0.5 * float(cfg.get("stance_width" if stand else "step_width",
                                 0.0) or 0.0)
    y_foot = (float(B["yw"]) + half_w, float(B["yw"]) - half_w)
    hard_contact = str(cfg.get("gait_contact", "ramp")) == "switch"
    chans = bench_channels(B)

    ext_P, ext_H = [], []
    solver = None if rigid else DeckSolver(B, dt)
    exc_ext = excitation_series(cfg, t_ext)
    exc = exc_ext[n_lead:]
    exc_x = cfg["exc_x_frac"] * L
    bench_pts_live = []; bench_hist = np.zeros((0, 3, n_steps))
    crowd_members, crowd_hist = [], []
    if solver is not None:
        if np.any(exc_ext != 0.0):
            ed = tuple(float(v) for v in
                       (cfg.get("exc_dir") or (0.0, 0.0, 1.0)))[:3]
            ext_P.append(solver.load_vector(exc_x, 0.0, ed))
            ext_H.append(exc_ext)
        if cfg.get("bench_force_live", True):
            bench_pts_live, bench_ext = bench_force_histories3(cfg, t_ext)
            bench_hist = bench_ext[:, :, n_lead:]
            for j, pt in enumerate(bench_pts_live):
                for a in range(3):
                    if np.any(bench_ext[j, a] != 0.0):
                        e3 = np.zeros(3); e3[a] = 1.0
                        ext_P.append(solver.load_vector(pt["x"], pt["y"], e3))
                        ext_H.append(bench_ext[j, a])

        def _sink(node, dof0, hist):
            idx, w = solver.eq_row([(node, dof0, 1.0)])
            if idx.size:
                P = np.zeros(solver.neq); P[idx] = w
                ext_P.append(P)
                ext_H.append(np.concatenate([np.zeros(n_lead), hist])
                             if n_lead else hist)
        crowd_members, crowd_hist = build_crowd(cfg, t, B, L, dt, sink=_sink)
    elif cfg.get("bench_force_live", True):
        bench_pts_live, bench_hist = bench_force_histories3(cfg, t)

    g_, b_ = 0.5, 0.25
    a0 = 1.0 / (b_ * dt * dt); a1 = g_ / (b_ * dt); a2 = 1.0 / (b_ * dt)
    a3 = 1.0 / (2 * b_) - 1.0; a4 = g_ / b_ - 1.0
    a5 = dt * (g_ / (2 * b_) - 1.0)
    Mh = np.diag(hum["M"]).copy()
    Kf, Cf = hum["K_free"], hum["C_free"]
    Ah = Kf + a1 * Cf + a0 * np.diag(Mh)
    Gh = np.linalg.inv(Ah)
    chk = int(cfg.get("_hsi_check_every", 0) or 0)
    resid = []
    kleg, cleg = hum["kleg"], hum["cleg"]
    hip_idx = [np.array([hdof(side, a) for a in range(3)]) for side in (0, 1)]

    q = np.zeros(N_HDOF); qd = np.zeros(N_HDOF); qdd = np.zeros(N_HDOF)
    acc_dyn = np.zeros((n_steps, N_HDOF))
    f_deck = np.zeros((n_steps, 2, 3))
    bench_acc = np.zeros((n_steps, len(chans)))
    if solver is not None:
        neq = solver.neq; Gb = solver.G; Cb = solver.C
        Md = solver.Mdiag
        Mfull = solver.M if Md is None else None
        u = np.zeros(neq); ud = np.zeros(neq); ud2 = np.zeros(neq)
        if ext_P:
            R = np.array(ext_P).T
            GbR = Gb @ R
            Hx = np.array(ext_H)
        else:
            GbR = None; Hx = None
        if n_lead:
            u, ud, ud2 = _deck_lead_in(
                solver, GbR, None if Hx is None else Hx[:, :n_lead])
            if Hx is not None:
                Hx = Hx[:, n_lead:]
        Rtap = np.zeros((len(chans), neq))
        for c, (bn, fr, sd, a, xc, yc) in enumerate(chans):
            cols, Ns = solver.point(xc, yc)
            Rtap[c, cols] += Ns[a]
        gn_cache = {}

        def GN(x, y):
            key = (round(float(x), 9), round(float(y), 9))
            hit = gn_cache.get(key)
            if hit is None:
                cols, Ns = solver.point(x, y)
                hit = (cols, Ns, Gb[:, cols] @ Ns.T)
                if len(gn_cache) > 2048:
                    gn_cache.clear()
                gn_cache[key] = hit
            return hit
        wnode = [solver.dof[i + 1][2] if B["kind"] != "shell" else None
                 for i in range(n_el + 1)]
    ez = np.array([0.0, 0.0, 1.0])

    every_rt = max(1, int(round((1.0 / cfg["anim_fps"]) / dt)))
    if n_steps // every_rt > cfg["anim_max_frames"]:
        every_rt = max(1, n_steps // cfg["anim_max_frames"])
    every = every_rt
    anim = (dict(t=[], beam=[], beamA=[], beamB=[], q=[], px=[], footL=[],
                 footR=[], exc=[], crowd_f=[], bench_f=[])
            if record_anim else None)

    def _nodal(eqs_dof):
        return [0.0 if e < 0 else float(u[e]) for e in eqs_dof]

    n_loop = 0 if (rigid and not record_anim) else n_steps
    for s in range(n_loop):
        tn = t[s]
        if stand:
            fxL = fxR = x_stand; fyL = fyR = 0.0; stL = stR = True
            wL = wR = 0.5; gL = gR = 1.0
        else:
            tg = tau[s]
            fxL, fyL, stL = foot_schedule("L", tg, cfg, seg)
            fxR, fyR, stR = foot_schedule("R", tg, cfg, seg)
            if hard_contact:
                n_st = (1 if stL else 0) + (1 if stR else 0) or 1
                wL = (1.0 / n_st) if stL else 0.0
                wR = (1.0 / n_st) if stR else 0.0
                gL, gR = (1.0 if stL else 0.0), (1.0 if stR else 0.0)
            else:
                wL = stance_weight("L", tg, cfg)
                wR = stance_weight("R", tg, cfg)
                gL = stance_weight("L", tg, cfg, "contact")
                gR = stance_weight("R", tg, cfg, "contact")
        feet = ((fxL, gL, wL, y_foot[0]), (fxR, gR, wR, y_foot[1]))

        hq = a1 * q + a4 * qd + a5 * qdd
        rh = Mh * (a0 * q + a2 * qd + a3 * qdd) + Cf @ hq
        if solver is not None:
            hu = a1 * u + a4 * ud + a5 * ud2
            wv = a0 * u + a2 * ud + a3 * ud2
            rb = (Md * wv if Md is not None else Mfull @ wv) + Cb @ hu
            ub = Gb @ rb
            if GbR is not None:
                ub += GbR @ Hx[:, s]
        f_known = [np.zeros(3), np.zeros(3)]
        con = []
        for side, (fx, g, w, fy) in enumerate(feet):
            on = (0.0 <= fx <= L)
            f_known[side] = -w * (P_w * ez + F3[:, s])
            if solver is not None and on and w > 0.0:
                ub += GN(fx, fy)[2] @ f_known[side]
            if g > 0.0:
                con.append((side, fx, g, on and solver is not None))
        if con:
            nc = 3 * len(con)
            Ghat = np.concatenate([g * (kleg[sd] + a1 * cleg[sd])
                                   for sd, _, g, _ in con])
            gam = np.concatenate([g * cleg[sd] for sd, _, g, _ in con])
            hips = np.concatenate([hip_idx[sd] for sd, _, _, _ in con])
            rho = hq[hips].copy()
            parts = []
            for k_, (sd, fx, g, on) in enumerate(con):
                if on:
                    cols, Ns, GNf = GN(fx, y_foot[sd])
                    rho[3 * k_:3 * k_ + 3] -= Ns @ hu[cols]
                    parts.append((k_, cols, Ns, GNf))
            gr = gam * rho
            for k_, cols, Ns, GNf in parts:
                ub -= GNf @ gr[3 * k_:3 * k_ + 3]
            rh[hips] += gr
            qb = Gh @ rh
            c0 = qb[hips].copy()
            D = np.zeros((nc, nc))
            for k_, cols, Ns, GNf in parts:
                c0[3 * k_:3 * k_ + 3] -= Ns @ ub[cols]
                for m_, cols2, Ns2, GNf2 in parts:
                    D[3 * k_:3 * k_ + 3, 3 * m_:3 * m_ + 3] = Ns @ GNf2[cols]
            E = Gh[np.ix_(hips, hips)]
            s_rel = np.linalg.solve(np.eye(nc) + (E + D) * Ghat[None, :], c0)
            fc = Ghat * s_rel
            qn = qb - Gh[:, hips] @ fc
            un = ub if solver is not None else None
            if parts:
                un = ub.copy()
                for k_, cols, Ns, GNf in parts:
                    un += GNf @ fc[3 * k_:3 * k_ + 3]
            if chk and solver is not None and s % chk == 0:
                nb = solver.neq
                Afull = np.zeros((nb + N_HDOF, nb + N_HDOF))
                Afull[:nb, :nb] = solver.Keff
                Afull[nb:, nb:] = Ah
                bfull = np.concatenate([solver.Keff @ ub, Ah @ qb])
                for k_, (sd, fx, g, on) in enumerate(con):
                    for a in range(3):
                        Gc = Ghat[3 * k_ + a]
                        hh = nb + hip_idx[sd][a]
                        Afull[hh, hh] += Gc
                        if on:
                            cols, Ns = solver.point(fx, y_foot[sd])
                            nrow = np.zeros(nb); nrow[cols] = Ns[a]
                            Afull[:nb, :nb] += Gc * np.outer(nrow, nrow)
                            Afull[:nb, hh] -= Gc * nrow
                            Afull[hh, :nb] -= Gc * nrow
                xd = np.linalg.solve(Afull, bfull)
                xc = np.concatenate([un, qn])
                resid.append(float(np.linalg.norm(xd - xc)
                                   / max(np.linalg.norm(xd), 1e-300)))
            for k_, (sd, fx, g, on) in enumerate(con):
                if on:
                    f_deck[s, sd] = (f_known[sd] + fc[3 * k_:3 * k_ + 3]
                                     - gr[3 * k_:3 * k_ + 3])
        else:
            qn = Gh @ rh
            un = ub if solver is not None else None
        for sd in (0, 1):
            fx, g, w, _fy = feet[sd]
            if g <= 0.0 and 0.0 <= fx <= L and w > 0.0:
                f_deck[s, sd] = f_known[sd]
        qdd_n = a0 * (qn - q) - a2 * qd - a3 * qdd
        qd = qd + dt * ((1 - g_) * qdd + g_ * qdd_n)
        q, qdd = qn, qdd_n
        acc_dyn[s] = qdd
        if solver is not None:
            ud2_n = a0 * (un - u) - a2 * ud - a3 * ud2
            ud = ud + dt * ((1 - g_) * ud2 + g_ * ud2_n)
            u, ud2 = un, ud2_n
            bench_acc[s] = Rtap @ ud2

        if record_anim and s % every == 0:
            anim["t"].append(tn)
            if solver is not None and B["kind"] != "shell":
                wz = _nodal(wnode)
                rx = _nodal([solver.dof[i + 1][3] for i in range(n_el + 1)])
                yed = B["yedge"]
                anim["beam"].append(wz)
                anim["beamA"].append([a - yed * r for a, r in zip(wz, rx)])
                anim["beamB"].append([a + yed * r for a, r in zip(wz, rx)])
            elif solver is not None:
                za = _nodal([solver.dof[B["nA"](i)][2]
                             for i in range(n_el + 1)])
                zb = _nodal([solver.dof[B["nB"](i)][2]
                             for i in range(n_el + 1)])
                anim["beam"].append([0.5 * (a + b) for a, b in zip(za, zb)])
                anim["beamA"].append(za); anim["beamB"].append(zb)
            else:
                z0 = [0.0] * (n_el + 1)
                anim["beam"].append(z0); anim["beamA"].append(z0)
                anim["beamB"].append(z0)
            anim["q"].append(q[2::3].copy())
            anim["px"].append(x_stand if stand else v * tn)
            anim["footL"].append((fxL, fyL, stL))
            anim["footR"].append((fxR, fyR, stR))
            anim["exc"].append(float(exc[s]))
            anim["crowd_f"].append([float(h[s]) for h in crowd_hist])
            anim["bench_f"].append([float(bench_hist[j, 2, s])
                                    for j in range(len(bench_pts_live))])

    acc3_dyn = acc_dyn.reshape(n_steps, N_LUMP, 3)
    acc3 = a_gait + acc3_dyn
    if stand and n_trim > 0:
        k0 = n_trim

        def _cut(x):
            return np.array(x, copy=True)
        t, tau = _cut(t[k0:]), _cut(tau[k0:])
        acc3, a_gait, acc3_dyn = (_cut(acc3[k0:]), _cut(a_gait[k0:]),
                                  _cut(acc3_dyn[k0:]))
        F3, f_deck, bench_acc = (_cut(F3[:, k0:]), _cut(f_deck[k0:]),
                                 _cut(bench_acc[k0:]))
    return dict(hsi_residual=resid, lead_s=float(T_lead),
                t=t, tau=tau, acc3=acc3, acc3_gait=a_gait,
                acc3_dyn=acc3_dyn, acc=acc3[:, :, 2], F_gait=F3,
                f_deck=f_deck, bench_acc=bench_acc, fs=1.0 / dt,
                true_freqs=B["freqs"], mode_labels=B["labels"],
                fe_freqs=B["freqs"], more=B.get("more", []),
                f_body=hum["f_body"], f_body3=hum["f_body3"], xs=xs,
                anim=anim, cfg=cfg, exc_x=float(exc_x), seg=seg,
                crowd_members=crowd_members, standing=stand,
                x_stand=x_stand if stand else None,
                bench_geom=[(fr, sd, a) for _, fr, sd, a, _, _ in chans])


def _white_noise(n, fs, dens_ug, seed):
    if dens_ug <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    return dens_ug * 1e-6 * G_ACC * np.sqrt(fs / 2.0) * rng.standard_normal(n)


def sensor_noise_of(cfg, sensor):
    if cfg.get("sens_noise") is not None:
        return float(cfg["sens_noise"])
    return float(cfg["sens_noise_wrist"] if "wrist" in sensor
                 else cfg["sens_noise_hip"])


def arm_swing_angle(cfg):
    th = float(cfg.get("arm_swing_deg", 20.0))
    if not cfg.get("arm_swing_auto", True):
        return th
    v_ref = float(cfg.get("arm_swing_v_ref", 1.5))
    v = float(cfg.get("walk_speed", v_ref))
    return float(np.clip(th * v / max(v_ref, 1e-6), 4.0, 45.0))


def arm_swing_kin(cfg, t, side, seg):
    l = float(seg["uparm"] + seg["forearm"])
    th = np.deg2rad(arm_swing_angle(cfg)) * float(cfg.get("arm_swing_gain", 1.0))
    if not cfg.get("arm_swing_on", True) or th <= 0.0:
        z = np.zeros_like(np.asarray(t, float))
        return z, z.copy()
    w1 = 2 * np.pi * 0.5 * float(cfg["pacing_freq"])
    ph = 0.0 if side == "L" else np.pi
    t = np.asarray(t, float)
    ax = -l * th * w1 * w1 * np.sin(w1 * t + ph)
    az = l * th * th * w1 * w1 * np.cos(2.0 * w1 * t + 2.0 * ph)
    if t.size:
        T_ramp = max(1e-9, 2.0 * np.pi / w1)
        g = np.clip(t / T_ramp, 0.0, 1.0)
        g = 0.5 - 0.5 * np.cos(np.pi * g)
        ax = ax * g
        az = az * g
    return ax, az


REP_SPEED = 1000000


REP_GAIT = 1000


REP_BENCH = 1000003


REP_CROWD = 2000000


REP_NOISE = 7368787


WALK_TRIM_S = 1.5


WALK_TRIM_MIN_TC = 3.0


TRIM_MIN_KEEP = 256


def walk_trim_s(cfg):
    if is_standing(cfg):
        return 0.0
    Tc = float(cfg["deck_L"]) / float(cfg["walk_speed"])
    return WALK_TRIM_S if Tc > WALK_TRIM_MIN_TC else 0.0


def realised_cadence(res, trim_s):
    fp = float(res["cfg"]["pacing_freq"])
    t = np.asarray(res["t"], float)
    tau = np.asarray(res.get("tau", t), float)
    fs = float(res["fs"])
    i0 = _trim_index(dict(fs=fs, trim_s=float(trim_s)), t.size)
    if t.size - i0 < 2 or t[-1] <= t[i0]:
        return fp
    return float(fp * (tau[-1] - tau[i0]) / (t[-1] - t[i0]))


def run_ensemble(cfg, sensors, record_anim=False):
    n = max(1, int(cfg["n_passes"]))
    stand = is_standing(cfg)
    trim = walk_trim_s(cfg)
    rep = int(cfg.get("rep_seed", 0) or 0)
    passes = []; anim = None; meta = {}
    for i in range(n):
        c = dict(cfg)
        c["gait_seed"] = int(cfg.get("gait_seed", 0)) + i + REP_GAIT * rep
        if i > 0 or rep:
            c["bench_seed"] = (int(cfg.get("bench_seed", 11)) + 7919 * i
                               + REP_BENCH * rep)
            c["crowd_seed"] = (int(cfg["crowd_seed"]) + 17 * i
                               + REP_CROWD * rep)
        if i > 0 and not stand:
            r = np.random.default_rng(4000 + i + REP_SPEED * rep)
            c["walk_speed"] = cfg["walk_speed"] * (1 + r.uniform(
                -cfg["pass_speed_jitter"], cfg["pass_speed_jitter"]))
        res = run_case(c, record_anim=(i == 0 and record_anim))
        if i == 0:
            anim = res["anim"]
            meta = dict(true_freqs=res["true_freqs"], fe_freqs=res["fe_freqs"],
                        mode_labels=res["mode_labels"],
                        more=res.get("more", []),
                        bench_geom=res["bench_geom"],
                        f_body=res["f_body"], f_body3=res["f_body3"],
                        xs=res["xs"], seg=res["seg"],
                        exc_x=res["exc_x"], crowd_members=res["crowd_members"],
                        cfg=res["cfg"], standing=stand,
                        x_stand=res.get("x_stand"))
        clean3 = {}
        for s in sensors:
            lump = SENSOR_DOF[s]
            clean3[s] = dict(X=res["acc3"][:, lump, 0],
                             Y=res["acc3"][:, lump, 1],
                             Z=res["acc3"][:, lump, 2])
        passes.append(dict(clean={s: clean3[s]["Z"] for s in sensors},
                           clean3=clean3,
                           bench=res["bench_acc"], fs=res["fs"],
                           standing=stand,
                           trim_s=trim,
                           fp=None if stand else res["cfg"]["pacing_freq"],
                           fp_real=(None if stand else
                                    realised_cadence(res, trim_s=trim)),
                           tau=np.asarray(res.get("tau", res["t"]), float)))
    x0 = passes[0]["clean"][sensors[0]]
    fs0 = passes[0]["fs"]
    i0 = _trim_index(passes[0], len(x0))
    grid = raw_psd(x0[i0:], fs0, cfg)[0]
    return dict(grid=grid, passes=passes, anim=anim, **meta)


def _trim_index(p, n):
    fs = float(p["fs"])
    tr = float(p.get("trim_s", WALK_TRIM_S))
    if not tr > 0:
        return 0
    i0 = int(tr * fs)
    return i0 if n - i0 >= TRIM_MIN_KEEP else 0


def gait_comb_kmax(cfg, fp):
    fp = max(float(fp), 0.2)
    k = int(np.ceil(2.0 * comb_top_hz(cfg) / fp))
    return int(min(max(k, 1), 200))


def comb_top_hz(cfg):
    top = float(cfg["psd_fmax"])
    if cfg.get("walk_force_on", True) and cfg.get("gait_hf_on", False):
        top = max(top, float(cfg.get("gait_hf_fmax", top)))
    return top + 1.0


def comb_line_freqs(cfg, fp, kmax=None):
    if kmax is None:
        kmax = cfg.get("notch_kmax")
    if kmax is None:
        kmax = gait_comb_kmax(cfg, fp)
    kmax = int(min(max(int(kmax), 1), 200))
    top = comb_top_hz(cfg)
    out = []
    for k in range(1, kmax + 1):
        c = k * 0.5 * float(fp)
        if c > top:
            break
        out.append(c)
    return out


COMB_GATE = dict(hann=0, whole=0, fallback=0)


def detect_comb_lines(sig, fs, cfg, lines):
    if not lines:
        return []
    npseg_h = int(min(8192, len(sig)))
    spacing = min((abs(b - a) for a, b in zip(lines, lines[1:])), default=0.0)
    if (len(lines) > 1 and spacing > 0.0 and npseg_h >= 64
            and 4.0 * float(fs) / npseg_h > spacing):
        keep = _detect_comb_whole_strides(sig, fs, cfg, lines, spacing)
        if keep is not None:
            COMB_GATE["whole"] += 1
            return keep
        COMB_GATE["fallback"] += 1
    else:
        COMB_GATE["hann"] += 1
    npseg = int(min(8192, len(sig)))
    if npseg < 64:
        return list(lines)
    f, P = sps.welch(sps.detrend(np.asarray(sig, float), type="linear"),
                     fs=fs, nperseg=npseg, noverlap=npseg // 2, detrend=False)
    df = f[1] - f[0]
    thr = 10.0 ** (float(cfg.get("comb_snr_db", 6.0)) / 10.0)
    half = max(1.5 * df, 0.15)
    off = np.ones(len(f), bool)
    for c in lines:
        off &= np.abs(f - c) > half
    top_c = max(lines) + 1.0
    off &= f <= top_c
    spacing = min((abs(b - a) for a, b in zip(lines, lines[1:])), default=1.0)
    band = max(8.0 * df, 3.0 * spacing, 1.5)
    keep = []
    for c in lines:
        m_line = np.abs(f - c) <= half
        m_near = off & (np.abs(f - c) <= band)
        if not m_line.any():
            continue
        if int(m_near.sum()) < 4:
            m_near = off
        if int(m_near.sum()) < 4:
            keep.append(c)
            continue
        if float(P[m_line].max()) >= thr * float(np.median(P[m_near])):
            keep.append(c)
    return keep


def _detect_comb_whole_strides(sig, fs, cfg, lines, f0):
    k_of = [c / f0 for c in lines]
    if any(abs(k - round(k)) > 1e-6 * max(1.0, k) for k in k_of):
        return None
    n_str = int(np.floor(len(sig) / float(fs) * f0 + 1e-9))
    if n_str < 2:
        return None
    n = int(round(n_str / f0 * float(fs)))
    n = min(n, len(sig))
    x = np.asarray(sig[:n], float)
    x = x - x.mean()
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / float(fs))
    df = f[1] - f[0]
    thr = 10.0 ** (float(cfg.get("comb_snr_db", 6.0)) / 10.0)
    on = np.zeros(len(f), bool)
    kb = {}
    for c in lines:
        k = int(round(c / df))
        if 0 < k < len(f):
            on[k] = True
            kb[c] = k
    off = ~on & (f > 0) & (f <= max(lines) + 1.0)
    band = max(8.0 * df, 3.0 * f0, 1.5)
    keep = []
    for c in lines:
        k = kb.get(c)
        if k is None:
            continue
        m_near = off & (np.abs(f - c) <= band)
        if int(m_near.sum()) < 4:
            m_near = off
        if int(m_near.sum()) < 4:
            keep.append(c)
            continue
        if float(P[k]) >= thr * float(np.median(P[m_near])):
            keep.append(c)
    return keep


def remove_gait_harmonics(sig, fs, fp, cfg, kmax=None, tau=None, lines=None,
                          return_lines=False):
    out, used = _remove_gait_harmonics(sig, fs, fp, cfg, kmax, tau, lines)
    return (out, used) if return_lines else out


def detect_gait_lines(sig, fs, fp, cfg, kmax=None, tau=None):
    return _remove_gait_harmonics(sig, fs, fp, cfg, kmax, tau,
                                  detect_only=True)[1]


def _remove_gait_harmonics(sig, fs, fp, cfg, kmax=None, tau=None,
                           lines_in=None, detect_only=False):
    if not cfg.get("notch_gait", True):
        return sig, []
    n = len(sig)
    if n < 16:
        return sig, []
    t = np.arange(n) / fs
    tr = None
    if tau is not None:
        tau = np.asarray(tau, float)
        if tau.size != n or not (tau[-1] > tau[0]):
            tau = None
    if tau is not None:
        if cfg.get("comb_order_track", True):
            tr = tau - tau[0]
        else:
            fp = float(fp) * (tau[-1] - tau[0]) / max(t[-1] - t[0], 1e-9)
    if lines_in is not None:
        lines = [float(c) for c in lines_in if c <= 0.49 * fs]
    else:
        lines = comb_line_freqs(cfg, fp, kmax)
        lines = [c for c in lines if c <= 0.49 * fs]
        if cfg.get("comb_detect", True):
            if tr is None:
                lines = detect_comb_lines(sig, fs, cfg, lines)
            else:
                tu = np.linspace(0.0, tr[-1], n)
                su = np.interp(tu, tr, np.asarray(sig, float))
                fs_tau = (n - 1) / tr[-1]
                lines = detect_comb_lines(su, fs_tau, cfg, lines)
    if detect_only:
        return sig, list(lines)
    if not lines:
        return sig, []
    tb = t if tr is None else tr
    short_hz = float(cfg.get("comb_short_hz", 8.0))
    cols_lo = []; cols_hi = []
    for c in lines:
        pair = [np.sin(2 * np.pi * c * tb), np.cos(2 * np.pi * c * tb)]
        (cols_lo if c <= short_hz else cols_hi).extend(pair)
    lim = max(4, int(0.15 * n))
    while len(cols_lo) + len(cols_hi) > lim and cols_hi:
        del cols_hi[-2:]
    while len(cols_lo) > lim and len(cols_lo) > 2:
        del cols_lo[-2:]
    out = sig.astype(float).copy()
    win_s = float(cfg.get("comb_win_s", 3.0))
    if win_s <= 0.0:
        cols = cols_lo + cols_hi
        if cols:
            X = np.column_stack(cols)
            coef, *_ = np.linalg.lstsq(X, out, rcond=None)
            out -= X @ coef
        return out, lines
    if cols_lo:
        X = np.column_stack(cols_lo)
        nseg = max(1, int(round(n / (win_s * fs))))
        edges = np.linspace(0, n, nseg + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a < X.shape[1] * 2:
                continue
            coef, *_ = np.linalg.lstsq(X[a:b], out[a:b], rcond=None)
            out[a:b] -= X[a:b] @ coef
    if cols_hi:
        X = np.column_stack(cols_hi)
        coef, *_ = np.linalg.lstsq(X, out, rcond=None)
        out -= X @ coef
    return out, lines


OBS_NFFT = 2 ** 17


AXIS_NAMES = ("X", "Y", "Z")


AXIS_SEED = (7001, 3001, 0)


def record_csd(chans, fs, cfg, nfft=None):
    x = np.array([sps.detrend(np.asarray(c, float), type="linear")
                  for c in chans])
    nch, n = x.shape
    seg_s = cfg.get("psd_seg_s", 30.0)
    lim = n if seg_s in (None, 0) else int(max(64, float(seg_s) * fs))
    npseg = int(min(lim, n))
    nf = npseg if not nfft else int(max(int(nfft), npseg))
    win = sps.get_window("hann", npseg)
    step = max(1, npseg // 2)
    starts = list(range(0, n - npseg + 1, step)) or [0]
    f = np.fft.rfftfreq(nf, 1.0 / fs)
    m = (f >= cfg["psd_fmin"]) & (f <= cfg["psd_fmax"])
    idx = np.nonzero(m)[0]
    G = np.zeros((idx.size, nch, nch), complex)
    for s0 in starts:
        X = np.fft.rfft(x[:, s0:s0 + npseg] * win, n=nf, axis=1)[:, idx]
        G += X.T[:, :, None] * np.conj(X.T[:, None, :])
    scale = 1.0 / (fs * float(np.sum(win ** 2)) * len(starts))
    G *= scale
    dbl = np.where((idx == 0) | ((nf % 2 == 0) & (idx == nf // 2)), 1.0, 2.0)
    G *= dbl[:, None, None]
    return f[idx], G, npseg / float(fs)


def sv1_of(G):
    return np.maximum(np.linalg.eigvalsh(G)[..., -1].real, 0.0)


def _device_axis_signal(ens, p, i, sensor, noise_ug, a, ax):
    cfg = ens["cfg"]
    x = p["clean3"][sensor][ax]
    sig = x + _white_noise(
        len(x), p["fs"], noise_ug,
        5000 + 17 * i + 101 * SENSOR_SEED[sensor]
        + 1000003 * int(cfg.get("noise_draw", 0))
        + REP_NOISE * int(cfg.get("rep_seed", 0) or 0) + AXIS_SEED[a])
    i0 = _trim_index(p, len(sig))
    return sig[i0:], i0


def _has_comb(p):
    return (not p.get("standing")) and bool(p.get("fp"))


def device_channels(ens, p, i, sensor, noise_ug, comb_lines=None,
                    comb_out=None):
    cfg = ens["cfg"]
    out = []
    for a, ax in enumerate(AXIS_NAMES):
        sig, i0 = _device_axis_signal(ens, p, i, sensor, noise_ug, a, ax)
        if _has_comb(p):
            tau = p.get("tau")
            key = (int(i), str(sensor), ax)
            given = None
            if comb_lines is not None:
                if key not in comb_lines:
                    raise KeyError("device_channels: no matched comb lines "
                                   "for %r" % (key,))
                given = comb_lines[key]
            sig, used = remove_gait_harmonics(
                sig, p["fs"], p["fp"], cfg,
                tau=None if tau is None else tau[i0:], lines=given,
                return_lines=True)
            if comb_out is not None:
                comb_out[key] = list(used)
        out.append(sig)
    return np.array(out)


def device_comb_detect(ens, p, i, sensor, noise_ug, out):
    if not _has_comb(p):
        return out
    cfg = ens["cfg"]
    tau = p.get("tau")
    for a, ax in enumerate(AXIS_NAMES):
        sig, i0 = _device_axis_signal(ens, p, i, sensor, noise_ug, a, ax)
        out[(int(i), str(sensor), ax)] = detect_gait_lines(
            sig, p["fs"], p["fp"], cfg,
            tau=None if tau is None else tau[i0:])
    return out


def _part_sensors(ens, part):
    return [s for s in ens["cfg"]["sensors"] if part in s] or [part]


def _part_noise(ens, part, noise_override=None):
    return tuple(float(noise_override) if noise_override is not None
                 else float(sensor_noise_of(ens["cfg"], s))
                 for s in _part_sensors(ens, part))


def ens_comb_detect(ens, part, noise_override=None):
    if ens.get("standing"):
        return {}
    sens = _part_sensors(ens, part)
    nzs = _part_noise(ens, part, noise_override)
    key = (part, nzs)
    cache = ens.setdefault("_det_cache", {})
    if key in cache:
        return cache[key]
    out = {}
    for i, p in enumerate(ens["passes"]):
        for s, nz in zip(sens, nzs):
            device_comb_detect(ens, p, i, s, nz, out)
    cache[key] = out
    return out


COMB_UNION_RTOL = 1e-9


def comb_union(a, b, rtol=COMB_UNION_RTOL):
    out = []
    for c in sorted([float(x) for x in a] + [float(x) for x in b]):
        if out and abs(c - out[-1]) <= rtol * max(abs(c), 1.0):
            continue
        out.append(c)
    return out


def _same_walk(flex, twin):
    if len(flex["passes"]) != len(twin["passes"]):
        raise ValueError("matched pair: %d flexible records, %d on the twin"
                         % (len(flex["passes"]), len(twin["passes"])))
    for i, (pf, pb) in enumerate(zip(flex["passes"], twin["passes"])):
        if _has_comb(pf) != _has_comb(pb):
            raise ValueError("matched pair: record %d has a comb on one side "
                             "only" % i)
        if not _has_comb(pf):
            continue
        if not np.isclose(float(pf["fp"]), float(pb["fp"]), rtol=1e-9,
                          atol=0.0):
            raise ValueError("matched pair: record %d cadence %r vs %r"
                             % (i, pf["fp"], pb["fp"]))
        tf, tb = pf.get("tau"), pb.get("tau")
        if (tf is None) != (tb is None) or (
                tf is not None and not (np.shape(tf) == np.shape(tb)
                                        and np.allclose(tf, tb, rtol=1e-9,
                                                        atol=1e-12))):
            raise ValueError("matched pair: record %d gait clocks differ" % i)


def matched_comb_lines(flex, twin, part, noise_override=None):
    if flex.get("standing") or twin.get("standing"):
        return {}
    _same_walk(flex, twin)
    a = ens_comb_detect(flex, part, noise_override)
    b = ens_comb_detect(twin, part, noise_override)
    if set(a) != set(b):
        raise KeyError("matched pair: the comb keys differ (%d flexible, %d "
                       "twin)" % (len(a), len(b)))
    return {k: comb_union(a[k], b[k]) for k in a}


def ens_device_spectra(ens, part, noise_override=None, nfft=OBS_NFFT,
                       comb_from=None, comb_twin=None):
    cfg = ens["cfg"]
    if cfg.get("cp_correct", False):
        raise ValueError("cp_correct: the contact-point correction was "
                         "derived for the vertical 5-DOF body and is not "
                         "defined for the triaxial model")
    if comb_from is not None and comb_twin is not None:
        raise ValueError("ens_device_spectra: comb_from (ens is the twin) or "
                         "comb_twin (ens is the flexible ensemble), not both")
    key = (part, None if noise_override is None else float(noise_override),
           int(nfft or 0))
    lines_in = None
    pair = None
    if not ens.get("standing"):
        if comb_from is not None and not comb_from.get("standing"):
            pair = ("B", comb_from, ens)
        elif comb_twin is not None and not comb_twin.get("standing"):
            pair = ("S", ens, comb_twin)
    if pair is not None:
        role, flex, twin = pair
        lines_in = matched_comb_lines(flex, twin, part, noise_override)
        pc = flex.setdefault("_pair_cache", {})
        pkey = (role, id(flex), id(twin)) + key
        hit = pc.get(pkey)
        if hit is not None and hit[0] is twin:
            return hit[1]
        cache = None
    else:
        cache = ens.setdefault("_spec_cache", {})
        if key in cache:
            return cache[key]
    sens = _part_sensors(ens, part)
    per = None; Gsum = {}; Tw = []; f = None; apsd = None
    lines_out = {}
    for i, p in enumerate(ens["passes"]):
        rec = None
        for s in sens:
            nz = (noise_override if noise_override is not None
                  else sensor_noise_of(cfg, s))
            ch = device_channels(ens, p, i, s, nz, comb_lines=lines_in,
                                 comb_out=lines_out)
            f, G, Twin = record_csd(ch, p["fs"], cfg, nfft=nfft)
            sv = sv1_of(G)
            rec = sv if rec is None else rec + sv
            Gsum[s] = G if s not in Gsum else Gsum[s] + G
            dg = np.real(np.einsum("kii->ki", G))
            apsd = dg if apsd is None else apsd + dg
        if s == sens[-1]:
            Tw.append(Twin)
        per = [] if per is None else per
        per.append(rec / len(sens))
    npass = len(ens["passes"])
    ens_sv1 = sum(sv1_of(G / npass) for G in Gsum.values()) / len(sens)
    out = dict(f=f, per=per, ens_sv1=ens_sv1, T_win=Tw,
               axis_psd=apsd / (npass * len(sens)), sensors=sens,
               comb_lines=lines_out)
    if cache is None:
        pc[pkey] = (twin, out)
    else:
        cache[key] = out
    return out


def ens_part_sv1(ens, part, noise_override=None, comb_from=None,
                 comb_twin=None):
    d = ens_device_spectra(ens, part, noise_override, comb_from=comb_from,
                           comb_twin=comb_twin)
    return ens["grid"], np.interp(ens["grid"], d["f"], d["ens_sv1"])


OBS_BAND_MAX_HZ = None


OBS_T1 = 27.5


OBS_T2 = 92.3


def obs_class(oi, t1=None, t2=None):
    t1 = OBS_T1 if t1 is None else t1
    t2 = OBS_T2 if t2 is None else t2
    if t1 is None or t2 is None:
        return "unclassified"
    oi = float(oi)
    if oi < t1:
        return "unobservable"
    if oi < t2:
        return "marginally observable"
    return "observable"


def half_power_band(f, sv1, f_pk, others=(), fmin=None, fmax=None,
                    max_half=None):
    f = np.asarray(f, float); sv = np.asarray(sv1, float)
    if f.size < 3:
        return (float("nan"), float("nan"))
    lo_lim = f[0] if fmin is None else max(f[0], float(fmin))
    hi_lim = f[-1] if fmax is None else min(f[-1], float(fmax))
    k = int(np.argmin(np.abs(f - float(f_pk))))
    while 0 < k < f.size - 1 and sv[k + 1] > sv[k]:
        k += 1
    while 0 < k < f.size - 1 and sv[k - 1] > sv[k]:
        k -= 1
    half = sv[k] / 2.0

    def cross(step):
        m = k
        while 0 < m < f.size - 1:
            n = m + step
            if n < 0 or n >= f.size:
                break
            if sv[n] <= half:
                a, b = sv[m], sv[n]
                if a == b:
                    return float(f[n])
                w = (a - half) / (a - b)
                return float(f[m] + w * (f[n] - f[m]))
            m = n
        return float(f[max(0, min(f.size - 1, m))])

    fL, fU = cross(-1), cross(+1)
    for fo in others:
        fo = float(fo)
        if not np.isfinite(fo) or fo == float(f_pk):
            continue
        a, b = sorted((float(f[k]), fo))
        m = (f >= a) & (f <= b)
        if int(m.sum()) < 3:
            continue
        fv = float(f[m][int(np.argmin(sv[m]))])
        if fo < f[k]:
            fL = max(fL, fv)
        else:
            fU = min(fU, fv)
    if max_half:
        fL = max(fL, float(f[k]) - float(max_half))
        fU = min(fU, float(f[k]) + float(max_half))
    fL = max(fL, lo_lim); fU = min(fU, hi_lim)
    if not (fU > fL):
        return (float("nan"), float("nan"))
    return (fL, fU)


OBS_SIGMA = 0.06


OBS_WALK_WIDTHS = 2.0


PHYSICS_REV = "rev2c"


def display_decode(b64, pk, n):
    import base64 as _b64
    if not b64:
        return np.full(n, np.nan)
    q = np.frombuffer(_b64.b64decode(b64), dtype="<u2").astype(float)
    out = float(pk) * 10.0 ** (-(q / 10.0) / 10.0)
    out[q == 65535] = np.nan
    return out


def display_smooth(y, sig_hz, df):
    y = np.asarray(y, float)
    if not (sig_hz > 0) or not (df > 0):
        return y.copy()
    s = sig_hz / df
    r = max(1, int(np.ceil(3 * s)))
    k = np.arange(-r, r + 1)
    w = np.exp(-0.5 * (k / s) ** 2)
    n = len(y)
    ok = np.isfinite(y) & (y > 0)
    lg = np.where(ok, np.log10(np.where(ok, y, 1.0)), 0.0)
    out = np.full(n, np.nan)
    for i in range(n):
        if not ok[i]:
            continue
        j = np.clip(i + k, 0, n - 1)
        m = ok[j]
        wt = w[m].sum()
        if wt > 0:
            out[i] = 10.0 ** ((w[m] * lg[j[m]]).sum() / wt)
    return out


def _curve_area(g, S, a, b):
    ok = np.isfinite(S)
    if ok.sum() < 2 or b <= a:
        return 0.0
    gg, ss = g[ok], S[ok]
    inside = (gg > a) & (gg < b)
    x = np.concatenate(([a], gg[inside], [b]))
    y = np.concatenate(([np.interp(a, gg, ss)], ss[inside],
                        [np.interp(b, gg, ss)]))
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


OBS_TIE = 1e-9


def obs_curve_mode(g, S, B, f_ic, fL, fU, h):
    g = np.asarray(g, float); S = np.asarray(S, float)
    B = np.asarray(B, float)
    ef, eb = _curve_area(g, S, fL, fU), _curve_area(g, B, fL, fU)
    WL = max(float(g[0]), min(fL, f_ic - 2.0 * h))
    WU = min(float(g[-1]), max(fU, f_ic + 2.0 * h))
    rec = dict(G=int(ef > eb), E_flex=ef, E_bg=eb, W=(WL, WU), h=h,
               W_walk=None, n_max=None,
               f_hat=None, S_peak=None, S_L=None, S_R=None, S_bg=None,
               F=None, P_own=None, P_bg=None, o=0.0)
    if not rec["G"]:
        return rec
    gt = lambda x, y: x > y * (1.0 + OBS_TIE)
    ge = lambda x, y: x >= y * (1.0 - OBS_TIE)
    n = len(g)
    lmax = []
    for k in range(1, n - 1):
        if not (WL <= g[k] <= WU):
            continue
        a, b_, c = S[k - 1], S[k], S[k + 1]
        if not (np.isfinite(a) and np.isfinite(b_) and np.isfinite(c)):
            continue
        if gt(b_, a) and ge(b_, c):
            lmax.append(k)
    rec["n_max"] = len(lmax)
    if not lmax:
        rec.update(F=0.0, P_own=0.0, P_bg=0.0)
        return rec
    w = WU - WL
    AL, AU = WL - OBS_WALK_WIDTHS * w, WU + OBS_WALK_WIDTHS * w
    lo = hi = None
    for j in range(n):
        if AL <= g[j] <= AU:
            if lo is None:
                lo = j
            hi = j
    rec["W_walk"] = (float(g[lo]), float(g[hi]))

    def own(k):
        pk_ = S[k]
        sl = S[k - 1]
        j = k - 2
        while j >= lo and np.isfinite(S[j]) and not gt(S[j], pk_):
            if S[j] < sl:
                sl = S[j]
            j -= 1
        sr = S[k + 1]
        j = k + 2
        while j <= hi and np.isfinite(S[j]) and not gt(S[j], pk_):
            if S[j] < sr:
                sr = S[j]
            j += 1
        return sl, sr, max(0.0, 1.0 - max(sl, sr) / pk_)

    best = SL = SR = Po = None
    for k in lmax:
        sl, sr, p = own(k)
        if (best is None or gt(p, Po)
                or (not gt(Po, p) and gt(S[k], S[best]))):
            best, SL, SR, Po = k, sl, sr, p
    fh, sp = float(g[best]), float(S[best])
    d = abs(fh - f_ic)
    edge = (f_ic - WL) if fh < f_ic else (WU - f_ic)
    F = 1.0 if d <= h else (max(0.0, 1.0 - (d - h) / (edge - h))
                            if edge > h else 0.0)
    Po = float(Po)
    sb = float(B[best]) if np.isfinite(B[best]) else 0.0
    Pb = max(0.0, 1.0 - sb / sp)
    o = (F * Po * Pb) ** (1.0 / 3.0) if F * Po * Pb > 0 else 0.0
    rec.update(f_hat=fh, S_peak=sp, S_L=float(SL), S_R=float(SR),
               S_bg=sb, F=F, P_own=Po, P_bg=Pb, o=o)
    return rec


def obs_oi(o):
    return float(np.floor(10000.0 * float(o) + 0.5)) / 100.0


OBS_PEAK_TOL = 1e-9


def obs_peak_groups(keys):
    groups, reps = [], []
    for i, kk in enumerate(keys):
        kk = [float(x) for x in kk]
        for gi, rp in enumerate(reps):
            if all(abs(a - b) <= OBS_PEAK_TOL * max(abs(a), abs(b))
                   for a, b in zip(kk, rp)):
                groups[gi].append(i)
                break
        else:
            reps.append(kk)
            groups.append([i])
    return groups


def obs_mean(ois, keys=None):
    if not ois:
        return 0.0
    vals = list(ois)
    if keys is not None:
        if len(keys) != len(ois):
            raise ValueError("obs_mean: %d values but %d peak keys"
                             % (len(ois), len(keys)))
        vals = []
        for grp in obs_peak_groups(keys):
            v = [float(ois[i]) for i in grp]
            if all(x == v[0] for x in v):
                vals.append(v[0])
            else:
                s = 0.0
                for x in v:
                    s += x
                vals.append(s / len(v))
    s = 0.0
    for v in vals:
        s += float(v)
    return float(np.floor(100.0 * s / len(vals) + 0.5)) / 100.0


def obs_curve_device(g, S, B, refs, bands, T_res, sigma=None):
    g = np.asarray(g, float)
    sig = OBS_SIGMA if sigma is None else float(sigma)
    df = float(g[1] - g[0])
    Ss = display_smooth(S, sig, df)
    Bs = display_smooth(B, sig, df)
    h = max(1.0 / float(T_res), df)
    if len(refs) != len(bands):
        raise ValueError("obs_curve_device: %d references but %d bands"
                         % (len(refs), len(bands)))
    pm = {}
    for (lbl, fic), (fL, fU) in zip(refs, bands):
        r = obs_curve_mode(g, Ss, Bs, float(fic), float(fL), float(fU), h)
        oi = obs_oi(r["o"])
        pm[str(lbl)] = dict(
            oi=oi, f=round(float(fic), 6), band=obs_class(oi),
            bnd=[round(float(fL), 6), round(float(fU), 6)],
            fh=(None if r["f_hat"] is None else round(r["f_hat"], 4)),
            ph=(None if r["S_peak"] is None else float("%.4g" % r["S_peak"])),
            G=r["G"], F=r["F"], Po=r["P_own"], Pb=r["P_bg"],
            W=[round(r["W"][0], 4), round(r["W"][1], 4)], h=round(h, 4),
            SL=r["S_L"], SR=r["S_R"], Sb=r["S_bg"],
            Ef=r["E_flex"], Eb=r["E_bg"])
    return pm, (Ss, Bs)


def curve_index_run(r, nf, sigma=None, seg_s=30.0):
    if r.get("failed"):
        return r
    g = np.linspace(float(r["fmin"]), float(r["fmax"]), int(nf))
    T = r.get("Tres") or min(float(r.get("Trec") or seg_s), float(seg_s))
    refs = [(x[0], x[1]) for x in r.get("refs") or []]
    bands = r.get("bands") or []
    keys = [(float(fic), float(b[0]), float(b[1]))
            for (_l, fic), b in zip(refs, bands)]
    r["n_peak"] = len(obs_peak_groups(keys))
    ps = []
    for dev in ("watch", "phone"):
        d = r.get(dev)
        if not isinstance(d, dict):
            continue
        if not d.get("c"):
            d.update(pmode={}, oi=0.0, pobs=0.0, band=obs_class(0.0),
                     gated=True)
            captures_from_index(d)
            continue
        S = display_decode(d["c"], d["pk"], len(g))
        B = (display_decode(d["rc"], d["rpk"], len(g)) if d.get("rc")
             else np.zeros(len(g)))
        pm, _ = obs_curve_device(g, S, B, refs, bands, T, sigma)
        d["pmode"] = pm
        ois = [pm[str(l)]["oi"] for l, _f in refs]
        d["oi"] = obs_mean(ois, keys)
        d["pobs"] = d["oi"] / 100.0
        d["band"] = obs_class(d["oi"])
        d["gated"] = True
        captures_from_index(d)
        ps.append(d["pobs"])
    if ps:
        s = 0.0
        for p in ps:
            s += p
        r["pobs"] = s / len(ps)
    return r


BG_DECK_KEYS = ("deck_E", "deck_rho", "deck_t", "deck_W",
                "deck_zeta", "sec_I_set", "sec_shape", "deck_n_modes",
                "bench_force_amps", "bench_n_src", "bench_T",
                "deck_fy", "deck_damp_model")


BG_CANON = dict(deck_E=30.0e9, deck_rho=2400.0, deck_t=0.20,
                deck_W=3.0, deck_zeta=0.02, sec_I_set=None,
                deck_n_modes=6)


def rigid_twin_cfg(cfg):
    c = dict(cfg)
    for k in BG_DECK_KEYS:
        if k in c and k in BG_CANON:
            c[k] = BG_CANON[k]
    c["rigid_base"] = True
    c["noise_draw"] = int(cfg.get("bg_noise_draw", 0))
    return c


def background_key(cfg):
    import json as _json
    c = rigid_twin_cfg(cfg)
    for k in BG_DECK_KEYS:
        c.pop(k, None)
    c.pop("rigid_base", None)
    c.pop("noise_draw", None)
    return _json.dumps(c, sort_keys=True, default=repr)


def captures_from_index(d, t1=None, t2=None):
    t1 = OBS_T1 if t1 is None else t1
    t2 = OBS_T2 if t2 is None else t2
    modes, caps, ferr, dhz = [], [], [], []
    for lbl, m in (d.get("pmode") or {}).items():
        if t1 is None or m.get("oi") is None or float(m["oi"]) < float(t1):
            continue
        if m.get("fh") is None or m.get("ph") is None:
            continue
        cls = "O" if (t2 is not None and float(m["oi"]) >= float(t2)) else "M"
        modes.append(lbl)
        caps.append([float(m["fh"]), float(m["ph"]), cls])
        dhz.append(abs(float(m["fh"]) - float(m["f"])))
        ferr.append(dhz[-1] / max(float(m["f"]), 1e-9))
    d["modes"] = modes
    d["caps"] = caps
    d["n_cap"] = len(caps)
    d["dhz"] = round(float(np.mean(dhz)), 4) if dhz else None
    d["ferr"] = round(100 * float(np.mean(ferr)), 3) if ferr else None
    return d


def _csd_matrix(chans, fs, nperseg, noverlap):
    x = np.asarray(chans, float)
    nch, n = x.shape
    nperseg = int(min(nperseg, n))
    step = max(1, nperseg - int(noverlap))
    win = sps.get_window("hann", nperseg)
    nf = nperseg // 2 + 1
    G = np.zeros((nf, nch, nch), complex)
    k = 0
    for s0 in range(0, n - nperseg + 1, step):
        seg = x[:, s0:s0 + nperseg]
        seg = seg - seg.mean(axis=1, keepdims=True)
        X = np.fft.rfft(seg * win, axis=1).T
        G += np.conj(X)[:, :, None] * X[:, None, :]
        k += 1
    G *= 1.0 / (fs * float(np.sum(win ** 2)) * max(k, 1))
    if nperseg % 2:
        G[1:] *= 2.0
    else:
        G[1:-1] *= 2.0
    return np.fft.rfftfreq(nperseg, 1.0 / fs), G


def _decimate(x, dec):
    dec = int(dec)
    while dec > 10:
        for fct in (10, 9, 8, 7, 6, 5, 4, 3, 2):
            if dec % fct == 0:
                x = sps.decimate(x, fct); dec //= fct
                break
        else:
            x = sps.decimate(x, dec); dec = 1
    if dec > 1:
        x = sps.decimate(x, dec)
    return x


def bench_force_points(cfg):
    case = int(cfg.get("bench_force_case", 1))
    if case not in (1, 2, 3):
        case = 1
    L = cfg["deck_L"]
    halfW = deck_width(cfg) / 2.0
    ef = float(cfg.get("bench_edge_frac", 0.90))
    fr = (0.25, 0.50, 0.75)
    if case == 1:
        ys = (-ef * halfW, 0.0, +ef * halfW)
    elif case == 2:
        ys = (0.0, 0.0, 0.0)
    else:
        ys = (ef * halfW, ef * halfW, ef * halfW)
    amps = _three_of(cfg.get("bench_force_amps"), cfg.get("bench_exc_amp", 120.0))
    frqs = _three_of(cfg.get("bench_force_freqs"), 2.0)
    return [dict(x=f * L, y=float(y), frac=f, label=f"P{i+1}",
                 amp=float(amps[i]), freq=float(frqs[i]))
            for i, (f, y) in enumerate(zip(fr, ys))]


def _three_of(val, default):
    if val is None:
        return [default, default, default]
    a = list(np.atleast_1d(np.asarray(val, dtype=float)))
    while len(a) < 3:
        a.append(a[-1])
    return a[:3]


def bench_force_histories(cfg, t):
    pts = bench_force_points(cfg)
    if not pts:
        return [], np.zeros((0, len(t)))
    dt = float(t[1] - t[0]) if len(t) > 1 else cfg["run_dt"]
    fs = 1.0 / dt
    avar = float(cfg.get("bench_force_amp_var", 0.0))
    ftype = cfg.get("bench_force_type", "random")
    out = np.zeros((len(pts), len(t)))
    for j, pt in enumerate(pts):
        rng_j = np.random.default_rng(int(cfg["bench_seed"]) + 100 * j)
        amp_j = pt["amp"] * (1 + rng_j.uniform(-avar, avar)) if avar \
            else pt["amp"]
        out[j] = _bench_force_history(ftype, amp_j, pt["freq"], len(t), dt,
                                      fs, cfg, rng_j, j=j)
    return pts, out


def _bench_force_history(ftype, amp, freq, n, dt, fs, cfg, rng, j=0):
    fc = min(cfg["psd_fmax"] * 1.5, 0.45 * fs)

    def _band_noise(gen):
        w = gen.standard_normal(n)
        w = sps.sosfilt(sps.butter(4, fc, btype="low", fs=fs, output="sos"), w)
        return w / (np.std(w) + 1e-12)

    amb = float(cfg.get("bench_ambient", 0.0))
    ambient = (amb * _band_noise(np.random.default_rng(
        int(cfg.get("bench_seed", 11)) * 7919 + 13 + 104729 * int(j)))
        if amb > 0.0 else 0.0)

    if ftype == "jump":
        out = np.zeros(n)
        t = np.arange(n) * dt
        fp = max(0.5, float(freq)); tc = -rng.uniform(0, 1.0 / fp)
        while tc < t[-1]:
            Tj = (1.0 / fp) * max(0.5, 1 + 0.05 * rng.standard_normal())
            Aj = max(0.3, 1 + 0.15 * rng.standard_normal())
            dur = 0.5 * Tj
            i0 = max(0, int(np.ceil(tc / dt)))
            i1 = min(n, int((tc + dur) / dt) + 1)
            if i1 > i0:
                tau = np.clip(t[i0:i1] - tc, 0.0, dur)
                out[i0:i1] -= amp * Aj * np.sin(np.pi * tau / dur)
            tc += Tj
        return out + ambient
    if ftype == "static":
        t_ramp = min(5.0, 0.1 * n * dt)
        k = max(1, int(t_ramp / dt))
        g = np.ones(n)
        k = min(k, n)
        g[:k] = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
        return -amp * g + ambient
    return amp * _band_noise(rng) + ambient


def run_benchmark_avt(cfg, verbose=True):
    cfg = dict(cfg)
    B = build_deck(cfg)
    dt = cfg["run_dt"]
    fs = 1.0 / dt
    f1 = max(B["freqs"][0], 0.2)
    T = cfg["bench_T"] if cfg["bench_T"] else max(120.0, 1000.0 / f1)
    n = int(round(T / dt))
    t_ax = np.arange(1, n + 1) * dt
    pts, H3 = bench_force_histories3(cfg, t_ax)
    sol = DeckSolver(B, dt)
    neq = sol.neq
    Pv, Hx = [], []
    for j, pt in enumerate(pts):
        for a in range(3):
            if np.any(H3[j, a] != 0.0):
                e3 = np.zeros(3); e3[a] = 1.0
                Pv.append(sol.load_vector(pt["x"], pt["y"], e3))
                Hx.append(H3[j, a])
    GbR = sol.G @ np.array(Pv).T if Pv else np.zeros((neq, 0))
    Hx = np.array(Hx) if Hx else np.zeros((0, n))
    chans = bench_channels(B)
    Rtap = np.zeros((len(chans), neq))
    for c, (bn, fr, sd, a, xc, yc) in enumerate(chans):
        cols, Ns = sol.point(xc, yc)
        Rtap[c, cols] += Ns[a]
    a0, a1, a2, a3, a4, a5 = sol.a0, sol.a1, sol.a2, sol.a3, sol.a4, sol.a5
    g_ = sol.gamma
    Gb, Cb, Md = sol.G, sol.C, sol.Mdiag
    Mfull = sol.M if Md is None else None
    u = np.zeros(neq); v = np.zeros(neq); ac = np.zeros(neq)
    Y = np.zeros((n, len(chans)))
    t0 = time.time()
    for s in range(n):
        hu = a1 * u + a4 * v + a5 * ac
        wv = a0 * u + a2 * v + a3 * ac
        rb = (Md * wv if Md is not None else Mfull @ wv) + Cb @ hu
        un = Gb @ rb + GbR @ Hx[:, s]
        an = a0 * (un - u) - a2 * v - a3 * ac
        v = v + dt * ((1 - g_) * ac + g_ * an)
        u, ac = un, an
        Y[s] = Rtap @ ac

    bn = float(cfg.get("bench_noise", 0.0) or 0.0)
    if bn > 0:
        for j in range(Y.shape[1]):
            Y[:, j] = Y[:, j] + _white_noise(len(Y), fs, bn,
                                             int(cfg["bench_seed"]) + 991 + j)

    dec = max(1, int(round(fs / max(50.0, 2.5 * cfg["psd_fmax"]))))
    ch = [_decimate(Y[:, j], dec) for j in range(Y.shape[1])]
    fsd = fs / dec
    if cfg.get("bench_nperseg"):
        npseg = int(cfg["bench_nperseg"])
    else:
        npseg = int(2 ** round(np.log2(max(64.0, float(
            cfg.get("bench_seg_s", 41.0)) * fsd))))
    npseg = int(min(npseg, len(ch[0])))
    npseg = min(npseg, 2 ** int(np.floor(
        np.log2(max(len(ch[0]) // 10, 16)))))
    f, G = _csd_matrix(ch, fsd, npseg, npseg // 2)
    m = (f >= cfg["psd_fmin"]) & (f <= cfg["psd_fmax"])
    f = f[m]; G = G[m]
    U, S, _ = np.linalg.svd(G)
    svall = S; U1 = U[:, :, 0]
    sv1 = svall[:, 0]
    n_avg = 1 + max(0, (len(ch[0]) - npseg) // (npseg // 2))
    geom = [(fr, sd, a) for _, fr, sd, a, _, _ in chans]
    sv1_axis, U1_axis, axis_idx = {}, {}, {}
    for a, ax in enumerate(AXES):
        idx = [k for k, g in enumerate(geom) if g[2] == a]
        if not idx:
            continue
        Ua, Sa, _ = np.linalg.svd(G[:, idx][:, :, idx])
        sv1_axis[ax] = Sa[:, 0]; U1_axis[ax] = Ua[:, :, 0]; axis_idx[ax] = idx
    rms = {ax: float(np.std(Y[:, [k for k, g in enumerate(geom)
                                  if g[2] == a]]))
           for a, ax in enumerate(AXES)}
    if verbose:
        case = int(cfg.get("bench_force_case", 1))
        cname = {1: "diagonal", 2: "centerline", 3: "one-edge"}.get(case, "?")
        src = (f"CASE {case} ({cname} 3-pt, "
               f"{cfg.get('bench_force_type','random')}, directions "
               f"{'/'.join(f'{d:g}' for d in cfg.get('bench_force_dir') or (0, 0, 1))})")
        amps_s = "/".join(f"{a:g}" for a in
                          _three_of(cfg.get("bench_force_amps"),
                                    cfg["bench_exc_amp"]))
        print(f"  BENCHMARK ambient vibration test [{time.time()-t0:.1f} s]: "
              f"T={T:.0f} s ({T*f1:.0f} cycles of {B['labels'][0]}) | "
              f"force = {src} @ {amps_s} N (uncorrelated) | "
              f"{len(chans)} channels (x, y, z at "
              f"{len(B['bench'])} tap points) | Welch: Hann, 50% overlap, "
              f"{n_avg} averages, df={f[1]-f[0]:.4f} Hz")
        print("    -> deck response RMS x/y/z "
              + "/".join(f"{rms[a]:.4f}" for a in AXES)
              + f" m/s2, SV1 peak {sv1.max():.3e} (m/s2)^2/Hz")
    return dict(f=f, G=G, sv1=sv1, svall=svall, U1=U1, Y=Y, fs=fs, T=T,
                n_avg=n_avg, df=float(f[1] - f[0]), rms=rms,
                sv1_axis=sv1_axis, U1_axis=U1_axis, axis_idx=axis_idx,
                bench_geom=geom,
                bench_meta=dict(L=float(cfg["deck_L"]),
                                yedge=float(B.get("yedge", 0.0)),
                                ztop=float(B.get("ztop", 0.0))),
                fe_freqs=B["freqs"], mode_labels=B["labels"],
                more=B.get("more", []), cfg=cfg)


def peak_refine(f, logs, k):
    if k <= 0 or k >= len(logs) - 1:
        return float(f[k])
    y0, y1, y2 = float(logs[k - 1]), float(logs[k]), float(logs[k + 1])
    den = y0 - 2.0 * y1 + y2
    if abs(den) < 1e-15:
        return float(f[k])
    d = 0.5 * (y0 - y2) / den
    d = max(-0.5, min(0.5, d))
    return float(f[k] + d * (f[1] - f[0]))


def assess_modes_sv1(f, sv1, fe_freqs, mode_labels, cfg, prom_db=2.0):
    f = np.asarray(f, float)
    sv1 = np.asarray(sv1, float)
    df = float(f[1] - f[0])
    exc = bench_excitation_lines(cfg)
    s = smooth_psd(sv1, df, min(cfg["psd_smooth_hz"], 3.0 * df))
    logs = np.log10(s + 1e-30)
    pk, props = find_peaks(logs, prominence=0.05)
    out = []
    for fn, lab in zip(fe_freqs, mode_labels):
        fn = float(fn)
        rec = dict(f_fe=fn, label=str(lab), observable=False, f_meas=None,
                   prom_db=0.0, reason="", k=None)
        if not (cfg["psd_fmin"] <= fn <= cfg["psd_fmax"]):
            rec["reason"] = "outside the analysis band"
            out.append(rec)
            continue
        win = max(4.0 * df, 0.04 * fn)
        near = [int(i) for i in pk if abs(float(f[i]) - fn) <= win]
        if not near:
            rec["reason"] = "no SV1 peak within %.2f Hz of it" % win
            out.append(rec)
            continue
        k = max(near, key=lambda j: logs[j])
        rec["k"] = k
        rec["prom_db"] = float(10.0 * props["prominences"][list(pk).index(k)])
        rec["f_meas"] = peak_refine(f, logs, k)
        tol = max(cfg["psd_reject_hz"], 2.0 * df)
        on_line = next((tg for fl, tg in exc
                        if abs(rec["f_meas"] - fl) < tol), None)
        if on_line:
            rec["reason"] = f"peak here is the shaker line {on_line}"
        elif rec["prom_db"] < prom_db:
            rec["reason"] = (f"peak only {rec['prom_db']:.1f} dB above the "
                             f"local floor")
        else:
            rec["observable"] = True
        out.append(rec)
    return out


REF_CURVE = {"L": "y", "A": "x"}


def bench_curve(bench, label):
    ax = REF_CURVE.get(str(label)[:1])
    sa = bench.get("sv1_axis") or {}
    if ax and ax in sa:
        return ax, np.asarray(sa[ax], float)
    return "all", np.asarray(bench["sv1"], float)


def benchmark_references(bench, cfg, prom_db=2.0):
    f = np.asarray(bench["f"], float)
    fe = list(bench["fe_freqs"]); lab = list(bench["mode_labels"])
    groups = {}
    for i, l in enumerate(lab):
        nm, _ = bench_curve(bench, l)
        groups.setdefault(nm, []).append(i)
    recs = [None] * len(lab)
    for nm, ids in groups.items():
        _, cv = bench_curve(bench, lab[ids[0]])
        for i, r in zip(ids, assess_modes_sv1(f, cv, [fe[i] for i in ids],
                                              [lab[i] for i in ids], cfg,
                                              prom_db)):
            r["curve"] = nm
            recs[i] = r
    refs, bands, curves = [], [], []
    ref_i = [i for i, r in enumerate(recs)
             if r["observable"] and r["f_meas"] is not None
             and cfg["psd_fmin"] <= r["f_meas"] <= cfg["psd_fmax"]]
    mh = OBS_BAND_MAX_HZ
    for i in ref_i:
        r = recs[i]
        _, cv = bench_curve(bench, r["label"])
        others = [recs[j]["f_meas"] for j in ref_i
                  if recs[j]["curve"] == r["curve"]]
        refs.append((r["label"], float(r["f_meas"])))
        bands.append(half_power_band(f, cv, r["f_meas"], others=others,
                                     fmin=cfg.get("psd_fmin"),
                                     fmax=cfg.get("psd_fmax"), max_half=mh))
        curves.append(r["curve"])
    return dict(assess=recs, refs=refs, bands=bands, curves=curves)


def bench_excitation_lines(cfg):
    out = []
    if cfg.get("bench_force_type") != "jump":
        return out
    seen = set()
    for fq in _three_of(cfg.get("bench_force_freqs"), 2.0):
        fq = float(fq)
        if fq <= 0:
            continue
        k = 1
        while k * fq <= cfg["psd_fmax"]:
            fl = round(k * fq, 6)
            if fl not in seen:
                seen.add(fl)
                out.append((fl, f"{k}xf_shaker"))
            k += 1
    return sorted(out)


def raw_psd(acc, fs, cfg):
    x = sps.detrend(acc, type="linear")
    seg_s = cfg.get("psd_seg_s", 30.0)
    lim = len(x) if seg_s in (None, 0) else int(max(64, float(seg_s) * fs))
    npseg = int(min(lim, len(x)))
    f, P = sps.welch(x, fs=fs, nperseg=npseg, noverlap=npseg // 2,
                     detrend=False)
    m = (f >= cfg["psd_fmin"]) & (f <= cfg["psd_fmax"])
    return f[m], P[m]


def smooth_psd(P, df, sigma_hz):
    if sigma_hz <= 0:
        return P
    return 10 ** gaussian_filter1d(np.log10(np.maximum(P, 1e-30)),
                                   sigma=sigma_hz / df)


def peak_sources(cfg, ens=None, fp=None):
    out = []
    cfg = apply_gait_stage(dict(cfg))
    stand = is_standing(cfg)
    fp = float(cfg["pacing_freq"] if fp is None else fp)
    fmax = float(cfg["psd_fmax"])
    if cfg.get("walk_force_on", True) and not stand:
        k = 1
        while k * 0.5 * fp <= fmax:
            fl = k * 0.5 * fp
            out.append((fl, "gait",
                        f"{k}x f_p/2" + ("  (= walking force "
                                         f"{k//2}x f_p)" if k % 2 == 0 else "")))
            k += 1
        if cfg.get("arm_swing_on", True) and fp <= fmax:
            out.append((0.5 * fp, "arm", "arm swing, fore-aft (x)"))
            out.append((fp, "arm", "arm swing, vertical (z)"))
    if ens is not None:
        fb3 = ens.get("f_body3")
        if fb3:
            for ax in AXES:
                for j, fb in enumerate(np.atleast_1d(fb3.get(ax, [])),
                                       start=1):
                    if float(fb) <= fmax:
                        out.append((float(fb), "body",
                                    f"carrier mode {ax}{j}"))
        else:
            for j, fb in enumerate(np.atleast_1d(ens.get("f_body", [])),
                                   start=1):
                if float(fb) <= fmax:
                    out.append((float(fb), "body", f"carrier mode {j}"))
    v = 0.0 if stand else float(cfg["walk_speed"])
    L = float(cfg["deck_L"])
    fd = v / max(L, 1e-9)
    for k in range(1, 4):
        if fd > 0 and k * fd <= fmax:
            out.append((k * fd, "drive", f"{k}x driving f_d = v/L"))
    if cfg.get("crowd_on") and cfg.get("crowd_amp", 0) > 0 and \
       cfg.get("crowd_type") in ("walk", "jump"):
        for k in range(1, 7):
            if k * cfg["crowd_freq"] <= fmax:
                out.append((k * cfg["crowd_freq"], "crowd", f"{k}x f_crowd"))
    for fl, tag in bench_excitation_lines(cfg):
        out.append((float(fl), "shaker", tag))
    if cfg.get("exc_type") == "sine" and cfg.get("exc_amp", 0) > 0:
        out.append((float(cfg["exc_freq"]), "shaker", "external sine"))
    if ens is not None and fd > 0 and 0.5 * fd >= float(cfg["psd_reject_hz"]):
        for n, fm in enumerate(ens.get("fe_freqs", []), start=1):
            if not (cfg["psd_fmin"] <= fm <= fmax):
                continue
            for sgn in (-1.0, 1.0):
                fl = float(fm) + sgn * 0.5 * fd
                if cfg["psd_fmin"] <= fl <= fmax:
                    out.append((fl, "pair",
                                f"{_mode_name(ens, n)} {'+' if sgn > 0 else '-'}"
                                " f_d/2 (camel hump)"))
    return sorted(out)


DRAW = dict(deck="#9aa6b2", edge="#56677a", shaker="#d9902f", leg="#3f6ea6",
            leg2="#8db0d6", foot="#5b6573", trunk="#a8452c", link="#5b6573",
            arm="#c25a3c", arm2="#e2a07c", skin="#e6c3a0")


SENSOR_COLORS = {"hipL": "#2f6fb5", "hipR": "#6fa3dc", "wristL": "#d63b25",
                 "wristR": "#f08a5d", "torso": "#7f7f7f"}


def _mode_name(base, n):
    labels = base.get("mode_labels")
    return labels[n - 1] if labels and n <= len(labels) else f"M{n}"


def drawn_gait(cfg):
    if is_standing(cfg) or not cfg.get("walk_force_on", True):
        bob = 0.0
    else:
        a1 = float(np.atleast_1d(cfg["gait_dlf"])[0])
        bob = a1 * G_ACC / (2 * np.pi * float(cfg["pacing_freq"])) ** 2
    swing = 0.0 if (is_standing(cfg) or not cfg.get("arm_swing_on", True)) \
        else float(cfg.get("arm_swing_gain", 1.0))
    return bob, swing


def leg_ik(hip, foot, l1, l2, knee_forward=True):
    hx, hy = hip; fx, fy = foot
    dx, dy = fx - hx, fy - hy
    d = np.hypot(dx, dy)
    d = min(d, (l1 + l2) * 0.999)
    d = max(d, abs(l1 - l2) + 1e-4)
    a = (d*d + l1*l1 - l2*l2) / (2 * d * l1)
    a = np.clip(a, -1, 1)
    base = np.arctan2(dy, dx)
    bend = np.arccos(a)
    ang = base + (bend if knee_forward else -bend)
    return (hx + l1 * np.cos(ang), hy + l1 * np.sin(ang))


def _cyl_mesh(p0, p1, r, m=8):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    ax = p1 - p0
    ln = np.linalg.norm(ax)
    ax = ax / ln if ln > 1e-9 else np.array([0.0, 0.0, 1.0])
    tmp = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(ax, tmp); u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    V = []
    for p in (p0, p1):
        for s in range(m):
            th = 2 * np.pi * s / m
            V.append(p + r * (np.cos(th) * u + np.sin(th) * w))
    V += [p0, p1]
    T = []
    for s in range(m):
        s2 = (s + 1) % m
        T += [(s, s2, m + s), (s2, m + s2, m + s),
              (2 * m, s2, s), (2 * m + 1, m + s, m + s2)]
    return V, T


def _sphere_mesh(c, r, nu=8, nv=6):
    c = np.asarray(c, float)
    V = []
    for iv in range(nv + 1):
        phi = np.pi * iv / nv
        for iu in range(nu):
            th = 2 * np.pi * iu / nu
            V.append(c + r * np.array([np.sin(phi) * np.cos(th),
                                       np.sin(phi) * np.sin(th),
                                       np.cos(phi)]))
    T = []
    for iv in range(nv):
        for iu in range(nu):
            a = iv * nu + iu; b = iv * nu + (iu + 1) % nu
            c2 = (iv + 1) * nu + iu; d = (iv + 1) * nu + (iu + 1) % nu
            T += [(a, b, c2), (b, d, c2)]
    return V, T


def _mesh_trace(parts, color, name=None, showlegend=False, opacity=1.0):
    X = []; Y = []; Z = []; I = []; J = []; K = []
    for V, T in parts:
        off = len(X)
        X += [float(v[0]) for v in V]
        Y += [float(v[1]) for v in V]
        Z += [float(v[2]) for v in V]
        for a, b, c_ in T:
            I.append(a + off); J.append(b + off); K.append(c_ + off)
    return go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color=color,
                     opacity=opacity, flatshading=True, name=name,
                     showlegend=showlegend, hoverinfo="skip",
                     lighting=dict(ambient=0.55, diffuse=0.75, specular=0.12))


def _section_rects(cfg):
    sec = cfg.get("deck_section", "rectangle")
    if sec == "rectangle":
        W, t = cfg["deck_W"], cfg["deck_t"]
        return [(-W / 2, W / 2, -t, 0.0)]
    if sec in ("channel", "tee"):
        W, tf = cfg["sec_flange_w"], cfg["sec_flange_t"]
        hw, tw = cfg["sec_web_h"], cfg["sec_web_t"]
        parts = [(-W / 2, W / 2, -tf, 0.0)]
        ycs = [0.0] if sec == "tee" else \
              [-(W / 2 - cfg["sec_web_off"]), +(W / 2 - cfg["sec_web_off"])]
        for yc in ycs:
            parts.append((yc - tw / 2, yc + tw / 2, -tf - hw, -tf))
        return parts
    if sec == "box":
        W, h, tw = cfg["sec_flange_w"], cfg["sec_box_h"], cfg["sec_box_wall"]
        return [(-W / 2, W / 2, -tw, 0.0),
                (-W / 2, W / 2, -h, -h + tw),
                (-W / 2, -W / 2 + tw, -h + tw, -tw),
                (W / 2 - tw, W / 2, -h + tw, -tw)]
    W = cfg["deck_W"]
    return [(-W / 2, W / 2, -section_depth(cfg), 0.0)]


def _extrude_part(xs, y0, y1, z0, z1, wfun):
    corners = ((y0, z0), (y1, z0), (y1, z1), (y0, z1))
    n = len(xs)
    V = []
    for xv in xs:
        for yc, zc in corners:
            V.append((xv, yc, zc + wfun(xv, yc)))
    T = []
    for s in range(n - 1):
        for e in range(4):
            a = s * 4 + e; b = s * 4 + (e + 1) % 4
            c_ = (s + 1) * 4 + e; d = (s + 1) * 4 + (e + 1) % 4
            T += [(a, b, c_), (b, d, c_)]
    T += [(0, 1, 2), (0, 2, 3)]
    m0 = (n - 1) * 4
    T += [(m0, m0 + 2, m0 + 1), (m0, m0 + 3, m0 + 2)]
    return V, T


def figure_3d(res):
    cfg, anim, seg = res["cfg"], res["anim"], res["seg"]
    if anim is None or not anim["t"]:
        return
    xs_full = np.asarray(res["xs"]); L = cfg["deck_L"]
    S = cfg["anim_disp_scale"]; dep = section_depth(cfg)
    sec = cfg.get("deck_section", "rectangle")
    W = cfg["deck_W"] if sec == "rectangle" else cfg["sec_flange_w"]
    halfW = W / 2.0
    grill = cfg.get("deck_model", "beam") in ("shell", "grillage")
    beamA_all = np.array(anim["beamA"]); beamB_all = np.array(anim["beamB"])
    shell = grill or bool(np.any(np.abs(beamA_all - beamB_all) > 1e-12))
    qs_all = np.array(anim["q"])
    yLg = (halfW - cfg["sec_web_off"]) if grill else halfW
    ywalk = float(np.clip(cfg["walk_y_frac"], -1, 1)) * (halfW - 0.35)
    Sb = cfg["anim_body_scale"]
    bob_amp, sw_amp = drawn_gait(cfg)
    stand = is_standing(cfg)
    L1, L2 = seg["thigh"], seg["shank"]
    UA, FA = seg["uparm"], seg["forearm"]
    leg_len, torso_h, head_h = seg["leg_len"], seg["torso"], seg["head"]
    crowd_members = res.get("crowd_members", [])
    rects = _section_rects(cfg)
    deck_col = DRAW["deck"]
    bench_pts = (bench_force_points(cfg)
                 if cfg.get("bench_force_live", True) else [])
    bf_type = cfg.get("bench_force_type", "random")
    bf_col = DRAW["shaker"]

    stride = max(1, len(anim["t"]) // 120)
    ks = list(range(0, len(anim["t"]), stride))

    def wfun_frame(k):
        zA = beamA_all[k] * S; zB = beamB_all[k] * S
        if shell and yLg > 1e-6:
            def wf(x, y):
                a = float(np.interp(x, xs_full, zA))
                b = float(np.interp(x, xs_full, zB))
                w = (y + yLg) / (2 * yLg)
                return a + (b - a) * w
        else:
            def wf(x, y):
                return float(np.interp(x, xs_full, zA))
        return wf

    def zdeck(k, x, y):
        return wfun_frame(k)(np.clip(x, 0, L), y) if 0 <= x <= L else 0.0

    def human_meshes(k):
        q = qs_all[k]; px = anim["px"][k]
        ph = 2 * np.pi * (0.5 * cfg["pacing_freq"]) * anim["t"][k]
        fxL, fyL, _ = anim["footL"][k]; fxR, fyR, _ = anim["footR"][k]
        _hw = 0.5 * float(cfg.get("stance_width" if is_standing(cfg)
                                  else "step_width", 0.0) or 0.0)
        yLt, yRt = ywalk + _hw, ywalk - _hw
        fzL = zdeck(k, fxL, yLt) + fyL
        fzR = zdeck(k, fxR, yRt) + fyR
        bob = bob_amp * np.cos(2.0 * ph)
        hip_z = min(fzL, fzR) + leg_len * 0.94 + bob + 0.5 * (q[0] + q[1]) * Sb
        hipL3 = np.array([px - 0.03, yLt, hip_z])
        hipR3 = np.array([px + 0.03, yRt, hip_z])
        sh_z = hip_z + torso_h + q[2] * Sb
        shL = np.array([px - 0.01, ywalk + 0.17, sh_z])
        shR = np.array([px + 0.05, ywalk - 0.17, sh_z])
        shC = 0.5 * (shL + shR); hipC = 0.5 * (hipL3 + hipR3)
        head_c = np.array([px + 0.02, ywalk, sh_z + head_h * 0.75])
        swL = sw_amp * np.sin(ph); swR = sw_amp * np.sin(ph + np.pi)
        elbL = np.array([shL[0] + UA * 0.85 * swL, ywalk + 0.21,
                         shL[2] - UA * 0.62])
        wrL = np.array([elbL[0] + FA * 1.05 * swL, ywalk + 0.21,
                        elbL[2] - FA * 0.72 + q[3] * Sb])
        elbR = np.array([shR[0] + UA * 0.85 * swR, ywalk - 0.21,
                         shR[2] - UA * 0.62])
        wrR = np.array([elbR[0] + FA * 1.05 * swR, ywalk - 0.21,
                        elbR[2] - FA * 0.72 + q[4] * Sb])
        kneeL = leg_ik((hipL3[0], hipL3[2]), (fxL, fzL), L1, L2)
        kneeR = leg_ik((hipR3[0], hipR3[2]), (fxR, fzR), L1, L2)
        kL3 = np.array([kneeL[0], yLt, kneeL[1]])
        kR3 = np.array([kneeR[0], yRt, kneeR[1]])
        fL3 = np.array([fxL, yLt, fzL + 0.035])
        fR3 = np.array([fxR, yRt, fzR + 0.035])

        groups = {}
        def add(colour, V, T):
            groups.setdefault(colour, []).append((V, T))
        add(DRAW["leg"], *_cyl_mesh(hipL3, kL3, 0.055))
        add(DRAW["leg"], *_cyl_mesh(kL3, fL3, 0.045))
        add(DRAW["leg2"], *_cyl_mesh(hipR3, kR3, 0.055))
        add(DRAW["leg2"], *_cyl_mesh(kR3, fR3, 0.045))
        add(DRAW["foot"], *_cyl_mesh(fL3 + [-0.04, 0, 0], fL3 + [0.17, 0, 0], 0.04))
        add(DRAW["foot"], *_cyl_mesh(fR3 + [-0.04, 0, 0], fR3 + [0.17, 0, 0], 0.04))
        add(DRAW["link"], *_cyl_mesh(hipL3, hipR3, 0.065))
        add(DRAW["trunk"], *_cyl_mesh(hipC, shC, 0.115))
        add(DRAW["link"], *_cyl_mesh(shL, shR, 0.05))
        add(DRAW["link"], *_cyl_mesh(shC, head_c - [0, 0, 0.10], 0.035))
        add(DRAW["arm"], *_cyl_mesh(shL, elbL, 0.042))
        add(DRAW["arm"], *_cyl_mesh(elbL, wrL, 0.036))
        add(DRAW["arm2"], *_cyl_mesh(shR, elbR, 0.042))
        add(DRAW["arm2"], *_cyl_mesh(elbR, wrR, 0.036))
        add(DRAW["skin"], *_sphere_mesh(head_c, 0.105))
        add(DRAW["skin"], *_sphere_mesh(wrL, 0.045))
        add(DRAW["skin"], *_sphere_mesh(wrR, 0.045))
        spos = {"hipL": hipL3 + [0, 0.09, 0], "hipR": hipR3 - [0, 0.09, 0],
                "wristL": wrL + [0, 0.06, 0], "wristR": wrR - [0, 0.06, 0]}
        return groups, spos

    def frame3d(k):
        t3 = []
        wf = wfun_frame(k)
        deck_parts = [_extrude_part(xs_full, y0, y1, z0, z1, wf)
                      for (y0, y1, z0, z1) in rects]
        t3.append(_mesh_trace(deck_parts, deck_col))
        for yy in (-halfW, halfW):
            t3.append(go.Scatter3d(
                x=xs_full, y=[yy] * len(xs_full),
                z=[wf(x, yy) for x in xs_full], mode="lines",
                line=dict(color=DRAW["edge"], width=3), hoverinfo="skip",
                showlegend=False))
        t3.append(go.Scatter3d(
            x=[0, 0, L, L], y=[-halfW, halfW, -halfW, halfW],
            z=[-dep - 0.05] * 4, mode="markers",
            marker=dict(symbol="diamond", size=5, color=DRAW["edge"]),
            hoverinfo="skip", showlegend=False))

        tk3 = anim["t"][k]
        sx = []; sy = []; sz = []
        cx3 = []; cy3 = []; cz3 = []; cw = []
        bxs = []; bys = []; bzs = []
        lx = []; ly = []; lz = []; lt = []
        bfk3 = anim["bench_f"][k] if k < len(anim.get("bench_f", [])) else []
        for pi, pt in enumerate(bench_pts):
            zt = zdeck(k, pt["x"], pt["y"])
            Fnow = abs(bfk3[pi]) if pi < len(bfk3) else pt["amp"]
            puls = 1.0 + 0.15 * np.tanh(
                (Fnow - pt["amp"]) / max(pt["amp"], 1.0))
            hlen = 0.60 * puls
            base = zt + 0.06; top = base + hlen
            sx += [pt["x"], pt["x"], None]
            sy += [pt["y"], pt["y"], None]
            sz += [top, base + 0.22, None]
            cx3.append(pt["x"]); cy3.append(pt["y"]); cz3.append(base + 0.14)
            cw.append(-1.0)
            bxs.append(pt["x"]); bys.append(pt["y"]); bzs.append(base)
            lx.append(pt["x"]); ly.append(pt["y"]); lz.append(top + 0.12)
            lab = f"shaker {pi + 1}"
            lt.append(lab)
        t3.append(go.Scatter3d(x=sx, y=sy, z=sz, mode="lines",
                  line=dict(color=bf_col, width=7), hoverinfo="skip",
                  showlegend=False))
        t3.append(go.Cone(x=cx3, y=cy3, z=cz3,
                  u=[0.0] * len(cx3), v=[0.0] * len(cx3), w=cw,
                  sizemode="absolute", sizeref=0.30, anchor="tip",
                  colorscale=[[0, bf_col], [1, bf_col]], showscale=False,
                  name="shaker",
                  showlegend=bool(bench_pts), hoverinfo="text", text=lt))
        t3.append(go.Scatter3d(x=bxs, y=bys, z=bzs, mode="markers",
                  marker=dict(color="white", size=5,
                  symbol=("square" if bf_type == "static" else "circle"),
                  line=dict(color=bf_col, width=2)),
                  hoverinfo="skip", showlegend=False))
        t3.append(go.Scatter3d(x=lx, y=ly, z=lz, mode="text", text=lt,
                  textfont=dict(color=bf_col, size=9), hoverinfo="skip",
                  showlegend=False))

        groups, spos = human_meshes(k)
        for colour, parts in groups.items():
            t3.append(_mesh_trace(parts, colour))
        for s_ in cfg["sensors"]:
            p = spos[s_]
            t3.append(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]],
                      mode="markers", marker=dict(color=SENSOR_COLORS[s_],
                      size=5, symbol="square",
                      line=dict(color=DRAW["edge"], width=1)),
                      name=s_, legendgroup=s_, showlegend=True,
                      hoverinfo="text", text=s_))

        if crowd_members:
            tk = anim["t"][k]
            parts = []
            cl1 = cl2 = 0.44
            for mem in crowd_members:
                pose = crowd_pose(mem, tk, L)
                cx = pose["x"]; cy = mem["y"]; jp = pose["jump"]
                cth, sth = np.cos(pose["theta"]), np.sin(pose["theta"])

                def bw(fwd, side):
                    return (cx + fwd * cth - side * sth,
                            cy + fwd * sth + side * cth)
                zt0 = zdeck(k, cx, cy)
                dz = jp["dz"] if jp else pose["hj"]
                hipz = zt0 + 0.83 + pose["bob"] + dz
                shz = hipz + 0.45
                hipC = np.array([cx, cy, hipz])
                for (f_fwd, lift, _st), sgn in zip(pose["feet"], (+1, -1)):
                    fxw, fyw = bw(f_fwd, sgn * 0.08)
                    if jp and jp["air"]:
                        fz_ = hipz - 0.81
                    else:
                        fz_ = zdeck(k, fxw, fyw) + lift
                    kn = leg_ik((0.0, hipz), (f_fwd, fz_), cl1, cl2)
                    kxw, kyw = bw(kn[0], sgn * 0.08)
                    hxw, hyw = bw(0.0, sgn * 0.08)
                    parts.append(_cyl_mesh((hxw, hyw, hipz),
                                           (kxw, kyw, kn[1]), 0.048))
                    parts.append(_cyl_mesh((kxw, kyw, kn[1]),
                                           (fxw, fyw, fz_ + 0.03), 0.042))
                    f2x, f2y = bw(f_fwd + 0.14, sgn * 0.08)
                    parts.append(_cyl_mesh((fxw, fyw, fz_ + 0.03),
                                           (f2x, f2y, fz_ + 0.03), 0.038))
                parts.append(_cyl_mesh(hipC, (cx, cy, hipz + 0.57), 0.10))
                parts.append(_sphere_mesh((cx, cy, hipz + 0.70), 0.10))
                if mem["type"] == "walk":
                    for sgn in (+1, -1):
                        sxw, syw = bw(0.0, sgn * 0.16)
                        hxw, hyw = bw(sgn * 0.30 * pose["armsw"], sgn * 0.19)
                        parts.append(_cyl_mesh(
                            (sxw, syw, shz),
                            (hxw, hyw, shz - 0.38 - 0.05 * sgn * pose["armsw"]),
                            0.035))
                elif jp:
                    tha = jp["arm"] * 1.15
                    for sgn in (+1, -1):
                        sxw, syw = bw(0.0, sgn * 0.16)
                        hxw, hyw = bw(0.34 * np.sin(tha), sgn * 0.20)
                        parts.append(_cyl_mesh(
                            (sxw, syw, shz),
                            (hxw, hyw, shz - 0.34 * np.cos(tha)), 0.035))
                else:
                    for sgn in (+1, -1):
                        sxw, syw = bw(0.0, sgn * 0.16)
                        hxw, hyw = bw(0.02, sgn * 0.20)
                        parts.append(_cyl_mesh((sxw, syw, shz),
                                               (hxw, hyw, shz - 0.36), 0.035))
            t3.append(_mesh_trace(parts, "#8d939b",
                                  name=f"crowd x{len(crowd_members)} "
                                       f"({cfg['crowd_type']})",
                                  showlegend=True, opacity=0.95))
        return t3

    frames = [go.Frame(data=frame3d(k), name=str(i))
              for i, k in enumerate(ks)]
    tt = np.asarray(anim["t"])[ks]
    frame_dt = float(np.median(np.diff(tt))) if len(tt) > 1 else 0.04
    frame_ms = max(15.0, 1000.0 * frame_dt / cfg["anim_time_scale"])

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=dict(text="span (m)"), range=[-1, L + 1]),
            yaxis=dict(title=dict(text="width (m)"),
                       range=[-halfW * 2.0, halfW * 2.0]),
            zaxis=dict(title=dict(text="height (m)"),
                       range=[-dep - 0.7, cfg["anim_ymax"]]),
            aspectmode="data", bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.55, y=-1.75, z=0.65),
                        projection=dict(type="orthographic"))),
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
        height=620, uirevision="keep", showlegend=False)
    return fig, frame_ms, [float(anim["t"][k]) for k in ks]
