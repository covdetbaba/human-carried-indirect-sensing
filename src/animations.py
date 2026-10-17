import json
import os
import tempfile

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

import human_di as hv
from explorer import base_cfg
from sheet_style import FONTS, SHEET_CSS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")
C = hv.DRAW
MAX_FRAMES = 360


STAND_WINDOW = (30.0, 40.0)


def simulate(mode):
    hv.SAVE_DIR = tempfile.mkdtemp(prefix="deck_anim_")
    cfg = base_cfg(mode)
    cfg["anim_max_frames"] = 100000
    res = hv.run_case(cfg, record_anim=True)
    if mode == "standing":
        trim(res, *STAND_WINDOW)
    return res


def trim(res, t0, t1):
    a = res["anim"]
    keep = [k for k, t in enumerate(a["t"]) if t0 <= t <= t1]
    n = len(a["t"])
    for key, v in list(a.items()):
        if isinstance(v, list) and len(v) == n:
            a[key] = [v[k] for k in keep]
    a["t"] = [t - t0 for t in a["t"]]


HEAD_H = 0.14
HEAD_W = 0.085
SHAFT = 0.42


def shaker_arrows(res, k, z):
    cfg, anim = res["cfg"], res["anim"]
    xs = np.asarray(res["xs"])
    pts = hv.bench_force_points(cfg) if cfg.get("bench_force_live", True) else []
    fk = anim["bench_f"][k] if k < len(anim.get("bench_f", [])) else []
    out = []
    for i, pt in enumerate(pts):
        zt = float(np.interp(pt["x"], xs, z))
        F = abs(float(fk[i])) if i < len(fk) else pt["amp"]
        puls = 1.0 + 0.15 * np.tanh((F - pt["amp"]) / max(pt["amp"], 1.0))
        neck = zt + HEAD_H
        out.append(dict(x=pt["x"], tip=zt, neck=neck, top=neck + SHAFT * puls,
                        head=[(pt["x"] - HEAD_W, neck), (pt["x"] + HEAD_W, neck),
                              (pt["x"], zt)]))
    return out


def poses(res):
    cfg, anim, seg = res["cfg"], res["anim"], res["seg"]
    xs = np.asarray(res["xs"]); L = cfg["deck_L"]
    S = cfg["anim_disp_scale"]; Sb = cfg["anim_body_scale"]
    L1, L2, UA, FA = seg["thigh"], seg["shank"], seg["uparm"], seg["forearm"]
    leg_len, torso_h, head_h = seg["leg_len"], seg["torso"], seg["head"]
    bob_amp, sw_amp = hv.drawn_gait(cfg)
    beam = np.array(anim["beam"]); qs = np.array(anim["q"])

    def top(x, z):
        return float(np.interp(np.clip(x, 0, L), xs, z)) if 0 <= x <= L else 0.0

    out = []
    for k in range(len(anim["t"])):
        z = beam[k] * S; q = qs[k]; px = anim["px"][k]
        ph = 2 * np.pi * (0.5 * cfg["pacing_freq"]) * anim["t"][k]
        fxL, fyL, _ = anim["footL"][k]; fxR, fyR, _ = anim["footR"][k]
        footL = (fxL, top(fxL, z) + fyL); footR = (fxR, top(fxR, z) + fyR)
        hip_y = (min(footL[1], footR[1]) + leg_len * 0.94
                 + bob_amp * np.cos(2.0 * ph) + 0.5 * (q[0] + q[1]) * Sb)
        hipL = (px - 0.04, hip_y); hipR = (px + 0.04, hip_y)
        sh_y = hip_y + torso_h + q[2] * Sb
        shL = (px - 0.02, sh_y); shR = (px + 0.06, sh_y)

        def arm(s, sw, w):
            e = (s[0] + UA * 0.85 * sw, s[1] - UA * 0.62)
            return e, (e[0] + FA * 1.05 * sw, e[1] - FA * 0.72 + w * Sb)
        eL, wL = arm(shL, sw_amp * np.sin(ph), q[3])
        eR, wR = arm(shR, sw_amp * np.sin(ph + np.pi), q[4])
        out.append(dict(
            t=float(anim["t"][k]), z=z,
            legL=[hipL, hv.leg_ik(hipL, footL, L1, L2), footL],
            legR=[hipR, hv.leg_ik(hipR, footR, L1, L2), footR],
            trunk=[(px, hip_y), (px + 0.02, sh_y)],
            armL=[shL, eL, wL], armR=[shR, eR, wR],
            head=(px + 0.02, sh_y + head_h * 0.6),
            sensors=dict(wristL=wL, wristR=wR, hipL=hipL, hipR=hipR)))
    return out


def figure_2d(res):
    cfg, anim = res["cfg"], res["anim"]
    xs = np.asarray(res["xs"]); L = cfg["deck_L"]
    t_sl = hv.section_depth(cfg)
    ps = poses(res)
    ks = list(range(0, len(ps), max(1, len(ps) // MAX_FRAMES)))

    def xy(seq):
        return dict(x=[p[0] for p in seq], y=[p[1] for p in seq])

    def frame(k):
        p = ps[k]; z = p["z"]
        tr = [go.Scatter(x=list(xs) + list(xs[::-1]),
                         y=list(z) + list((z - t_sl)[::-1]), mode="lines",
                         fill="toself", fillcolor=C["deck"],
                         fillpattern=dict(shape="/", fgcolor=C["edge"],
                                          bgcolor=C["deck"], size=7,
                                          solidity=0.12),
                         line=dict(color=C["edge"], width=1.2),
                         hoverinfo="skip")]
        ar = shaker_arrows(res, k, z)
        sx, sy, hx, hy = [], [], [], []
        for a in ar:
            sx += [a["x"], a["x"], None]; sy += [a["top"], a["neck"], None]
            hx += [q[0] for q in a["head"]] + [a["head"][0][0], None]
            hy += [q[1] for q in a["head"]] + [a["head"][0][1], None]
        tr.append(go.Scatter(x=sx, y=sy, mode="lines",
                             line=dict(color=C["shaker"], width=5),
                             hoverinfo="skip"))
        tr.append(go.Scatter(x=hx, y=hy, mode="lines", fill="toself",
                             fillcolor=C["shaker"],
                             line=dict(color=C["shaker"], width=1),
                             hoverinfo="skip"))
        for key, col, w in (("legR", C["leg2"], 5), ("armR", C["arm2"], 4),
                            ("legL", C["leg"], 6), ("trunk", C["trunk"], 9),
                            ("armL", C["arm"], 5)):
            tr.append(go.Scatter(**xy(p[key]), mode="lines",
                                 line=dict(color=col, width=w), hoverinfo="skip"))
        tr.append(go.Scatter(x=[p["head"][0]], y=[p["head"][1]], mode="markers",
                             marker=dict(size=20, color=C["skin"],
                                         line=dict(color=C["edge"], width=1.2)),
                             hoverinfo="skip"))
        names = dict(wristL="watch, left wrist", wristR="watch, right wrist",
                     hipL="phone, left hip", hipR="phone, right hip")
        for s in ("hipR", "wristR", "hipL", "wristL"):
            q = p["sensors"][s]
            tr.append(go.Scatter(x=[q[0]], y=[q[1]], mode="markers",
                                 marker=dict(symbol="square", size=12,
                                             color=hv.SENSOR_COLORS[s],
                                             line=dict(color=C["edge"],
                                                       width=1)),
                                 hovertext=names[s], hoverinfo="text"))
        return tr

    frames = [go.Frame(data=frame(k), name=str(i)) for i, k in enumerate(ks)]
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.add_trace(go.Scatter(x=[0, L], y=[-t_sl - 0.06] * 2, mode="markers",
                             marker=dict(symbol="triangle-up", size=14,
                                         color=C["edge"]), hoverinfo="skip"))
    fig.update_layout(
        xaxis=dict(title=dict(text="span (m)"), range=[-1.0, L + 1.0],
                   constrain="domain"),
        yaxis=dict(title=dict(text="height (m)"),
                   range=[-0.6, 2.2],
                   scaleanchor="x", scaleratio=1.0),
        height=430, margin=dict(l=60, r=10, t=10, b=46), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    tt = np.asarray(anim["t"])[ks]
    frame_ms = 1000.0 * float(np.median(np.diff(tt))) / cfg["anim_time_scale"]
    return fig, frame_ms, [float(x) for x in tt]


ANIM_CSS = """
.player{display:flex;flex-wrap:wrap;align-items:center;gap:10px;border:1px solid var(--rule);border-bottom:0;border-radius:8px 8px 0 0;padding:8px 12px;background:var(--panel);font:13px var(--sans)}
.player input[type=range]{accent-color:var(--accent)}
.player .t{min-width:76px;color:var(--blue);font-weight:600;font-variant-numeric:tabular-nums}
.figbox{border:1px solid var(--rule);border-radius:0 0 8px 8px;background:#fff;padding:4px}
"""

ANIM_JS = r"""
(function(){
var D=JSON.parse(document.getElementById('fig').textContent),T=D.times,N=T.length,base=D.frame_ms,k=0,on=false,tm=null;
var $=function(i){return document.getElementById(i);};
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function theme(){
  var ink=css('--ink'),grid=css('--gridline'),ax={color:ink,linecolor:css('--rule'),gridcolor:grid,showline:true,mirror:true,ticks:'outside',tickcolor:css('--rule'),zeroline:false};
  var u={'font.family':'IBM Plex Sans, system-ui, sans-serif','font.color':ink,'font.size':12};
  if(D.three){['xaxis','yaxis','zaxis'].forEach(function(a){u['scene.'+a+'.color']=ink;u['scene.'+a+'.gridcolor']=css('--gridline');u['scene.'+a+'.backgroundcolor']='rgba(0,0,0,0)';});}
  else{Object.keys(ax).forEach(function(p){u['xaxis.'+p]=ax[p];u['yaxis.'+p]=ax[p];});}
  return Plotly.relayout('plot',u);
}
function show(i){k=i;Plotly.animate('plot',[String(i)],{mode:'immediate',frame:{duration:0,redraw:true},transition:{duration:0}});
  $('t').textContent='t = '+T[i].toFixed(2)+' s';$('pos').value=i;}
function loop(){if(!on){return;}show((k+1)%N);tm=setTimeout(loop,base/parseFloat($('sp').value));}
function size(){
  var el=$('plot'),xr=D.fig.layout.xaxis.range,yr=D.fig.layout.yaxis.range,w=el.clientWidth-70;
  var h=Math.round(Math.max(150,w*(yr[1]-yr[0])/(xr[1]-xr[0])+56));
  el.style.height=h+'px';return h;
}
function fit(){
  if(D.three){return Promise.resolve();}
  var h=size();
  return Plotly.relayout('plot',{height:h,'xaxis.range':D.fig.layout.xaxis.range,'yaxis.range':D.fig.layout.yaxis.range});
}
if(!D.three){D.fig.layout.height=size();}
Plotly.newPlot('plot',D.fig.data,D.fig.layout,{displaylogo:false,responsive:true}).then(function(){
  return Plotly.addFrames('plot',D.fig.frames);}).then(theme).then(fit).then(function(){show(0);});
window.addEventListener('resize',function(){clearTimeout(window._fz);window._fz=setTimeout(fit,150);});
$('play').onclick=function(){if(on){return;}on=true;this.classList.add('on');loop();};
$('pause').onclick=function(){on=false;clearTimeout(tm);$('play').classList.remove('on');};
$('pos').max=N-1;$('pos').oninput=function(){$('pause').onclick();show(parseInt(this.value,10));};
$('sp').oninput=function(){$('spv').textContent=parseFloat(this.value).toFixed(1)+'x';};
})();
"""

PAGES = [("walk-2d.html", "Walking, side view"), ("walk-3d.html", "Walking, 3D"),
         ("stand-2d.html", "Standing, side view"), ("stand-3d.html", "Standing, 3D")]


def write_anim_page(fname, fig, frame_ms, times, title, lead, note, three, sheet):
    cur = ' class="cur"'
    nav = "".join(f'<a href="{f}"{cur if f == fname else ""}>{t}</a>'
                  for f, t in PAGES)
    blob = json.dumps(dict(fig=fig.to_plotly_json(), frame_ms=frame_ms,
                           times=times, three=three),
                      cls=PlotlyJSONEncoder).replace("</", "<\\/")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>{FONTS}<style>{SHEET_CSS}{ANIM_CSS}</style>
<script>{get_plotlyjs()}</script></head><body><div class="sheet">
<header class="head"><div><div class="kicker">Animation</div><h1>{title}</h1>
<p class="lead">{lead}</p></div>
<table class="tblock"><tr><th>Deck</th><td>14 m span, 3 m wide, 0.2 m slab</td></tr>
<tr><th>Person</th><td>75 kg, 1.75 m, 15 DOF</td></tr><tr><th>Devices</th><td>watch on both wrists, phone on both hips</td></tr>
<tr><th>By</th><td>J. Babazade</td></tr></table></header>
<nav class="nav"><a href="index.html">Index</a>{nav}</nav>
<section><div class="player"><button id="play">Play</button><button id="pause">Pause</button>
<input type="range" id="pos" min="0" value="0" style="flex:1;min-width:160px"><span class="t" id="t"></span>
<span>speed</span><input type="range" id="sp" min="0.1" max="3" step="0.1" value="1"><span id="spv">1.0x</span></div>
<div class="figbox"><div id="plot"></div></div><p class="hint">{note}</p></section>
<footer class="foot">© 2026 Jovdat Babazade. All rights reserved. Free to run and study with credit to the author.</footer>
</div><script type="application/json" id="fig">{blob}</script><script>{ANIM_JS}</script></body></html>"""
    with open(os.path.join(DOCS, fname), "w", encoding="utf-8") as fh:
        fh.write(html)
    print("  wrote", fname)


def write_gif(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    cfg = res["cfg"]; L = cfg["deck_L"]; xs = np.asarray(res["xs"])
    t_sl = hv.section_depth(cfg); ps = poses(res); t = res["t"]
    watch = res["acc3"][:, 3, 2]; phone = res["acc3"][:, 0, 2]
    paper, ink, red = "#ffffff", "#1f2a36", "#0e7c86"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": ink, "axes.labelcolor": ink,
                         "xtick.color": ink, "ytick.color": ink,
                         "axes.edgecolor": ink, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(8.4, 4.5), dpi=160, facecolor=paper)
    fig.add_artist(Rectangle((0.008, 0.012), 0.984, 0.976, transform=fig.transFigure,
                             fill=False, lw=1.0, ec="#ddd6c8"))
    fig.add_artist(Rectangle((0.008, 0.885), 0.984, 0.103, transform=fig.transFigure,
                             fill=False, lw=0.8, ec="#ddd6c8"))
    fig.text(0.025, 0.925, "Person on a flexible deck", fontsize=12.5,
             fontweight="bold")
    fig.text(0.025, 0.897, "walking %g m/s  |  %g m span  |  three ambient shakers"
             "  |  deck motion x%g" % (cfg["walk_speed"], L, cfg["anim_disp_scale"]),
             fontsize=8, color="#5b6470")
    clock = fig.text(0.975, 0.925, "", fontsize=11, color=red, ha="right",
                     fontweight="bold")
    ax = fig.add_axes([0.03, 0.40, 0.94, 0.47], facecolor=paper)
    ax2 = fig.add_axes([0.085, 0.10, 0.885, 0.22], facecolor=paper)
    ax.set_xlim(-0.6, L + 0.6); ax.set_ylim(-0.42, 2.0); ax.set_aspect("equal")
    ax.axis("off")
    deck, = ax.fill([], [], facecolor=C["deck"], edgecolor=C["edge"],
                    hatch="////", lw=1)
    ax.plot([0, L], [-t_sl - 0.07] * 2, "^", color=C["edge"], ms=11)
    from matplotlib.patches import Polygon
    shafts, heads = [], []
    for a in shaker_arrows(res, 0, ps[0]["z"]):
        shafts.append(ax.plot([], [], color=C["shaker"], lw=3.2,
                              solid_capstyle="butt")[0])
        hp = Polygon(a["head"], closed=True, fc=C["shaker"], ec=C["shaker"], lw=0.6)
        ax.add_patch(hp); heads.append(hp)
    lines = {k: ax.plot([], [], color=c, lw=w, solid_capstyle="round")[0]
             for k, c, w in (("legR", C["leg2"], 4), ("armR", C["arm2"], 3),
                             ("legL", C["leg"], 5), ("trunk", C["trunk"], 7),
                             ("armL", C["arm"], 4))}
    head, = ax.plot([], [], "o", ms=12, mfc=C["skin"], mec=C["edge"])
    sens = {s: ax.plot([], [], "s", ms=6, color=hv.SENSOR_COLORS[s],
                       mec=C["edge"])[0]
            for s in ("hipR", "wristR", "hipL", "wristL")}
    lim = 1.1 * max(abs(watch).max(), abs(phone).max())
    ax2.set_xlim(t[0], t[-1]); ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("time (s)", labelpad=1); ax2.set_ylabel("z (m/s²)")
    ax2.grid(color="#e3e6ea", lw=0.6)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    lw_, = ax2.plot([], [], color=hv.SENSOR_COLORS["wristL"], lw=0.9,
                    label="watch, left wrist")
    lp_, = ax2.plot([], [], color=hv.SENSOR_COLORS["hipL"], lw=0.9,
                    label="phone, left hip")
    cur = ax2.axvline(t[0], color=ink, lw=0.8, alpha=0.6)
    ax2.legend(loc="lower right", bbox_to_anchor=(1, 0.98), ncol=2, fontsize=8,
               frameon=False, handlelength=1.5)

    def draw(k):
        p = ps[k]; z = p["z"]
        deck.set_xy(np.c_[np.r_[xs, xs[::-1]], np.r_[z, (z - t_sl)[::-1]]])
        for sh, hp, a in zip(shafts, heads, shaker_arrows(res, k, z)):
            sh.set_data([a["x"], a["x"]], [a["top"], a["neck"]])
            hp.set_xy(a["head"])
        for key, ln in lines.items():
            q = np.array(p[key]); ln.set_data(q[:, 0], q[:, 1])
        head.set_data([p["head"][0]], [p["head"][1]])
        for s, mk in sens.items():
            mk.set_data([p["sensors"][s][0]], [p["sensors"][s][1]])
        m = t <= p["t"]
        lw_.set_data(t[m], watch[m]); lp_.set_data(t[m], phone[m])
        cur.set_xdata([p["t"], p["t"]])
        clock.set_text("t = %.2f s" % p["t"])
        return []
    from PIL import Image
    dt = float(np.median(np.diff([q["t"] for q in ps])))
    rgb = []
    for k in range(len(ps)):
        draw(k)
        fig.canvas.draw()
        rgb.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
    plt.close(fig)
    probe = Image.new("RGB", (rgb[0].width, rgb[0].height * 3))
    for n, k in enumerate((0, len(rgb) // 2, len(rgb) - 1)):
        probe.paste(rgb[k], (0, n * rgb[0].height))
    pal = probe.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in rgb]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(round(1000 * dt)), loop=0, disposal=1)
    print("  wrote", os.path.relpath(path, ROOT))


def main():
    os.makedirs(DOCS, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "media"), exist_ok=True)
    for mode, stem in (("walking", "walk"), ("standing", "stand")):
        res = simulate(mode)
        stand = mode == "standing"
        title2 = "Standing, side view" if stand else "Walking, side view"
        title3 = "Standing, 3D" if stand else "Walking, 3D"
        lead = ("The person stands at mid-span while three shakers excite the deck."
                if stand else
                "One crossing at 1.5 m/s while three shakers excite the deck.")
        fig, ms, tt = figure_2d(res)
        write_anim_page(f"{stem}-2d.html", fig, ms, tt, title2, lead,
                        ("A 10 s window of the 60 s record. " if stand else "")
                        + "Deck motion is drawn %g times larger. The person is "
                        "to scale. Squares are the devices. Arrows are the three "
                        "shakers, resting on the deck and loading it all the time." %
                        res["cfg"]["anim_disp_scale"], False, "06" if stand else "04")
        fig, ms, tt = hv.figure_3d(res)
        write_anim_page(f"{stem}-3d.html", fig, ms, tt, title3, lead,
                        ("A 10 s window of the 60 s record. " if stand else "")
                        + "Drag to rotate, scroll to zoom. Deck motion is drawn "
                        "%g times larger. Arrows mark the three shakers." %
                        res["cfg"]["anim_disp_scale"],
                        True, "07" if stand else "05")
        if not stand:
            write_gif(res, os.path.join(ROOT, "media", "crossing.gif"))


if __name__ == "__main__":
    main()
