FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:'
         'wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" '
         'rel="stylesheet">')

SHEET_CSS = """
:root{color-scheme:light;--paper:#ffffff;--panel:#ffffff;--hi:#eef3fa;--ink:#1b1f24;--ink2:#5b6470;--rule:#d6d9de;
--soft:#e9ecef;--red:#c23b3b;--blue:#2563a8;--gridline:#e3e6ea;
--ramp:#e8a33d,#d1495b,#8e3b8f,#2563a8,#1b2a41;
--mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;--sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
--cond:var(--sans)}
*{box-sizing:border-box}
html,body{background:#ffffff}
body{margin:0;color:var(--ink);font:15px/1.6 var(--sans)}
.sheet{max-width:1440px;margin:0 auto;padding:28px 32px 22px}
@media (max-width:700px){.sheet{padding:18px 16px}}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;flex-wrap:wrap;border-bottom:1px solid var(--rule);padding-bottom:18px}
.kicker{font:500 13px var(--sans);color:var(--blue)}
h1{font:600 32px/1.15 var(--sans);margin:4px 0 8px;letter-spacing:-.01em}
@media (max-width:700px){h1{font-size:26px}}
.lead{margin:0;max-width:680px;color:var(--ink2)}
.tblock{border-collapse:collapse;font:13px var(--sans);min-width:280px}
.tblock th,.tblock td{border-bottom:1px solid var(--soft);padding:5px 10px;text-align:left}
.tblock th{font-weight:500;color:var(--ink2);width:90px}
.nav{display:flex;gap:4px;margin:16px 0 4px;flex-wrap:wrap}
.nav a{font:500 13px var(--sans);color:var(--ink2);text-decoration:none;padding:6px 12px;border:1px solid var(--rule);border-radius:6px;background:#fff}
.nav a.cur{background:var(--blue);border-color:var(--blue);color:#fff}
.nav a:hover:not(.cur){border-color:var(--blue);color:var(--blue)}
section{margin-top:30px}
h2{font:600 19px var(--sans);margin:0 0 8px;display:flex;align-items:center;gap:10px}
h2 .num{display:inline-flex;width:24px;height:24px;border-radius:50%;background:var(--hi);color:var(--blue);align-items:center;justify-content:center;font:600 12px var(--sans)}
.hint{font-size:13.5px;color:var(--ink2);margin:4px 0 10px;max-width:920px}
button{font:500 13px var(--sans);color:var(--ink);background:#fff;border:1px solid var(--rule);border-radius:6px;padding:6px 12px;cursor:pointer}
button:hover:not(:disabled){border-color:var(--blue);color:var(--blue)}
button.on{background:var(--blue);border-color:var(--blue);color:#fff}
button:disabled{opacity:.55;cursor:default}
.cmd{display:none;font:12.5px var(--mono);background:#f6f8fa;border:1px solid var(--rule);border-radius:6px;padding:8px 10px;margin:0;white-space:pre-wrap}
.foot{margin-top:32px;padding-top:12px;border-top:1px solid var(--rule);font-size:12.5px;color:var(--ink2)}
a{color:var(--blue)}
"""
