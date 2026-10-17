FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:'
         'wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" '
         'rel="stylesheet">')

SHEET_CSS = """
:root{color-scheme:light;--paper:#f4f1ea;--panel:#ffffff;--ink:#1f2a36;--ink2:#5c6670;--rule:#ddd6c8;--soft:#ece7dd;
--accent:#0e7c86;--hi:#e3f1f2;--blue:var(--accent);--red:#c0392b;--gridline:#e6e2da;
--ramp:#e0a030,#d0603a,#a03c70,#3a6f9a,#16324f;
--mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;--sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
--cond:var(--sans)}
*{box-sizing:border-box}
html,body{background:var(--paper)}
body{margin:0;color:var(--ink);font:15px/1.6 var(--sans)}
.sheet{max-width:1440px;margin:0 auto;padding:24px 32px 22px}
@media (max-width:700px){.sheet{padding:14px 12px}}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;flex-wrap:wrap;background:var(--panel);
border:1px solid var(--rule);border-top:4px solid var(--accent);border-radius:8px;padding:18px 22px}
.kicker{font:600 13px var(--sans);color:var(--accent)}
h1{font:600 32px/1.15 var(--sans);margin:4px 0 8px;letter-spacing:-.01em;color:var(--ink)}
@media (max-width:700px){h1{font-size:26px}}
.lead{margin:0;max-width:680px;color:var(--ink2)}
.tblock{border-collapse:collapse;font:13px var(--sans);min-width:280px}
.tblock th,.tblock td{border-bottom:1px solid var(--soft);padding:5px 10px;text-align:left}
.tblock th{font-weight:500;color:var(--ink2);width:90px}
.nav{display:flex;gap:6px;margin:16px 0 4px;flex-wrap:wrap}
.nav a{font:500 13px var(--sans);color:var(--ink);text-decoration:none;padding:6px 13px;border:1px solid var(--rule);border-radius:6px;background:var(--panel)}
.nav a.cur{background:var(--accent);border-color:var(--accent);color:#fff}
.nav a:hover:not(.cur){border-color:var(--accent);color:var(--accent)}
section{margin-top:30px}
h2{font:600 19px var(--sans);margin:0 0 8px;display:flex;align-items:center;gap:10px}
h2 .num{display:inline-flex;width:24px;height:24px;border-radius:5px;background:var(--accent);color:#fff;align-items:center;justify-content:center;font:600 12px var(--sans)}
.hint{font-size:13.5px;color:var(--ink2);margin:4px 0 10px;max-width:920px}
button{font:500 13px var(--sans);color:var(--ink);background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:6px 12px;cursor:pointer}
button:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
button:disabled{opacity:.55;cursor:default}
.cmd{display:none;font:12.5px var(--mono);background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:8px 10px;margin:0;white-space:pre-wrap}
.foot{margin-top:32px;padding-top:12px;border-top:1px solid var(--rule);font-size:12.5px;color:var(--ink2)}
a{color:var(--accent)}
"""
