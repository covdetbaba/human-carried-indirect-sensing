FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:'
         'wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600;700&'
         'family=IBM+Plex+Sans:wght@400;500&display=swap" rel="stylesheet">')

SHEET_CSS = """
:root{--paper:#f2eee3;--panel:#faf8f2;--hi:#efe3cf;--ink:#1a2532;--ink2:#4d5966;--rule:#1a2532;
--red:#b83a1f;--grid:rgba(26,37,50,.055);--gridmaj:rgba(26,37,50,.10);--gridline:rgba(26,37,50,.22);
--ramp:#d9a032,#c9572b,#8a2f4f,#2f5585,#121c27;
--mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;--sans:"IBM Plex Sans",system-ui,sans-serif;
--cond:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif}
@media (prefers-color-scheme:dark){:root{--paper:#0e2740;--panel:#113050;--hi:#1a4166;--ink:#e6eef6;--ink2:#a8bccf;
--rule:#9fc0de;--red:#ff8f6e;--grid:rgba(255,255,255,.045);--gridmaj:rgba(255,255,255,.09);--gridline:rgba(230,238,246,.22);
--ramp:#ffd27a,#ff9466,#ef6f9a,#86b6ff,#f2f6fa}}
*{box-sizing:border-box}
html{background:var(--paper)}
body{margin:0;color:var(--ink);font:15px/1.55 var(--sans);background-color:var(--paper);
background-image:linear-gradient(var(--gridmaj) 1px,transparent 1px),linear-gradient(90deg,var(--gridmaj) 1px,transparent 1px),
linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);
background-size:120px 120px,120px 120px,24px 24px,24px 24px}
.sheet{max-width:1500px;margin:18px auto;padding:22px 26px 18px;border:1.5px solid var(--rule);outline:1px solid var(--rule);
outline-offset:5px;background:transparent}
@media (max-width:700px){.sheet{margin:10px 8px;padding:16px 14px;outline-offset:3px}}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;flex-wrap:wrap;border-bottom:1.5px solid var(--rule);padding-bottom:16px}
.kicker{font:500 11px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--red)}
h1{font:700 40px/1.05 var(--cond);letter-spacing:.01em;text-transform:uppercase;margin:6px 0 8px}
@media (max-width:700px){h1{font-size:30px}}
.lead{margin:0;max-width:640px;color:var(--ink2)}
.tblock{border-collapse:collapse;font:12px var(--mono);min-width:290px}
.tblock th,.tblock td{border:1px solid var(--rule);padding:4px 9px;text-align:left}
.tblock th{text-transform:uppercase;letter-spacing:.08em;font-weight:500;color:var(--ink2);width:84px}
.nav{display:flex;gap:0;margin:14px 0 4px;border:1px solid var(--rule);width:max-content;max-width:100%;flex-wrap:wrap}
.nav a{font:500 12px var(--mono);text-transform:uppercase;letter-spacing:.08em;color:var(--ink);text-decoration:none;padding:6px 14px;border-right:1px solid var(--rule)}
.nav a:last-child{border-right:0}
.nav a.cur,.nav a:hover{background:var(--ink);color:var(--paper)}
section{margin-top:26px}
h2{font:600 19px var(--cond);text-transform:uppercase;letter-spacing:.06em;margin:0 0 8px;display:flex;align-items:center;gap:10px}
h2 .num{display:inline-flex;width:26px;height:26px;border:1.5px solid var(--rule);border-radius:50%;align-items:center;justify-content:center;font:600 12px var(--mono)}
.hint{font-size:13px;color:var(--ink2);margin:4px 0 10px;max-width:900px}
button{font:500 12px var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--ink);background:var(--panel);
border:1px solid var(--rule);border-radius:0;padding:6px 12px;cursor:pointer}
button:hover:not(:disabled),button.on{background:var(--ink);color:var(--paper)}
button:disabled{opacity:.55;cursor:default}
.cmd{display:none;font:12px var(--mono);background:var(--panel);border:1px dashed var(--rule);padding:8px 10px;margin:0;white-space:pre-wrap}
.foot{margin-top:28px;padding-top:10px;border-top:1.5px solid var(--rule);font:11.5px var(--mono);color:var(--ink2)}
a{color:var(--red)}
"""
