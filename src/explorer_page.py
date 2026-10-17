import json

from sheet_style import FONTS, SHEET_CSS

PAGE_CSS = """
.controls{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:0;border-top:1px solid var(--rule);border-left:1px solid var(--rule)}
.prow{display:grid;grid-template-columns:22px 1fr 150px;align-items:center;gap:8px;padding:7px 10px;border-right:1px solid var(--rule);border-bottom:1px solid var(--rule);background:var(--panel)}
.prow.sweep{background:var(--hi)}
.prow label{font:500 13px/1.2 var(--sans);cursor:pointer}
.prow .off{color:var(--red);font:600 11px var(--mono);margin-left:6px}
.prow select{font:12px var(--mono);color:var(--ink);background:transparent;border:1px solid var(--rule);border-radius:0;padding:3px 4px;width:100%}
.prow input[type=radio]{accent-color:var(--red);margin:0}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:14px 0 10px}
.status{font:12.5px/1.5 var(--mono);color:var(--ink2);margin:12px 0 0;min-height:1.5em}
.status b{color:var(--red);font-weight:600}
.note{font:14px/1.5 var(--sans);margin:4px 0 0}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font:11.5px var(--mono);border:1px solid var(--rule);padding:3px 7px 3px 5px;cursor:pointer;display:inline-flex;align-items:center;gap:6px;background:var(--panel);color:var(--ink)}
.chip i{display:inline-block;width:14px;height:3px}
.chip.hide{opacity:.35}
.plots{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
@media (max-width:1100px){.plots{grid-template-columns:1fr}}
.cell{position:relative;border:1px solid var(--rule);background:var(--panel);padding:30px 6px 4px}
.cell .tag{position:absolute;top:0;left:0;font:600 11px var(--mono);letter-spacing:.06em;padding:5px 9px;border-right:1px solid var(--rule);border-bottom:1px solid var(--rule);background:var(--paper)}
.cell .plot{height:340px}
table.modes{border-collapse:collapse;font:12px var(--mono);width:100%;background:var(--panel)}
table.modes th,table.modes td{border:1px solid var(--rule);padding:4px 7px;text-align:right;white-space:nowrap}
table.modes th:first-child,table.modes td:first-child{text-align:left}
table.modes th{font-weight:600;background:var(--paper)}
.scroll{overflow-x:auto}
"""

PAGE_JS = r"""
(function(){
var M=JSON.parse(document.getElementById('meta').textContent);
var RUNS=JSON.parse(document.getElementById('runs').textContent);
var P=M.params, NF=M.nf, F=[], i;
for(i=0;i<NF;i++){F.push(M.fmin+(M.fmax-M.fmin)*i/(NF-1));}
var state={sweep:P[0],sel:{},rigid:false,hidden:{},mark:null};
P.forEach(function(p){state.sel[p]=M.base[p];});
var $=function(id){return document.getElementById(id);};

function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function keyOf(sel){var k=[];P.forEach(function(p){if(sel[p]!==M.base[p]){k.push(p+'='+sel[p]);}});return k.length?k.join('+'):'base';}
function dec(s,pk){
  var b=atob(s),n=b.length/2,y=new Float64Array(n),j,q;
  for(j=0;j<n;j++){q=b.charCodeAt(2*j)|(b.charCodeAt(2*j+1)<<8);y[j]=(q===65535)?NaN:pk*Math.pow(10,-q/100);}
  return smooth(y);
}
function smooth(y){
  var df=(M.fmax-M.fmin)/(NF-1),s=M.smooth/df,h=Math.ceil(3*s),w=[],k,j,a,b,o=new Float64Array(y.length);
  for(k=-h;k<=h;k++){w.push(Math.exp(-0.5*k*k/(s*s)));}
  for(j=0;j<y.length;j++){a=0;b=0;for(k=-h;k<=h;k++){var v=y[j+k];if(v>0){a+=w[k+h]*v;b+=w[k+h];}}o[j]=b?a/b:NaN;}
  return o;
}
function ramp(n){
  var st=css('--ramp').split(',').map(function(c){c=c.trim();return[parseInt(c.substr(1,2),16),parseInt(c.substr(3,2),16),parseInt(c.substr(5,2),16)];});
  var out=[],t,u,a,b,j;
  for(j=0;j<n;j++){t=n>1?j/(n-1)*(st.length-1):0;u=Math.min(Math.floor(t),st.length-2);a=st[u];b=st[u+1];t-=u;
    out.push('rgb('+[0,1,2].map(function(c){return Math.round(a[c]+(b[c]-a[c])*t);}).join(',')+')');}
  return out;
}
function curvesFor(){
  var p=state.sweep,n=M.labels[p].length,list=[];
  for(var k=0;k<n;k++){var s=Object.assign({},state.sel);s[p]=k;list.push({i:k,key:keyOf(s),run:RUNS[keyOf(s)]});}
  return list;
}
function axis(t,log){
  var ink=css('--ink'),grid=css('--gridline');
  return {title:{text:t,font:{size:11}},type:log?'log':'linear',showline:true,linecolor:ink,linewidth:1,mirror:true,
    ticks:'outside',tickcolor:ink,ticklen:4,gridcolor:grid,griddash:'dot',zeroline:false,exponentformat:'power',
    range:log?undefined:[0,M.fmax],dtick:log?1:undefined};
}
function draw(){
  var list=curvesFor(),cols=ramp(list.length),have=list.filter(function(c){return c.run;});
  var ink=css('--ink'),mark=state.mark;
  if(mark===null||!list[mark]||!list[mark].run){mark=(list[state.sel[state.sweep]]&&list[state.sel[state.sweep]].run)?state.sel[state.sweep]:(have[0]?have[0].i:null);}
  var spec=[['deck','cb','pkb',null],['watch','c','pk','rc'],['phone','c','pk','rc']];
  spec.forEach(function(sp,pi){
    var data=[];
    list.forEach(function(c){
      if(!c.run||state.hidden[c.i]){return;}
      var src=sp[0]==='deck'?c.run:c.run[sp[0]];
      var lab=M.labels[state.sweep][c.i];
      if(state.rigid&&sp[3]){
        data.push({x:F,y:dec(src.rc,src.rpk),mode:'lines',line:{color:cols[c.i],width:1,dash:'dot'},opacity:.75,hoverinfo:'skip',showlegend:false});
      }
      data.push({x:F,y:dec(src[sp[1]],src[sp[2]]),mode:'lines',name:lab,line:{color:cols[c.i],width:c.i===mark?2.2:1.3},
        hovertemplate:lab+'<br>%{x:.2f} Hz<br>%{y:.3e}<extra></extra>',showlegend:false});
    });
    var shapes=[],notes=[];
    if(mark!==null&&list[mark].run){
      list[mark].run.modes.forEach(function(m){
        shapes.push({type:'line',xref:'x',yref:'paper',x0:m[1],x1:m[1],y0:0,y1:1,line:{color:css('--red'),width:1,dash:'dash'},opacity:.7});
        notes.push({x:m[1],y:1,xref:'x',yref:'paper',yanchor:'bottom',text:m[0],showarrow:false,font:{size:10,color:css('--red')}});
      });
    }
    var lay={margin:{l:58,r:10,t:16,b:40},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
      font:{family:'IBM Plex Mono, ui-monospace, monospace',size:11,color:ink},
      xaxis:axis('Frequency (Hz)',false),yaxis:axis(pi===0?'Deck PSD ((m/s²)²/Hz)':'Device PSD ((m/s²)²/Hz)',true),
      shapes:shapes,annotations:notes,hovermode:'closest',showlegend:false};
    Plotly.react('plot'+pi,data,lay,{displaylogo:false,responsive:true,modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']});
  });
  chips(list,cols,mark);
  table(list,cols);
  status(list,have);
}
function chips(list,cols,mark){
  var box=$('chips');box.innerHTML='';
  list.forEach(function(c){
    var b=document.createElement('span');b.className='chip'+(state.hidden[c.i]?' hide':'');
    b.innerHTML='<i style="background:'+cols[c.i]+'"></i>'+M.labels[state.sweep][c.i]+(c.i===mark?' ◆':'')+(c.run?'':' (not stored)');
    b.title='Click to show or hide. Double click to mark its modes.';
    b.onclick=function(){state.hidden[c.i]=!state.hidden[c.i];draw();};
    b.ondblclick=function(){state.hidden[c.i]=false;state.mark=c.i;draw();};
    box.appendChild(b);
  });
}
function table(list,cols){
  var labs=[],seen={};
  list.forEach(function(c){if(c.run){c.run.modes.forEach(function(m){if(!seen[m[0]]){seen[m[0]]=1;labs.push(m[0]);}});}});
  var ref={};list.forEach(function(c){if(c.run){c.run.modes.forEach(function(m){if(!(m[0] in ref)){ref[m[0]]=m[1];}});}});
  labs.sort(function(a,b){return ref[a]-ref[b];});
  var h='<tr><th>'+M.name[state.sweep]+'</th>'+labs.map(function(l){return '<th>'+l+'</th>';}).join('')+'</tr>';
  list.forEach(function(c){
    if(!c.run){return;}
    var f={};c.run.modes.forEach(function(m){f[m[0]]=m[1];});
    h+='<tr><td><i style="display:inline-block;width:10px;height:3px;margin-right:6px;vertical-align:middle;background:'+cols[c.i]+'"></i>'+M.labels[state.sweep][c.i]+'</td>'+
      labs.map(function(l){return '<td>'+(l in f?f[l].toFixed(2):'')+'</td>';}).join('')+'</tr>';
  });
  $('modes').innerHTML=h;
}
function status(list,have){
  var off=P.filter(function(p){return p!==state.sweep&&state.sel[p]!==M.base[p];}).length;
  var miss=list.length-have.length;
  var s='Sweeping '+M.name[state.sweep].toLowerCase()+' over '+list.length+' levels. ';
  s+=off?off+' other parameter'+(off>1?'s':'')+' off baseline. ':'Everything else at baseline. ';
  if(miss){s+='<b>'+miss+' level'+(miss>1?'s are':' is')+' not stored.</b> ';}
  $('status').innerHTML=s;
  $('note').textContent=M.note[state.sweep];
  var live=/^https?:$/.test(location.protocol)&&/^(localhost|127\.0\.0\.1)$/.test(location.hostname);
  $('compute').style.display=miss?'inline-block':'none';
  $('compute').textContent=live?'Compute missing levels':'Missing levels need the local server';
  $('compute').disabled=!live;
  $('cmd').style.display=(miss&&!live)?'block':'none';
}
function compute(){
  var b=$('compute');b.disabled=true;b.textContent='Computing... (a few minutes)';
  fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sweep:state.sweep,sel:state.sel})}).then(function(r){return r.json();}).then(function(d){
      (d.runs||[]).forEach(function(r){RUNS[r.key]=r;});
      if(d.error){alert(d.error);}
      draw();
    }).catch(function(e){alert('Server not reachable: '+e);draw();});
}
function controls(){
  var box=$('params');box.innerHTML='';
  P.forEach(function(p){
    var row=document.createElement('div');row.className='prow'+(p===state.sweep?' sweep':'');
    var r=document.createElement('input');r.type='radio';r.name='sweep';r.id='r_'+p;r.checked=p===state.sweep;
    r.onchange=function(){state.sweep=p;state.hidden={};state.mark=null;controls();draw();};
    var l=document.createElement('label');l.htmlFor='r_'+p;l.innerHTML=M.name[p]+(state.sel[p]!==M.base[p]?'<span class="off">off base</span>':'');
    var s=document.createElement('select');s.disabled=p===state.sweep;
    M.labels[p].forEach(function(t,k){var o=document.createElement('option');o.value=k;o.textContent=t+(k===M.base[p]?'  (base)':'');s.appendChild(o);});
    s.value=state.sel[p];
    s.onchange=function(){state.sel[p]=parseInt(s.value,10);state.mark=null;controls();draw();};
    row.appendChild(r);row.appendChild(l);row.appendChild(s);box.appendChild(row);
  });
}
$('rigid').onclick=function(){state.rigid=!state.rigid;this.classList.toggle('on',state.rigid);this.textContent='Rigid floor: '+(state.rigid?'on':'off');draw();};
$('reset').onclick=function(){P.forEach(function(p){state.sel[p]=M.base[p];});state.hidden={};state.mark=null;controls();draw();};
$('compute').onclick=compute;
if(window.matchMedia){window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);}
controls();draw();
})();
"""

BODY = """
<div class="sheet">
<header class="head">
  <div>
    <div class="kicker">sheet __SHEET__ / parameter explorer</div>
    <h1>__TITLE__</h1>
    <p class="lead">__LEAD__</p>
  </div>
  <table class="tblock">
    <tr><th>project</th><td>Human-carried indirect sensing</td></tr>
    <tr><th>records</th><td>__RECORDS__</td></tr>
    <tr><th>stored</th><td>__N__ configurations</td></tr>
    <tr><th>band</th><td>__FMIN__ to __FMAX__ Hz</td></tr>
    <tr><th>by</th><td>J. Babazade</td></tr>
  </table>
</header>
<nav class="nav"><a href="index.html">Index</a><a href="explorer-walking.html"__WCUR__>Walking</a><a href="explorer-standing.html"__SCUR__>Standing</a></nav>

<section>
  <h2><span class="num">1</span>Set the deck and the person</h2>
  <p class="hint">Pick one parameter to sweep. Set the others with the menus. Combinations with up to two parameters off baseline are stored.</p>
  <div class="controls" id="params"></div>
  <div class="bar"><button id="reset">Reset to baseline</button><button id="compute"></button></div>
  <pre class="cmd" id="cmd">__CMD__</pre>
  <p class="status" id="status"></p>
</section>

<section>
  <h2><span class="num">2</span>Spectra</h2>
  <p class="note" id="note"></p>
  <div class="bar"><button id="rigid">Rigid floor: off</button><div class="chips" id="chips"></div></div>
  <div class="plots">
    <div class="cell"><span class="tag">A / DECK</span><div class="plot" id="plot0"></div></div>
    <div class="cell"><span class="tag">B / SMARTWATCH (WRIST)</span><div class="plot" id="plot1"></div></div>
    <div class="cell"><span class="tag">C / SMARTPHONE (HIP)</span><div class="plot" id="plot2"></div></div>
  </div>
  <p class="hint">A is the deck alone under the ambient shakers. B and C are the devices on the person, as the first singular value of x, y and z over __RECORDS__, left and right averaged. Dashed red lines mark the deck modes of the level shown with ◆. Double click a level to move them.</p>
</section>

<section>
  <h2><span class="num">3</span>Mode frequencies (Hz)</h2>
  <div class="scroll"><table class="modes" id="modes"></table></div>
</section>

<footer class="foot">© 2026 Jovdat Babazade. All rights reserved. Free to run and study with credit to the author.</footer>
</div>
"""


def write_page(path, meta, runs):
    import plotly.offline
    stand = meta["mode"] == "standing"
    slim = {r["key"]: {k: r[k] for k in ("cb", "pkb", "modes", "watch",
                                         "phone")}
            for r in runs if not r.get("failed")}
    records = (f"{meta['passes']} standing records of {meta['stand_T']:g} s"
               if stand else f"{meta['passes']} crossings")
    rep = {
        "__SHEET__": "03" if stand else "02",
        "__TITLE__": "Standing person" if stand else "Walking person",
        "__LEAD__": ("The person stands at mid-span while the deck vibrates."
                     if stand else
                     "The person walks across the deck while it vibrates."),
        "__RECORDS__": records,
        "__N__": str(len(slim)),
        "__FMIN__": f"{meta['fmin']:g}", "__FMAX__": f"{meta['fmax']:g}",
        "__CMD__": "Run locally:  cd src  &&  " + meta["cmd"],
        "__WCUR__": "" if stand else ' class="cur"',
        "__SCUR__": ' class="cur"' if stand else "",
    }
    body = BODY
    for k, v in rep.items():
        body = body.replace(k, v)

    def blob(x):
        return json.dumps(x, separators=(",", ":"), ensure_ascii=False) \
            .replace("</", "<\\/")
    html = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, "
            "initial-scale=1\">"
            f"<title>{rep['__TITLE__']} explorer</title>{FONTS}"
            f"<style>{SHEET_CSS}{PAGE_CSS}</style>"
            f"<script>{plotly.offline.get_plotlyjs()}</script></head><body>"
            f"{body}"
            f"<script type=\"application/json\" id=\"meta\">{blob(meta)}</script>"
            f"<script type=\"application/json\" id=\"runs\">{blob(slim)}</script>"
            f"<script>{PAGE_JS}</script></body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
