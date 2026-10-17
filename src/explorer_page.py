import json

from sheet_style import FONTS, SHEET_CSS

PAGE_CSS = """
:root{--fam-gait:#c98a12;--fam-arm:#c9572b;--fam-body:#2f5585;--fam-drive:#6c7785;
--cls-u:#b83a1f;--cls-m:#c98a12;--cls-o:#2e7d4f}
@media (prefers-color-scheme:dark){:root{--fam-gait:#ffd27a;--fam-arm:#ff9466;--fam-body:#86b6ff;--fam-drive:#b9c6d3;
--cls-u:#ff8f6e;--cls-m:#ffd27a;--cls-o:#7fe0a6}}
.controls{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:0;border-top:1px solid var(--rule);border-left:1px solid var(--rule)}
.prow{display:grid;grid-template-columns:22px 1fr 150px;align-items:center;gap:8px;padding:7px 10px;border-right:1px solid var(--rule);border-bottom:1px solid var(--rule);background:var(--panel)}
.prow.sweep{background:var(--hi)}
.prow label{font:500 13px/1.2 var(--sans);cursor:pointer}
.prow .off{color:var(--red);font:600 11px var(--mono);margin-left:6px}
select{font:12px var(--mono);color:var(--ink);background:var(--panel);border:1px solid var(--rule);border-radius:0;padding:3px 4px}
.prow select{width:100%}
.prow input[type=radio]{accent-color:var(--red);margin:0}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0 10px}
.bar .lbl{font:500 11px var(--mono);text-transform:uppercase;letter-spacing:.08em;color:var(--ink2)}
.status{font:12.5px/1.5 var(--mono);color:var(--ink2);margin:12px 0 0;min-height:1.5em}
.status b{color:var(--red);font-weight:600}
.note{font:14px/1.5 var(--sans);margin:4px 0 0}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font:11.5px var(--mono);border:1px solid var(--rule);padding:3px 7px 3px 5px;cursor:pointer;display:inline-flex;align-items:center;gap:6px;background:var(--panel);color:var(--ink)}
.chip i{display:inline-block;width:14px;height:3px}
.chip.hide{opacity:.35}
.chip.on{background:var(--ink);color:var(--paper)}
.plots{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
@media (max-width:1100px){.plots{grid-template-columns:1fr}}
.cell{position:relative;border:1px solid var(--rule);background:var(--panel);padding:30px 6px 4px}
.cell .tag{position:absolute;top:0;left:0;font:600 11px var(--mono);letter-spacing:.06em;padding:5px 9px;border-right:1px solid var(--rule);border-bottom:1px solid var(--rule);background:var(--paper)}
.cell .plot{height:340px}
#plot0{cursor:crosshair}
table.t{border-collapse:collapse;font:12px var(--mono);width:100%;background:var(--panel)}
table.t th,table.t td{border:1px solid var(--rule);padding:4px 7px;text-align:right;white-space:nowrap}
table.t th:first-child,table.t td:first-child{text-align:left}
table.t th{font-weight:600;background:var(--paper)}
table.t td.wrap{white-space:normal;text-align:left}
.scroll{overflow-x:auto}
.two{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:14px;align-items:start}
@media (max-width:1000px){.two{grid-template-columns:1fr}}
.shape{height:330px}
.sw{display:inline-block;width:12px;height:3px;margin-right:7px;vertical-align:middle}
.cls{font-weight:600}
.cls.u{color:var(--cls-u)}.cls.m{color:var(--cls-m)}.cls.o{color:var(--cls-o)}
.ok{color:var(--cls-o);font-weight:600}.miss{color:var(--cls-u);font-weight:600}
.eq{font:13px var(--mono);border:1px dashed var(--rule);padding:8px 10px;background:var(--panel);margin:6px 0 10px;max-width:900px}
"""

PAGE_JS = r"""
(function(){
var M=JSON.parse(document.getElementById('meta').textContent);
var RUNS=JSON.parse(document.getElementById('runs').textContent);
var P=M.params, NF=M.nf, F=[], i;
for(i=0;i<NF;i++){F.push(M.fmin+(M.fmax-M.fmin)*i/(NF-1));}
var DF=(M.fmax-M.fmin)/(NF-1), CAP=0.35, PROM=2.0;
var state={sweep:P[0],sel:{},rigid:false,gait:false,caps:true,hidden:{},mark:null,picks:[],ms:null,sigma:M.smooth};
P.forEach(function(p){state.sel[p]=M.base[p];});
var $=function(id){return document.getElementById(id);};
var CACHE={};
var FAMS={gait:'Gait comb',arm:'Arm swing',body:'Body modes',drive:'Crossing rate'};
var FAMNOTE={gait:'Multiples of half the step rate. They move with cadence.',
  arm:'Arm swing at the stride and step rates.',
  body:'Resonances of the body itself. They move with body mass and stiffness, not with the deck.',
  drive:'Speed over span, v/L, and its multiples.'};

function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function keyOf(sel){var k=[];P.forEach(function(p){if(sel[p]!==M.base[p]){k.push(p+'='+sel[p]);}});return k.length?k.join('+'):'base';}
function raw(s,pk){
  var b=atob(s),n=b.length/2,y=new Float64Array(n),j,q;
  for(j=0;j<n;j++){q=b.charCodeAt(2*j)|(b.charCodeAt(2*j+1)<<8);y[j]=(q===65535)?NaN:pk*Math.pow(10,-q/100);}
  return y;
}
function smooth(y){
  if(!(state.sigma>0)){return y;}
  var s=state.sigma/DF,h=Math.ceil(3*s),w=[],k,j,a,b,o=new Float64Array(y.length);
  for(k=-h;k<=h;k++){w.push(Math.exp(-0.5*k*k/(s*s)));}
  for(j=0;j<y.length;j++){a=0;b=0;for(k=-h;k<=h;k++){var v=y[j+k];if(v>0){a+=w[k+h]*v;b+=w[k+h];}}o[j]=b?a/b:NaN;}
  return o;
}
function curve(key,dev,fld,sm){
  var id=key+'|'+dev+'|'+fld+'|'+(sm?state.sigma:'raw');
  if(!CACHE[id]){var r=RUNS[key],src=dev==='deck'?r:r[dev];
    var y=raw(src[fld==='c'?(dev==='deck'?'cb':'c'):'rc'],src[fld==='c'?(dev==='deck'?'pkb':'pk'):'rpk']);
    CACHE[id]=sm?smooth(y):y;}
  return CACHE[id];
}
function ramp(n){
  var st=css('--ramp').split(',').map(function(c){c=c.trim();return[parseInt(c.substr(1,2),16),parseInt(c.substr(3,2),16),parseInt(c.substr(5,2),16)];});
  var out=[],t,u,a,b,j;
  for(j=0;j<n;j++){t=n>1?j/(n-1)*(st.length-1):0;u=Math.min(Math.floor(t),st.length-2);a=st[u];b=st[u+1];t-=u;
    out.push('rgb('+[0,1,2].map(function(c){return Math.round(a[c]+(b[c]-a[c])*t);}).join(',')+')');}
  return out;
}
function cls(oi){return oi>=M.t2?['o','observable']:(oi>=M.t1?['m','marginal']:['u','unobservable']);}
function lab(i){return M.labels[state.sweep][i];}
function list(){
  var p=state.sweep,n=M.labels[p].length,out=[];
  for(var k=0;k<n;k++){var s=Object.assign({},state.sel);s[p]=k;var key=keyOf(s);out.push({i:k,key:key,run:RUNS[key]});}
  return out;
}
function markRow(L){
  var m=state.mark;
  if(m===null||!L[m]||!L[m].run){m=L[state.sel[state.sweep]]&&L[state.sel[state.sweep]].run?state.sel[state.sweep]:null;
    if(m===null){L.forEach(function(c){if(m===null&&c.run){m=c.i;}});}}
  state.mark=m;return m===null?null:L[m];
}
function axis(t,log){
  var ink=css('--ink'),grid=css('--gridline');
  return {title:{text:t,font:{size:11}},type:log?'log':'linear',showline:true,linecolor:ink,linewidth:1,mirror:true,
    ticks:'outside',tickcolor:ink,ticklen:4,gridcolor:grid,griddash:'dot',zeroline:false,exponentformat:'power',
    range:log?undefined:[0,M.fmax],dtick:log?1:undefined};
}
function localPeaks(y,minProm){
  var out=[],n=y.length,w=Math.max(3,Math.round(n/40));
  for(var i=1;i<n-1;i++){
    var a=y[i-1],b=y[i],c=y[i+1];
    if(!(b>a&&b>=c)){continue;}
    var lo=Math.max(0,i-5*w),hi=Math.min(n,i+5*w),bg=[];
    for(var j=lo;j<hi;j++){if(Math.abs(j-i)>2*w&&y[j]>0){bg.push(y[j]);}}
    if(bg.length<3){continue;}
    bg.sort(function(p,q){return p-q;});
    var db=10*Math.log10(b/bg[Math.floor(bg.length/2)]);
    if(db>=minProm){out.push({f:F[i],db:db});}
  }
  return out;
}
function nearestPeak(y,fq,tol,minProm){
  var best=null,bd=1e9;
  localPeaks(y,minProm).forEach(function(p){var d=Math.abs(p.f-fq);if(d<=tol&&d<bd){bd=d;best=p;}});
  return best;
}
function draw(){
  var L=list(),cols=ramp(L.length),have=L.filter(function(c){return c.run;}),mk=markRow(L);
  var ink=css('--ink'),red=css('--red');
  var spec=['deck','watch','phone'];
  spec.forEach(function(dev,pi){
    var data=[];
    L.forEach(function(c){
      if(!c.run||state.hidden[c.i]){return;}
      if(state.rigid&&dev!=='deck'){
        data.push({x:F,y:curve(c.key,dev,'r',true),mode:'lines',line:{color:cols[c.i],width:1,dash:'dot'},opacity:.75,hoverinfo:'skip',showlegend:false});
      }
      data.push({x:F,y:curve(c.key,dev,'c',true),mode:'lines',line:{color:cols[c.i],width:(mk&&c.i===mk.i)?2.3:1.3},
        hovertemplate:lab(c.i)+'<br>%{x:.2f} Hz<br>%{y:.3e}<extra></extra>',showlegend:false});
      if(state.caps&&dev!=='deck'){
        var cp=c.run[dev].caps||[];
        var yc=curve(c.key,dev,'c',true);
        if(cp.length){data.push({x:cp.map(function(q){return q[0];}),y:cp.map(function(q){var j=Math.round((q[0]-M.fmin)/DF);return yc[Math.max(0,Math.min(NF-1,j))];}),mode:'markers',
          marker:{symbol:cp.map(function(q){return q[2]==='O'?'diamond':'diamond-open';}),size:9,color:cols[c.i],line:{color:cols[c.i],width:1.5}},
          hovertemplate:lab(c.i)+'<br>captured %{x:.2f} Hz<extra></extra>',showlegend:false});}
      }
    });
    var shapes=[],notes=[];
    if(mk){mk.run.refs.forEach(function(m){
      shapes.push({type:'line',xref:'x',yref:'paper',x0:m[1],x1:m[1],y0:0,y1:1,line:{color:red,width:1,dash:'dash'},opacity:.55});
      notes.push({x:m[1],y:1,xref:'x',yref:'paper',yanchor:'bottom',text:m[0],showarrow:false,font:{size:10,color:red}});
    });}
    if(state.gait&&mk&&dev!=='deck'){mk.run.hum.forEach(function(h){
      shapes.push({type:'line',xref:'x',yref:'paper',x0:h[0],x1:h[0],y0:0,y1:0.92,line:{color:css('--fam-'+h[1]),width:1,dash:'dot'},opacity:.9});
    });}
    state.picks.forEach(function(fq){
      shapes.push({type:'line',xref:'x',yref:'paper',x0:fq,x1:fq,y0:0,y1:1,line:{color:ink,width:1.6},opacity:.8});
    });
    var lay={margin:{l:58,r:10,t:16,b:40},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
      font:{family:'IBM Plex Mono, ui-monospace, monospace',size:11,color:ink},
      xaxis:axis('Frequency (Hz)',false),yaxis:axis(pi===0?'Deck PSD ((m/s²)²/Hz)':'Device PSD ((m/s²)²/Hz)',true),
      shapes:shapes,annotations:notes,hovermode:'closest',showlegend:false};
    Plotly.react('plot'+pi,data,lay,{displaylogo:false,responsive:true,modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']}).catch(function(){});
  });
  chips(L,cols,mk);picks(mk);gaitTable(mk);obsTables(L,cols,mk);freqTable(L,cols);status(L,have);
}
function chips(L,cols,mk){
  var box=$('chips');box.innerHTML='';
  L.forEach(function(c){
    var b=document.createElement('span');b.className='chip'+(state.hidden[c.i]?' hide':'');
    b.innerHTML='<i style="background:'+cols[c.i]+'"></i>'+lab(c.i)+(c.run?'':' (not stored)');
    b.title='Show or hide this level';
    b.onclick=function(){state.hidden[c.i]=!state.hidden[c.i];draw();};
    box.appendChild(b);
  });
  var s=$('mark');s.innerHTML='';
  L.forEach(function(c){if(c.run){var o=document.createElement('option');o.value=c.i;o.textContent=lab(c.i);s.appendChild(o);}});
  if(mk){s.value=mk.i;}
}
/* ---- picking and mode shapes ---- */
function modeOf(mk,fq){
  var best=null,bd=1e9;
  mk.run.refs.forEach(function(m){var d=Math.abs(m[1]-fq);if(d<bd){bd=d;best=m;}});
  var tol=Math.max(CAP,2/mk.run.Tres);
  return best?{lab:best[0],f:best[1],d:bd,sure:bd<=tol}:null;
}
function picks(mk){
  var box=$('picks');
  if(!state.picks.length){box.innerHTML='<span class="lbl">No mode selected. Click a peak on the deck graph.</span>';}
  else{box.innerHTML='<span class="lbl">Selected</span>';
    state.picks.forEach(function(fq){
      var m=mk?modeOf(mk,fq):null,b=document.createElement('span');
      b.className='chip'+(state.ms===fq?' on':'');
      b.innerHTML=fq.toFixed(2)+' Hz'+(m?' · '+m.lab+(m.sure?'':' ?'):'')+'  <b title="remove">×</b>';
      b.onclick=function(e){if(e.target.tagName==='B'){togglePick(fq);}else{state.ms=fq;draw();}};
      box.appendChild(b);
    });
    var c=document.createElement('button');c.textContent='Clear';c.onclick=function(){state.picks=[];state.ms=null;draw();};box.appendChild(c);
  }
  detectTable(mk);
  shapeView(mk);
}
function togglePick(fq){
  var hit=state.picks.findIndex(function(v){return Math.abs(v-fq)<CAP/2;});
  if(hit>=0){if(state.ms===state.picks[hit]){state.ms=null;}state.picks.splice(hit,1);}
  else{state.picks.push(fq);state.picks.sort(function(a,b){return a-b;});state.ms=fq;}
  draw();
}
function onDeckClick(e){
  var L=list(),mk=markRow(L);if(!mk||!e.points||!e.points.length){return;}
  var fq=e.points[0].x;
  var hit=state.picks.findIndex(function(v){return Math.abs(v-fq)<CAP/2;});
  if(hit>=0){togglePick(state.picks[hit]);return;}
  var p=nearestPeak(curve(mk.key,'deck','c',true),fq,CAP/2,0);
  togglePick(p?p.f:fq);
}
function detectTable(mk){
  var box=$('detect');
  if(!state.picks.length||!mk){box.innerHTML='';return;}
  var tol=Math.max(CAP,2/mk.run.Tres),h='<tr><th>selected (Hz)</th><th>mode</th><th>watch Δf</th><th>watch dB</th><th>watch</th><th>phone Δf</th><th>phone dB</th><th>phone</th></tr>';
  state.picks.forEach(function(fq){
    var m=modeOf(mk,fq),row='<td>'+fq.toFixed(2)+'</td><td>'+(m?m.lab+(m.sure?'':' ?'):'')+'</td>';
    ['watch','phone'].forEach(function(dev){
      var p=nearestPeak(curve(mk.key,dev,'c',false),fq,tol,-99),ok=p&&p.db>=PROM;
      row+='<td>'+(p?(p.f-fq>=0?'+':'')+(p.f-fq).toFixed(3):'')+'</td><td>'+(p?p.db.toFixed(1):'')+'</td><td class="'+(ok?'ok':'miss')+'">'+(ok?'found':'missed')+'</td>';
    });
    h+='<tr>'+row+'</tr>';
  });
  box.innerHTML='<table class="t">'+h+'</table><p class="hint">Found means a peak within ±'+tol.toFixed(2)+' Hz that stands at least '+PROM+' dB above its neighbourhood, on the device curve as recorded at '+lab(mk.i)+'.</p>';
}
var MS={timer:null,st:null};
function shapeView(mk){
  var box=$('shape'),txt=$('shapetxt');
  if(MS.timer){clearInterval(MS.timer);MS.timer=null;}
  if(state.ms===null||!mk){box.style.display='none';txt.textContent='';return;}
  var m=modeOf(mk,state.ms);if(!m){box.style.display='none';return;}
  var g=/^([A-Za-z]+)\s*(\d+)/.exec(m.lab)||[],kind=(g[1]||'B').charAt(0).toUpperCase(),n=Number(g[2]||1);
  var Lg=mk.run.L,W=mk.run.W,amp=kind==='L'?0.3*W:(kind==='A'?0.05*Lg:0.16*Lg),NX=61,NY=13,X=[],Y=[];
  for(var a=0;a<NX;a++){X.push(a/(NX-1));}for(var b=0;b<NY;b++){Y.push(b/(NY-1));}
  MS.st={kind:kind,n:n,L:Lg,W:W,amp:amp,X:X,Y:Y,ph:0};
  var name={T:'torsion',L:'lateral bending',A:'axial'}[kind]||'vertical bending';
  txt.innerHTML='<b>'+m.lab+'</b> at '+m.f.toFixed(2)+' Hz, '+name+', order '+n+'. '+(m.sure?'':'<span class="miss">The selected peak is '+m.d.toFixed(2)+' Hz from the nearest deck mode. It may not be a deck mode.</span>');
  box.style.display='block';
  var fr=msXYZ(MS.st),ink=css('--ink'),zr=(kind==='L'||kind==='A')?0.35*W:1.35*amp;
  var tr=[{type:'surface',x:fr.x,y:fr.y,z:fr.z,cmin:-amp,cmax:amp,showscale:false,hoverinfo:'skip',
      colorscale:[[0,'#2f5585'],[0.5,'#9aa6b2'],[1,'#d9472a']],lighting:{ambient:.65,diffuse:.8,specular:.15}},
    {type:'scatter3d',mode:'lines',x:fr.x[0],y:fr.y[0],z:fr.z[0],line:{color:'#2f5585',width:6},hoverinfo:'skip'},
    {type:'scatter3d',mode:'lines',x:fr.x[NY-1],y:fr.y[NY-1],z:fr.z[NY-1],line:{color:'#b83a1f',width:6},hoverinfo:'skip'},
    {type:'scatter3d',mode:'lines',x:[0,Lg,Lg,0,0],y:[-W/2,-W/2,W/2,W/2,-W/2],z:[0,0,0,0,0],line:{color:ink,width:2,dash:'dash'},hoverinfo:'skip'}];
  var nx=[],ny=[],nz=[];for(var k=1;k<n;k++){nx.push(Lg*k/n,Lg*k/n,null);ny.push(-W/2,W/2,null);nz.push(0,0,null);}
  tr.push({type:'scatter3d',mode:'lines',x:nx,y:ny,z:nz,line:{color:css('--cls-o'),width:5},hoverinfo:'skip'});
  var ax={color:ink,gridcolor:css('--hi'),backgroundcolor:'rgba(0,0,0,0)',zeroline:false};
  Plotly.newPlot('shapeplot',tr,{paper_bgcolor:'rgba(0,0,0,0)',margin:{l:0,r:0,t:0,b:0},showlegend:false,
    font:{family:'IBM Plex Mono, monospace',size:10,color:ink},
    scene:{aspectmode:'manual',aspectratio:{x:3.2,y:0.9,z:0.8},camera:{eye:{x:1.5,y:-1.8,z:0.9}},
      xaxis:Object.assign({title:{text:'span (m)'},range:[-0.02*Lg,1.02*Lg]},ax),
      yaxis:Object.assign({title:{text:'width (m)'},range:kind==='L'?[-0.9*W-amp,0.9*W+amp]:[-0.9*W,0.9*W]},ax),
      zaxis:Object.assign({title:{text:''},range:[-zr,zr],showticklabels:false},ax)}},{displaylogo:false,responsive:true});
  MS.busy=false;
  MS.timer=setInterval(function(){if(MS.busy){return;}MS.busy=true;MS.st.ph+=0.26;var f=msXYZ(MS.st);
    Plotly.restyle('shapeplot',{x:[f.x,f.x[0],f.x[NY-1]],y:[f.y,f.y[0],f.y[NY-1]],z:[f.z,f.z[0],f.z[NY-1]]},[0,1,2])
      .then(function(){MS.busy=false;},function(){MS.busy=false;});},90);
}
function msXYZ(st){
  var A=st.amp*Math.cos(st.ph),X=[],Y=[],Z=[];
  for(var b=0;b<st.Y.length;b++){var rx=[],ry=[],rz=[],y0=(st.Y[b]-0.5)*st.W;
    for(var a=0;a<st.X.length;a++){var x0=st.X[a]*st.L,s=Math.sin(st.n*Math.PI*st.X[a]),dx=0,dy=0,dz=0;
      if(st.kind==='T'){dz=A*s*(2*st.Y[b]-1);}else if(st.kind==='L'){dy=A*s;}
      else if(st.kind==='A'){dx=A*Math.sin((2*st.n-1)*0.5*Math.PI*st.X[a]);}else{dz=A*s;}
      rx.push(x0+dx);ry.push(y0+dy);rz.push(dz);}
    X.push(rx);Y.push(ry);Z.push(rz);}
  return {x:X,y:Y,z:Z};
}
/* ---- person's own lines ---- */
function gaitTable(mk){
  var box=$('gait');
  if(!mk){box.innerHTML='';return;}
  var r=mk.run,g={};
  r.hum.forEach(function(h){(g[h[1]]=g[h[1]]||[]).push(h[0]);});
  var h='<tr><th>family</th><th>frequencies (Hz)</th><th>why</th></tr>';
  ['gait','arm','body','drive'].forEach(function(f){
    if(!g[f]){return;}
    h+='<tr><td><span class="sw" style="background:'+css('--fam-'+f)+'"></span>'+FAMS[f]+'</td><td class="wrap">'+
      g[f].map(function(x){return x.toFixed(2);}).join('  ')+'</td><td class="wrap">'+FAMNOTE[f]+'</td></tr>';
  });
  box.innerHTML='<p class="hint">'+(r.fp?'Step rate '+r.fp.toFixed(2)+' Hz at '+lab(mk.i)+'. ':'Standing still: no gait, only the body modes. ')+
    'None of these lines belong to the deck.</p><div class="scroll"><table class="t">'+h+'</table></div>';
}
/* ---- observability ---- */
function obsCell(oi,extra){var c=cls(oi);return '<td class="cls '+c[0]+'" title="'+(extra||'')+'">'+oi.toFixed(1)+'</td>';}
function obsTables(L,cols,mk){
  var h='<tr><th>'+M.name[state.sweep]+'</th><th>watch OI</th><th>class</th><th>phone OI</th><th>class</th></tr>';
  L.forEach(function(c){if(!c.run){return;}
    var w=c.run.watch.oi,p=c.run.phone.oi;
    h+='<tr><td><span class="sw" style="background:'+cols[c.i]+'"></span>'+lab(c.i)+'</td>'+obsCell(w)+'<td class="cls '+cls(w)[0]+'">'+cls(w)[1]+'</td>'+obsCell(p)+'<td class="cls '+cls(p)[0]+'">'+cls(p)[1]+'</td></tr>';
  });
  $('obs1').innerHTML=h;
  if(!mk){$('obs2').innerHTML='';return;}
  var r=mk.run,k='<tr><th>mode</th><th>f (Hz)</th><th>watch OI</th><th>phone OI</th></tr>';
  r.refs.forEach(function(m){
    var a=r.watch.pmode[m[0]]||{oi:0},b=r.phone.pmode[m[0]]||{oi:0};
    var t=function(x){return x.F===undefined?'':'F '+x.F.toFixed(2)+'  Pown '+(x.Po||0).toFixed(2)+'  Pbg '+(x.Pb||0).toFixed(2);};
    k+='<tr><td>'+m[0]+'</td><td>'+m[1].toFixed(2)+'</td>'+obsCell(a.oi,t(a))+obsCell(b.oi,t(b))+'</tr>';
  });
  $('obs2').innerHTML=k;
  $('obs2cap').textContent='Per mode at '+lab(mk.i)+'. Hover a value to see its three factors.';
}
function freqTable(L,cols){
  var labs=[],seen={},ref={};
  L.forEach(function(c){if(c.run){c.run.refs.forEach(function(m){if(!seen[m[0]]){seen[m[0]]=1;labs.push(m[0]);ref[m[0]]=m[1];}});}});
  labs.sort(function(a,b){return ref[a]-ref[b];});
  var h='<tr><th>'+M.name[state.sweep]+'</th>'+labs.map(function(l){return '<th>'+l+'</th>';}).join('')+'</tr>';
  L.forEach(function(c){if(!c.run){return;}var f={};c.run.refs.forEach(function(m){f[m[0]]=m[1];});
    h+='<tr><td><span class="sw" style="background:'+cols[c.i]+'"></span>'+lab(c.i)+'</td>'+labs.map(function(l){return '<td>'+(l in f?f[l].toFixed(2):'')+'</td>';}).join('')+'</tr>';});
  $('modes').innerHTML=h;
}
function status(L,have){
  var off=P.filter(function(p){return p!==state.sweep&&state.sel[p]!==M.base[p];}).length,miss=L.length-have.length;
  var s='Sweeping '+M.name[state.sweep].toLowerCase()+' over '+L.length+' levels. ';
  s+=off?off+' other parameter'+(off>1?'s':'')+' off baseline. ':'Everything else at baseline. ';
  if(miss){s+='<b>'+miss+' level'+(miss>1?'s are':' is')+' not stored.</b> ';}
  $('status').innerHTML=s;$('note').textContent=M.note[state.sweep];
  var live=/^https?:$/.test(location.protocol)&&/^(localhost|127\.0\.0\.1)$/.test(location.hostname);
  $('compute').style.display=miss?'inline-block':'none';
  $('compute').textContent=live?'Compute missing levels':'Missing levels need the local server';
  $('compute').disabled=!live;$('cmd').style.display=(miss&&!live)?'block':'none';
}
function compute(){
  var b=$('compute');b.disabled=true;b.textContent='Computing...';
  fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sweep:state.sweep,sel:state.sel})})
    .then(function(r){return r.json();}).then(function(d){(d.runs||[]).forEach(function(r){RUNS[r.key]=r;});if(d.error){alert(d.error);}draw();})
    .catch(function(e){alert('Server not reachable: '+e);draw();});
}
function controls(){
  var box=$('params');box.innerHTML='';
  P.forEach(function(p){
    var row=document.createElement('div');row.className='prow'+(p===state.sweep?' sweep':'');
    var r=document.createElement('input');r.type='radio';r.name='sweep';r.id='r_'+p;r.checked=p===state.sweep;
    r.onchange=function(){state.sweep=p;state.hidden={};state.mark=null;state.picks=[];state.ms=null;controls();draw();};
    var l=document.createElement('label');l.htmlFor='r_'+p;l.innerHTML=M.name[p]+(state.sel[p]!==M.base[p]?'<span class="off">off base</span>':'');
    var s=document.createElement('select');s.disabled=p===state.sweep;
    M.labels[p].forEach(function(t,k){var o=document.createElement('option');o.value=k;o.textContent=t+(k===M.base[p]?'  (base)':'');s.appendChild(o);});
    s.value=state.sel[p];
    s.onchange=function(){state.sel[p]=parseInt(s.value,10);state.mark=null;controls();draw();};
    row.appendChild(r);row.appendChild(l);row.appendChild(s);box.appendChild(row);
  });
}
function toggle(id,key,name){$(id).onclick=function(){state[key]=!state[key];this.classList.toggle('on',state[key]);this.textContent=name+': '+(state[key]?'on':'off');draw();};}
toggle('rigid','rigid','Rigid floor');toggle('gaitbtn','gait','Gait lines');toggle('capbtn','caps','Captured modes');
$('mark').onchange=function(){state.mark=parseInt(this.value,10);draw();};
$('sig').oninput=function(){state.sigma=parseFloat(this.value);$('sigv').textContent=state.sigma>0?state.sigma.toFixed(2)+' Hz':'off';};
$('sig').onchange=function(){CACHE={};draw();};
$('allon').onclick=function(){state.hidden={};draw();};
$('alloff').onclick=function(){M.labels[state.sweep].forEach(function(_,k){state.hidden[k]=true;});draw();};
$('reset').onclick=function(){P.forEach(function(p){state.sel[p]=M.base[p];});state.hidden={};state.mark=null;state.picks=[];state.ms=null;controls();draw();};
$('compute').onclick=compute;
if(window.matchMedia){window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);}
controls();draw();
$('plot0').on('plotly_click',onDeckClick);
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
  <div class="bar"><button id="rigid">Rigid floor: off</button><button id="gaitbtn">Gait lines: off</button><button id="capbtn" class="on">Captured modes: on</button>
    <span class="lbl">detail level</span><select id="mark"></select>
    <span class="lbl">smoothing</span><input type="range" id="sig" min="0" max="0.5" step="0.01" value="__SIG__" style="width:130px;accent-color:var(--red)"><span class="lbl" id="sigv">__SIG__ Hz</span></div>
  <div class="bar"><button id="allon">Show all</button><button id="alloff">Hide all</button><div class="chips" id="chips"></div></div>
  <div class="plots">
    <div class="cell"><span class="tag">A / DECK</span><div class="plot" id="plot0"></div></div>
    <div class="cell"><span class="tag">B / SMARTWATCH (WRIST)</span><div class="plot" id="plot1"></div></div>
    <div class="cell"><span class="tag">C / SMARTPHONE (HIP)</span><div class="plot" id="plot2"></div></div>
  </div>
  <div class="bar chips" id="picks"></div>
  <p class="hint">A is the deck alone under the ambient shakers. B and C are the devices on the person, as the first singular value of x, y and z over __RECORDS__, left and right averaged. Red dashed lines are the deck modes of the detail level. Diamonds are captured modes, filled when observable. Dotted curves are the same records on a rigid floor.</p>
</section>

<section>
  <h2><span class="num">3</span>Selected modes</h2>
  <p class="hint">Click a peak on the deck graph. It snaps to the nearest maximum, is checked on both devices and its mode shape is drawn.</p>
  <div class="two">
    <div class="scroll" id="detect"></div>
    <div><div class="cell" id="shape" style="display:none;padding:4px"><div class="shape" id="shapeplot"></div></div><p class="hint" id="shapetxt"></p></div>
  </div>
</section>

<section>
  <h2><span class="num">4</span>The person's own lines</h2>
  <div id="gait"></div>
</section>

<section>
  <h2><span class="num">5</span>Observability</h2>
  <div class="eq">OI = 100 · G · (F · P<sub>own</sub> · P<sub>bg</sub>)<sup>1/3</sup> &nbsp; per mode and device. &nbsp; Unobservable below __T1__, observable from __T2__. Scored at σ = __SIG__ Hz.</div>
  <p class="hint">F is frequency agreement with the deck mode. P<sub>own</sub> is the peak's prominence on the device curve. P<sub>bg</sub> is its height over the rigid-floor curve. G is zero when the device curve does not rise above the rigid floor in the mode's band. A device's OI is the mean over the deck modes.</p>
  <div class="two">
    <div class="scroll"><table class="t" id="obs1"></table></div>
    <div><div class="scroll"><table class="t" id="obs2"></table></div><p class="hint" id="obs2cap"></p></div>
  </div>
</section>

<section>
  <h2><span class="num">6</span>Mode frequencies (Hz)</h2>
  <div class="scroll"><table class="t" id="modes"></table></div>
</section>

<footer class="foot">© 2026 Jovdat Babazade. All rights reserved. Free to run and study with credit to the author.</footer>
</div>
"""

KEEP = ("cb", "pkb", "refs", "Tres", "L", "W", "fp", "hum", "watch", "phone")


def write_page(path, meta, runs):
    from plotly.offline import get_plotlyjs
    stand = meta["mode"] == "standing"
    slim = {r["key"]: {k: r[k] for k in KEEP}
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
        "__T1__": f"{meta['t1']:g}", "__T2__": f"{meta['t2']:g}",
        "__SIG__": f"{meta['smooth']:.2f}",
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
            f"<script>{get_plotlyjs()}</script></head><body>"
            f"{body}"
            f"<script type=\"application/json\" id=\"meta\">{blob(meta)}</script>"
            f"<script type=\"application/json\" id=\"runs\">{blob(slim)}</script>"
            f"<script>{PAGE_JS}</script></body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
