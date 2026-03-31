// ui.js — all UI render functions, overlays, tabs, and canvas drawing
// Requires: data.js (all data vars), ground.js (updateScene, updateBars, bars, camera)

function toggleSquad(i,el){
  if(selectedSquad.has(i)) selectedSquad.delete(i);
  else selectedSquad.add(i);
  el.classList.toggle('on');
}

function applyTeamSelection(){
  var ourSel=document.querySelector('#sidebar select:nth-child(2)');
  var oppSel=document.querySelector('#sidebar select:nth-child(4)');
  var ourTeam=ourSel?ourSel.value:'Quetta Gladiators';
  var oppTeam=oppSel?oppSel.value:'Lahore Qalandars';

  selectedOurTeam=ourTeam;
  selectedOppTeam=oppTeam;

  var ourData=TEAM_ROSTERS[ourTeam];
  var oppData=OPP_ROSTERS[oppTeam]||OPP_ROSTERS['Lahore Qalandars'];
  if(!ourData) return;

  // Update xiOpts players for selected team
  xiOpts[0].players=ourData.players.map(function(p){return {n:p.n,p:p.p};});
  xiOpts[1].players=ourData.spinXI.map(function(p){return {n:p.n,p:p.p};});
  xiOpts[2].players=ourData.paceXI.map(function(p){return {n:p.n,p:p.p};});

  // Update opposition batting order
  oppBat=oppData;

  // Rebuild over plan with correct bowlers
  var newBowlers=ourData.bowlers;
  for(var i=0;i<20;i++){
    overPlan[i].bowler=newBowlers[planB[i]%newBowlers.length];
  }

  // Update fullSquad list
  var allPlayers=ourData.players.map(function(p){return p.n;});
  fullSquad.length=0;
  allPlayers.forEach(function(n){fullSquad.push(n);});

  // Refresh UI
  if(curTab===0){renderSidebar();renderRight();renderOverlays();renderPL();}
  else if(curTab===1){renderSidebar();renderRight();}
  else if(curTab===2){renderSidebar();renderOverlays();renderOppPL();}
}

function generateBrief(){
  var ourSel=document.querySelector('#sidebar select:nth-child(2)');
  var oppSel=document.querySelector('#sidebar select:nth-child(4)');
  var venSel=document.querySelector('#sidebar select:nth-child(6)');
  var ourTeam=ourSel?ourSel.value:'Quetta Gladiators';
  curOppTeam=oppSel?oppSel.value:'Lahore Qalandars';
  curVenue=venSel?venSel.value:'National Stadium, Karachi';

  var ourData=teams[ourTeam]||teams['Quetta Gladiators'];
  var oppData=teams[curOppTeam]||teams['Lahore Qalandars'];

  var lsOur=document.getElementById('ls-our');
  var lsOpp=document.getElementById('ls-opp');
  lsOur.textContent=ourData.abbr;lsOur.style.background=ourData.bg;lsOur.style.borderColor=ourData.color+'55';
  lsOpp.textContent=oppData.abbr;lsOpp.style.background=oppData.bg;lsOpp.style.borderColor=oppData.color+'55';
  document.getElementById('ls-sub').textContent=ourData.abbr+' vs '+oppData.abbr+' — Match Brief';

  var ls=document.getElementById('loading-screen');
  var fill=document.getElementById('ls-fill');
  ls.style.display='flex';
  fill.style.animation='none';
  fill.offsetHeight;
  fill.style.animation='lsfill 2.2s ease-in-out forwards';

  setTimeout(function(){
    ls.style.display='none';
    briefGenerated=true;
    curBriefFormation=1;
    renderSidebar();
  },2300);
}

function intensityToHex(v){
  if(v<0.35) return '#4CAF50';
  if(v<0.65) return '#FF9800';
  if(v<0.85) return '#FF5722';
  return '#F44336';
}

function buildPitchHeatmapHTML(venue,opp){
  var data=getPitchData(venue||curVenue,opp||curOppTeam);
  var zones=[
    {label:'FULL TOSS',key:'fullToss'},
    {label:'YORKER',key:'yorker'},
    {label:'GOOD LEN',key:'goodLen'},
    {label:'SHORT/GD',key:'shortOfGood'},
    {label:'SHORT',key:'short'},
  ];
  var maxCount=Math.max.apply(null,zones.map(function(z){return data[z.key];}));
  if(maxCount<1)maxCount=1;
  var total=zones.reduce(function(a,z){return a+data[z.key];},0)||1;

  var ph=420,pw=200,svgW=220,px=10;

  // Build gradient stops
  var gradStops=zones.map(function(z,i){
    var pct=Math.round((i/zones.length)*100);
    var intensity=data[z.key]/maxCount;
    var color=intensityToHex(intensity);
    var opacity=(0.3+intensity*0.6).toFixed(2);
    return '<stop offset="'+pct+'%" stop-color="'+color+'" stop-opacity="'+opacity+'"/>';
  }).join('');

  // Find max zone for radial glow
  var maxZoneIdx=0;var maxZoneCount=0;
  zones.forEach(function(z,i){if(data[z.key]>maxZoneCount){maxZoneCount=data[z.key];maxZoneIdx=i;}});
  var glowY=Math.round(18+(maxZoneIdx/zones.length)*(ph-36)+(ph-36)/zones.length/2);

  // Build scatter dots + hover hit-rects per zone
  var svgZones='';var hoverRects='';var yy=18;
  zones.forEach(function(z,zi){
    var cnt=data[z.key];
    var zh=Math.max(18,Math.round((cnt/total)*(ph-36))+14);
    for(var d=0;d<Math.min(cnt,10);d++){
      var dx=px+10+Math.random()*(pw-20);
      var dy=yy+6+Math.random()*(zh-12);
      svgZones+='<circle cx="'+dx+'" cy="'+dy+'" r="2.5" fill="rgba(255,255,255,0.75)"/>';
    }
    // Invisible hit-rect spanning full SVG width for easy hover
    var intensity=cnt/maxCount;
    var tipColor=intensityToHex(intensity);
    hoverRects+='<rect x="0" y="'+yy+'" width="'+svgW+'" height="'+zh+'" fill="transparent" style="cursor:pointer"'+
      ' onmouseover="(function(el,e){var t=document.getElementById(\'phm-tip\');if(!t)return;t.style.display=\'block\';t.innerHTML=\'<span style=&quot;color:'+tipColor+'&quot;>'+z.label+'</span><br>'+cnt+' wickets in last 10 matches\';var r=el.closest(\'.phm-wrap\');if(r){var rb=r.getBoundingClientRect();var tx=Math.min(e.clientX-rb.left+12,rb.width-120);t.style.top=(e.clientY-rb.top-30)+\'px\';t.style.left=tx+\'px\';}else{t.style.top=\'50%\';t.style.left=\'50%\';}})(this,event)"'+
      ' onmouseout="var t=document.getElementById(\'phm-tip\');if(t)t.style.display=\'none\'"/>';
    yy+=zh;
  });

  var pitchSVG='<svg viewBox="0 0 '+svgW+' '+ph+'" xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;height:auto">'+
    '<defs>'+
    '<linearGradient id="pg" x1="0" y1="0" x2="0" y2="1">'+
    '<stop offset="0%" stop-color="#B8934A"/><stop offset="50%" stop-color="#D4B882"/><stop offset="100%" stop-color="#B8934A"/>'+
    '</linearGradient>'+
    '<linearGradient id="zg" x1="0" y1="0" x2="0" y2="1">'+gradStops+'</linearGradient>'+
    '<radialGradient id="rg" cx="50%" cy="'+Math.round((glowY/ph)*100)+'%" r="40%">'+
    '<stop offset="0%" stop-color="#F44336" stop-opacity="0.4"/><stop offset="100%" stop-color="#F44336" stop-opacity="0"/>'+
    '</radialGradient>'+
    '</defs>'+
    // Grass background (full SVG width)
    '<rect width="'+svgW+'" height="'+ph+'" fill="#0d1f0d" rx="3"/>'+
    // Outfield stripe lines for depth
    '<line x1="0" y1="50" x2="'+svgW+'" y2="50" stroke="rgba(255,255,255,0.03)" stroke-width="14"/>'+
    '<line x1="0" y1="100" x2="'+svgW+'" y2="100" stroke="rgba(255,255,255,0.03)" stroke-width="14"/>'+
    '<line x1="0" y1="150" x2="'+svgW+'" y2="150" stroke="rgba(255,255,255,0.03)" stroke-width="14"/>'+
    // Pitch surface (centered)
    '<rect x="'+px+'" width="'+pw+'" height="'+ph+'" fill="url(#pg)"/>'+
    '<rect x="'+(px+4)+'" y="18" width="'+(pw-8)+'" height="'+(ph-36)+'" fill="url(#zg)" rx="2"/>'+
    '<rect x="'+(px+4)+'" y="18" width="'+(pw-8)+'" height="'+(ph-36)+'" fill="url(#rg)" rx="2"/>'+
    svgZones+hoverRects+
    // Crease lines
    '<line x1="'+px+'" y1="16" x2="'+(px+pw)+'" y2="16" stroke="rgba(255,255,255,0.8)" stroke-width="1.2"/>'+
    '<line x1="'+px+'" y1="'+(ph-16)+'" x2="'+(px+pw)+'" y2="'+(ph-16)+'" stroke="rgba(255,255,255,0.8)" stroke-width="1.2"/>'+
    // Stumps top
    '<line x1="'+(px+pw/2-5)+'" y1="12" x2="'+(px+pw/2-5)+'" y2="20" stroke="white" stroke-width="1.5"/>'+
    '<line x1="'+(px+pw/2)+'" y1="12" x2="'+(px+pw/2)+'" y2="20" stroke="white" stroke-width="1.5"/>'+
    '<line x1="'+(px+pw/2+5)+'" y1="12" x2="'+(px+pw/2+5)+'" y2="20" stroke="white" stroke-width="1.5"/>'+
    // Stumps bottom
    '<line x1="'+(px+pw/2-5)+'" y1="'+(ph-20)+'" x2="'+(px+pw/2-5)+'" y2="'+(ph-12)+'" stroke="white" stroke-width="1.5"/>'+
    '<line x1="'+(px+pw/2)+'" y1="'+(ph-20)+'" x2="'+(px+pw/2)+'" y2="'+(ph-12)+'" stroke="white" stroke-width="1.5"/>'+
    '<line x1="'+(px+pw/2+5)+'" y1="'+(ph-20)+'" x2="'+(px+pw/2+5)+'" y2="'+(ph-12)+'" stroke="white" stroke-width="1.5"/>'+
    // Pitch border
    '<rect x="'+px+'" width="'+pw+'" height="'+ph+'" fill="none" stroke="rgba(255,215,0,0.3)" stroke-width="1"/>'+
    // Labels
    '<text x="'+(px+2)+'" y="10" font-family="Orbitron" font-size="4" fill="rgba(255,255,255,0.4)">LEG</text>'+
    '<text x="'+(px+pw-14)+'" y="10" font-family="Orbitron" font-size="4" fill="rgba(255,255,255,0.4)">OFF</text>'+
    '<text x="'+(px+pw/2)+'" y="'+(ph-4)+'" font-family="Orbitron" font-size="4" fill="rgba(255,215,0,0.5)" text-anchor="middle">MID</text>'+
    '</svg>';

  return '<div style="margin-bottom:4px">'+
    '<div style="font-family:Rajdhani;font-size:7px;color:rgba(255,255,255,0.25);letter-spacing:1px;margin-bottom:4px">WICKET HEATMAP · LAST 10 MATCHES · '+((venue||curVenue).split(',')[0].toUpperCase())+'</div>'+
    '<div class="phm-wrap" style="position:relative">'+
      '<div class="phm-pitch">'+pitchSVG+'</div>'+
      '<div id="phm-tip" style="display:none;position:absolute;left:70px;top:30px;background:rgba(10,5,25,0.92);border:1px solid rgba(255,215,0,0.25);border-radius:6px;padding:6px 10px;font-family:Rajdhani;font-size:10px;color:rgba(255,255,255,0.85);white-space:nowrap;pointer-events:none;z-index:20;line-height:1.6"></div>'+
    '</div>'+
    '<div style="display:flex;gap:8px;margin-top:5px;flex-wrap:wrap">'+
    '<span style="display:flex;align-items:center;gap:3px;font-family:Rajdhani;font-size:6px;color:rgba(255,255,255,0.3)"><span style="width:6px;height:6px;border-radius:50%;background:#4CAF50;display:inline-block"></span>Low Risk</span>'+
    '<span style="display:flex;align-items:center;gap:3px;font-family:Rajdhani;font-size:6px;color:rgba(255,255,255,0.3)"><span style="width:6px;height:6px;border-radius:50%;background:#FF9800;display:inline-block"></span>Moderate</span>'+
    '<span style="display:flex;align-items:center;gap:3px;font-family:Rajdhani;font-size:6px;color:rgba(255,255,255,0.3)"><span style="width:6px;height:6px;border-radius:50%;background:#F44336;display:inline-block"></span>High Danger</span>'+
    '</div></div>';
}

function renderBriefSidebar(){
  var el=document.getElementById('sidebar');
  var ourData=teams[selectedOurTeam]||teams['Quetta Gladiators'];
  var oppData=teams[curOppTeam]||teams[selectedOppTeam]||teams['Lahore Qalandars'];
  // Use xiOpts (live from backend after brief) for the current formation tab
  var xiIdx=Math.max(0,Math.min(curBriefFormation-1,xiOpts.length-1));
  var curXIPlayers=xiOpts[xiIdx]?xiOpts[xiIdx].players:[];
  var curXIName=xiOpts[xiIdx]?xiOpts[xiIdx].name:'Formation';
  var curXIColor=xiOpts[xiIdx]?xiOpts[xiIdx].color:'#FFD700';

  el.innerHTML='<div class="brief-panel">'+
    '<div class="bp-header">'+
      '<div class="bp-teams">'+
        '<div class="bp-logo" style="background:'+(ourData.bg||'#23074F')+'">'+ourData.abbr+'</div>'+
        '<div><div class="bp-vs">'+ourData.abbr+' vs '+oppData.abbr+'</div>'+
        '<div class="bp-venue">'+curVenue.split(',')[0]+'</div></div>'+
        '<div class="bp-logo" style="background:'+(oppData.bg||'#003300')+'">'+oppData.abbr+'</div>'+
      '</div>'+
      '<div style="display:flex;align-items:center;gap:6px;margin-top:6px;padding:5px 8px;background:rgba(255,255,255,0.03);border-radius:6px;border:1px solid '+curXIColor+'22">'+
        '<div style="width:7px;height:7px;border-radius:50%;background:'+curXIColor+';flex-shrink:0"></div>'+
        '<div style="font-family:Orbitron;font-size:7px;font-weight:700;letter-spacing:1.5px;color:'+curXIColor+'">'+curXIName.toUpperCase()+'</div>'+
        '<div style="font-family:Rajdhani;font-size:7px;color:rgba(255,255,255,0.2);margin-left:auto">Select dot to switch</div>'+
      '</div>'+
    '</div>'+
    '<div class="bp-body">'+
      '<div class="bp-section" style="color:'+curXIColor+'88">▸ '+curXIName+'</div>'+
      curXIPlayers.map(function(p,i){var role=(p.role||'bat').toUpperCase();return '<div class="bp-player xi-on"><span class="bp-num">'+(i+1)+'</span><span class="bp-pname">'+(p.n||p.name||'')+'</span><span class="bp-role '+role.toLowerCase()+'">'+role+'</span></div>';}).join('')+
      '<div class="bp-section" style="margin-top:12px">Opposition — '+oppData.abbr+'</div>'+
      oppBat.map(function(b,i){var d=(b.danger||'low').toUpperCase();var r=(b.style||'BAT').toUpperCase();return '<div class="bp-player"><span class="bp-num">'+(i+1)+'</span><span class="bp-pname">'+b.name+'</span><span class="bp-role bat">'+r+'</span><span class="bat-danger '+(b.danger||'low')+'" style="font-size:5px;padding:1px 4px">'+d+'</span></div>';}).join('')+
    '</div></div>';
}

function renderTabs(){
  var subRow='';
  if(curTab===0){
    subRow='<div class="xi-dot-row">'+xiOpts.map(function(o,i){return '<div class="xi-dot '+(i===curXI?'act':'')+'" style="background:'+o.color+';'+(i===curXI?'box-shadow:0 0 8px '+o.color+'88':'')+'" onclick="curXI='+i+';curBriefFormation='+(i+1)+';renderSidebar();renderOverlays();renderPL();renderTabs()"></div>';}).join('')+'</div>'+
      '<div style="font-family:Rajdhani;font-size:9px;font-weight:600;letter-spacing:1.5px;color:'+xiOpts[curXI].color+';margin-top:3px">'+xiOpts[curXI].name.toUpperCase()+'</div>';
  } else if(curTab===1){
    subRow='<div class="xi-dot-row"><button class="vb act" onclick="setView(\'3d\',this)">3D-Perspective</button><button class="vb" onclick="setView(\'top\',this)">Top Down</button></div>';
  }
  var tabHTML='<div class="tab-row">'+TABS.map(function(t,i){return '<button class="tbtn '+(i===curTab?'active':'')+'" onclick="switchTab('+i+')">'+t+'</button>';}).join('')+'</div>'+subRow;
  document.getElementById('tab-area').innerHTML=tabHTML;
  document.getElementById('tab-area-full').innerHTML='<div class="tab-row">'+TABS.map(function(t,i){return '<button class="tbtn '+(i===curTab?'active':'')+'" onclick="switchTab('+i+')">'+t+'</button>';}).join('')+'</div>';
  var vbar=document.getElementById('vbar');
  if(vbar) vbar.classList.toggle('hidden',curTab!==0);
}

function switchTab(idx){
  curTab=idx;renderTabs();
  var is3=idx<3;
  document.getElementById('layout-3col').classList.toggle('hidden',!is3);
  document.getElementById('layout-full').classList.toggle('hidden',is3);
  var zc=document.getElementById('zoom-ctrl');
  var zh=document.getElementById('zoom-hint');
  if(zc) zc.style.display=(idx===1)?'flex':'none';
  if(zh) zh.style.display=(idx===1)?'block':'none';
  // Ball-ring canvas: always clear — not used on any tab
  var _brc=document.getElementById('ball-ring-canvas');
  if(_brc){var _brcCtx=_brc.getContext('2d');_brcCtx.clearRect(0,0,_brc.width,_brc.height);}
  if(is3){renderSidebar();renderRight();renderOverlays();updateScene();}
  else{renderScenarios();}
  // PDF button: only on Final Playing XI tab (idx === 0)
  var _pdf=document.getElementById('pdf-btn-overlay');
  if(_pdf) _pdf.style.display=(idx===0)?'block':'none';
}

function renderSidebar(){
  var el=document.getElementById('sidebar');
  if(curTab===0){
    if(briefGenerated){renderBriefSidebar();return;}
    var ourTeamOpts=['Quetta Gladiators','Lahore Qalandars','Islamabad United','Karachi Kings','Peshawar Zalmi','Multan Sultans','Rawalpindiz','Hyderabad Kingsmen'];
    var oppTeamOpts=['Lahore Qalandars','Quetta Gladiators','Islamabad United','Karachi Kings','Peshawar Zalmi','Multan Sultans','Rawalpindiz','Hyderabad Kingsmen'];
    el.innerHTML='<div class="st">Match Setup</div>'+
      '<span class="sl">Our Team</span><select class="ss" onchange="applyTeamSelection()">'+ourTeamOpts.map(function(t){return '<option'+(t===selectedOurTeam?' selected':'')+'>'+t+'</option>';}).join('')+'</select>'+
      '<span class="sl">Opposition</span><select class="ss" onchange="applyTeamSelection()">'+oppTeamOpts.map(function(t){return '<option'+(t===selectedOppTeam?' selected':'')+'>'+t+'</option>';}).join('')+'</select>'+
      '<span class="sl">Venue</span><select class="ss"><option>National Stadium, Karachi</option><option>Gaddafi Stadium, Lahore</option><option>Rawalpindi Cricket Stadium</option><option>Multan Cricket Stadium</option><option>Arbab Niaz Stadium, Peshawar</option><option>Iqbal Stadium, Faisalabad</option></select>'+
      '<span class="sl">Date & Time</span><input class="ss" type="datetime-local" value="2026-03-25T19:00" style="font-size:9px">'+
      '<span class="sl">Squad (tap to toggle)</span>'+
      '<div class="sp">'+fullSquad.map(function(n,i){return '<span class="spl '+(selectedSquad.has(i)?'on':'')+'" onclick="toggleSquad('+i+',this)">'+n+'</span>';}).join('')+'</div>'+
      '<button class="sbtn gold" onclick="generateBrief()">Generate Brief</button>';
  } else if(curTab===1){
    el.innerHTML='<div class="st">Over-by-Over Plan</div>'+overPlan.map(function(o){return '<div class="opi '+(o.over===curOver?'act':'')+'" onclick="curOver='+o.over+';renderSidebar();updateBars()"><div class="opi-h"><span class="opi-o">Over '+o.over+'</span><span class="opi-ph '+o.phase+'">'+(o.phase==='pp'?'PP':o.phase==='mid'?'MID':'DEATH')+'</span></div><div class="opi-b">'+o.bowler+'</div><div class="opi-n">'+o.note+'</div></div>';}).join('');
  } else if(curTab===2){
    el.innerHTML='<div class="st">Opposition Batting Order</div><div style="font-family:Rajdhani;font-size:9px;color:rgba(255,255,255,0.2);margin-bottom:8px">'+selectedOppTeam.toUpperCase()+'</div>'+oppBat.map(function(b,i){return '<div class="bat-item"><span class="bat-num">'+(i+1)+'</span><span class="bat-name">'+b.name+'</span><span class="bat-danger '+(b.danger||'low')+'">'+(b.danger||'low').toUpperCase()+'</span></div>';}).join('');
  }
}

function renderRight(){
  var el=document.getElementById('right-panel');
  if(!briefGenerated){
    el.innerHTML=
      '<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:32px 16px;text-align:center;gap:16px">'+
        '<div style="width:52px;height:52px;border-radius:50%;border:2px solid rgba(255,215,0,0.15);display:flex;align-items:center;justify-content:center">'+
          '<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="rgba(255,215,0,0.35)" stroke-width="1.5">'+
            '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'+
          '</svg>'+
        '</div>'+
        '<div style="font-family:Orbitron;font-size:8px;font-weight:700;color:rgba(255,215,0,0.4);letter-spacing:2px">AWAITING BRIEF</div>'+
        '<div style="font-family:Rajdhani;font-size:9px;color:rgba(255,255,255,0.18);line-height:1.6;max-width:140px">'+
          'Select match parameters and click<br><span style="color:rgba(255,215,0,0.35)">Generate Brief</span> to load<br>Playing XI, analysis & intel'+
        '</div>'+
        '<div style="width:40px;height:1px;background:rgba(255,215,0,0.1)"></div>'+
        '<div style="font-family:Rajdhani;font-size:7px;color:rgba(255,255,255,0.1);letter-spacing:1.5px">PSL DECISION INTELLIGENCE</div>'+
      '</div>';
    return;
  }
  if(curTab===0) el.innerHTML=renderXIRight();
  else if(curTab===1) el.innerHTML=renderBowlRight();
  else if(curTab===2) el.innerHTML=renderOppRight();
}

function renderXIRight(){
  var roleMap={
    'Babar Azam':'bat','Saud Shakeel':'bat','Sarfaraz Ahmed':'wk',
    'Di Warner':'bat','Shadab Khan':'all','W. Hasaranga':'all',
    'Imad Wasim':'all','Zahid Mahmood':'bowl','Naseem Shah':'bowl',
    'Rashid Khan':'all','Karv Bhan':'bowl','Rasa Yesea':'all',
    'Rase Ronme':'bowl','Rashs Vassa':'bowl','Shaheen Afridi':'bowl',
    'Iftikhar Ahmed':'all','Mohammad Nawaz':'all','Azam Khan':'bat'
  };
  var curPlayers=xiOpts[curXI].players;
  var roleLabels={bat:'BAT',bowl:'BOWL',all:'ALL',wk:'WK'};
  var xiListHTML=curPlayers.map(function(p,i){
    var role=roleMap[p.n]||'bat';
    return '<div class="xi-list-row"><span class="xi-list-num">'+(i+1)+'</span><span class="xi-list-name">'+p.n+'</span><span class="xi-list-role '+role+'">'+roleLabels[role]+'</span></div>';
  }).join('');

  var _wnd=window._liveWeather;
  var _windKph=_wnd&&_wnd.wind_kph?Math.round(_wnd.wind_kph):0;
  var _windDir=_wnd&&_wnd.wind_dir?_wnd.wind_dir:'--';
  var _windLabel=_windDir!=='--'?(_windDir+' '+_windKph+' km/h'):'No live wind data';
  var _windColor=_windKph>25?'rgba(244,67,54,0.7)':_windKph>12?'rgba(255,152,0,0.6)':'rgba(0,191,255,0.5)';
  var _swingNote=_wnd&&_wnd.swing_bonus>1.1?'Humidity drives swing — use new ball early':'Standard swing conditions';
  return '<div class="rc"><div class="rt"><span class="d"></span>Interactive Blueprint</div>'+
    '<div class="wind-v">'+
    [15,38,62,82].map(function(t,i){return '<div class="wa" style="top:'+t+'%;left:'+(5+i*15)+'%;animation-delay:'+(i*.4)+'s"><svg viewBox="0 0 35 10"><path d="M0,5 L25,5" stroke="'+_windColor+'" stroke-width="1.5" fill="none" opacity="'+(0.3+i*0.1)+'"/><polygon points="25,2 33,5 25,8" fill="'+_windColor+'" opacity="'+(0.3+i*0.1)+'"/></svg></div>';}).join('')+
    '<div style="position:absolute;bottom:3px;right:5px;font-family:Rajdhani;font-size:6px;color:'+_windColor+'">'+_windLabel+'</div>'+
    '</div>'+
    '<div style="font-family:Rajdhani;font-size:7px;color:rgba(255,255,255,0.15);margin-top:3px;text-align:center">'+_swingNote+'</div>'+
    '</div>'+
    '<div class="rc"><div class="rt"><span class="d"></span>Weather Hub</div>'+
    (function(){
      var w=window._liveWeather;
      var hum=w?Math.round(w.humidity):68;
      var temp=w?Math.round(w.temp):22;
      var dewOver=w?w.dew_onset_over:0;
      var spin=w?Math.round((1-w.spinner_penalty)*100):0;
      var humOff=Math.round(100-hum);
      var spinColor=spin>40?'#F44336':spin>20?'#FF8C00':'#4CAF50';
      var warnings=w&&w.warnings&&w.warnings.length?'<div style="font-family:Rajdhani;font-size:7px;color:#FFC107;margin-top:4px">⚠ '+w.warnings.join(' · ')+'</div>':'';
      return '<div class="wm"><div class="wm-r"><svg viewBox="0 0 40 40"><circle class="tk" cx="20" cy="20" r="16"/><circle class="fl" cx="20" cy="20" r="16" stroke="#8B5CF6" stroke-dasharray="100" stroke-dashoffset="'+humOff+'"/></svg></div><div><div class="wm-v" style="color:#8B5CF6">'+hum+'%</div><div class="wm-l">Humidity</div></div></div>'+
        '<div class="wm"><div class="wm-r"><svg viewBox="0 0 40 40"><circle class="tk" cx="20" cy="20" r="16"/><circle class="fl" cx="20" cy="20" r="16" stroke="#FFD700" stroke-dasharray="100" stroke-dashoffset="'+(100-Math.min(100,Math.round(temp/50*100)))+'"/></svg></div><div><div class="wm-v" style="color:#FFD700">'+temp+'°C</div><div class="wm-l">Temperature</div></div></div>'+
        '<div class="wm"><div style="width:30px;height:30px;border-radius:50%;border:1.5px solid '+spinColor+';background:rgba(76,175,80,0.06);display:flex;align-items:center;justify-content:center;font-family:Orbitron;font-size:7px;font-weight:700;color:'+spinColor+'">'+spin+'%</div><div><div class="wm-v" style="color:'+spinColor+'">'+(dewOver>0?'Ov '+dewOver:'No Dew')+'</div><div class="wm-l">Dew Onset</div></div></div>'+
        warnings;
    })()+
    '</div>'+
    '<div class="rc"><div class="rt"><span class="d"></span>Soil Health</div>'+
    '<div style="text-align:center;padding:6px 0"><span class="soil-num">4.2</span><span class="soil-max">/5</span></div>'+
    '<div class="soil-conds">'+
    '<div class="soil-c"><div class="soil-ic">🪨</div><div class="soil-lb">Cracked</div></div>'+
    '<div class="soil-c"><div class="soil-ic">🌾</div><div class="soil-lb">Dry Grass</div></div>'+
    '</div></div>'+
    '<div class="rc"><div class="rt"><span class="d"></span>Playing XI — '+xiOpts[curXI].name+'</div>'+
    xiListHTML+'</div>';
}

function renderBowlRight(){
  var barHTML=Array.from({length:20},function(_,i){
    var p=20+Math.random()*70,s=10+Math.random()*50,d=5+Math.random()*35;
    return '<div class="gbc-g"><div class="gbc-b" style="width:6px;height:'+p+'%;background:linear-gradient(180deg,#8B5CF6,#6D28D9)"></div><div class="gbc-b" style="width:6px;height:'+s+'%;background:linear-gradient(180deg,#FFD700,#FF8C00)"></div><div class="gbc-b" style="width:6px;height:'+d+'%;background:linear-gradient(180deg,#F44336,#C62828)"></div></div>';
  }).join('');
  return '<div class="rc"><div class="rt"><span class="d"></span>Wicket Heatmap</div>'+buildPitchHeatmapHTML()+'</div>'+
    (function(){
      var pp=0,mid=0,death=0;
      overPlan.forEach(function(o){
        var ph=(o.phase||'').toLowerCase();
        if(ph==='pp'||ph.indexOf('power')>=0) pp++;
        else if(ph==='death'||ph.indexOf('death')>=0) death++;
        else mid++;
      });
      return '<div class="rc"><div class="rt"><span class="d"></span>Bowling Phase Summary</div>'+
      '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;text-align:center">'+
      '<div style="padding:6px;background:rgba(0,229,255,0.04);border-radius:6px;border:1px solid rgba(0,229,255,0.08)"><div style="font-family:Orbitron;font-size:12px;font-weight:800;color:#00E5FF">'+pp+'</div><div style="font-family:Rajdhani;font-size:6px;letter-spacing:1.5px;color:rgba(255,255,255,0.25)">POWERPLAY</div></div>'+
      '<div style="padding:6px;background:rgba(255,215,0,0.03);border-radius:6px;border:1px solid rgba(255,215,0,0.06)"><div style="font-family:Orbitron;font-size:12px;font-weight:800;color:#FFD700">'+mid+'</div><div style="font-family:Rajdhani;font-size:6px;letter-spacing:1.5px;color:rgba(255,255,255,0.25)">MIDDLE</div></div>'+
      '<div style="padding:6px;background:rgba(244,67,54,0.04);border-radius:6px;border:1px solid rgba(244,67,54,0.06)"><div style="font-family:Orbitron;font-size:12px;font-weight:800;color:#F44336">'+death+'</div><div style="font-family:Rajdhani;font-size:6px;letter-spacing:1.5px;color:rgba(255,255,255,0.25)">DEATH</div></div>'+
      '</div></div>';
    })()+
    '<div class="rc"><div class="rt"><span class="d"></span>Bowler Allocation</div>'+(function(){
      var map={};
      overPlan.forEach(function(o){
        var k=o.bowler||'?';
        if(!map[k]) map[k]={name:k,overs:[],total:0};
        map[k].overs.push(o.over);
        map[k].total++;
      });
      return Object.keys(map).map(function(k){var b=map[k];return '<div class="ba-row"><span class="ba-name">'+b.name+'</span><div class="ba-overs">'+b.overs.map(function(o){return '<span class="ba-ov">'+o+'</span>';}).join('')+'</div><span class="ba-total">'+b.total+' overs</span></div>';}).join('');
    }())+'</div>'+
    '<div class="rc"><div class="rt"><span class="d"></span>Attack Chart</div>'+
    '<div class="gbc">'+barHTML+'</div>'+
    '<div class="gbc-labels"><span>1</span><span>5</span><span>10</span><span>15</span><span>20</span></div>'+
    '<div style="display:flex;gap:8px;margin-top:4px">'+
    '<span style="display:flex;align-items:center;gap:3px;font-size:7px;color:rgba(255,255,255,0.3)"><span style="width:6px;height:6px;border-radius:1px;background:#8B5CF6"></span>Pace</span>'+
    '<span style="display:flex;align-items:center;gap:3px;font-size:7px;color:rgba(255,255,255,0.3)"><span style="width:6px;height:6px;border-radius:1px;background:#FFD700"></span>Spin</span>'+
    '<span style="display:flex;align-items:center;gap:3px;font-size:7px;color:rgba(255,255,255,0.3)"><span style="width:6px;height:6px;border-radius:1px;background:#F44336"></span>Danger</span>'+
    '</div></div>';
}

function renderOppRight(){
  return '<div class="rc"><div class="rt"><span class="d"></span>Key Notes & Arrival</div>'+
    oppBat.map(function(b){
      var danger=b.danger||'low';
      var arrival=b.arrival||b.arrival_over_range||'';
      var note=b.notes||b.note||'';
      var srTag=b.career_sr?'<span style="font-family:Orbitron;font-size:7px;color:rgba(255,255,255,0.3);margin-left:4px">SR '+b.career_sr+'</span>':'';
      var conf=b.confidence?'<span style="font-family:Rajdhani;font-size:6px;color:rgba(255,255,255,0.2);margin-left:4px">['+b.confidence+']</span>':'';
      return '<div class="kn '+danger+'"><div style="display:flex;justify-content:space-between;align-items:center"><div class="kn-conf '+danger+'">'+b.name+srTag+conf+'</div><span class="arr-badge">Arrives: '+arrival+'</span></div><div class="kn-text">'+note+'</div></div>';
    }).join('')+
    '</div>';
}

function renderOverlays(){
  var el=document.getElementById('overlays');
  var plEl=document.getElementById('player-labels');
  var zlEl=document.getElementById('zl');

  if(curTab===0){
    var _tossRec=(window._liveToss&&window._liveToss.recommendation)||'BAT FIRST';
    var _tossWhy=(window._liveToss&&window._liveToss.reasoning&&window._liveToss.reasoning.length)?window._liveToss.reasoning.join(' '):'Dew factor significant in 2nd innings at this venue. Set a target and defend. Pitch will slow down under lights.';
    var _tossDL=(window._liveToss&&window._liveToss.dl_note)?'<div class="toss-dl" style="font-family:Rajdhani;font-size:7px;color:#FFC107;margin-top:4px;padding-top:4px;border-top:1px solid rgba(255,215,0,0.1)">'+window._liveToss.dl_note+'</div>':'';
    el.innerHTML='<div class="toss-card"><div class="toss-t"><span class="d"></span>Toss Recommendation</div><div class="toss-rec">'+_tossRec+'</div><div class="toss-why">'+_tossWhy+'</div>'+_tossDL+'</div>';
    plEl.style.display='block';zlEl.classList.add('hidden');renderPL();
  } else if(curTab===1){
    var _alertTxt=window._liveTacticalAlert||'Monitor opening bowler costs — escalate to backup if 15+ in first over with new ball.';
    el.innerHTML='<div class="alert-card"><div class="alert-t">⚠ TACTICAL ALERT</div><div class="alert-txt">'+_alertTxt+'</div></div><div class="cont-card"><div class="cont-t">Contingency Notes</div>'+(contingency.length?contingency.map(function(n){return '<div class="cont-item"><span style="color:#FFC107">⚡</span><span class="cont-text">'+n+'</span></div>';}).join(''):'<div class="cont-item" style="color:rgba(255,255,255,0.2);font-family:Rajdhani;font-size:8px">Generate brief to load live contingencies</div>')+'</div>';
    plEl.style.display='none';zlEl.classList.remove('hidden');
    zlEl.innerHTML='<span><span class="zd" style="background:#F44336"></span>Fine Leg 7%</span><span><span class="zd" style="background:#FF8C00"></span>Square Leg 10%</span><span><span class="zd" style="background:#00E5FF"></span>Mid Wicket 15%</span><span><span class="zd" style="background:#4CAF50"></span>Long On 13%</span><span><span class="zd" style="background:#FFD700"></span>Long Off 22%</span><span><span class="zd" style="background:#8B5CF6"></span>Cover 12%</span>';
  } else if(curTab===2){
    el.innerHTML='<div class="matchup-overlay">'+matchupNotes.map(function(n){return '<div class="mn-card '+n.conf+'"><span class="mn-lvl '+n.conf+'">'+(n.conf==='high'?'High':'Medium')+'</span><span class="mn-balls">'+n.balls+' balls</span><div class="mn-txt">'+n.text+'</div></div>';}).join('')+'</div>';
    plEl.style.display='block';zlEl.classList.add('hidden');renderOppPL();
  }
}

function renderPL(){
  var el=document.getElementById('player-labels');
  var ps=xiOpts[curXI].players;
  el.innerHTML=ps.map(function(p,i){
    var initials=p.n.split(' ').map(function(w){return w[0];}).join('').slice(0,2);
    var avHtml;
    if(p.p){
      avHtml='<div class="av" style="position:relative;overflow:hidden">'+
        '<img src="'+p.p+'" style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;object-position:top center;border-radius:50%" onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'" alt="">'+
        '<span style="position:absolute;top:0;left:0;width:100%;height:100%;display:none;align-items:center;justify-content:center;font-family:Orbitron;font-size:9px;font-weight:700;color:#FFD700">'+initials+'</span>'+
        '</div>';
    }else{
      avHtml='<div class="av">'+initials+'</div>';
    }
    return '<div class="pl" id="pl-'+i+'">'+avHtml+'<div class="nf">'+p.n.split(' ').pop()+'</div></div>';
  }).join('');
}

function renderOppPL(){
  var el=document.getElementById('player-labels');
  el.innerHTML=oppBat.map(function(b,i){
    var initials=b.name.split(' ').map(function(w){return w[0];}).join('').slice(0,2);
    var danger=b.danger||'low';
    return '<div class="pl" id="pl-'+i+'"><div class="av">'+initials+'</div><div class="nf">'+b.name.split(' ').pop()+'</div><div class="db '+danger+'">'+danger.toUpperCase()+'</div></div>';
  }).join('');
}

function _roleClass(role){
  if(!role) return 'anc';
  var r=role.toLowerCase();
  if(r.indexOf('agg')>=0||r.indexOf('attack')>=0) return 'agg';
  if(r.indexOf('fin')>=0) return 'fin';
  if(r.indexOf('sup')>=0) return 'sup';
  return 'anc';
}

function renderScenarios(){
  document.getElementById('scenarios-content').innerHTML=
    '<div class="scenarios-grid">'+scenarios.map(function(s){
      var cls=s.cls||(s.id?s.id.toLowerCase():'a');
      var weatherLine=s.weather_note?'<div style="font-family:Rajdhani;font-size:7px;color:rgba(0,229,255,0.5);margin-top:3px">🌧 '+s.weather_note+'</div>':'';
      return '<div class="sc-card '+cls+'"><div class="sc-title" style="color:'+s.color+'">'+s.title+'<span class="sc-tag">['+s.id+']</span></div><div class="sc-desc">'+(s.desc||s.description||'')+'</div><div class="sc-trigger"><strong>Trigger:</strong> '+s.trigger+'</div>'+weatherLine+
        (s.players||[]).map(function(p,i){
          var pCls=p.cls||_roleClass(p.role);
          var pName=p.name||p.player_name||'';
          var pNote=p.note||p.instruction||'';
          return '<div class="sc-player"><span class="sc-pnum">'+(i+1)+'</span><div><span class="sc-pname">'+pName+'</span><span class="sc-prole '+pCls+'">'+p.role+'</span><div class="sc-pnote">'+pNote+'</div></div></div>';
        }).join('')+
        '</div>';
    }).join('')+'</div>'+
    '<div style="font-family:Orbitron;font-size:9px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#FFD700;margin:16px 0 12px;display:flex;align-items:center;gap:6px"><span style="width:5px;height:5px;border-radius:50%;background:#FFD700"></span>Key Matchup Notes</div>'+
    '<div class="matchup-bar">'+matchupNotes.map(function(n){
      var txt=n.text||n.note||'';
      var conf=n.conf||'med';
      var bwr=n.bowler?'<span style="color:rgba(139,92,246,0.7)">'+n.bowler+'</span> vs ':'';
      var bat=n.batter?'<span style="color:rgba(255,215,0,0.7)">'+n.batter+'</span>':'';
      var matchup=(bwr||bat)?'<div style="font-family:Rajdhani;font-size:7px;color:rgba(255,255,255,0.3);margin-bottom:2px">'+bwr+bat+'</div>':'';
      return '<div class="mn '+conf+'"><div style="display:flex;align-items:center;gap:6px;margin-bottom:4px"><span class="mn-level '+conf+'">'+(conf==='high'?'High':'Medium')+'</span><span class="mn-balls">'+(n.balls||0)+' balls</span></div>'+matchup+'<div class="mn-text">'+txt+'</div></div>';
    }).join('')+'</div>';
}

// 120-ball boundary probability ring (canvas overlay)
function drawBallRing(){
  var canvas=document.getElementById('ball-ring-canvas');
  if(!canvas)return;
  var W=canvas.offsetWidth||canvas.parentElement.clientWidth;
  var H=canvas.offsetHeight||canvas.parentElement.clientHeight;
  canvas.width=W;canvas.height=H;
  var ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,W,H);

  var cx=W/2,cy=H*0.52;
  var ringR=Math.min(W,H)*0.30;
  var barMaxH=Math.min(W,H)*0.055;
  var barW=Math.max(2,(2*Math.PI*ringR/120)*0.72);

  ctx.beginPath();ctx.arc(cx,cy,ringR,0,Math.PI*2);
  ctx.strokeStyle='rgba(255,255,255,0.04)';ctx.lineWidth=1;ctx.stroke();

  var phases=[
    {start:0,end:36,color:'rgba(0,229,255,0.04)'},
    {start:36,end:90,color:'rgba(255,215,0,0.03)'},
    {start:90,end:120,color:'rgba(244,67,54,0.05)'}
  ];
  phases.forEach(function(ph){
    var a0=(ph.start/120)*Math.PI*2-Math.PI/2;
    var a1=(ph.end/120)*Math.PI*2-Math.PI/2;
    ctx.beginPath();ctx.arc(cx,cy,ringR+barMaxH+4,a0,a1);
    ctx.arc(cx,cy,ringR-4,a1,a0,true);
    ctx.closePath();ctx.fillStyle=ph.color;ctx.fill();
  });

  [0,36,90].forEach(function(b){
    var a=(b/120)*Math.PI*2-Math.PI/2;
    ctx.beginPath();
    ctx.moveTo(cx+Math.cos(a)*(ringR-8),cy+Math.sin(a)*(ringR-8));
    ctx.lineTo(cx+Math.cos(a)*(ringR+barMaxH+10),cy+Math.sin(a)*(ringR+barMaxH+10));
    ctx.strokeStyle='rgba(255,255,255,0.08)';ctx.lineWidth=0.8;ctx.stroke();
  });

  var phaseLabels=[
    {ball:18,label:'POWERPLAY',color:'rgba(0,229,255,0.6)'},
    {ball:63,label:'MIDDLE',color:'rgba(255,215,0,0.5)'},
    {ball:105,label:'DEATH',color:'rgba(244,67,54,0.65)'}
  ];
  phaseLabels.forEach(function(pl){
    var a=(pl.ball/120)*Math.PI*2-Math.PI/2;
    var lr=ringR+barMaxH+18;
    ctx.save();ctx.translate(cx+Math.cos(a)*lr,cy+Math.sin(a)*lr);ctx.rotate(a+Math.PI/2);
    ctx.font='500 8px Rajdhani, sans-serif';
    ctx.fillStyle=pl.color;ctx.textAlign='center';ctx.fillText(pl.label,0,0);ctx.restore();
  });

  for(var ov=0;ov<20;ov++){
    var a=(ov*6/120)*Math.PI*2-Math.PI/2;
    ctx.beginPath();
    ctx.moveTo(cx+Math.cos(a)*(ringR-6),cy+Math.sin(a)*(ringR-6));
    ctx.lineTo(cx+Math.cos(a)*(ringR+2),cy+Math.sin(a)*(ringR+2));
    ctx.strokeStyle='rgba(255,255,255,0.12)';ctx.lineWidth=ov%5===0?1.2:0.6;ctx.stroke();
    if(ov%5===0&&ov>0){
      var nr=ringR-14;
      ctx.font='bold 7px Orbitron, sans-serif';
      ctx.fillStyle='rgba(255,255,255,0.18)';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(ov,cx+Math.cos(a)*nr,cy+Math.sin(a)*nr);
    }
  }

  ballProbs.forEach(function(p,b){
    var a=(b/120)*Math.PI*2-Math.PI/2;
    var h=p*barMaxH;
    var r0=ringR,r1=ringR+h;
    var hw=barW/2;
    var pa=a+Math.PI/2;
    var x0=cx+Math.cos(a)*r0,y0=cy+Math.sin(a)*r0;
    var x1=cx+Math.cos(a)*r1,y1=cy+Math.sin(a)*r1;
    var col;
    if(p<0.25) col='rgba(76,175,80,0.85)';
    else if(p<0.50) col='rgba(255,215,0,0.88)';
    else if(p<0.75) col='rgba(255,140,0,0.88)';
    else col='rgba(244,67,54,0.90)';
    ctx.beginPath();
    ctx.moveTo(x0+Math.cos(pa)*hw,y0+Math.sin(pa)*hw);
    ctx.lineTo(x0-Math.cos(pa)*hw,y0-Math.sin(pa)*hw);
    ctx.lineTo(x1-Math.cos(pa)*hw,y1-Math.sin(pa)*hw);
    ctx.lineTo(x1+Math.cos(pa)*hw,y1+Math.sin(pa)*hw);
    ctx.closePath();ctx.fillStyle=col;ctx.fill();
    if(p>0.65){
      ctx.beginPath();ctx.arc(x1,y1,barW*0.7,0,Math.PI*2);
      ctx.fillStyle=p>0.80?'rgba(244,67,54,0.5)':'rgba(255,140,0,0.35)';ctx.fill();
    }
  });

  ctx.font='bold 9px Orbitron, sans-serif';
  ctx.fillStyle='rgba(255,255,255,0.12)';ctx.textAlign='center';ctx.textBaseline='middle';
  ctx.fillText('120 BALLS',cx,cy-6);
  ctx.font='500 7px Rajdhani, sans-serif';
  ctx.fillStyle='rgba(255,255,255,0.08)';
  ctx.fillText('BOUNDARY RISK',cx,cy+6);
}

// Redraw ring on resize
window.addEventListener('resize',function(){
  if(curTab===1) setTimeout(drawBallRing,50);
});
