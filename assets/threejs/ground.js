// ground.js — Three.js 3D scene, camera, animation, and scene update functions
// Requires: Three.js (CDN), data.js (ballProbs, playerPos, oppBat, xiOpts, curTab, curXI, curOver)

var container=document.getElementById('cc');
var W=container.clientWidth||window.innerWidth-480;
var H=container.clientHeight||window.innerHeight-48;
if(W<100)W=window.innerWidth-480;
if(H<100)H=window.innerHeight-48;

var scene=new THREE.Scene();
scene.fog=new THREE.FogExp2(0x080114,0.012);

var camera=new THREE.PerspectiveCamera(42,W/H,0.1,200);
camera.position.set(0,42,52);

var renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
renderer.setSize(W,H);
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.domElement.style.position='absolute';
renderer.domElement.style.top='0';
renderer.domElement.style.left='0';
container.appendChild(renderer.domElement);

// Resize handler
window.addEventListener('resize',function(){
  var w=container.clientWidth||window.innerWidth-480;
  var h=container.clientHeight||window.innerHeight-48;
  camera.aspect=w/h;
  camera.updateProjectionMatrix();
  renderer.setSize(w,h);
});

// Lighting
scene.add(new THREE.AmbientLight(0xD4E8C2,0.42));
var dl=new THREE.DirectionalLight(0xFFFDE7,1.05);dl.position.set(15,35,15);scene.add(dl);
var dl2=new THREE.DirectionalLight(0xC8E6C9,0.28);dl2.position.set(-12,20,-12);scene.add(dl2);
var dlFill=new THREE.DirectionalLight(0xE8F5E9,0.16);dlFill.position.set(0,30,-20);scene.add(dlFill);

// Dark outer ground base
var gnd=new THREE.Mesh(new THREE.PlaneGeometry(120,120),new THREE.MeshPhongMaterial({color:0x050010}));
gnd.rotation.x=-Math.PI/2;gnd.position.y=-0.22;scene.add(gnd);

// Outfield
var outfield=new THREE.Mesh(new THREE.CylinderGeometry(20,20,0.28,128),new THREE.MeshPhongMaterial({color:0x3A7D35,emissive:0x1C4A18,emissiveIntensity:0.10,shininess:6}));
scene.add(outfield);

// Mowing stripes texture
(function(){
  var size=512;
  var cv=document.createElement('canvas');cv.width=size;cv.height=size;
  var ctx=cv.getContext('2d');
  var cx=size/2,cy=size/2,R=size/2;
  ctx.fillStyle='#3A7D35';ctx.fillRect(0,0,size,size);
  var numBands=16;
  for(var i=numBands;i>=0;i--){
    var r1=((i+1)/numBands)*R;
    var light=(i%2===0);
    ctx.beginPath();ctx.arc(cx,cy,r1,0,Math.PI*2);ctx.closePath();
    ctx.fillStyle=light?'#4A9643':'#2D6B28';ctx.fill();
  }
  var stripeW=size/20;
  for(var s=0;s<20;s++){
    if(s%2===0){
      ctx.save();
      ctx.beginPath();ctx.arc(cx,cy,R,0,Math.PI*2);ctx.clip();
      ctx.fillStyle='rgba(80,160,70,0.13)';
      ctx.fillRect(s*stripeW,0,stripeW,size);
      ctx.restore();
    }
  }
  for(var n=0;n<1200;n++){
    var nx=Math.random()*size,ny=Math.random()*size;
    var dist=Math.hypot(nx-cx,ny-cy);
    if(dist<R){
      ctx.fillStyle='rgba('+(Math.random()>0.5?90:30)+','+(Math.random()>0.5?140:80)+','+(Math.random()>0.5?60:30)+',0.06)';
      ctx.fillRect(nx,ny,2,2);
    }
  }
  var vg=ctx.createRadialGradient(cx,cy,R*0.5,cx,cy,R);
  vg.addColorStop(0,'rgba(0,0,0,0)');vg.addColorStop(1,'rgba(0,0,0,0.35)');
  ctx.fillStyle=vg;ctx.beginPath();ctx.arc(cx,cy,R,0,Math.PI*2);ctx.fill();
  var tex=new THREE.CanvasTexture(cv);
  tex.wrapS=tex.wrapT=THREE.ClampToEdgeWrapping;
  outfield.material=new THREE.MeshPhongMaterial({map:tex,shininess:5,specular:new THREE.Color(0x1A3A18)});
  outfield.material.needsUpdate=true;
})();

// Infield (30-yard circle)
var infield=new THREE.Mesh(new THREE.CylinderGeometry(11,11,0.31,96),new THREE.MeshPhongMaterial({color:0x428C3C,emissive:0x1E4F1A,emissiveIntensity:0.09,shininess:5}));
scene.add(infield);

// Boundary rope
var bR=new THREE.Mesh(new THREE.TorusGeometry(20.3,0.09,10,128),new THREE.MeshBasicMaterial({color:0xFFFFFF}));
bR.rotation.x=Math.PI/2;bR.position.y=0.2;scene.add(bR);

// 30yd circle line
var iR=new THREE.Mesh(new THREE.TorusGeometry(11,0.05,8,96),new THREE.MeshBasicMaterial({color:0xFFFFFF,transparent:true,opacity:0.25}));
iR.rotation.x=Math.PI/2;iR.position.y=0.22;scene.add(iR);

// Subtle mid-radius cut ring markers
for(var _rm=0;_rm<32;_rm++){
  if(_rm%2===0){
    var a0=(_rm/32)*Math.PI*2,a1=((_rm+0.7)/32)*Math.PI*2;
    var arcGeo=new THREE.TorusGeometry(15,0.04,4,32,a1-a0);
    var arcM=new THREE.Mesh(arcGeo,new THREE.MeshBasicMaterial({color:0x5AA850,transparent:true,opacity:0.12}));
    arcM.rotation.x=Math.PI/2;arcM.rotation.z=a0;arcM.position.y=0.21;scene.add(arcM);
  }
}

// Pitch
var pitchM=new THREE.Mesh(new THREE.BoxGeometry(2,0.05,6.5),new THREE.MeshPhongMaterial({color:0xC5A87D,emissive:0x8B7355,emissiveIntensity:0.2}));
pitchM.position.y=0.2;scene.add(pitchM);
var cM=new THREE.MeshBasicMaterial({color:0xFFFFFF,transparent:true,opacity:0.5});
[-2.4,2.4].forEach(function(z){
  var c=new THREE.Mesh(new THREE.BoxGeometry(1.5,0.02,0.06),cM);
  c.position.set(0,0.25,z);scene.add(c);
});

// 16 stadium stands
for(var _st=0;_st<16;_st++){
  var _a=(_st/16)*Math.PI*2,_d=24;
  var _s=new THREE.Mesh(new THREE.BoxGeometry(8,4,3),new THREE.MeshPhongMaterial({color:0x140536,emissive:0x0E0226,emissiveIntensity:0.3,transparent:true,opacity:0.7}));
  _s.position.set(Math.cos(_a)*_d,2,Math.sin(_a)*_d);_s.lookAt(0,2,0);scene.add(_s);
  var _e=new THREE.Mesh(new THREE.BoxGeometry(8.2,0.1,0.1),new THREE.MeshBasicMaterial({color:0x8B5CF6,transparent:true,opacity:0.35}));
  _e.position.set(Math.cos(_a)*_d,4.1,Math.sin(_a)*_d);_e.lookAt(0,4.1,0);scene.add(_e);
}

// 6 floodlight towers
var floodlightPositions=[];
for(var _fl=0;_fl<6;_fl++){
  var _fa=(_fl/6)*Math.PI*2;
  floodlightPositions.push([Math.cos(_fa)*30,Math.sin(_fa)*30]);
}
floodlightPositions.forEach(function(fp){
  var px=fp[0],pz=fp[1];
  var towerGroup=new THREE.Group();
  var pole=new THREE.Mesh(new THREE.CylinderGeometry(0.08,0.22,22,10),new THREE.MeshPhongMaterial({color:0x8899AA,specular:0x444444,shininess:40}));
  pole.position.set(0,11,0);towerGroup.add(pole);
  for(var lvl=0;lvl<3;lvl++){
    var y=5+lvl*5;
    var brace=new THREE.Mesh(new THREE.CylinderGeometry(0.03,0.03,2.2,4),new THREE.MeshPhongMaterial({color:0x778899}));
    brace.position.set(0,y,0);brace.rotation.z=Math.PI/4;towerGroup.add(brace);
    var brace2=brace.clone();brace2.rotation.z=-Math.PI/4;towerGroup.add(brace2);
  }
  var armAngle=Math.atan2(-pz,-px);
  var arm=new THREE.Mesh(new THREE.CylinderGeometry(0.05,0.07,5,6),new THREE.MeshPhongMaterial({color:0x778899}));
  arm.rotation.z=Math.PI/2;arm.position.set(0,22,0);
  var armPivot=new THREE.Group();armPivot.add(arm);armPivot.rotation.y=armAngle;
  arm.position.set(2.5,0,0);towerGroup.add(armPivot);
  var housingBar=new THREE.Mesh(new THREE.BoxGeometry(2.8,0.18,0.18),new THREE.MeshPhongMaterial({color:0x334455}));
  var hbGroup=new THREE.Group();hbGroup.rotation.y=armAngle;hbGroup.add(housingBar);
  housingBar.position.set(5,0,0);housingBar.position.y=22-0.5;towerGroup.add(hbGroup);
  for(var f=0;f<8;f++){
    var frac=(f/7)-0.5;
    var back=new THREE.Mesh(new THREE.BoxGeometry(0.28,0.2,0.12),new THREE.MeshPhongMaterial({color:0x223344}));
    var lens=new THREE.Mesh(new THREE.BoxGeometry(0.22,0.14,0.06),new THREE.MeshBasicMaterial({color:0xFFF8E1,transparent:true,opacity:0.95}));
    lens.position.z=0.08;
    var fixture=new THREE.Group();fixture.add(back);fixture.add(lens);
    var dx=Math.cos(armAngle)*frac*2.4+Math.cos(armAngle)*5;
    var dz=Math.sin(armAngle)*frac*2.4+Math.sin(armAngle)*5;
    fixture.position.set(dx,21.6,dz);fixture.rotation.x=Math.PI/6;fixture.rotation.y=armAngle;
    towerGroup.add(fixture);
  }
  var pl1=new THREE.PointLight(0xFFFDE0,0.55,70);pl1.position.set(Math.cos(armAngle)*4,21,Math.sin(armAngle)*4);towerGroup.add(pl1);
  var pl2=new THREE.PointLight(0xFFF8D0,0.3,55);pl2.position.set(Math.cos(armAngle)*2,19,Math.sin(armAngle)*2);towerGroup.add(pl2);
  var glowSphere=new THREE.Mesh(new THREE.SphereGeometry(0.35,8,8),new THREE.MeshBasicMaterial({color:0xFFFBCC,transparent:true,opacity:0.55}));
  glowSphere.position.set(Math.cos(armAngle)*5,21.8,Math.sin(armAngle)*5);towerGroup.add(glowSphere);
  towerGroup.position.set(px,0,pz);scene.add(towerGroup);
});

// Glow ring
var gR=new THREE.Mesh(new THREE.TorusGeometry(20.5,0.4,8,64),new THREE.MeshBasicMaterial({color:0x4CAF50,transparent:true,opacity:0.08}));
gR.rotation.x=Math.PI/2;gR.position.y=0.15;scene.add(gR);

// Wind system
var windG=new THREE.Group();scene.add(windG);
var windDots=[];
var windDir=new THREE.Vector3(1,0,0.3).normalize();
var windStreams=[];
for(var _ws=0;_ws<12;_ws++){
  var sx=-16+Math.random()*32,sz=-14+Math.random()*28,wy=1.5+Math.random()*7;
  var pts=[];
  for(var j=0;j<8;j++) pts.push(new THREE.Vector3(sx+j*2.8*windDir.x,wy+Math.sin(j*0.7)*0.5,sz+j*2.8*windDir.z));
  var geo=new THREE.BufferGeometry().setFromPoints(pts);
  var line=new THREE.Line(geo,new THREE.LineBasicMaterial({color:0xAADDFF,transparent:true,opacity:0.18+Math.random()*0.14}));
  windG.add(line);
  var cone=new THREE.Mesh(new THREE.ConeGeometry(0.22,0.7,6),new THREE.MeshBasicMaterial({color:0xCCEEFF,transparent:true,opacity:0.55}));
  cone.rotation.z=-Math.PI/2;
  var endPt=pts[pts.length-1];cone.position.copy(endPt);windG.add(cone);
  windStreams.push({line:line,cone:cone,pts:pts,baseY:wy,phase:Math.random()*Math.PI*2});
}
for(var _wd=0;_wd<500;_wd++){
  var streamIdx=Math.floor(Math.random()*12);
  var stream=windStreams[streamIdx];
  var ptIdx=Math.floor(Math.random()*stream.pts.length);
  var ref=stream.pts[ptIdx];
  var ox=(Math.random()-0.5)*6,oy=(Math.random()-0.5)*3,oz=(Math.random()-0.5)*5;
  var dot=new THREE.Mesh(new THREE.SphereGeometry(0.06+Math.random()*0.1,5,5),new THREE.MeshBasicMaterial({color:0xDDEEFF,transparent:true,opacity:0.18+Math.random()*0.28}));
  var bx=ref.x+ox,by=ref.y+oy,bz=ref.z+oz;
  dot.position.set(bx,by,bz);windG.add(dot);
  windDots.push({mesh:dot,baseX:bx,baseY:by,baseZ:bz,speed:0.4+Math.random()*1.8,drift:3+Math.random()*5,phase:Math.random()*Math.PI*2});
}

// 120-ball bars around field
var barG=new THREE.Group();barG.visible=false;scene.add(barG);
var bars=[];
var BAR_RADIUS=21;
var BAR_MAX_H=7;
for(var _b=0;_b<120;_b++){
  var _ba=(_b/120)*Math.PI*2;
  var _prob=ballProbs[_b];
  var _h=0.3+_prob*BAR_MAX_H;
  var _col;
  if(_prob<0.25) _col=0x4CAF50;
  else if(_prob<0.50) _col=0xFFD700;
  else if(_prob<0.75) _col=0xFF8C00;
  else _col=0xF44336;
  var _bx=Math.cos(_ba)*BAR_RADIUS,_bz=Math.sin(_ba)*BAR_RADIUS;
  var _mesh=new THREE.Mesh(new THREE.BoxGeometry(0.28,_h,0.28),new THREE.MeshPhongMaterial({color:_col,emissive:_col,emissiveIntensity:0.35,transparent:true,opacity:0.88}));
  _mesh.position.set(_bx,0.15+_h/2,_bz);
  barG.add(_mesh);
  bars.push({mesh:_mesh,baseH:_h,prob:_prob,angle:_ba,ball:_b});
}

// Rebuild bar geometries/colours from the current ballProbs array.
// Called after _applyBriefData() updates ballProbs with live bowler stats.
function refreshBallProbs(){
  bars.forEach(function(b){
    var prob=ballProbs[b.ball]||0.35;
    var h=0.3+prob*BAR_MAX_H;
    var col;
    if(prob<0.25)      col=0x4CAF50;
    else if(prob<0.50) col=0xFFD700;
    else if(prob<0.75) col=0xFF8C00;
    else               col=0xF44336;
    // Swap geometry (dispose old to free GPU memory)
    b.mesh.geometry.dispose();
    b.mesh.geometry=new THREE.BoxGeometry(0.28,h,0.28);
    b.mesh.material.color.setHex(col);
    b.mesh.material.emissive.setHex(col);
    b.mesh.position.y=0.15+h/2;
    b.baseH=h;
    b.prob=prob;
  });
}

// Bowling zones
var bzG=new THREE.Group();bzG.visible=false;scene.add(bzG);
[{z:-2,c:0x8B5CF6,o:.12},{z:.5,c:0xFF8C00,o:.1},{z:2.8,c:0xF44336,o:.15}].forEach(function(d){
  var p=new THREE.Mesh(new THREE.PlaneGeometry(1.8,2),new THREE.MeshBasicMaterial({color:d.c,transparent:true,opacity:d.o,side:THREE.DoubleSide}));
  p.rotation.x=-Math.PI/2;p.position.set(0,0.22,d.z);bzG.add(p);
});
[{x:-0.3,z:-1.8,c:0x8B5CF6},{x:0.2,z:-1.2,c:0x8B5CF6},{x:0.5,z:0.3,c:0xFF8C00},{x:-0.3,z:0.8,c:0xFF8C00},{x:0.1,z:2.5,c:0xF44336},{x:-0.4,z:2.8,c:0xF44336},{x:0.3,z:-0.5,c:0x8B5CF6},{x:-0.5,z:1.5,c:0xFF8C00}].forEach(function(d){
  var m=new THREE.Mesh(new THREE.SphereGeometry(0.1,8,8),new THREE.MeshBasicMaterial({color:d.c}));
  m.position.set(d.x,0.35,d.z);bzG.add(m);
});

// Camera constants and state
var DEF_Y=42, DEF_ORBIT=52;
var targetY=DEF_Y, targetZ=DEF_ORBIT, orbitRadius=DEF_ORBIT;
var baseY=DEF_Y, baseZ=DEF_ORBIT, zoomLevel=1;

function updateScene(){
  barG.visible=(curTab===1);
  bzG.visible=(curTab===1);
  var ps=(curTab===1)?2.5:1;
  pitchM.scale.set(ps,1,ps);
}

function updateBars(){
  bars.forEach(function(b){
    var ballOver=Math.floor(b.ball/6)+1;
    var boost=(ballOver===curOver)?0.4:0;
    b.mesh.scale.y=1+boost;
    b.mesh.material.emissiveIntensity=0.35+boost;
  });
}

function updatePL(){
  if(curTab>2)return;
  var rect=container.getBoundingClientRect();
  var ps=(curTab===2)?oppBat:xiOpts[curXI].players;
  ps.forEach(function(p,i){
    if(i>=playerPos.length)return;
    var pos=playerPos[i];
    var v=new THREE.Vector3(pos.x,1.5,pos.z);v.project(camera);
    var x=(v.x*.5+.5)*rect.width,y=(-v.y*.5+.5)*rect.height;
    var el=document.getElementById('pl-'+i);
    if(el){el.style.left=(x-22)+'px';el.style.top=(y-30)+'px';el.style.opacity=v.z<1?1:0;}
  });
}

function animate(){
  requestAnimationFrame(animate);
  var t=Date.now()*.001;
  camera.position.y+=(targetY-camera.position.y)*0.03;
  camera.position.z+=(targetZ-camera.position.z)*0.03;
  camera.position.x=Math.sin(t*0.06)*orbitRadius;
  camera.position.z=Math.cos(t*0.06)*orbitRadius;
  camera.lookAt(0,0,0);
  gR.material.opacity=.06+Math.sin(t)*.03;
  windDots.forEach(function(d){
    var travel=((t*d.speed+d.phase)%d.drift)/d.drift;
    d.mesh.position.x=d.baseX+travel*d.drift*windDir.x;
    d.mesh.position.z=d.baseZ+travel*d.drift*windDir.z;
    d.mesh.position.y=d.baseY+Math.sin(t*0.8+d.phase)*0.3;
    var fade=travel<0.1?travel/0.1:travel>0.85?(1-travel)/0.15:1;
    d.mesh.material.opacity=(0.06+Math.sin(t*2+d.phase)*0.04)*fade;
  });
  windStreams.forEach(function(s){
    s.line.position.y=Math.sin(t*0.4+s.phase)*0.3;
    s.cone.position.y=s.pts[s.pts.length-1].y+Math.sin(t*0.4+s.phase)*0.3;
  });
  if(curTab===1){
    bars.forEach(function(b){
      var ballOver=Math.floor(b.ball/6)+1;
      var isSelected=(ballOver===curOver);
      var breath=isSelected
        ?1+Math.sin(t*2.5+b.angle)*0.25
        :1+Math.sin(t*1.2+b.angle*2)*0.06;
      b.mesh.scale.y=breath;
    });
  }
  renderer.render(scene,camera);
  updatePL();
}
