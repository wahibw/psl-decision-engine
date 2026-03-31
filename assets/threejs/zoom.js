// zoom.js — zoom/view controls and wheel listener
// Camera vars (DEF_Y, DEF_ORBIT, targetY, targetZ, orbitRadius) are defined in ground.js

function setView(v,el){
  document.querySelectorAll('.vb').forEach(function(b){b.classList.remove('act');});
  el.classList.add('act');
  if(v==='top'){targetY=60;orbitRadius=1;}
  else{targetY=DEF_Y;orbitRadius=DEF_ORBIT;}
}

function applyZoom(){}
function zoomStep(){}
function zoomReset(){}

// Wheel scroll disabled — fixed camera angle on all tabs
container.addEventListener('wheel',function(e){e.preventDefault();},{passive:false});
