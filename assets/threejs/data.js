// data.js — all static data, constants, and lookup tables
// All top-level vars use var for global accessibility across script files

var TABS=['Final Playing XI','Bowling Attack','Opposition Batting','Batting Scenarios'];
var curTab=0, curXI=0, curOver=1;

// Full squad for sidebar
var fullSquad=['Babar Azam','Saud Shakeel','Shadab Khan','Naseem Shah','Rashid Khan','Sarfaraz Ahmed','Di Warner','Rasa Yesea','Rase Ronme','Rashs Vassa','W. Hasaranga','Karv Bhan','Mohammad Nawaz','Faheem Ashraf','Iftikhar Ahmed','Azam Khan'];
// Select all squad members by default so backend has full pool for 3 distinct XIs
var selectedSquad=new Set([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]);

var xiOpts=[
  {name:'Primary XI',color:'#4CAF50',players:[{n:'Babar Azam',p:'https://i.pravatar.cc/200?img=12'},{n:'Saud Shakeel',p:'https://i.pravatar.cc/200?img=32'},{n:'Shadab Khan',p:'https://i.pravatar.cc/200?img=14'},{n:'Naseem Shah',p:'https://i.pravatar.cc/200?img=51'},{n:'Rashid Khan',p:'https://i.pravatar.cc/200?img=52'},{n:'Sarfaraz Ahmed',p:'https://i.pravatar.cc/200?img=57'},{n:'Di Warner',p:'https://i.pravatar.cc/200?img=60'},{n:'Rasa Yesea',p:'https://i.pravatar.cc/200?img=33'},{n:'W. Hasaranga',p:'https://i.pravatar.cc/200?img=11'},{n:'Rase Ronme',p:'https://i.pravatar.cc/200?img=15'},{n:'Karv Bhan',p:'https://i.pravatar.cc/200?img=22'}]},
  {name:'Spin Heavy',color:'#FFD700',players:[{n:'Babar Azam',p:'https://i.pravatar.cc/200?img=12'},{n:'Saud Shakeel',p:'https://i.pravatar.cc/200?img=32'},{n:'Shadab Khan',p:'https://i.pravatar.cc/200?img=14'},{n:'Rashid Khan',p:'https://i.pravatar.cc/200?img=52'},{n:'Sarfaraz Ahmed',p:'https://i.pravatar.cc/200?img=57'},{n:'Di Warner',p:'https://i.pravatar.cc/200?img=60'},{n:'W. Hasaranga',p:'https://i.pravatar.cc/200?img=11'},{n:'Imad Wasim',p:'https://i.pravatar.cc/200?img=36'},{n:'Zahid Mahmood',p:'https://i.pravatar.cc/200?img=40'},{n:'Naseem Shah',p:'https://i.pravatar.cc/200?img=51'},{n:'Karv Bhan',p:'https://i.pravatar.cc/200?img=22'}]},
  {name:'Pace Heavy',color:'#F44336',players:[{n:'Babar Azam',p:'https://i.pravatar.cc/200?img=12'},{n:'Saud Shakeel',p:'https://i.pravatar.cc/200?img=32'},{n:'Naseem Shah',p:'https://i.pravatar.cc/200?img=51'},{n:'Rashid Khan',p:'https://i.pravatar.cc/200?img=52'},{n:'Sarfaraz Ahmed',p:'https://i.pravatar.cc/200?img=57'},{n:'Di Warner',p:'https://i.pravatar.cc/200?img=60'},{n:'Rasa Yesea',p:'https://i.pravatar.cc/200?img=33'},{n:'Rase Ronme',p:'https://i.pravatar.cc/200?img=15'},{n:'Rashs Vassa',p:'https://i.pravatar.cc/200?img=44'},{n:'Shaheen Afridi',p:'https://i.pravatar.cc/200?img=48'},{n:'Karv Bhan',p:'https://i.pravatar.cc/200?img=22'}]}
];

var overPlan=[];
var bwlrs=['Shaheen Afridi','Naseem Shah','Rashid Khan','Shadab Khan','W. Hasaranga','Rasa Yesea'];
var planB=[0,1,0,1,2,3,2,4,3,2,4,3,1,2,4,5,1,0,1,0];
var planP=['pp','pp','pp','pp','pp','pp','mid','mid','mid','mid','mid','mid','mid','mid','mid','mid','death','death','death','death'];
var planN=['Target off-stump','Short of good length','Yorker if set','Mix pace','Googly first up','Attack stumps','Continue spin','Flight the ball','Bowl dry','Build dots','Change pace','Target weakness','Reverse swing','Final spin','Defensive field','Medium pace','Yorkers only','Slower bouncer','Wide yorker','Full & straight'];
for(var _i=0;_i<20;_i++) overPlan.push({over:_i+1,bowler:bwlrs[planB[_i]],phase:planP[_i],note:planN[_i]});

var bowlerAlloc=[
  {name:'Shaheen Afridi',overs:[1,3,18,20],total:4},{name:'Naseem Shah',overs:[2,4,13,17,19],total:4},
  {name:'Rashid Khan',overs:[5,7,10,14],total:4},{name:'Shadab Khan',overs:[6,9,12],total:3},
  {name:'W. Hasaranga',overs:[8,11,15],total:3},{name:'Rasa Yesea',overs:[16],total:1}
];

var oppBat=[
  {name:'Usman Khan',danger:'high',form:'Good form',arrival:'opens',notes:'Elite death hitter (SR 169). 34 avg, 144 SR last 10.',last10:{runs:340,sr:144,avg:34}},
  {name:'Saim Ayub',danger:'med',form:'Good form',arrival:'opens',notes:'Left-hand bat. 17 avg, 125 SR last 10.',last10:{runs:170,sr:125,avg:17}},
  {name:'Maaz Sadaqat',danger:'high',form:'50',arrival:'1-4',notes:'Elite death hitter (SR 211). 39 avg, 166 SR.',last10:{runs:390,sr:166,avg:39}},
  {name:'Sharjeel Khan',danger:'high',form:'63',arrival:'4-8',notes:'Elite death hitter (SR 203). 18 avg, 119 SR.',last10:{runs:180,sr:119,avg:18}},
  {name:'Kusal Perera',danger:'low',form:'50',arrival:'6-10',notes:'Left-hand bat. 25 avg, 130 SR.',last10:{runs:200,sr:130,avg:25}},
  {name:'Glenn Maxwell',danger:'low',form:'50',arrival:'8-12',notes:'Career PSL SR 120.',last10:{runs:250,sr:120,avg:25}},
  {name:'Marnus Labuschagne',danger:'low',form:'50',arrival:'10-14',notes:'Career PSL SR 120.',last10:{runs:220,sr:110,avg:22}},
  {name:'Akif Javed',danger:'low',form:'Good form',arrival:'12-16',notes:'Career PSL SR 58. 7 avg, 70 SR.',last10:{runs:70,sr:70,avg:7}}
];

var matchupNotes=[
  {conf:'high',balls:35,text:'Shafique has edge vs Ahmed: SR 123 in 35 PSL balls. Protect Ahmed at non-striker end.'},
  {conf:'med',balls:10,text:'Tariq dominates Shafique: 1 wkt in 10 balls (SR 100). Bowl Tariq early — key wicket.'},
  {conf:'med',balls:19,text:'Ahmed dominates Zaman: 1 wkt in 19 balls (SR 110). Bowl Ahmed early to Zaman.'},
  {conf:'med',balls:8,text:'Zaman has edge vs Tariq: SR 162 in 8 balls. Protect Tariq — avoid this matchup.'}
];

var contingency=['If Abrar goes for 12+ in over 1, bring Usman Tariq for over 3 instead.','If Abrar concedes 14+ before over 17, switch to Usman Tariq for remaining death overs.'];

var scenarios=[
  {id:'A',title:'Ideal Start',color:'#4CAF50',cls:'a',desc:'Platform set — accelerate immediately and target 190+.',trigger:'60+ on board after 10 overs with 2 or fewer wickets',players:[{name:'Rilee Rossouw',role:'Aggressor',cls:'agg',note:'Keep going — platform set. Target 150+ SR.'},{name:'Ben McDermott',role:'Aggressor',cls:'agg',note:'Platform set. Target McDermott scoring at 150+ SR.'},{name:'Bismillah Khan',role:'Aggressor',cls:'agg',note:'Positive intent from ball 1. Accelerate immediately.'}]},
  {id:'B',title:'Tough Start (Collapse)',color:'#FFC107',cls:'b',desc:'Rebuild first, accelerate later — get to over 15 with 4 wickets in hand.',trigger:'3+ wickets lost by end of powerplay with under 35',players:[{name:'Rilee Rossouw',role:'Anchor',cls:'anc',note:'Anchor role — minimum 15 balls before taking risks.'},{name:'Saud Shakeel',role:'Anchor',cls:'anc',note:'Anchor — minimum 15 balls before risks. Get to over 10 with 4+ wickets.'}]},
  {id:'C',title:'Death Chase',color:'#F44336',cls:'c',desc:'Clear the boundary from ball 1 — all or nothing.',trigger:'Innings 2, over 15+, required rate above 12 per over',players:[{name:'Rilee Rossouw',role:'Aggressor',cls:'agg',note:'Accelerate NOW. Must clear boundary, no half-measures.'},{name:'Ben McDermott',role:'Aggressor',cls:'agg',note:'Already in — accelerate NOW.'},{name:'Usman Tariq',role:'Finisher',cls:'fin',note:'Play big shots. Tariq to clear the boundary.'}]},
  {id:'D',title:'Conservative Build',color:'#00E5FF',cls:'d',desc:'Wickets in hand are the asset — build to over 15 then release.',trigger:'Low pitch score (CRR under 7.5 at over 12)',players:[{name:'Rilee Rossouw',role:'Anchor',cls:'anc',note:'Patience. Play straight, rotate strike, no loose shots before 12.'},{name:'Saud Shakeel',role:'Anchor',cls:'anc',note:'Play straight, rotate strike.'},{name:'Tom Curran',role:'Support',cls:'sup',note:'Solid contribution. Avoid big shot until over 14.'}]}
];

// Player positions — spread across full ground (radius up to 17), no overlaps
var playerPos=[
  {x:0,   z:-15},
  {x:-10, z:-12},
  {x:10,  z:-12},
  {x:-16, z:-5},
  {x:16,  z:-5},
  {x:-17, z:4},
  {x:17,  z:4},
  {x:-10, z:12},
  {x:10,  z:12},
  {x:0,   z:16},
  {x:0,   z:0},
];

// Team & venue data
var teams={
  'Quetta Gladiators':{abbr:'QG',color:'#8B5CF6',bg:'linear-gradient(135deg,#23074F,#8B5CF6)'},
  'Lahore Qalandars':{abbr:'LQ',color:'#00C853',bg:'linear-gradient(135deg,#003300,#00C853)'},
  'Islamabad United':{abbr:'IU',color:'#F44336',bg:'linear-gradient(135deg,#4a0000,#F44336)'},
  'Karachi Kings':{abbr:'KK',color:'#00E5FF',bg:'linear-gradient(135deg,#003344,#00E5FF)'},
  'Peshawar Zalmi':{abbr:'PZ',color:'#FF8C00',bg:'linear-gradient(135deg,#3a1a00,#FF8C00)'},
  'Multan Sultans':{abbr:'MS',color:'#FFD700',bg:'linear-gradient(135deg,#2a1a00,#FFD700)'},
  'Rawalpindiz':{abbr:'RW',color:'#00BCD4',bg:'linear-gradient(135deg,#002a2f,#00BCD4)'},
  'Hyderabad Kingsmen':{abbr:'HK',color:'#FF6F00',bg:'linear-gradient(135deg,#2a1a00,#FF6F00)'}
};

// Pitch heatmap data per venue+opposition
function getPitchData(venue,opp){
  var seed=(venue+opp).split('').reduce(function(a,c){return a+c.charCodeAt(0);},0);
  var rng=function(n){return((seed*9301+49297*n)%233280)/233280;};
  return {
    fullToss: Math.round(rng(1)*8+1),
    yorker:   Math.round(rng(2)*12+3),
    goodLen:  Math.round(rng(3)*22+8),
    shortOfGood: Math.round(rng(4)*18+5),
    short:    Math.round(rng(5)*10+1),
    wideOut:  Math.round(rng(6)*14+2),
    wideIn:   Math.round(rng(7)*10+1),
  };
}

// Brief / formation state
var briefGenerated=false;
var curBriefFormation=1;
var curOppTeam='Lahore Qalandars';
var curVenue='National Stadium, Karachi';

// Enriched XI formations with roles for brief panel
var formations=[
  {idx:1,name:'Spin Heavy',color:'#8B5CF6',tag:'SPIN',
   players:[
    {n:'Babar Azam',role:'BAT'},{n:'Saud Shakeel',role:'BAT'},{n:'Sarfaraz Ahmed',role:'WK'},
    {n:'Di Warner',role:'BAT'},{n:'Shadab Khan',role:'ALL'},{n:'W. Hasaranga',role:'ALL'},
    {n:'Imad Wasim',role:'ALL'},{n:'Zahid Mahmood',role:'BOWL'},{n:'Naseem Shah',role:'BOWL'},
    {n:'Rashid Khan',role:'BOWL'},{n:'Karv Bhan',role:'BOWL'}
  ]},
  {idx:2,name:'Pace Heavy',color:'#F44336',tag:'PACE',
   players:[
    {n:'Babar Azam',role:'BAT'},{n:'Saud Shakeel',role:'BAT'},{n:'Sarfaraz Ahmed',role:'WK'},
    {n:'Di Warner',role:'BAT'},{n:'Rashid Khan',role:'ALL'},{n:'Rasa Yesea',role:'ALL'},
    {n:'Rase Ronme',role:'BOWL'},{n:'Rashs Vassa',role:'BOWL'},{n:'Naseem Shah',role:'BOWL'},
    {n:'Shaheen Afridi',role:'BOWL'},{n:'Karv Bhan',role:'BOWL'}
  ]}
];

// 120-ball boundary probability generator
function genBallProbs(){
  var probs=[];
  for(var b=0;b<120;b++){
    var ov=Math.floor(b/6);
    var base,noise;
    if(ov<6){
      if(ov===0) base=0.45+Math.sin(b*0.9)*0.12;
      else if(ov<3) base=0.35+Math.sin(b*0.7+1)*0.1;
      else base=0.28+Math.sin(b*0.6)*0.09;
      noise=0.06;
    } else if(ov<10){
      base=0.18+Math.sin(b*0.4+2)*0.07;
      noise=0.05;
    } else if(ov<14){
      var prog=(ov-10)/4;
      base=0.20+prog*0.14+Math.sin(b*0.5)*0.06;
      noise=0.06;
    } else if(ov===14||ov===15){
      base=0.38+Math.sin(b*0.8)*0.08;
      noise=0.07;
    } else {
      var prog2=(ov-15)/5;
      base=0.52+prog2*0.28+Math.sin(b*1.2)*0.1;
      if(ov>=18) base=Math.min(0.92,base+0.12);
      noise=0.08;
    }
    probs.push(Math.max(0.05,Math.min(0.97,base+(Math.random()*2-1)*noise)));
  }
  return probs;
}
var ballProbs=genBallProbs();

// ─── Team/opposition selection state ───
var selectedOurTeam='Quetta Gladiators';
var selectedOppTeam='Lahore Qalandars';

// ─── Full team roster data for dynamic XI updates ───
var TEAM_ROSTERS={
  'Quetta Gladiators':{
    abbr:'QG',color:'#8B5CF6',
    players:[
      {n:'Babar Azam',p:'https://i.pravatar.cc/200?img=12',role:'BAT'},
      {n:'Saud Shakeel',p:'https://i.pravatar.cc/200?img=32',role:'BAT'},
      {n:'Sarfaraz Ahmed',p:'https://i.pravatar.cc/200?img=57',role:'WK'},
      {n:'Di Warner',p:'https://i.pravatar.cc/200?img=60',role:'BAT'},
      {n:'Shadab Khan',p:'https://i.pravatar.cc/200?img=14',role:'ALL'},
      {n:'W. Hasaranga',p:'https://i.pravatar.cc/200?img=11',role:'ALL'},
      {n:'Mohammad Nawaz',p:'https://i.pravatar.cc/200?img=36',role:'ALL'},
      {n:'Naseem Shah',p:'https://i.pravatar.cc/200?img=51',role:'BOWL'},
      {n:'Shaheen Afridi',p:'https://i.pravatar.cc/200?img=48',role:'BOWL'},
      {n:'Rashid Khan',p:'https://i.pravatar.cc/200?img=52',role:'BOWL'},
      {n:'Karv Bhan',p:'https://i.pravatar.cc/200?img=22',role:'BOWL'},
    ],
    spinXI:[
      {n:'Babar Azam',p:'https://i.pravatar.cc/200?img=12'},{n:'Saud Shakeel',p:'https://i.pravatar.cc/200?img=32'},
      {n:'Sarfaraz Ahmed',p:'https://i.pravatar.cc/200?img=57'},{n:'Di Warner',p:'https://i.pravatar.cc/200?img=60'},
      {n:'Shadab Khan',p:'https://i.pravatar.cc/200?img=14'},{n:'W. Hasaranga',p:'https://i.pravatar.cc/200?img=11'},
      {n:'Imad Wasim',p:'https://i.pravatar.cc/200?img=36'},{n:'Zahid Mahmood',p:'https://i.pravatar.cc/200?img=40'},
      {n:'Rashid Khan',p:'https://i.pravatar.cc/200?img=52'},{n:'Naseem Shah',p:'https://i.pravatar.cc/200?img=51'},
      {n:'Karv Bhan',p:'https://i.pravatar.cc/200?img=22'},
    ],
    paceXI:[
      {n:'Babar Azam',p:'https://i.pravatar.cc/200?img=12'},{n:'Saud Shakeel',p:'https://i.pravatar.cc/200?img=32'},
      {n:'Sarfaraz Ahmed',p:'https://i.pravatar.cc/200?img=57'},{n:'Di Warner',p:'https://i.pravatar.cc/200?img=60'},
      {n:'Rashid Khan',p:'https://i.pravatar.cc/200?img=52'},{n:'Rasa Yesea',p:'https://i.pravatar.cc/200?img=33'},
      {n:'Rase Ronme',p:'https://i.pravatar.cc/200?img=15'},{n:'Rashs Vassa',p:'https://i.pravatar.cc/200?img=44'},
      {n:'Naseem Shah',p:'https://i.pravatar.cc/200?img=51'},{n:'Shaheen Afridi',p:'https://i.pravatar.cc/200?img=48'},
      {n:'Karv Bhan',p:'https://i.pravatar.cc/200?img=22'},
    ],
    bowlers:['Shaheen Afridi','Naseem Shah','Rashid Khan','Shadab Khan','W. Hasaranga','Rasa Yesea'],
  },
  'Lahore Qalandars':{
    abbr:'LQ',color:'#00C853',
    players:[
      {n:'Fakhar Zaman',p:'https://i.pravatar.cc/200?img=13',role:'BAT'},
      {n:'Abdullah Shafique',p:'https://i.pravatar.cc/200?img=23',role:'BAT'},
      {n:'Sikandar Raza',p:'https://i.pravatar.cc/200?img=34',role:'ALL'},
      {n:'David Wiese',p:'https://i.pravatar.cc/200?img=45',role:'ALL'},
      {n:'Haris Rauf',p:'https://i.pravatar.cc/200?img=53',role:'BOWL'},
      {n:'Zaman Khan',p:'https://i.pravatar.cc/200?img=56',role:'BOWL'},
      {n:'Liam Dawson',p:'https://i.pravatar.cc/200?img=61',role:'ALL'},
      {n:'Mohammad Hafeez',p:'https://i.pravatar.cc/200?img=65',role:'ALL'},
      {n:'Shaheen Afridi',p:'https://i.pravatar.cc/200?img=48',role:'BOWL'},
      {n:'Rashid Khan',p:'https://i.pravatar.cc/200?img=52',role:'BOWL'},
      {n:'Dilbar Hussain',p:'https://i.pravatar.cc/200?img=71',role:'BOWL'},
    ],
    spinXI:[
      {n:'Fakhar Zaman',p:'https://i.pravatar.cc/200?img=13'},{n:'Abdullah Shafique',p:'https://i.pravatar.cc/200?img=23'},
      {n:'Sikandar Raza',p:'https://i.pravatar.cc/200?img=34'},{n:'Liam Dawson',p:'https://i.pravatar.cc/200?img=61'},
      {n:'Mohammad Hafeez',p:'https://i.pravatar.cc/200?img=65'},{n:'David Wiese',p:'https://i.pravatar.cc/200?img=45'},
      {n:'Haris Rauf',p:'https://i.pravatar.cc/200?img=53'},{n:'Shaheen Afridi',p:'https://i.pravatar.cc/200?img=48'},
      {n:'Rashid Khan',p:'https://i.pravatar.cc/200?img=52'},{n:'Zaman Khan',p:'https://i.pravatar.cc/200?img=56'},
      {n:'Dilbar Hussain',p:'https://i.pravatar.cc/200?img=71'},
    ],
    paceXI:[
      {n:'Fakhar Zaman',p:'https://i.pravatar.cc/200?img=13'},{n:'Abdullah Shafique',p:'https://i.pravatar.cc/200?img=23'},
      {n:'Sikandar Raza',p:'https://i.pravatar.cc/200?img=34'},{n:'David Wiese',p:'https://i.pravatar.cc/200?img=45'},
      {n:'Haris Rauf',p:'https://i.pravatar.cc/200?img=53'},{n:'Zaman Khan',p:'https://i.pravatar.cc/200?img=56'},
      {n:'Shaheen Afridi',p:'https://i.pravatar.cc/200?img=48'},{n:'Rashid Khan',p:'https://i.pravatar.cc/200?img=52'},
      {n:'Dilbar Hussain',p:'https://i.pravatar.cc/200?img=71'},{n:'Mohammad Hafeez',p:'https://i.pravatar.cc/200?img=65'},
      {n:'Liam Dawson',p:'https://i.pravatar.cc/200?img=61'},
    ],
    bowlers:['Shaheen Afridi','Haris Rauf','Zaman Khan','Sikandar Raza','Liam Dawson','Dilbar Hussain'],
  },
  'Islamabad United':{
    abbr:'IU',color:'#F44336',
    players:[
      {n:'Paul Stirling',p:'https://i.pravatar.cc/200?img=17',role:'BAT'},
      {n:'Alex Hales',p:'https://i.pravatar.cc/200?img=27',role:'BAT'},
      {n:'Asif Ali',p:'https://i.pravatar.cc/200?img=37',role:'BAT'},
      {n:'Shadab Khan',p:'https://i.pravatar.cc/200?img=14',role:'ALL'},
      {n:'Faheem Ashraf',p:'https://i.pravatar.cc/200?img=47',role:'ALL'},
      {n:'Azam Khan',p:'https://i.pravatar.cc/200?img=55',role:'WK'},
      {n:'Rumman Raees',p:'https://i.pravatar.cc/200?img=63',role:'BOWL'},
      {n:'Akif Javed',p:'https://i.pravatar.cc/200?img=67',role:'BOWL'},
      {n:'Mohammad Wasim',p:'https://i.pravatar.cc/200?img=72',role:'BOWL'},
      {n:'Waqas Maqsood',p:'https://i.pravatar.cc/200?img=75',role:'BOWL'},
      {n:'Usman Qadir',p:'https://i.pravatar.cc/200?img=78',role:'BOWL'},
    ],
    spinXI:[
      {n:'Paul Stirling',p:'https://i.pravatar.cc/200?img=17'},{n:'Alex Hales',p:'https://i.pravatar.cc/200?img=27'},
      {n:'Asif Ali',p:'https://i.pravatar.cc/200?img=37'},{n:'Shadab Khan',p:'https://i.pravatar.cc/200?img=14'},
      {n:'Usman Qadir',p:'https://i.pravatar.cc/200?img=78'},{n:'Faheem Ashraf',p:'https://i.pravatar.cc/200?img=47'},
      {n:'Azam Khan',p:'https://i.pravatar.cc/200?img=55'},{n:'Rumman Raees',p:'https://i.pravatar.cc/200?img=63'},
      {n:'Akif Javed',p:'https://i.pravatar.cc/200?img=67'},{n:'Mohammad Wasim',p:'https://i.pravatar.cc/200?img=72'},
      {n:'Waqas Maqsood',p:'https://i.pravatar.cc/200?img=75'},
    ],
    paceXI:[
      {n:'Paul Stirling',p:'https://i.pravatar.cc/200?img=17'},{n:'Alex Hales',p:'https://i.pravatar.cc/200?img=27'},
      {n:'Asif Ali',p:'https://i.pravatar.cc/200?img=37'},{n:'Shadab Khan',p:'https://i.pravatar.cc/200?img=14'},
      {n:'Faheem Ashraf',p:'https://i.pravatar.cc/200?img=47'},{n:'Azam Khan',p:'https://i.pravatar.cc/200?img=55'},
      {n:'Rumman Raees',p:'https://i.pravatar.cc/200?img=63'},{n:'Akif Javed',p:'https://i.pravatar.cc/200?img=67'},
      {n:'Mohammad Wasim',p:'https://i.pravatar.cc/200?img=72'},{n:'Waqas Maqsood',p:'https://i.pravatar.cc/200?img=75'},
      {n:'Usman Qadir',p:'https://i.pravatar.cc/200?img=78'},
    ],
    bowlers:['Mohammad Wasim','Akif Javed','Rumman Raees','Shadab Khan','Usman Qadir','Waqas Maqsood'],
  },
  'Karachi Kings':{
    abbr:'KK',color:'#00E5FF',
    players:[
      {n:'Sharjeel Khan',p:'https://i.pravatar.cc/200?img=18',role:'BAT'},
      {n:'Babar Azam',p:'https://i.pravatar.cc/200?img=12',role:'BAT'},
      {n:'Joe Clarke',p:'https://i.pravatar.cc/200?img=28',role:'WK'},
      {n:'Mohammad Nabi',p:'https://i.pravatar.cc/200?img=38',role:'ALL'},
      {n:'Imad Wasim',p:'https://i.pravatar.cc/200?img=36',role:'ALL'},
      {n:'Arshad Iqbal',p:'https://i.pravatar.cc/200?img=46',role:'BOWL'},
      {n:'Mohammad Amir',p:'https://i.pravatar.cc/200?img=54',role:'BOWL'},
      {n:'Umaid Asif',p:'https://i.pravatar.cc/200?img=62',role:'BOWL'},
      {n:'Qasim Akram',p:'https://i.pravatar.cc/200?img=68',role:'ALL'},
      {n:'Chris Jordan',p:'https://i.pravatar.cc/200?img=73',role:'BOWL'},
      {n:'Zahid Mahmood',p:'https://i.pravatar.cc/200?img=40',role:'BOWL'},
    ],
    spinXI:[
      {n:'Sharjeel Khan',p:'https://i.pravatar.cc/200?img=18'},{n:'Babar Azam',p:'https://i.pravatar.cc/200?img=12'},
      {n:'Joe Clarke',p:'https://i.pravatar.cc/200?img=28'},{n:'Mohammad Nabi',p:'https://i.pravatar.cc/200?img=38'},
      {n:'Imad Wasim',p:'https://i.pravatar.cc/200?img=36'},{n:'Zahid Mahmood',p:'https://i.pravatar.cc/200?img=40'},
      {n:'Qasim Akram',p:'https://i.pravatar.cc/200?img=68'},{n:'Arshad Iqbal',p:'https://i.pravatar.cc/200?img=46'},
      {n:'Mohammad Amir',p:'https://i.pravatar.cc/200?img=54'},{n:'Umaid Asif',p:'https://i.pravatar.cc/200?img=62'},
      {n:'Chris Jordan',p:'https://i.pravatar.cc/200?img=73'},
    ],
    paceXI:[
      {n:'Sharjeel Khan',p:'https://i.pravatar.cc/200?img=18'},{n:'Babar Azam',p:'https://i.pravatar.cc/200?img=12'},
      {n:'Joe Clarke',p:'https://i.pravatar.cc/200?img=28'},{n:'Mohammad Nabi',p:'https://i.pravatar.cc/200?img=38'},
      {n:'Imad Wasim',p:'https://i.pravatar.cc/200?img=36'},{n:'Arshad Iqbal',p:'https://i.pravatar.cc/200?img=46'},
      {n:'Mohammad Amir',p:'https://i.pravatar.cc/200?img=54'},{n:'Umaid Asif',p:'https://i.pravatar.cc/200?img=62'},
      {n:'Chris Jordan',p:'https://i.pravatar.cc/200?img=73'},{n:'Qasim Akram',p:'https://i.pravatar.cc/200?img=68'},
      {n:'Zahid Mahmood',p:'https://i.pravatar.cc/200?img=40'},
    ],
    bowlers:['Mohammad Amir','Arshad Iqbal','Chris Jordan','Imad Wasim','Zahid Mahmood','Mohammad Nabi'],
  },
  'Peshawar Zalmi':{
    abbr:'PZ',color:'#FF8C00',
    players:[
      {n:'Kamran Akmal',p:'https://i.pravatar.cc/200?img=19',role:'WK'},
      {n:'Tom Kohler-Cadmore',p:'https://i.pravatar.cc/200?img=29',role:'BAT'},
      {n:'Shoaib Malik',p:'https://i.pravatar.cc/200?img=39',role:'ALL'},
      {n:'Liam Livingstone',p:'https://i.pravatar.cc/200?img=49',role:'ALL'},
      {n:'Wahab Riaz',p:'https://i.pravatar.cc/200?img=58',role:'BOWL'},
      {n:'Salman Irshad',p:'https://i.pravatar.cc/200?img=64',role:'BOWL'},
      {n:'Amad Butt',p:'https://i.pravatar.cc/200?img=69',role:'BOWL'},
      {n:'Usman Qadir',p:'https://i.pravatar.cc/200?img=78',role:'BOWL'},
      {n:'Saim Ayub',p:'https://i.pravatar.cc/200?img=79',role:'BAT'},
      {n:'Mohammad Imran',p:'https://i.pravatar.cc/200?img=80',role:'BOWL'},
      {n:'Rovman Powell',p:'https://i.pravatar.cc/200?img=81',role:'BAT'},
    ],
    spinXI:[
      {n:'Kamran Akmal',p:'https://i.pravatar.cc/200?img=19'},{n:'Tom Kohler-Cadmore',p:'https://i.pravatar.cc/200?img=29'},
      {n:'Shoaib Malik',p:'https://i.pravatar.cc/200?img=39'},{n:'Liam Livingstone',p:'https://i.pravatar.cc/200?img=49'},
      {n:'Saim Ayub',p:'https://i.pravatar.cc/200?img=79'},{n:'Usman Qadir',p:'https://i.pravatar.cc/200?img=78'},
      {n:'Rovman Powell',p:'https://i.pravatar.cc/200?img=81'},{n:'Wahab Riaz',p:'https://i.pravatar.cc/200?img=58'},
      {n:'Salman Irshad',p:'https://i.pravatar.cc/200?img=64'},{n:'Amad Butt',p:'https://i.pravatar.cc/200?img=69'},
      {n:'Mohammad Imran',p:'https://i.pravatar.cc/200?img=80'},
    ],
    paceXI:[
      {n:'Kamran Akmal',p:'https://i.pravatar.cc/200?img=19'},{n:'Tom Kohler-Cadmore',p:'https://i.pravatar.cc/200?img=29'},
      {n:'Shoaib Malik',p:'https://i.pravatar.cc/200?img=39'},{n:'Liam Livingstone',p:'https://i.pravatar.cc/200?img=49'},
      {n:'Rovman Powell',p:'https://i.pravatar.cc/200?img=81'},{n:'Wahab Riaz',p:'https://i.pravatar.cc/200?img=58'},
      {n:'Salman Irshad',p:'https://i.pravatar.cc/200?img=64'},{n:'Amad Butt',p:'https://i.pravatar.cc/200?img=69'},
      {n:'Mohammad Imran',p:'https://i.pravatar.cc/200?img=80'},{n:'Saim Ayub',p:'https://i.pravatar.cc/200?img=79'},
      {n:'Usman Qadir',p:'https://i.pravatar.cc/200?img=78'},
    ],
    bowlers:['Wahab Riaz','Salman Irshad','Amad Butt','Usman Qadir','Mohammad Imran','Liam Livingstone'],
  },
  'Multan Sultans':{
    abbr:'MS',color:'#FFD700',
    players:[
      {n:'Mohammad Rizwan',p:'https://i.pravatar.cc/200?img=20',role:'WK'},
      {n:'Shan Masood',p:'https://i.pravatar.cc/200?img=30',role:'BAT'},
      {n:'Rilee Rossouw',p:'https://i.pravatar.cc/200?img=41',role:'BAT'},
      {n:'Tim David',p:'https://i.pravatar.cc/200?img=50',role:'BAT'},
      {n:'Khushdil Shah',p:'https://i.pravatar.cc/200?img=59',role:'ALL'},
      {n:'Imran Tahir',p:'https://i.pravatar.cc/200?img=66',role:'BOWL'},
      {n:'Shahnawaz Dahani',p:'https://i.pravatar.cc/200?img=70',role:'BOWL'},
      {n:'Ihsanullah',p:'https://i.pravatar.cc/200?img=74',role:'BOWL'},
      {n:'Abbas Afridi',p:'https://i.pravatar.cc/200?img=76',role:'BOWL'},
      {n:'Usama Mir',p:'https://i.pravatar.cc/200?img=77',role:'BOWL'},
      {n:'David Willey',p:'https://i.pravatar.cc/200?img=82',role:'ALL'},
    ],
    spinXI:[
      {n:'Mohammad Rizwan',p:'https://i.pravatar.cc/200?img=20'},{n:'Shan Masood',p:'https://i.pravatar.cc/200?img=30'},
      {n:'Rilee Rossouw',p:'https://i.pravatar.cc/200?img=41'},{n:'Tim David',p:'https://i.pravatar.cc/200?img=50'},
      {n:'Khushdil Shah',p:'https://i.pravatar.cc/200?img=59'},{n:'Imran Tahir',p:'https://i.pravatar.cc/200?img=66'},
      {n:'Usama Mir',p:'https://i.pravatar.cc/200?img=77'},{n:'Shahnawaz Dahani',p:'https://i.pravatar.cc/200?img=70'},
      {n:'Ihsanullah',p:'https://i.pravatar.cc/200?img=74'},{n:'Abbas Afridi',p:'https://i.pravatar.cc/200?img=76'},
      {n:'David Willey',p:'https://i.pravatar.cc/200?img=82'},
    ],
    paceXI:[
      {n:'Mohammad Rizwan',p:'https://i.pravatar.cc/200?img=20'},{n:'Shan Masood',p:'https://i.pravatar.cc/200?img=30'},
      {n:'Rilee Rossouw',p:'https://i.pravatar.cc/200?img=41'},{n:'Tim David',p:'https://i.pravatar.cc/200?img=50'},
      {n:'Khushdil Shah',p:'https://i.pravatar.cc/200?img=59'},{n:'Shahnawaz Dahani',p:'https://i.pravatar.cc/200?img=70'},
      {n:'Ihsanullah',p:'https://i.pravatar.cc/200?img=74'},{n:'Abbas Afridi',p:'https://i.pravatar.cc/200?img=76'},
      {n:'David Willey',p:'https://i.pravatar.cc/200?img=82'},{n:'Imran Tahir',p:'https://i.pravatar.cc/200?img=66'},
      {n:'Usama Mir',p:'https://i.pravatar.cc/200?img=77'},
    ],
    bowlers:['Shahnawaz Dahani','Ihsanullah','Abbas Afridi','Imran Tahir','Usama Mir','David Willey'],
  },
  'Rawalpindiz':{
    abbr:'RW',color:'#00BCD4',
    players:[
      {n:'Zain Abbas',p:'',role:'BAT'},{n:'Usman Khan',p:'',role:'BAT'},{n:'Salman Agha',p:'',role:'ALL'},
      {n:'Iftikhar Ahmed',p:'',role:'ALL'},{n:'Mohammad Nawaz',p:'',role:'ALL'},{n:'Rohail Nazir',p:'',role:'WK'},
      {n:'Hasan Ali',p:'',role:'BOWL'},{n:'Sohail Khan',p:'',role:'BOWL'},{n:'Mohammad Hasnain',p:'',role:'BOWL'},
      {n:'Abbas Afridi',p:'',role:'BOWL'},{n:'Mehran Mumtaz',p:'',role:'BOWL'},
    ],
    spinXI:[
      {n:'Zain Abbas',p:''},{n:'Usman Khan',p:''},{n:'Salman Agha',p:''},{n:'Iftikhar Ahmed',p:''},
      {n:'Mohammad Nawaz',p:''},{n:'Rohail Nazir',p:''},{n:'Mehran Mumtaz',p:''},
      {n:'Hasan Ali',p:''},{n:'Abbas Afridi',p:''},{n:'Mohammad Hasnain',p:''},{n:'Sohail Khan',p:''},
    ],
    paceXI:[
      {n:'Zain Abbas',p:''},{n:'Usman Khan',p:''},{n:'Salman Agha',p:''},{n:'Iftikhar Ahmed',p:''},
      {n:'Rohail Nazir',p:''},{n:'Hasan Ali',p:''},{n:'Sohail Khan',p:''},
      {n:'Mohammad Hasnain',p:''},{n:'Abbas Afridi',p:''},{n:'Mohammad Nawaz',p:''},{n:'Mehran Mumtaz',p:''},
    ],
    bowlers:['Hasan Ali','Mohammad Hasnain','Abbas Afridi','Mohammad Nawaz','Sohail Khan','Mehran Mumtaz'],
  },
  'Hyderabad Kingsmen':{
    abbr:'HK',color:'#FF6F00',
    players:[
      {n:'Sahibzada Farhan',p:'',role:'BAT'},{n:'Haider Ali',p:'',role:'BAT'},{n:'Khushdil Shah',p:'',role:'ALL'},
      {n:'Saud Shakeel',p:'',role:'BAT'},{n:'Asad Shafiq',p:'',role:'BAT'},{n:'Sarfaraz Ahmed',p:'',role:'WK'},
      {n:'Noman Ali',p:'',role:'BOWL'},{n:'Abrar Ahmed',p:'',role:'BOWL'},{n:'Mohammad Ali',p:'',role:'BOWL'},
      {n:'Zaman Khan',p:'',role:'BOWL'},{n:'Naseem Shah',p:'',role:'BOWL'},
    ],
    spinXI:[
      {n:'Sahibzada Farhan',p:''},{n:'Haider Ali',p:''},{n:'Saud Shakeel',p:''},{n:'Asad Shafiq',p:''},
      {n:'Khushdil Shah',p:''},{n:'Sarfaraz Ahmed',p:''},{n:'Noman Ali',p:''},
      {n:'Abrar Ahmed',p:''},{n:'Zaman Khan',p:''},{n:'Mohammad Ali',p:''},{n:'Naseem Shah',p:''},
    ],
    paceXI:[
      {n:'Sahibzada Farhan',p:''},{n:'Haider Ali',p:''},{n:'Saud Shakeel',p:''},{n:'Khushdil Shah',p:''},
      {n:'Sarfaraz Ahmed',p:''},{n:'Naseem Shah',p:''},{n:'Zaman Khan',p:''},
      {n:'Mohammad Ali',p:''},{n:'Noman Ali',p:''},{n:'Abrar Ahmed',p:''},{n:'Asad Shafiq',p:''},
    ],
    bowlers:['Naseem Shah','Zaman Khan','Mohammad Ali','Noman Ali','Abrar Ahmed','Khushdil Shah'],
  }
};

// ─── Opposition batting rosters per opponent team ───
var OPP_ROSTERS={
  'Lahore Qalandars':[
    {name:'Fakhar Zaman',danger:'high',arrival:'opens',notes:'SR 148 last 10. Sweeps well against spin.',last10:{runs:380,sr:148,avg:38}},
    {name:'Abdullah Shafique',danger:'med',arrival:'opens',notes:'Anchor. Builds pressure slowly, 118 SR.',last10:{runs:290,sr:118,avg:29}},
    {name:'Sikandar Raza',danger:'high',arrival:'1-4',notes:'Dangerous 360 hitter. SR 162 in PSL.',last10:{runs:360,sr:162,avg:36}},
    {name:'David Wiese',danger:'med',arrival:'4-8',notes:'Finisher. SR 138 in death overs.',last10:{runs:200,sr:138,avg:25}},
    {name:'Liam Dawson',danger:'low',arrival:'6-10',notes:'Spin all-rounder. Steady scorer.',last10:{runs:150,sr:115,avg:20}},
    {name:'Haris Rauf',danger:'low',arrival:'lower',notes:'Tail. Focus on dot balls.',last10:{runs:40,sr:80,avg:8}},
    {name:'Shaheen Afridi',danger:'low',arrival:'lower',notes:'Tail. 1 boundary per 20 balls.',last10:{runs:30,sr:60,avg:5}},
    {name:'Zaman Khan',danger:'low',arrival:'lower',notes:'Tail end. Target for dot balls.',last10:{runs:20,sr:50,avg:4}},
  ],
  'Islamabad United':[
    {name:'Paul Stirling',danger:'high',arrival:'opens',notes:'Explosive opener. SR 165 powerplay.',last10:{runs:400,sr:165,avg:40}},
    {name:'Alex Hales',danger:'high',arrival:'opens',notes:'Big hitter. 24 sixes last 10 matches.',last10:{runs:420,sr:158,avg:42}},
    {name:'Asif Ali',danger:'high',arrival:'4-8',notes:'Death specialist. SR 203 overs 16-20.',last10:{runs:280,sr:203,avg:28}},
    {name:'Shadab Khan',danger:'med',arrival:'4-8',notes:'Dangerous in middle overs. SR 142.',last10:{runs:220,sr:142,avg:22}},
    {name:'Azam Khan',danger:'med',arrival:'6-10',notes:'Power hitter. Takes on short ball.',last10:{runs:250,sr:145,avg:25}},
    {name:'Faheem Ashraf',danger:'med',arrival:'8-12',notes:'Finisher. Useful lower order runs.',last10:{runs:180,sr:130,avg:20}},
    {name:'Akif Javed',danger:'low',arrival:'lower',notes:'Tail. 1 boundary per 15 balls.',last10:{runs:50,sr:75,avg:8}},
    {name:'Mohammad Wasim',danger:'low',arrival:'lower',notes:'Tail. Focus on yorkers.',last10:{runs:30,sr:60,avg:5}},
  ],
  'Karachi Kings':[
    {name:'Sharjeel Khan',danger:'high',arrival:'opens',notes:'Aggressive opener. 28 sixes last 10.',last10:{runs:390,sr:170,avg:39}},
    {name:'Babar Azam',danger:'high',arrival:'opens',notes:'World class. All-round game. SR 132.',last10:{runs:450,sr:132,avg:45}},
    {name:'Joe Clarke',danger:'med',arrival:'1-4',notes:'WK-BAT. Solid middle order.',last10:{runs:280,sr:128,avg:28}},
    {name:'Mohammad Nabi',danger:'med',arrival:'4-8',notes:'Experience. Useful in pressure overs.',last10:{runs:210,sr:120,avg:21}},
    {name:'Imad Wasim',danger:'med',arrival:'6-10',notes:'Spin all-rounder. 6th gear hitter.',last10:{runs:190,sr:135,avg:19}},
    {name:'Qasim Akram',danger:'low',arrival:'8-12',notes:'Useful cameo. Plays spin well.',last10:{runs:120,sr:110,avg:15}},
    {name:'Arshad Iqbal',danger:'low',arrival:'lower',notes:'Tail. Bowl yorkers to him.',last10:{runs:40,sr:70,avg:7}},
    {name:'Mohammad Amir',danger:'low',arrival:'lower',notes:'Tail. Experienced but weak hitter.',last10:{runs:25,sr:55,avg:4}},
  ],
  'Peshawar Zalmi':[
    {name:'Kamran Akmal',danger:'high',arrival:'opens',notes:'Aggressive WK. SR 155 powerplay.',last10:{runs:380,sr:155,avg:38}},
    {name:'Tom Kohler-Cadmore',danger:'med',arrival:'opens',notes:'Solid opener. SR 124.',last10:{runs:300,sr:124,avg:30}},
    {name:'Liam Livingstone',danger:'high',arrival:'1-4',notes:'360 hitter. SR 168 in PSL.',last10:{runs:350,sr:168,avg:35}},
    {name:'Rovman Powell',danger:'high',arrival:'4-8',notes:'Caribbean power. 22 sixes last 10.',last10:{runs:320,sr:175,avg:32}},
    {name:'Shoaib Malik',danger:'med',arrival:'4-8',notes:'Experience. Dangerous in 15+.',last10:{runs:240,sr:138,avg:24}},
    {name:'Saim Ayub',danger:'med',arrival:'6-10',notes:'Young talent. Improving rapidly.',last10:{runs:200,sr:130,avg:20}},
    {name:'Wahab Riaz',danger:'low',arrival:'lower',notes:'Tail. Bowl straight at stumps.',last10:{runs:35,sr:65,avg:6}},
    {name:'Usman Qadir',danger:'low',arrival:'lower',notes:'Spin bowler. Minimal batting threat.',last10:{runs:20,sr:50,avg:4}},
  ],
  'Multan Sultans':[
    {name:'Mohammad Rizwan',danger:'high',arrival:'opens',notes:'World-class WK. SR 138, anchor + finisher.',last10:{runs:480,sr:138,avg:48}},
    {name:'Shan Masood',danger:'med',arrival:'opens',notes:'Compact opener. Difficult to dislodge.',last10:{runs:310,sr:122,avg:31}},
    {name:'Rilee Rossouw',danger:'high',arrival:'1-4',notes:'South African power. SR 182 death overs.',last10:{runs:400,sr:165,avg:40}},
    {name:'Tim David',danger:'high',arrival:'4-8',notes:'Singapore finisher. SR 190 overs 16-20.',last10:{runs:360,sr:190,avg:36}},
    {name:'Khushdil Shah',danger:'med',arrival:'6-10',notes:'Left-hand slogger. Dangerous with room.',last10:{runs:250,sr:148,avg:25}},
    {name:'David Willey',danger:'med',arrival:'8-12',notes:'Useful lower order runs and pace.',last10:{runs:180,sr:132,avg:18}},
    {name:'Imran Tahir',danger:'low',arrival:'lower',notes:'Tail. Bowl length, no room.',last10:{runs:20,sr:45,avg:4}},
    {name:'Shahnawaz Dahani',danger:'low',arrival:'lower',notes:'Tail. Quick but poor hitter.',last10:{runs:30,sr:58,avg:5}},
  ],
  'Quetta Gladiators':[
    {name:'Babar Azam',danger:'high',arrival:'opens',notes:'World class. All-round game.',last10:{runs:450,sr:132,avg:45}},
    {name:'Saud Shakeel',danger:'med',arrival:'opens',notes:'Anchor. 17 avg, 125 SR.',last10:{runs:170,sr:125,avg:17}},
    {name:'Di Warner',danger:'high',arrival:'1-4',notes:'Aggressive. Elite death hitter.',last10:{runs:390,sr:166,avg:39}},
    {name:'Shadab Khan',danger:'med',arrival:'4-8',notes:'Dangerous middle overs. SR 142.',last10:{runs:220,sr:142,avg:22}},
    {name:'Sarfaraz Ahmed',danger:'low',arrival:'6-10',notes:'WK. Steady scorer. SR 118.',last10:{runs:200,sr:118,avg:20}},
    {name:'W. Hasaranga',danger:'med',arrival:'8-12',notes:'Spin all-rounder. Useful cameo.',last10:{runs:180,sr:125,avg:18}},
    {name:'Naseem Shah',danger:'low',arrival:'lower',notes:'Tail. Bowl yorkers.',last10:{runs:40,sr:70,avg:7}},
    {name:'Shaheen Afridi',danger:'low',arrival:'lower',notes:'Tail. Experienced but weak hitter.',last10:{runs:25,sr:55,avg:4}},
  ],
  'Rawalpindiz':[
    {name:'Usman Khan',danger:'high',arrival:'opens',notes:'SR 165 powerplay. Dangerous in first 6.',last10:{runs:390,sr:165,avg:39}},
    {name:'Zain Abbas',danger:'med',arrival:'opens',notes:'Steady opener. SR 128.',last10:{runs:280,sr:128,avg:28}},
    {name:'Salman Agha',danger:'high',arrival:'1-4',notes:'All-rounder. SR 148 in middle overs.',last10:{runs:320,sr:148,avg:32}},
    {name:'Iftikhar Ahmed',danger:'high',arrival:'4-8',notes:'Death specialist. SR 185 overs 16-20.',last10:{runs:300,sr:185,avg:30}},
    {name:'Mohammad Nawaz',danger:'med',arrival:'6-10',notes:'Left-arm spin all-rounder. Useful hitter.',last10:{runs:200,sr:132,avg:20}},
    {name:'Rohail Nazir',danger:'med',arrival:'8-12',notes:'WK. Solid lower order contributor.',last10:{runs:180,sr:120,avg:18}},
    {name:'Hasan Ali',danger:'low',arrival:'lower',notes:'Tail. Bowl yorkers to him.',last10:{runs:35,sr:65,avg:6}},
    {name:'Mohammad Hasnain',danger:'low',arrival:'lower',notes:'Tail. Quick but minimal bat.',last10:{runs:20,sr:55,avg:4}},
  ],
  'Hyderabad Kingsmen':[
    {name:'Sahibzada Farhan',danger:'high',arrival:'opens',notes:'SR 158 powerplay. Big square boundaries threat.',last10:{runs:410,sr:158,avg:41}},
    {name:'Haider Ali',danger:'high',arrival:'opens',notes:'Explosive opener. 26 sixes last 10.',last10:{runs:380,sr:162,avg:38}},
    {name:'Saud Shakeel',danger:'med',arrival:'1-4',notes:'Anchor. Builds partnerships. SR 126.',last10:{runs:290,sr:126,avg:29}},
    {name:'Khushdil Shah',danger:'high',arrival:'4-8',notes:'Left-hand slogger. SR 168 death overs.',last10:{runs:340,sr:168,avg:34}},
    {name:'Asad Shafiq',danger:'med',arrival:'6-10',notes:'Experience. Difficult to remove in 10+.',last10:{runs:230,sr:122,avg:23}},
    {name:'Sarfaraz Ahmed',danger:'med',arrival:'8-12',notes:'WK. Steady scorer and finisher.',last10:{runs:200,sr:130,avg:20}},
    {name:'Noman Ali',danger:'low',arrival:'lower',notes:'Tail. Spin bowler. Minimal batting threat.',last10:{runs:25,sr:48,avg:4}},
    {name:'Abrar Ahmed',danger:'low',arrival:'lower',notes:'Tail. Bowl yorkers and short balls.',last10:{runs:20,sr:52,avg:4}},
  ]
};
