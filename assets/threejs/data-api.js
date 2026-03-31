// data-api.js — live data API layer
// Fetches /api/prep-meta on load to populate real squads/teams/venues.
// Overrides generateBrief() with full /api/prep-brief POST.
// All backend data flows into shared globals so renderOverlays, renderSidebar
// etc. pick them up automatically.

// ─── Live data globals ───────────────────────────────────────────────────────
var _squadByTeam = {};          // from /api/prep-meta
var _liveWeather = null;        // from brief response
var _liveToss = null;           // from brief response
var _liveTacticalAlert = null;  // from brief.key_decisions[0]

// ─── Colour mapping for XI options ───────────────────────────────────────────
var _XI_COLORS  = ['#4CAF50', '#FFD700', '#F44336'];
var _XI_NAMES   = ['Primary XI', 'Spin Heavy', 'Pace Heavy'];

// ─── Helper: map backend danger string to UI danger class ────────────────────
function _normDanger(d) {
  if (!d) return 'low';
  var dl = d.toLowerCase();
  if (dl === 'high' || dl === 'very high') return 'high';
  if (dl === 'medium' || dl === 'med' || dl === 'moderate') return 'med';
  return 'low';
}

// ─── Helper: map backend role label to roleMap key ───────────────────────────
function _normRole(r) {
  if (!r) return 'bat';
  var rl = r.toLowerCase();
  if (rl === 'wk' || rl === 'wicketkeeper' || rl === 'keeper') return 'wk';
  if (rl.indexOf('all') >= 0) return 'all';
  if (rl === 'bowl' || rl === 'bowler') return 'bowl';
  return 'bat';
}

// ─── INIT: fetch /api/prep-meta to get real teams, venues, squads ─────────────
function initPrepMeta() {
  fetch('/api/prep-meta')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      _squadByTeam = data.squad_by_team || {};

      // Update sidebar dropdowns if already rendered
      _refreshDropdowns(data.teams || [], data.venues || []);

      // Seed fullSquad for the initially selected team
      _applySquadForTeam(selectedOurTeam);
    })
    .catch(function(err) {
      console.warn('prep-meta load failed, using hardcoded squads:', err);
    });
}

// Refresh the Our Team / Opposition / Venue selects with real backend data
function _refreshDropdowns(teams, venues) {
  var ourSel = document.querySelector('#sidebar select:nth-child(2)');
  var oppSel = document.querySelector('#sidebar select:nth-child(4)');
  var venSel = document.querySelector('#sidebar select:nth-child(6)');
  if (!ourSel || !oppSel) return;

  if (teams.length) {
    var ourOpts = teams.map(function(t) {
      return '<option' + (t === selectedOurTeam ? ' selected' : '') + '>' + t + '</option>';
    }).join('');
    ourSel.innerHTML = ourOpts;

    var oppOpts = teams.map(function(t) {
      return '<option' + (t === selectedOppTeam ? ' selected' : '') + '>' + t + '</option>';
    }).join('');
    oppSel.innerHTML = oppOpts;
  }

  if (venues.length && venSel) {
    venSel.innerHTML = venues.map(function(v) {
      return '<option>' + v + '</option>';
    }).join('');
  }
}

// Update fullSquad from the cached squad_by_team for a given team
function _applySquadForTeam(teamName) {
  var sq = _squadByTeam[teamName];
  if (!sq || !sq.length) return;
  fullSquad.length = 0;
  sq.forEach(function(n) { fullSquad.push(n); });
  // Select all squad members so backend receives the full pool for 3 distinct XIs
  selectedSquad.clear();
  for (var j = 0; j < fullSquad.length; j++) selectedSquad.add(j);
}

// ─── Patch applyTeamSelection to pull squads from meta ───────────────────────
var _origApplyTeam = window.applyTeamSelection;
window.applyTeamSelection = function() {
  // Capture selections before calling original
  var ourSel = document.querySelector('#sidebar select:nth-child(2)');
  var oppSel = document.querySelector('#sidebar select:nth-child(4)');
  if (ourSel) selectedOurTeam = ourSel.value;
  if (oppSel) selectedOppTeam = oppSel.value;

  // Apply live squad from meta (overrides TEAM_ROSTERS)
  if (_squadByTeam[selectedOurTeam] && _squadByTeam[selectedOurTeam].length) {
    _applySquadForTeam(selectedOurTeam);
    // Update OPP_ROSTERS with live squad_by_team data if available
    if (_squadByTeam[selectedOppTeam]) {
      var liveSq = _squadByTeam[selectedOppTeam];
      // Build minimal oppBat entries from live squad
      oppBat = liveSq.slice(0, 8).map(function(name, i) {
        // Try to keep existing OPP_ROSTERS entry if name matches
        var existing = (OPP_ROSTERS[selectedOppTeam] || []).find(function(x){ return x.name === name; });
        return existing || {
          name: name, danger: i < 3 ? 'med' : 'low',
          arrival: i === 0 ? 'opens' : i < 4 ? '1-6' : '6-14',
          notes: '', career_sr: 0
        };
      });
    }
    // Refresh UI
    if (curTab === 0) { renderSidebar(); renderRight(); renderOverlays(); renderPL(); }
    else if (curTab === 1) { renderSidebar(); renderRight(); }
    else if (curTab === 2) { renderSidebar(); renderOverlays(); renderOppPL(); }
    return; // skip original (it uses TEAM_ROSTERS which may be stale)
  }

  // Fallback to original (uses hardcoded TEAM_ROSTERS)
  if (_origApplyTeam) _origApplyTeam();
};

// ─── Override generateBrief with full API call ────────────────────────────────
window.generateBrief = function() {
  var ourSel = document.querySelector('#sidebar select:nth-child(2)');
  var oppSel = document.querySelector('#sidebar select:nth-child(4)');
  var venSel = document.querySelector('#sidebar select:nth-child(6)');
  var dtInput = document.querySelector('#sidebar .ss[type="datetime-local"]');

  var ourTeam   = ourSel  ? ourSel.value  : selectedOurTeam;
  var oppTeam   = oppSel  ? oppSel.value  : selectedOppTeam;
  var venue     = venSel  ? venSel.value  : curVenue;
  var matchDt   = dtInput ? dtInput.value : '2026-03-25T19:00';

  selectedOurTeam = ourTeam;
  selectedOppTeam = oppTeam;
  curOppTeam      = oppTeam;
  curVenue        = venue;

  var ourData = teams[ourTeam] || teams['Quetta Gladiators'];
  var oppData = teams[oppTeam] || teams['Lahore Qalandars'];

  // Show loading screen
  var lsOur = document.getElementById('ls-our');
  var lsOpp = document.getElementById('ls-opp');
  lsOur.textContent = ourData.abbr;
  lsOur.style.background = ourData.bg;
  lsOur.style.borderColor = ourData.color + '55';
  lsOpp.textContent = oppData.abbr;
  lsOpp.style.background = oppData.bg;
  lsOpp.style.borderColor = oppData.color + '55';
  document.getElementById('ls-sub').textContent = ourData.abbr + ' vs ' + oppData.abbr + ' — Match Brief';

  var ls   = document.getElementById('loading-screen');
  var fill = document.getElementById('ls-fill');
  ls.style.display = 'flex';
  fill.style.animation = 'none';
  fill.offsetHeight; // reflow
  fill.style.animation = 'lsfill 2.2s ease-in-out forwards';

  // Send the full available squad so the backend can generate 3 genuinely distinct XIs.
  // selectedSquad toggles mark which players are available (unselected = unavailable/injured).
  // We always include all selected players — the backend picks the best 11 for each option.
  var squad = [];
  selectedSquad.forEach(function(i) {
    if (fullSquad[i]) squad.push(fullSquad[i]);
  });
  // Hard fallback: if toggles somehow excluded too many, send the entire squad
  if (squad.length < 13) {
    squad = fullSquad.slice();
  }

  // POST to backend
  fetch('/api/prep-brief', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      our_team:       ourTeam,
      opposition:     oppTeam,
      venue:          venue,
      squad:          squad,
      match_datetime: matchDt
    })
  })
  .then(function(r) {
    if (!r.ok) return r.json().then(function(e){ throw new Error(e.error || 'API '+r.status); });
    return r.json();
  })
  .then(function(data) {
    _applyBriefData(data);
    _finishBrief();
  })
  .catch(function(err) {
    console.warn('[prep-brief] API failed — using hardcoded data:', err);
    _finishBrief();
  });
};

// ─── Apply all backend brief data to UI globals ───────────────────────────────
function _applyBriefData(data) {

  // ── Toss ──────────────────────────────────────────────────────────────────
  if (data.toss) {
    _liveToss = data.toss;
  }

  // ── Weather ───────────────────────────────────────────────────────────────
  if (data.weather) {
    _liveWeather = data.weather;
    // Ensure wind fields are present with safe defaults
    if (_liveWeather.wind_kph === undefined) _liveWeather.wind_kph = 0;
    if (!_liveWeather.wind_dir) _liveWeather.wind_dir = '';
  }

  // ── Tactical alert from key_decisions ────────────────────────────────────
  if (data.key_decisions && data.key_decisions.length) {
    _liveTacticalAlert = data.key_decisions[0];
  }

  // ── Contingencies ─────────────────────────────────────────────────────────
  if (data.contingencies && data.contingencies.length) {
    contingency = data.contingencies;
  }

  // ── Opposition batting order ──────────────────────────────────────────────
  if (data.opposition && data.opposition.length) {
    oppBat = data.opposition.map(function(b) {
      return {
        name:            b.name,
        danger:          _normDanger(b.danger),
        arrival:         b.arrival || '',
        arrival_over_range: b.arrival || '',
        notes:           b.note || '',
        note:            b.note || '',
        career_sr:       b.career_sr || 0,
        death_sr:        b.death_sr  || 0,
        confidence:      b.confidence || '',
        style:           b.style || ''
      };
    });
  }

  // ── Matchup notes ─────────────────────────────────────────────────────────
  if (data.matchup_notes && data.matchup_notes.length) {
    matchupNotes = data.matchup_notes.map(function(m) {
      return {
        conf:    m.conf || 'med',
        balls:   m.balls || 0,
        text:    m.note || m.text || '',
        note:    m.note || m.text || '',
        batter:  m.batter || '',
        bowler:  m.bowler || '',
        advantage: m.advantage || ''
      };
    });
  }

  // ── Over plan ─────────────────────────────────────────────────────────────
  if (data.over_plan && data.over_plan.length) {
    overPlan = data.over_plan.map(function(o) {
      var ph = (o.phase || 'mid').toLowerCase();
      if (ph.indexOf('pp') >= 0 || ph.indexOf('power') >= 0) ph = 'pp';
      else if (ph.indexOf('death') >= 0) ph = 'death';
      else ph = 'mid';
      return {
        over:   o.over,
        bowler: o.bowler || '',
        backup: o.backup || '',
        phase:  ph,
        note:   o.note || ''
      };
    });
    curOver = 1;
  }

  // ── XI options ────────────────────────────────────────────────────────────
  if (data.xi_options && data.xi_options.length) {
    data.xi_options.forEach(function(opt, idx) {
      if (!xiOpts[idx]) return;
      xiOpts[idx].name  = _XI_NAMES[idx];
      xiOpts[idx].color = _XI_COLORS[idx];
      xiOpts[idx].description = opt.description || '';
      xiOpts[idx].overseas_count = opt.overseas_count;
      xiOpts[idx].constraint_note = opt.constraint_note || '';
      // Map players: {pos, name, role, key_stat, source} → {n, p}
      if (opt.players && opt.players.length) {
        var _tw = selectedOurTeam.split(' ');
        var _tf = (_tw.length >= 2 ? _tw[0] + '-' + _tw[1].toLowerCase() : selectedOurTeam.toLowerCase().replace(/ /g, '-'));
        xiOpts[idx].players = opt.players.map(function(p) {
          var nm = p.name || p.player_name || '';
          return {
            n:        nm,
            role:     _normRole(p.role),
            key_stat: p.key_stat || '',
            source:   p.source || '',
            p: nm ? '/assets/players/' + _tf + '/' + nm.toLowerCase().replace(/ /g, '-') + '.png' : ''
          };
        });
      }
    });
    // Update formations for brief panel
    data.xi_options.forEach(function(opt, idx) {
      if (!formations[idx]) return;
      formations[idx].name = opt.label || formations[idx].name;
      if (opt.players && opt.players.length) {
        formations[idx].players = opt.players.map(function(p) {
          return { n: p.name || '', role: (p.role || 'BAT').toUpperCase() };
        });
      }
    });
  }

  // ── Batting scenarios ─────────────────────────────────────────────────────
  if (data.scenarios && data.scenarios.length) {
    scenarios = data.scenarios.map(function(s) {
      return {
        id:          s.id || 'A',
        title:       s.title || s.name || '',
        color:       s.color || '#FFD700',
        cls:         s.id ? s.id.toLowerCase() : 'a',
        desc:        s.desc || s.description || '',
        trigger:     s.trigger || '',
        key_message: s.key_message || '',
        weather_note:s.weather_note || '',
        players: (s.players || []).map(function(p) {
          return {
            name:  p.name || p.player_name || '',
            role:  p.role || p.role_in_card || '',
            cls:   _roleClass ? _roleClass(p.role||p.role_in_card||'') : 'anc',
            note:  p.note || p.instruction || ''
          };
        })
      };
    });
  }

  // ── Ball probs — live bowler-stats-driven bar chart ───────────────────────
  if (data.ball_probs && data.ball_probs.length === 120) {
    ballProbs.length = 0;
    data.ball_probs.forEach(function(p) { ballProbs.push(p); });
    // Rebuild 3D bars with real bowler economy data
    if (typeof refreshBallProbs === 'function') refreshBallProbs();
    console.info('[prep-brief] Ball probs updated from live bowler stats');
  }

  // ── Data tier warnings ────────────────────────────────────────────────────
  if (data.data_tier_notes && data.data_tier_notes.length) {
    window._dataWarnings = data.data_tier_notes;
    console.info('[prep-brief] Data confidence notes:', data.data_tier_notes.join(' | '));
  }
}

// ─── Finish loading: hide screen, update UI ───────────────────────────────────
function _finishBrief() {
  setTimeout(function() {
    document.getElementById('loading-screen').style.display = 'none';
    briefGenerated    = true;
    curBriefFormation = 1;
    curXI             = 0;
    renderSidebar();
    renderRight();
    renderOverlays();
    renderTabs();
  }, 2300);
}

// ─── Boot: load meta immediately ─────────────────────────────────────────────
initPrepMeta();
