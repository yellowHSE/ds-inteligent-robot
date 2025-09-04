import { db } from './firebase-config.js';
import {
  ref, onValue, query, orderByKey, limitToLast, get, child
} from 'https://www.gstatic.com/firebasejs/12.1.0/firebase-database.js';

// ---------- Utils ----------
const $  = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const fmt = (v) => (v===undefined||v===null ? '' : String(v));
const fmtDate = (iso) => { try{ return new Date(iso).toLocaleString('ko-KR'); }catch{ return iso } };
const fmtKRW = (n) => n===undefined||n===null||n==='' ? '' : (new Intl.NumberFormat('ko-KR').format(Number(n)) + '원');

// 라벨 매핑
const labelStatus = (s) => ({
  ENTRY: '입차',
  EXIT: '출차',
  EXIT_REQUESTED: '출차 요청',
  EXIT_IN_PROGRESS: '출차 처리중',
  EXIT_FAILED: '출차 실패'
}[String(s||'').toUpperCase()] || s);

const labelClass = (c) => ({
  compact: '경차',
  general: '일반',
}[String(c||'').toLowerCase()] || c);

const occPill = (occ) => occ
  ? '<span class="pill warn">사용중</span>'
  : '<span class="pill ok">비어있는</span>';

// ---------- DOM refs ----------
let classFilterEl, plateSearchEl, statTotalEl, statEmptyEl, statOccupiedEl, statUpdatedEl;
let slotsTableBody, carinfoTableBody, slotMap, ctxMap, robotListEl;

// 내부 상태
let filterClass = 'all';
let plateQuery  = '';
let slots = {};
let filtered = [];
let sortKey = 'slot_id';
let sortAsc = true;

// ---------- Boot ----------
function boot(){
  initUIRefs();
  wireEvents();
  applyFilter();
  ensureChartHeights();
  resizeMapCanvas();
  subscribeRTDB();
  debugReadOnce();
}
if (document.readyState === 'loading') window.addEventListener('DOMContentLoaded', boot); else boot();

// ---------- Init ----------
function need(id){
  const el = document.getElementById(id);
  if (!el) console.info(`[UI] optional element #${id} not found`);
  return el;
}
function initUIRefs(){
  classFilterEl    = need('classFilter');
  plateSearchEl    = need('plateSearch');
  statTotalEl      = need('statTotal');
  statEmptyEl      = need('statEmpty');
  statOccupiedEl   = need('statOccupied');
  statUpdatedEl    = need('statUpdated');
  slotsTableBody   = need('slotsTable')?.querySelector('tbody') || null;
  carinfoTableBody = need('carinfoTable')?.querySelector('tbody') || null;
  slotMap          = need('slotMap');
  ctxMap           = slotMap ? slotMap.getContext('2d') : null;
  robotListEl      = need('robotList');
}
function wireEvents(){
  if (classFilterEl){
    classFilterEl.addEventListener('change', ()=>{ filterClass = classFilterEl.value; applyFilter(); });
  }
  if (plateSearchEl){
    plateSearchEl.addEventListener('input', ()=>{ plateQuery = plateSearchEl.value.trim(); applyFilter(); });
  }
  $$('#slotsTable thead th[data-sort]').forEach(th=>{
    th.addEventListener('click', ()=>{
      const key = th.dataset.sort;
      if (sortKey === key) sortAsc = !sortAsc; else { sortKey = key; sortAsc = true; }
      sortAndRender();
    });
  });
  let _rzTimer;
  window.addEventListener('resize', ()=>{
    clearTimeout(_rzTimer);
    _rzTimer = setTimeout(resizeMapCanvas, 120);
  });
}

// ---------- Helpers ----------
function slotSortKey(id){
  const m = String(id).match(/^([A-Za-z]+)-(\d+)$/);
  return m ? [m[1], Number(m[2])] : [String(id), 0];
}
function currentClass(){ return classFilterEl?.value ?? filterClass; }
function currentPlate(){ return (plateSearchEl?.value ?? plateQuery).trim(); }

// ---------- Stats / Filter / Sort ----------
function computeStats(current){
  const vals  = Object.entries(current).map(([slot_id,v])=>({slot_id, ...v}));
  const total = vals.length;
  const empty = vals.filter(v=>!v.occupied).length;
  const occ   = total - empty;

  const classes = Array.from(new Set(vals.map(v=>v.class || 'unknown'))).sort();
  if (classFilterEl){
    const keep = classFilterEl.value;
    classFilterEl.innerHTML =
      `<option value="all">전체</option>` +
      classes.map(c=>`<option value="${c}">${labelClass(c)}</option>`).join('');
    if (classes.includes(keep)) classFilterEl.value = keep;
  }

  if (statTotalEl)    statTotalEl.textContent = total;
  if (statEmptyEl)    statEmptyEl.textContent = empty;
  if (statOccupiedEl) statOccupiedEl.textContent = occ;

  const now = new Date();
  if (statUpdatedEl){
    const d = now.toLocaleDateString('ko-KR');
    const t = now.toLocaleTimeString('ko-KR', { hour12: false });
    statUpdatedEl.innerHTML = `${d}<br>${t}`;
  }

  const emptyByClass = {};
  vals.forEach(v=>{
    const c = v.class || 'unknown';
    emptyByClass[c] = emptyByClass[c] || 0;
    if (!v.occupied) emptyByClass[c]++;
  });
  return { total, empty, occ, emptyByClass };
}

function applyFilter(){
  const cls = currentClass();
  const q   = currentPlate();
  const vals = Object.entries(slots).map(([slot_id,v])=>({slot_id,...v}));
  filtered = vals.filter(v=>{
    const byClass = (cls==='all' || (v.class||'unknown')===cls);
    const byPlate = !q || String(v.occupied_by||'').includes(q);
    return byClass && byPlate;
  });
  sortAndRender();
}

function sortAndRender(){
  filtered.sort((a,b)=>{
    let av,bv;
    if (sortKey==='slot_id'){
      const A=slotSortKey(a.slot_id), B=slotSortKey(b.slot_id);
      av=A[0]; bv=B[0]; if(av===bv){ av=A[1]; bv=B[1]; }
    } else if (sortKey==='class'){ av=a.class||''; bv=b.class||''; }
    else if (sortKey==='occupied'){ av=Number(!!a.occupied); bv=Number(!!b.occupied); }
    else { av=a[sortKey]; bv=b[sortKey]; }
    if (av<bv) return sortAsc?-1:1;
    if (av>bv) return sortAsc?1:-1;
    return 0;
  });
  renderSlotsTable();
  renderMap();
}

// ---------- Tables ----------
function renderSlotsTable(){
  if (!slotsTableBody) return;
  slotsTableBody.innerHTML = filtered.map(v=>{
    return `
      <tr>
        <td>${fmt(v.slot_id)}</td>
        <td>${labelClass(v.class||'unknown')}</td>
        <td>${occPill(!!v.occupied)}</td>
        <td>${fmt(v.occupied_by)}</td>
      </tr>`;
  }).join('');
}

// ---------- Charts ----------
let pieOverall, barByClass;
let barFilterSelection = null;

function ensureChartHeights(){
  const pieEl = document.getElementById('pieOverall');
  const barEl = document.getElementById('barByClass');
  if (pieEl && !pieEl.style.height) pieEl.style.height = '240px';
  if (barEl && !barEl.style.height) barEl.style.height = '260px';
}

function renderCharts(stats){
  const pieEl = document.getElementById('pieOverall');
  const barEl = document.getElementById('barByClass');
  if (!pieEl || !barEl || typeof Chart === 'undefined') return;
  ensureChartHeights();

  // 파이 차트: 한국어 라벨
  if(!pieOverall){
    pieOverall = new Chart(pieEl, {
      type:'pie',
      data:{ labels:['사용중','비어있음'], datasets:[{ data:[stats.occ, stats.empty] }] },
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{ legend:{ position:'top', labels:{ color:'#c7d1e0', boxWidth:10 } } }
      }
    });
  } else {
    pieOverall.data.datasets[0].data = [stats.occ, stats.empty];
    pieOverall.update();
  }

  // 막대 차트: 클래스 한글화
  const labels = Object.keys(stats.emptyByClass).sort().map(labelClass);
  const data   = Object.keys(stats.emptyByClass).sort().map(k=>stats.emptyByClass[k]);

  if(!barByClass){
    barByClass = new Chart(barEl, {
      type:'bar',
      data:{ labels, datasets:[{ label:'비어있는 슬롯', data }] },
      options:{
        responsive:true, maintainAspectRatio:false,
        scales:{ x:{ ticks:{ color:'#c7d1e0' } }, y:{ ticks:{ color:'#c7d1e0' }, beginAtZero:true, precision:0 } },
        plugins:{ legend:{ display:false } }
      }
    });
    barEl.addEventListener('click', (evt)=>{
      const pts = barByClass.getElementsAtEventForMode(evt, 'nearest', {intersect:true}, true);
      if(!pts.length) return;
      const idx = pts[0].index;
      const labelKo = barByClass.data.labels[idx];
      // 역매핑: ko -> key
      const mapping = { 경차:'compact', 일반:'general'};
      const labelKey = mapping[labelKo] || labelKo;
      if (barFilterSelection === labelKey){ barFilterSelection = null; filterClass = 'all'; }
      else { barFilterSelection = labelKey; filterClass = labelKey; }
      if (classFilterEl) classFilterEl.value = filterClass;
      applyFilter();
    });
  } else {
    barByClass.data.labels = labels;
    barByClass.data.datasets[0].data = data;
    barByClass.update();
  }
}

// ---------- Map ----------
function renderMap(){
  if (!slotMap || !ctxMap) return;
  const W = slotMap.width, H = slotMap.height;
  ctxMap.clearRect(0,0,W,H);

  const vals  = filtered.length ? filtered : Object.entries(slots).map(([slot_id,v])=>({slot_id,...v}));
  const poses = vals.map(v=>v.pose||{}).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));
  if(!poses.length){ ctxMap.fillStyle='#6b778a'; ctxMap.fillText('pose 데이터가 없습니다.', 10, 20); return; }

  const pad=30;
  const minX=Math.min(...poses.map(p=>p.x)), maxX=Math.max(...poses.map(p=>p.x));
  const minY=Math.min(...poses.map(p=>p.y)), maxY=Math.max(...poses.map(p=>p.y));
  const spanX=Math.max(1e-6, maxX-minX), spanY=Math.max(1e-6, maxY-minY);
  const sx=(W-2*pad)/spanX, sy=(H-2*pad)/spanY;
  const scale=Math.min(sx,sy);
  const toPix = (p)=>({ x: pad+(p.x-minX)*scale, y: H-pad-(p.y-minY)*scale });

  // 추천 하이라이트
  const cls = currentClass();
  const empties = vals.filter(v=>!v.occupied && (cls==='all' || (v.class||'unknown')===cls));
  empties.sort((a,b)=>{ const A=slotSortKey(a.slot_id), B=slotSortKey(b.slot_id); return A[0]===B[0] ? A[1]-B[1] : (A[0]<B[0]?-1:1); });
  const highlightId = empties.length ? empties[0].slot_id : null;

  // 그리드
  ctxMap.strokeStyle='#1f2534'; ctxMap.lineWidth=1;
  for(let gx=0; gx<3; gx++){ ctxMap.beginPath(); ctxMap.moveTo(pad+(W-2*pad)*gx/3,pad); ctxMap.lineTo(pad+(W-2*pad)*gx/3,H-pad); ctxMap.stroke(); }
  for(let gy=0; gy<3; gy++){ ctxMap.beginPath(); ctxMap.moveTo(pad,pad+(H-2*pad)*gy/3); ctxMap.lineTo(W-pad,pad+(H-2*pad)*gy/3); ctxMap.stroke(); }

  // 포인트
  vals.forEach(v=>{
    const p=v.pose||{}; if(!Number.isFinite(p.x)||!Number.isFinite(p.y)) return;
    const q=toPix(p), r=7;
    ctxMap.beginPath();
    ctxMap.fillStyle = v.occupied ? '#ef6a6a' : '#27c093';
    ctxMap.arc(q.x,q.y,r,0,Math.PI*2); ctxMap.fill();
    if (v.slot_id===highlightId){
      ctxMap.lineWidth=3; ctxMap.strokeStyle='#f7d26a';
      ctxMap.beginPath(); ctxMap.arc(q.x,q.y,r+3,0,Math.PI*2); ctxMap.stroke();
    }
    ctxMap.fillStyle='#c7d1e0'; ctxMap.font='12px ui-monospace, Menlo, monospace';
    ctxMap.fillText(String(v.slot_id), q.x+10, q.y-8);
  });
}

function resizeMapCanvas(){
  if (!slotMap) return;
  const rect = slotMap.getBoundingClientRect();
  const desiredW = Math.max(560, Math.floor(rect.width));
  const desiredH = Math.min(480, Math.max(360, Math.floor(desiredW * 0.65)));
  if (slotMap.width !== desiredW || slotMap.height !== desiredH){
    slotMap.width = desiredW;
    slotMap.height = desiredH;
  }
  renderMap();
}

// ---------- Robots ----------
function renderRobots(obj){
  if (!robotListEl) return;
  const entries = Object.entries(obj || {}).sort(([a],[b]) => a.localeCompare(b));
  if (!entries.length){
    robotListEl.innerHTML = `<div class="muted small">로봇 데이터가 없습니다.</div>`;
    return;
  }

  robotListEl.innerHTML = entries.map(([name, v])=>{
    const online = (v?.status === true || v?.status === 'true');
    const batt   = Math.max(0, Math.min(100, Number(v?.battery ?? 0)));
    const barCls = (!online || batt < 20) ? 'bbar low' : 'bbar';
    const pill   = online ? '<span class="pill ok robot-pill">대기중</span>'
                          : '<span class="pill warn robot-pill">작업중</span>';
    return `
      <div class="robot-row">
        <div class="robot-name">${fmt(name)}</div>
        <div>${pill}</div>
        <div class="robot-batt">
          <div class="${barCls}"><div class="fill" style="width:${batt}%;"></div></div>
          <div class="bpct">${batt}%</div>
        </div>
      </div>`;
  }).join('');
}

// ---------- RTDB ----------
function subscribeRTDB(){
  // /slots
  onValue(ref(db, '/slots'), (snap)=>{
    slots = snap.val() || {};
    const stats = computeStats(slots);
    applyFilter();
    renderCharts(stats);
    resizeMapCanvas();
  }, (err)=>console.error('[RTDB]/slots error', err));

  // /records  (최근 50개, 요금+slot 표시, 라벨 한글화)
  onValue(query(ref(db, '/records'), orderByKey(), limitToLast(50)), (snap)=>{
    const val = snap.val() || {};
    const rows = Object.entries(val).sort((a,b)=> a[0]<b[0]?1:-1);
    if (carinfoTableBody){
      carinfoTableBody.innerHTML = rows.map(([id,v])=>{
        const when   = v.entry_time_kst || v.created_at_kst || v.created_at || '';
        const klass  = v.vehicle_class || v.car_class || '';
        const fare   = v.fare || {};
        const total   = (v.total_fee ?? fare.total_fee ?? 0);
        const parking = (fare.parking_fee ?? 0);
        const valet   = (fare.valet_fee ?? 0);
        const units   = (fare.billed_units ?? 0);
        const dur     = (fare.duration_minutes ?? 0);
        const slot    = (v.slot_id ?? v.slot_id_last ?? v.exit_slot_id ?? '');

        return `
          <tr>
            <td>${fmtDate(when)}</td>
            <td>${fmt(v.license_plate)}</td>
            <td>${labelClass(klass)}</td>
            <td>${labelStatus(v.status)}</td>
            <td>${fmt(slot)}</td>
            <td>${fmtKRW(total)}</td>
            <td>${fmtKRW(parking)}</td>
            <td>${fmtKRW(valet)}</td>
            <td>${units}</td>
            <td>${dur}</td>
            <td class="muted">${id}</td>
          </tr>`;
      }).join('');
    }
  }, (err)=>console.error('[RTDB]/records error', err));

  // /robot_status
  onValue(ref(db, '/robot_status'), (snap)=>{
    renderRobots(snap.val() || {});
  }, (err)=>console.error('[RTDB]/robot_status error', err));
}

// ---------- One-shot debug ----------
async function debugReadOnce(){
  try{
    const s  = await get(child(ref(db), '/slots'));
    const r  = await get(child(ref(db), '/records'));
    const rb = await get(child(ref(db), '/robot_status'));
    console.info('[DEBUG]/slots exists =', s.exists());
    console.info('[DEBUG]/records exists =', r.exists());
    console.info('[DEBUG]/robot_status exists =', rb.exists());
  }catch(e){
    console.error('[DEBUG] read once failed', e);
  }
}
