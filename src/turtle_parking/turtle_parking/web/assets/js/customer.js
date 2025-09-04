// 고객 출차 요청 + 실시간 요금 감지(같은 번호면 즉시/업데이트 대기)
// 하드코딩 제거 버전
import { db } from './firebase-config.js';
import {
  ref, get, update, query, orderByChild, equalTo, serverTimestamp,
  onValue, off
} from 'https://www.gstatic.com/firebasejs/12.1.0/firebase-database.js';

console.log('customer.js v2025-09-03 live-fare-watch');

const $ = s => document.querySelector(s);

// --- DOM refs ---
const exitForm   = $('#exitForm');
const inputPlate = $('#plate');
const submitBtn  = $('#submitBtn');

const resultSec  = $('#result');
const reqIdSpan  = $('#reqId');
const reqPlate   = $('#reqPlate');
const reqTime    = $('#reqTime');
const newBtn     = $('#newBtn');

const fareBox      = $('#fareBox');
const fareRecId    = $('#fareRecId');
const farePlate    = $('#farePlate');
const fareEntry    = $('#fareEntryTime');
const fareDuration = $('#fareDuration');
const fareGrace    = $('#fareGrace');
const fareBillable = $('#fareBillable');
const fareUnit     = $('#fareUnit');
const fareUnitPrice= $('#fareUnitPrice');
const fareUnits    = $('#fareUnits');
const fareParking  = $('#fareParking');
const fareValet    = $('#fareValet');
const fareTotal    = $('#fareTotal');
const fareFormula  = $('#fareFormula');

// --- helpers ---
const params = new URLSearchParams(location.search);
if (params.get('plate')) inputPlate.value = params.get('plate');

function normalizePlate(s){
  return (s || '')
    .toUpperCase().replace(/\s+/g,'')
    .replace(/[^\p{Script=Hangul}0-9]/gu,''); // 한글·숫자만
}
const nowISO = () => new Date().toISOString();
const fmtKRW = (n) => new Intl.NumberFormat('ko-KR').format(Number(n||0)) + '원';
const toKST   = (t) => { try{ const d=new Date(t); return isNaN(d)? String(t||'-') : d.toLocaleString('ko-KR'); }catch{ return String(t||'-'); } };
const hasFare = (v) => !!(v && (v.total_fee !== undefined || v.fare));

function showForm(){
  exitForm.classList.remove('hidden');
  resultSec.classList.add('hidden');
  fareBox.classList.add('hidden');
}
function showWaiting(recId, plate){
  exitForm.classList.add('hidden');
  resultSec.classList.remove('hidden');
  fareBox.classList.add('hidden');
  reqIdSpan.textContent = recId || '-';
  reqPlate.textContent  = plate || '-';
  reqTime.textContent   = new Date().toLocaleString('ko-KR');
}

function renderFareInline(rec){
  const fare = rec.fare || {};
  const unitM = fare.unit_minutes ?? 0;
  const unitP = fare.unit_price   ?? 0;
  const dur   = fare.duration_minutes ?? 0;
  const grace = fare.grace_minutes ?? 0;
  const billable = fare.billable_minutes ?? Math.max(0, dur - grace);
  const billedUnits = fare.billed_units ?? (unitM > 0 ? Math.ceil(billable / unitM) : 0);
  const parkingFee  = fare.parking_fee  ?? billedUnits * unitP;
  const valetFee    = fare.valet_fee    ?? 0;
  const total       = rec.total_fee ?? fare.total_fee ?? (parkingFee + valetFee);

  fareRecId.textContent     = rec.key || '-';
  farePlate.textContent     = rec.license_plate || '-';
  fareEntry.textContent     = toKST(fare.entry_time || rec.entry_time || rec.created_at || '-');
  fareDuration.textContent  = `${dur}분`;
  fareGrace.textContent     = `${grace}분`;
  fareBillable.textContent  = `${billable}분`;
  fareUnit.textContent      = `${unitM || '-'}분`;
  fareUnitPrice.textContent = fmtKRW(unitP);
  fareUnits.textContent     = `${billedUnits}단위`;
  fareParking.textContent   = fmtKRW(parkingFee);
  fareValet.textContent     = fmtKRW(valetFee);
  fareTotal.textContent     = fmtKRW(total);

  const calcPart  = fmtKRW(billedUnits * unitP);
  const valetPart = fmtKRW(valetFee);
  const calcShown = fmtKRW(valetFee + billedUnits * unitP);
  fareFormula.textContent   = `${valetPart} + (${billedUnits} × ${fmtKRW(unitP)} = ${calcPart}) = ${calcShown}`;

  resultSec.classList.add('hidden');
  fareBox.classList.remove('hidden');
}

function safeRecRef(path){
  if (!path.startsWith('/records/')) {
    console.error('❌ blocked write path:', path);
    throw new Error('Writes are allowed only under /records/*');
  }
  return ref(db, path);
}

// 최신 레코드를 "활성 우선 → 최신"으로 선택
function sortCandidates(items){
  const isActive = (st)=> (st||'').toUpperCase() !== 'EXIT';
  items.sort((a,b)=>{
    const aActive = isActive(a.status), bActive = isActive(b.status);
    if (aActive !== bActive) return aActive ? -1 : 1;
    const at = Date.parse(a.entry_time || a.created_at || 0) || 0;
    const bt = Date.parse(b.entry_time || b.created_at || 0) || 0;
    return bt - at;
  });
  return items;
}

async function pickLatestRecordByPlate(plateRaw){
  // 정확 일치(우선)
  const q1 = query(ref(db,'/records'), orderByChild('license_plate'), equalTo(plateRaw));
  const s1 = await get(q1);
  if (s1.exists()){
    const items = Object.entries(s1.val()).map(([key,rec])=>({ key, ...rec }));
    return sortCandidates(items)[0] || null;
  }
  // 정규화 일치
  const snap = await get(ref(db,'/records'));
  if (!snap.exists()) return null;
  const plateNorm = normalizePlate(plateRaw);
  const items = [];
  for (const [key,rec] of Object.entries(snap.val())){
    const lp = rec?.license_plate ?? '';
    if (normalizePlate(lp) === plateNorm) items.push({ key, ...rec });
  }
  return sortCandidates(items)[0] || null;
}

/* ----- 실시간 구독 + 폴백 ----- */
let pathListener  = null;
let plateListener = null;
let pollTimer     = null;
function detachAll(){
  if (pathListener){  off(pathListener.ref,  'value', pathListener.cb);  pathListener = null; }
  if (plateListener){ off(plateListener.ref, 'value', plateListener.cb); plateListener = null; }
  if (pollTimer){ clearInterval(pollTimer); pollTimer = null; }
}

/**
 * 같은 번호판의 어떤 레코드든 요금이 나오면 즉시 표시.
 * - 개별 레코드 path 실시간 구독
 * - 번호판 equalTo 실시간 구독
 * - 2초 폴백 폴링(최대 60회)
 */
function waitFareForPlate(recKey, plate){
  detachAll();

  // 1) 개별 path
  const r = ref(db, `/records/${recKey}`);
  const cb1 = (snap)=>{
    if(!snap.exists()) return;
    const cur = { key: recKey, ...snap.val() };
    if (hasFare(cur) && String(cur.status).toUpperCase() === 'EXIT'){
      renderFareInline(cur); detachAll();
    }
  };
  onValue(r, cb1);
  pathListener = { ref: r, cb: cb1 };

  // 2) 번호판 equalTo
  const q = query(ref(db,'/records'), orderByChild('license_plate'), equalTo(plate));
  const cb2 = (snap)=>{
    if(!snap.exists()) return;
    const list = Object.entries(snap.val()).map(([k,v])=>({ key:k, ...v }))
                   .sort((a,b)=>(Date.parse(b.created_at||0)||0) - (Date.parse(a.created_at||0)||0));
    const hit = list.find(v => hasFare(v) && String(v.status).toUpperCase() === 'EXIT');
    if (hit){ renderFareInline(hit); detachAll(); }
  };
  onValue(q, cb2);
  plateListener = { ref: q, cb: cb2 };

  // 3) 폴백 폴링(2초 간격, 최대 60회)
  let tries = 0;
  pollTimer = setInterval(async ()=>{
    tries++;
    try{
      const s = await get(r);
      if (s.exists()){
        const cur = { key: recKey, ...s.val() };
        if (hasFare(cur) && String(cur.status).toUpperCase() === 'EXIT'){
          renderFareInline(cur); detachAll();
        }
      }
      if (tries >= 60) detachAll();
    }catch(e){
      console.warn('[poll] fail', e);
    }
  }, 2000);
}

/* ----- 제출 흐름 ----- */
exitForm?.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const typed = (inputPlate.value || '').trim();
  if (!typed){ inputPlate.focus(); return; }
  submitBtn.disabled = true;

  try{
    const rec = await pickLatestRecordByPlate(typed);
    if (!rec){
      alert('해당 차량 번호의 레코드를 찾을 수 없습니다.');
      return;
    }

    // 이미 EXIT & 요금 존재 → 즉시 표시
    if (String(rec.status).toUpperCase() === 'EXIT' && hasFare(rec)){
      renderFareInline(rec);
      return;
    }

    // 접수 화면 전환
    showWaiting(rec.key, rec.license_plate || typed);

    const path = `/records/${rec.key}`;

    // 혹시 이미 계산이 완료되어 있으면 즉시 표시
    const curSnap = await get(ref(db, path));
    if (curSnap.exists()){
      const cur = { key: rec.key, ...curSnap.val() };
      if (String(cur.status).toUpperCase() === 'EXIT' && hasFare(cur)){
        renderFareInline(cur);
        return;
      }
    }

    // 아직 EXIT 아니면 EXIT_REQUESTED로 전이
    if (String(rec.status).toUpperCase() !== 'EXIT'){
      await update(safeRecRef(path), {
        status: 'EXIT_REQUESTED',
        exit_request_at: nowISO(),
        exit_request_ts: serverTimestamp()
      });
      console.info('[update] EXIT_REQUESTED set for', rec.key);
    }

    // 동일 번호판에서 요금이 뜨는 순간을 실시간 감시
    waitFareForPlate(rec.key, rec.license_plate || typed);

  }catch(err){
    console.error(err);
    alert('요청 처리 중 오류가 발생했습니다.\n' + (err?.message || ''));
  }finally{
    submitBtn.disabled = false;
  }
});

/* ----- 리셋 ----- */
function resetForm(){
  detachAll();
  inputPlate.value = '';
  showForm();
  inputPlate.focus();
}
newBtn?.addEventListener('click', resetForm);

// 초기 상태
showForm();
