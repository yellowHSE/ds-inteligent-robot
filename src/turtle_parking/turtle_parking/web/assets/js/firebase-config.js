// Firebase Web SDK (ESM, v12.1.0)
import { initializeApp } from 'https://www.gstatic.com/firebasejs/12.1.0/firebase-app.js';
import { getAuth, signInAnonymously, onAuthStateChanged } from 'https://www.gstatic.com/firebasejs/12.1.0/firebase-auth.js';
import { getDatabase } from 'https://www.gstatic.com/firebasejs/12.1.0/firebase-database.js';

export const firebaseConfig = {
  apiKey: "AIzaSyBXbxmZgU_1-kLpP8FdI_SwV7OjKDHZLHs",
  authDomain: "ds-intelligent-robot.firebaseapp.com",
  databaseURL: "https://ds-intelligent-robot-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "ds-intelligent-robot",
  storageBucket: "ds-intelligent-robot.appspot.com",   // ← 형식 고정
  messagingSenderId: "549203645073",
  appId: "1:549203645073:web:50c65cf5d84a3247bd6530",
  measurementId: "G-JTMDP3NDN5"
};

export const app  = initializeApp(firebaseConfig);
export const auth = getAuth(app);

// 익명 로그인 (Console → Authentication → Anonymous ON 필수)
await signInAnonymously(auth);
onAuthStateChanged(auth, (u) => console.log('[auth] uid:', u?.uid));

export const db   = getDatabase(app);

// (옵션) Analytics: HTTPS 또는 localhost에서만 안전하게 로드
if (location.protocol === 'https:' || location.hostname === 'localhost') {
  const { getAnalytics, isSupported } = await import('https://www.gstatic.com/firebasejs/12.1.0/firebase-analytics.js');
  if (await isSupported()) getAnalytics(app);
}

// 디버깅 편의
window.db = db;