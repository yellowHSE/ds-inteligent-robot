# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

"""
exit_request_watcher.py
- Firebase /records 에서 status == EXIT_REQUESTED 레코드를 폴링
- 감지 즉시 EXIT_IN_PROGRESS로 전이
- delay_sec 뒤에:
  1) 요금 정산(발렛 기본금 + 주차비) 계산
  2) records: status=EXIT, slot_id=null, slot_id_last 보존, fare breakdown 저장
  3) slots/{slot_id}: occupied=false, occupied_by=null
"""

import rclpy
from rclpy.node import Node

import firebase_admin
from firebase_admin import credentials, db
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

KST = ZoneInfo("Asia/Seoul") if ZoneInfo else None

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _from_epoch_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

def _parse_iso_guess(s: str, assume_kst: bool) -> Optional[datetime]:
    """ISO 비슷한 문자열 파싱. tz 없으면 KST 또는 UTC로 가정."""
    if not s:
        return None
    try:
        ss = s.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            if assume_kst and KST:
                dt = dt.replace(tzinfo=KST)
            else:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def _pick_entry_dt(rec: Dict[str, Any]) -> Tuple[Optional[datetime], str, str]:
    """
    entry 시각을 다음 우선순위로 선택:
      1) entry_ts / entry_time_ts / created_at_ts (epoch ms)
      2) *_kst → KST로 파싱
      3) entry_time / created_at / entered_at → tz 없으면 KST로 가정
    반환: (UTC datetime or None, source_key, tz_assumption)
    """
    for k in ("entry_ts", "entry_time_ts", "created_at_ts"):
        v = rec.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return (_from_epoch_ms(int(v)), k, "epoch_ms")

    for k in ("entry_time_kst", "created_at_kst"):
        v = rec.get(k)
        if isinstance(v, str):
            dt = _parse_iso_guess(v, assume_kst=True)
            if dt:
                return (dt, k, "Asia/Seoul")

    for k in ("entry_time", "created_at", "entered_at"):
        v = rec.get(k)
        if isinstance(v, str):
            dt = _parse_iso_guess(v, assume_kst=True)
            if dt:
                return (dt, k, "Asia/Seoul")

    return (None, "unknown", "n/a")


class ExitRequestWatcher(Node):
    def __init__(self):
        super().__init__('exit_request_watcher')

        # ---------- Parameters ----------
        self.declare_parameter('poll_sec', 1.0)
        self.declare_parameter('delay_sec', 10.0)

        # 요금 정책(원)
        self.declare_parameter('valet_fee', 10000)   # 발렛 기본금
        self.declare_parameter('unit_minutes', 10)   # 과금 단위(분)
        self.declare_parameter('unit_price', 1000)   # 단위당 가격
        self.declare_parameter('grace_minutes', 5)   # 무료 유예(분)
        self.declare_parameter('rounding', 'ceil')   # ceil|floor|round
        self.declare_parameter('min_parking_fee', 0) # 최저 주차비

        poll_sec = float(self.get_parameter('poll_sec').value)
        self.delay_sec = float(self.get_parameter('delay_sec').value)

        self.valet_fee = int(self.get_parameter('valet_fee').value)
        self.unit_minutes = int(self.get_parameter('unit_minutes').value)
        self.unit_price = int(self.get_parameter('unit_price').value)
        self.grace_minutes = int(self.get_parameter('grace_minutes').value)
        self.rounding = str(self.get_parameter('rounding').value).lower()
        self.min_parking_fee = int(self.get_parameter('min_parking_fee').value)

        # ---------- Firebase ----------
        pkg_root = Path(__file__).resolve().parent
        cred_path = str(pkg_root / "keys" / "service-account.json")
        db_url = "https://ds-intelligent-robot-default-rtdb.asia-southeast1.firebasedatabase.app"

        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {"databaseURL": db_url})

        self.ref_records = db.reference('/records')
        self.ref_slots_root = db.reference('/slots')

        # one-shot 타이머 저장
        self._pending_timers: Dict[str, any] = {}

        # 주기 폴링
        self.create_timer(poll_sec, self.poll_records)

        self.get_logger().info(
            f"ExitRequestWatcher started. poll={poll_sec}s delay={self.delay_sec}s "
            f"fare(valet={self.valet_fee}, {self.unit_minutes}m/{self.unit_price}₩, "
            f"grace={self.grace_minutes}m, round={self.rounding})"
        )

    def poll_records(self):
        try:
            snap = self.ref_records.get()
            if not snap:
                return

            for key, rec in snap.items():
                if rec.get("status") == "EXIT_REQUESTED":
                    if key in self._pending_timers:
                        continue
                    # 중복 방지 위해 즉시 IN_PROGRESS 전이
                    self.ref_records.child(key).update({
                        "status": "EXIT_IN_PROGRESS",
                        "exit_in_progress_at": now_iso()
                    })
                    # delay_sec 뒤 finalize
                    timer = self.create_timer(self.delay_sec, lambda k=key: self._finalize_exit(k))
                    self._pending_timers[key] = timer

        except Exception as e:
            self.get_logger().error(f"poll failed: {e}")

    def _calc_fee(self, rec: dict) -> Tuple[int, dict]:
        entry_dt, src, tz_assume = _pick_entry_dt(rec)
        now_dt = datetime.now(timezone.utc)

        if entry_dt is None:
            dur_raw = 0
        else:
            dur_raw = int((now_dt - entry_dt).total_seconds() // 60)

        # 음수 방지(타임존/시계 문제)
        dur_min = max(0, dur_raw)

        billable_min = max(0, dur_min - self.grace_minutes)

        if self.unit_minutes <= 0:
            billed_units = 0
        else:
            units = billable_min / self.unit_minutes
            if self.rounding == 'floor':
                billed_units = int(units // 1)
            elif self.rounding == 'round':
                billed_units = int(round(units))
            else:
                # ceil
                billed_units = int(-(-billable_min // self.unit_minutes)) if billable_min > 0 else 0

        parking_fee = billed_units * self.unit_price
        if parking_fee < self.min_parking_fee:
            parking_fee = self.min_parking_fee if billable_min > 0 else 0

        total = self.valet_fee + parking_fee

        breakdown = {
            "calc_at": now_iso(),
            "timestamp_source": src,
            "tz_assumption": tz_assume,
            "entry_time": rec.get(src) if src in rec else (rec.get("entry_time") or rec.get("created_at")),
            "duration_raw_minutes": dur_raw,
            "duration_minutes": dur_min,
            "grace_minutes": self.grace_minutes,
            "billable_minutes": billable_min,
            "unit_minutes": self.unit_minutes,
            "unit_price": self.unit_price,
            "rounding": self.rounding,
            "billed_units": billed_units,
            "parking_fee": parking_fee,
            "valet_fee": self.valet_fee,
            "min_parking_fee": self.min_parking_fee,
            "total_fee": total,
            "currency": "KRW"
        }
        return total, breakdown

    def _finalize_exit(self, rec_key: str):
        # one-shot
        timer = self._pending_timers.get(rec_key)
        if timer:
            timer.cancel()
            self._pending_timers.pop(rec_key, None)

        try:
            rec_ref = self.ref_records.child(rec_key)
            rec = rec_ref.get()
            if not rec or rec.get("status") not in ("EXIT_REQUESTED", "EXIT_IN_PROGRESS"):
                return

            slot_id = rec.get("slot_id")

            total, fare = self._calc_fee(rec)

            # slot_id를 null로 비우기 전에 last 필드로 보존
            rec_ref.update({
                "status": "EXIT",
                "exited_at": now_iso(),
                "slot_id_last": slot_id,
                "slot_id": None,
                "fare": fare,
                "total_fee": total,
                "fare_ready": True
            })

            # 슬롯 해제
            if slot_id:
                slot_ref = self.ref_slots_root.child(str(slot_id))
                if slot_ref.get() is not None:
                    slot_ref.update({
                        "occupied": False,
                        "occupied_by": None,
                        "updated_at": now_iso()
                    })

            self.get_logger().info(f"✅ EXIT+fare: rec={rec_key} total={total} KRW")

        except Exception as e:
            self.get_logger().error(f"Finalize failed for {rec_key}: {e}")
            try:
                self.ref_records.child(rec_key).update({
                    "status": "EXIT_FAILED",
                    "exit_failed_at": now_iso(),
                    "exit_failed_reason": str(e)
                })
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = ExitRequestWatcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
