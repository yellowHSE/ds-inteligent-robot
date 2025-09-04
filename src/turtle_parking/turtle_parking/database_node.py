#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
)
from geometry_msgs.msg import Pose2D

import firebase_admin
from firebase_admin import credentials, db

# srv
from parking_msgs.srv import CarSize
from parking_msgs.srv import CarInfo  # 번호판 업데이트용 서비스

# -----------------------------
# 유틸
# -----------------------------
def slot_sort_key(slot_id: str):
    m = re.match(r'([A-Za-z]+)-(\d+)', slot_id)
    if m:
        return (m.group(1), int(m.group(2)))
    return (slot_id, 0)

def ts_triplet() -> Tuple[str, str, int]:
    now_utc = datetime.now(timezone.utc)
    iso_utc = now_utc.isoformat(timespec="seconds").replace("+00:00", "Z")
    kst = now_utc.astimezone(ZoneInfo("Asia/Seoul"))
    human = kst.strftime("%Y-%m-%d %H:%M:%S")
    epochms = int(now_utc.timestamp() * 1000)
    return iso_utc, human, epochms

def _alloc_human_id() -> str:
    """REC-YYYYMMDD-NNNN 형태의 사람친화 ID를 트랜잭션으로 발급"""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    ctr_ref = db.reference(f"/meta/seq/records/{today}")

    def txn(cur):
        return int(cur or 0) + 1

    n = ctr_ref.transaction(txn)
    return f"REC-{today}-{n:04d}"

# -----------------------------
# 노드
# -----------------------------
class DatabaseNode(Node):
    def __init__(self):
        super().__init__("database_node")

        # Firebase 초기화
        pkg_root = Path(__file__).resolve().parent
        cred_path = str(pkg_root / "keys" / "service-account.json")
        db_url = "https://ds-intelligent-robot-default-rtdb.asia-southeast1.firebasedatabase.app"

        cred = credentials.Certificate(cred_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {"databaseURL": db_url})

        # 파라미터
        self.declare_parameter("desired_class", "general")  # /parking/free_slot 산출에 사용

        # QoS (마지막 메시지 보존 + 신뢰형)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # 퍼블리셔: /parking/free_slot (Pose2D)
        self.pub_free_slot = self.create_publisher(Pose2D, "parking/free_slot", qos)

        # 타이머: 3초마다 슬롯 읽고 가장 앞번호 빈 슬롯 퍼블리시(옵션)
        self.timer = self.create_timer(3.0, self.read_slots)

        # 서비스 서버
        self.srv_carsize = self.create_service(CarSize, "/car_size", self.handle_carsize)
        self.srv_carinfo = self.create_service(CarInfo, "/parking_in/car_info_in", self.handle_carinfo)

        self.get_logger().info("Database node started. Services: /car_size, /parking_in/car_info_in")

    # -----------------------------
    # 멀티패스 원자 업데이트
    # -----------------------------
    def _atomic_update(self, updates: dict) -> None:
        db.reference("/").update(updates)

    # -----------------------------
    # 슬롯 읽기 → 로그 출력(옵션)
    # -----------------------------
    @staticmethod
    def dict_to_pose2d(pose_dict: dict) -> Pose2D:
        msg = Pose2D()
        msg.x = float(pose_dict.get("x", 0.0))
        msg.y = float(pose_dict.get("y", 0.0))
        msg.theta = float(pose_dict.get("theta", 0.0))
        return msg

    def read_slots(self):
        try:
            ref = db.reference("/slots")
            slots = ref.get()
            if not slots:
                self.get_logger().warn("[DB] No slots data found")
                return

            desired_class = self.get_parameter("desired_class").get_parameter_value().string_value

            occupied_list = []
            empty_list = []
            empty_candidates_for_class = []  # (slot_id, data)

            for slot_id, slot_data in slots.items():
                class_type = slot_data.get("class", "unknown")
                occupied = slot_data.get("occupied", False)
                pose = slot_data.get("pose", {})
                occupied_by = slot_data.get("occupied_by", None)

                if occupied:
                    occupied_list.append(f"{slot_id} (class={class_type}, 차량번호={occupied_by}, pose={pose})")
                else:
                    empty_list.append(f"{slot_id} (class={class_type}, pose={pose})")
                    if class_type == desired_class:
                        empty_candidates_for_class.append((slot_id, slot_data))

            # 요약 로그
            self.get_logger().info("=== Parking Slots ===")
            self.get_logger().info(f"빈 슬롯 ({len(empty_list)}):")
            for s in sorted(empty_list, key=lambda x: slot_sort_key(x.split()[0])):
                self.get_logger().info(f"  - {s}")
            self.get_logger().info(f"주차된 슬롯 ({len(occupied_list)}):")
            for s in sorted(occupied_list, key=lambda x: slot_sort_key(x.split()[0])):
                self.get_logger().info(f"  - {s}")

            # 필요 시 퍼블리시 로직 활성화 가능
            # if empty_candidates_for_class:
            #     empty_candidates_for_class.sort(key=lambda t: slot_sort_key(t[0]))
            #     pick_id, pick_data = empty_candidates_for_class[0]
            #     pose_msg = self.dict_to_pose2d(pick_data.get("pose", {}))
            #     self.pub_free_slot.publish(pose_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to read slots: {e}")

    # -----------------------------
    # 내부 유틸: 클래스별 빈 슬롯 선택
    # -----------------------------
    def _pick_empty_slot_by_class(self, desired_class: str) -> Optional[str]:
        """원하는 class의 '비어있는' 슬롯 중 가장 앞 번호 반환. 없으면 None"""
        ref = db.reference("/slots")
        slots = ref.get() or {}

        def is_empty_and_class(sd):
            return (sd.get("class", "unknown") == desired_class) and (not sd.get("occupied", False))

        candidates = [(sid, sdata) for sid, sdata in slots.items() if is_empty_and_class(sdata)]
        if not candidates:
            return None

        candidates.sort(key=lambda t: slot_sort_key(t[0]))  # A-1, A-2 ...
        return candidates[0][0]

    # -----------------------------
    # 서비스 핸들러: CarSize → /records 생성
    # -----------------------------
    def handle_carsize(self, request: CarSize.Request, response: CarSize.Response):
        size_raw = (request.size or "").strip()
        vehicle_class = size_raw.lower()

        if vehicle_class not in ("compact", "general"):
            response.success = False
            self.get_logger().warn(f"[CarSize] reject: invalid size '{size_raw}' (allowed: compact/general)")
            return response

        try:
            # 1) 클래스에 맞는 가장 앞번호 빈 슬롯 선택
            slot_id = self._pick_empty_slot_by_class(vehicle_class)
            if not slot_id:
                response.success = False
                self.get_logger().warn(f"[CarSize] no empty slot for class='{vehicle_class}'")
                return response

            # 2) /records/<human_id> 생성
            human_id = _alloc_human_id()
            entry_iso, entry_kst, entry_ms = ts_triplet()
            created_iso, created_kst, created_ms = ts_triplet()

            rec_ref = db.reference(f"/records/{human_id}")
            rec_ref.set({
                "id": human_id,
                "status": "ENTRY",
                "vehicle_class": vehicle_class,
                "slot_id": slot_id,
                "entry_time_kst": entry_kst,
                "created_at_kst": created_kst,
            })

            self.get_logger().info(
                f"[CarSize] ENTRY record created: id={human_id}, class={vehicle_class}, slot={slot_id}, at={created_kst} KST"
            )
            response.success = True
            return response

        except Exception as e:
            self.get_logger().error(f"[CarSize] handling failed: {e}")
            response.success = False
            return response

    # -----------------------------
    # 서비스 핸들러: CarInfo → records + slots 동시 갱신
    # -----------------------------
    def handle_carinfo(self, request: CarInfo.Request, response: CarInfo.Response):
        plate = (request.license_plate or "").strip()
        if not plate:
            self.get_logger().warn("[CarInfo] empty plate text")
            response.success = False
            return response

        try:
            records_ref = db.reference("/records")

            # 1) status == ENTRY 우선 조회(인덱스 권장: rules에 .indexOn: ["status"])
            try:
                snapshot = records_ref.order_by_child("status").equal_to("ENTRY").get()
            except Exception as qe:
                self.get_logger().warn(f"[CarInfo] query by status failed, fallback to full scan: {qe}")
                snapshot = records_ref.get()

            # 2) license_plate 미기입(없거나 빈 문자열) 레코드 선택
            target_key = None
            target_rec = None
            if isinstance(snapshot, dict):
                for k, rec in snapshot.items():
                    if rec and rec.get("status") == "ENTRY" and not rec.get("license_plate"):
                        target_key, target_rec = k, rec
                        break
                if not target_key:
                    allrecs = records_ref.get() or {}
                    for k in sorted(allrecs.keys()):
                        rec = allrecs[k]
                        if rec and rec.get("status") == "ENTRY" and not rec.get("license_plate"):
                            target_key, target_rec = k, rec
                            break

            if not target_key:
                self.get_logger().warn("[CarInfo] no ENTRY record without license_plate")
                response.success = False
                return response

            # 3) 대상 레코드의 slot_id 확인
            slot_id = (target_rec or {}).get("slot_id")
            if not slot_id:
                # 슬롯을 모르면 records만 갱신
                self._atomic_update({f"records/{target_key}/license_plate": plate})
                self.get_logger().info(
                    f"[CarInfo] license only updated (record has no slot_id). record='{target_key}', plate='{plate}'"
                )
                response.success = True
                return response

            # 4) 슬롯 충돌 체크(이미 다른 번호면 실패)
            slot_ref = db.reference(f"/slots/{slot_id}")
            slot_data = slot_ref.get() or {}
            occupied_by = slot_data.get("occupied_by")
            if occupied_by and occupied_by != plate:
                self.get_logger().warn(
                    f"[CarInfo] slot '{slot_id}' already occupied_by='{occupied_by}', incoming='{plate}' → abort"
                )
                response.success = False
                return response

            # 5) 원자적 멀티패스 업데이트 (records + slots 동시)
            updates = {
                f"records/{target_key}/license_plate": plate,
                f"slots/{slot_id}/occupied_by": plate,
                f"slots/{slot_id}/occupied": True,
                # 타임스탬프 기록을 원하면 아래 라인 활성화
                # f"records/{target_key}/plate_updated_kst": ts_triplet()[1],
            }
            self._atomic_update(updates)

            self.get_logger().info(
                f"[CarInfo] record='{target_key}', slot='{slot_id}' ← plate='{plate}' (records+slots updated atomically)"
            )
            response.success = True
            return response

        except Exception as e:
            self.get_logger().error(f"[CarInfo] handling failed: {e}")
            response.success = False
            return response

# -----------------------------
# main
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = DatabaseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
