#!/usr/bin/env python3
import json
import math
from enum import Enum, auto
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion
from builtin_interfaces.msg import Duration as BiDuration
from action_msgs.msg import GoalStatus
from std_srvs.srv import Trigger

from nav2_msgs.action import NavigateToPose, Spin, BackUp
import tf2_ros

# 커스텀 서비스
from parking_msgs.srv import LiftYaw

# -------------------- 유틸 --------------------
PI = math.pi
TWO_PI = 2.0 * math.pi
DEFAULT_YAW = 1.57  # 서비스 오기 전 임시 yaw

def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    half = yaw * 0.5
    q.z = math.sin(half); q.w = math.cos(half)
    q.x = 0.0; q.y = 0.0
    return q

def normalize_angle(a: float) -> float:
    return (a + PI) % (2.0 * PI) - PI

def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

class Phase(Enum):
    IDLE = auto()
    NAVIGATING = auto()
    PAUSED_NAV = auto()   # 사람 근접으로 네비만 일시정지
    ALIGNING = auto()
    BACKING_UP = auto()

# -------------------- 노드 --------------------
class LiftNavigationNode(Node):
    def __init__(self):
        super().__init__('lift_navigation_node')

        # -------- 파라미터 --------
        self.declare_parameter('objects_topic', '/parking_in/objects_in_map')
        self.declare_parameter('final_car_topic', '/parking_in/final_car_pose')
        self.declare_parameter('nav_action_name', 'navigate_to_pose')
        self.declare_parameter('spin_action_name', 'spin')
        self.declare_parameter('backup_action_name', 'backup')

        # 사람 감지 정/재개
        self.declare_parameter('stop_distance', 1.5)
        self.declare_parameter('resume_distance', 2.5)
        self.declare_parameter('stop_consecutive', 5)
        self.declare_parameter('resume_consecutive', 5)

        # 목표 생성 보정
        self.declare_parameter('approach_ahead_m', 0.70)
        self.declare_parameter('car_length_m', 0.40)

        # 네비게이션 목표 yaw(자세)은 고정
        self.declare_parameter('nav_goal_yaw_fixed', -1.57)  # rad, 예: -90도

        # 정렬(Spin)
        self.declare_parameter('yaw_tolerance_rad', 0.05)  # ≈2.9°
        self.declare_parameter('spin_timeout_sec', 10)
        self.declare_parameter('cw_only', True)            # 시계방향만 회전
        self.declare_parameter('spin_max_retries', 1)      # 보정 1회만

        # 후진(BackUp)
        self.declare_parameter('backup_distance', 0.50)
        self.declare_parameter('backup_speed', 0.10)
        self.declare_parameter('backup_timeout_sec', 10)

        # -------- 파라미터 값 --------
        self.objects_topic = str(self.get_parameter('objects_topic').value)
        self.final_car_topic = str(self.get_parameter('final_car_topic').value)
        self.nav_action_name = str(self.get_parameter('nav_action_name').value)
        self.spin_action_name = str(self.get_parameter('spin_action_name').value)
        self.backup_action_name = str(self.get_parameter('backup_action_name').value)

        self.stop_distance = float(self.get_parameter('stop_distance').value)
        self.resume_distance = float(self.get_parameter('resume_distance').value)
        self.stop_consecutive = int(self.get_parameter('stop_consecutive').value)
        self.resume_consecutive = int(self.get_parameter('resume_consecutive').value)

        self.approach_ahead_m = float(self.get_parameter('approach_ahead_m').value)
        self.car_length_m = float(self.get_parameter('car_length_m').value)

        self.nav_goal_yaw_fixed = float(self.get_parameter('nav_goal_yaw_fixed').value)

        self.yaw_tolerance_rad = float(self.get_parameter('yaw_tolerance_rad').value)
        self.spin_timeout_sec = int(self.get_parameter('spin_timeout_sec').value)
        self.cw_only = bool(self.get_parameter('cw_only').value)
        self.spin_max_retries = int(self.get_parameter('spin_max_retries').value)

        self.backup_distance = float(self.get_parameter('backup_distance').value)
        self.backup_speed = float(self.get_parameter('backup_speed').value)
        self.backup_timeout_sec = int(self.get_parameter('backup_timeout_sec').value)

        # -------- TF/액션/통신 --------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._nav_client = ActionClient(self, NavigateToPose, self.nav_action_name)
        self._spin_client = ActionClient(self, Spin, self.spin_action_name)
        self._backup_client = ActionClient(self, BackUp, self.backup_action_name)

        self.sub_obj = self.create_subscription(String, self.objects_topic, self.objects_cb, 10)
        self.sub_car = self.create_subscription(PoseStamped, self.final_car_topic, self.final_car_cb, 1)

        self.lift_yaw_srv = self.create_service(LiftYaw, '/parking_in/lift_yaw', self.lift_yaw_cb)

        self.perception_start_cli = self.create_client(Trigger, '/parking_in/start_perception')
        self.lift_stop_cli  = self.create_client(Trigger, '/parking_in/lift_stop')
        self.lift_reset_cli = self.create_client(Trigger, '/parking_in/reset_perception')
        self.lift_done_cli  = self.create_client(Trigger, '/parking_in/lift_done')

        # -------- 상태 --------
        self._phase: Phase = Phase.IDLE
        self._goal_pose: Optional[PoseStamped] = None
        self._target_yaw: Optional[float] = None  # 서비스로 받은 yaw (정렬용)
        self._nav_goal_handle = None
        self._spin_goal_handle = None
        self._backup_goal_handle = None
        self._spin_attempts = 0

        # 사람 감지 디바운스
        self._stop_count = 0
        self._resume_count = 0

        # 타이머: 메인 루프
        self.timer = self.create_timer(0.2, self.tick)

        self.get_logger().info('LiftNavigationNode: FSM=IDLE. Call /parking_in/lift_yaw then publish final_car_pose to start.')

    # -------------------- 서비스: yaw 수신 & 퍼셉션 시작 --------------------
    def lift_yaw_cb(self, req: LiftYaw.Request, res: LiftYaw.Response):
        try:
            # 받은 yaw는 "위치 계산"과 "정렬(Spin)"에 그대로 사용
            self._target_yaw = float(req.yaw)
            self.get_logger().info(f'[LIFT_YAW] target yaw set: {self._target_yaw:.3f} rad')

            # Perception 시작
            if not self.perception_start_cli.service_is_ready():
                self.perception_start_cli.wait_for_service(timeout_sec=2.0)
            if not self.perception_start_cli.service_is_ready():
                res.accepted = False
                res.message = 'start_perception service unavailable.'
                return res

            fut = self.perception_start_cli.call_async(Trigger.Request())
            def _after_start(_):
                try:
                    start_res = fut.result()
                    if start_res and start_res.success:
                        self.get_logger().info('[START] Perception started.')
                    else:
                        self.get_logger().warn(f"[START] Perception start failed: {getattr(start_res,'message','')}")
                except Exception as e:
                    self.get_logger().warn(f'[START] call failed: {e}')
            fut.add_done_callback(_after_start)

            res.accepted = True
            res.message = 'Yaw accepted & perception started.'
            return res
        except Exception as e:
            res.accepted = False
            res.message = f'Error: {e}'
            return res

    # -------------------- 최종 목표 수신 → NAVIGATING 진입 --------------------
    def final_car_cb(self, ps: PoseStamped):
        """
        입력 ps는 '차의 앞(프런트) 좌표'라고 가정.
        위치 계산: 받은 yaw(self._target_yaw)를 사용해 방향 벡터를 만들고,
                  중심에서 (half_len + approach_ahead) 만큼 그 방향으로 이동.
        자세(orientation): 네비 목표는 고정 yaw(nav_goal_yaw_fixed) 사용.
        """
        # 위치 계산용 yaw는 서비스로 받은 yaw (없으면 DEFAULT_YAW)
        yaw_pos = self._target_yaw if self._target_yaw is not None else DEFAULT_YAW
        fx, fy = math.cos(yaw_pos), math.sin(yaw_pos)

        # 차량 중심 보정 및 목표 위치 산출
        half_len = 0.5 * self.car_length_m
        radius_from_center = half_len + self.approach_ahead_m

        cx = float(ps.pose.position.x)
        cy = float(ps.pose.position.y - half_len)

        gx = cx + radius_from_center * fx
        gy = cy + radius_from_center * fy

        # 네비 목표 자세는 고정 yaw
        yaw_nav = self.nav_goal_yaw_fixed

        goal = PoseStamped()
        goal.header = ps.header
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.position.z = ps.pose.position.z
        goal.pose.orientation = yaw_to_quaternion(yaw_nav)  # ★ orientation은 고정 yaw

        # 상태 리셋 후 NAVIGATING으로
        self._cancel_all_actions()
        self._goal_pose = goal
        self._phase = Phase.NAVIGATING
        self._spin_attempts = 0

        self.get_logger().info(
            f'[TARGET] front=({ps.pose.position.x:.2f},{ps.pose.position.y:.2f}) '
            f'center=({cx:.2f},{cy:.2f}) → goal=({gx:.2f},{gy:.2f}); '
            f'pos_yaw(받은 yaw)={yaw_pos:.3f} rad, nav_yaw(고정)={yaw_nav:.3f} rad'
        )

    # -------------------- 사람 감지 처리 (NAVIGATING에서만 일시정지/재개) --------------------
    def objects_cb(self, msg: String):
        # NAVIGATING/PAUSED_NAV 외 상태에서는 사람 메시지 무시
        if self._phase not in (Phase.NAVIGATING, Phase.PAUSED_NAV):
            return

        try:
            data = json.loads(msg.data)
            if not isinstance(data, list) or not data or not isinstance(data[0], dict):
                return
            obj = data[0]
        except Exception:
            return

        humans = obj.get('human', []) or []
        min_d = self._min_human_distance(humans)

        if min_d is None:
            if self._phase == Phase.PAUSED_NAV:
                self._resume_count += 1
                self._stop_count = 0
                if self._resume_count >= self.resume_consecutive:
                    self._phase = Phase.NAVIGATING
                    self.get_logger().info('[RESUME] humans cleared → NAVIGATING')
            else:
                self._resume_count = 0
                self._stop_count = 0
            return

        if min_d <= self.stop_distance:
            self._stop_count += 1
            self._resume_count = 0
            if self._phase == Phase.NAVIGATING and self._stop_count >= self.stop_consecutive:
                self._cancel_nav_only()
                self._phase = Phase.PAUSED_NAV
                self.get_logger().info(f'[PAUSE] human {min_d:.2f}m ≤ {self.stop_distance}m → PAUSED_NAV')
        elif min_d >= self.resume_distance:
            if self._phase == Phase.PAUSED_NAV:
                self._resume_count += 1
                self._stop_count = 0
                if self._resume_count >= self.resume_consecutive:
                    self._phase = Phase.NAVIGATING
                    self.get_logger().info('[RESUME] humans far → NAVIGATING')
        else:
            self._stop_count = 0
            self._resume_count = 0

    # -------------------- 메인 루프 --------------------
    def tick(self):
        if self._phase == Phase.IDLE:
            return

        if self._phase == Phase.NAVIGATING:
            if self._goal_pose is None:
                return
            if not self._nav_client.wait_for_server(timeout_sec=0.0):
                return
            if self._nav_goal_handle is None:
                # 네비 시작
                goal = NavigateToPose.Goal()
                goal.pose = self._goal_pose
                self.get_logger().info(f'[NAV] start → ({goal.pose.pose.position.x:.2f}, {goal.pose.pose.position.y:.2f})')
                self._nav_client.send_goal_async(goal).add_done_callback(self._on_nav_goal_resp)
            return

        if self._phase == Phase.ALIGNING:
            # 정렬 완료 체크 → 완료면 BACKING_UP으로
            tgt = self._target_yaw if self._target_yaw is not None else DEFAULT_YAW
            if self._is_aligned(tgt, self.yaw_tolerance_rad):
                self.get_logger().info('[ALIGN] OK → BACKING_UP')
                self._phase = Phase.BACKING_UP
                return
            # 정렬 필요 & spin 아직 미보냄 → 1회만 보정
            if self._spin_goal_handle is None and self._spin_attempts < self.spin_max_retries:
                self._spin_attempts += 1
                self._send_spin_to(tgt)
            return

        if self._phase == Phase.BACKING_UP:
            if not self._backup_client.wait_for_server(timeout_sec=0.0):
                return
            if self._backup_goal_handle is None:
                self._send_backup_once(self.backup_distance, self.backup_speed, self.backup_timeout_sec)
            return

    # -------------------- NAV 콜백 --------------------
    def _on_nav_goal_resp(self, fut):
        try:
            gh = fut.result()
        except Exception as e:
            self.get_logger().warn(f'[NAV] send failed: {e}')
            self._nav_goal_handle = None
            return

        if not gh.accepted:
            self.get_logger().warn('[NAV] goal REJECTED')
            self._nav_goal_handle = None
            return
        self._nav_goal_handle = gh
        self.get_logger().info('[NAV] goal ACCEPTED')
        gh.get_result_async().add_done_callback(self._on_nav_result)

    def _on_nav_result(self, fut):
        try:
            status = fut.result().status
        except Exception as e:
            self.get_logger().warn(f'[NAV] result error: {e}')
            self._nav_goal_handle = None
            return

        self._nav_goal_handle = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('[NAV] SUCCEEDED → ALIGNING')
            self._phase = Phase.ALIGNING
        else:
            self.get_logger().warn(f'[NAV] not succeeded (status={status}); stay IDLE')
            self._phase = Phase.IDLE

    # -------------------- SPIN(정렬) --------------------
    def _is_aligned(self, target_yaw: float, tol: float) -> bool:
        cur = self._get_robot_yaw_in_map()
        if cur is None:
            return False
        err = abs(normalize_angle(target_yaw - cur))
        self.get_logger().info(f'[ALIGNCHK] cur={cur:.3f}, target={target_yaw:.3f}, err={err:.3f} (tol={tol:.3f})')
        return err <= tol

    def _send_spin_to(self, target_yaw: float):
        if not self._spin_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn('[SPIN] server not available')
            return
        cur = self._get_robot_yaw_in_map()
        if cur is None:
            self.get_logger().warn('[SPIN] current yaw unavailable')
            return
        diff = normalize_angle(target_yaw - cur)  # [-pi, pi)
        if self.cw_only and diff > 0.0:
            diff -= TWO_PI  # 시계방향만
        self.get_logger().info(f'[SPIN] rotate={diff:.3f} rad (cw_only={self.cw_only})')
        goal = Spin.Goal()
        goal.target_yaw = float(diff)
        goal.time_allowance = BiDuration(sec=max(5, int(self.spin_timeout_sec)))
        self._spin_client.send_goal_async(goal).add_done_callback(self._on_spin_goal_resp)

    def _on_spin_goal_resp(self, fut):
        try:
            gh = fut.result()
        except Exception as e:
            self.get_logger().warn(f'[SPIN] send failed: {e}')
            self._spin_goal_handle = None
            return
        if not gh.accepted:
            self.get_logger().warn('[SPIN] goal REJECTED')
            self._spin_goal_handle = None
            return
        self._spin_goal_handle = gh
        self.get_logger().info('[SPIN] goal ACCEPTED')
        gh.get_result_async().add_done_callback(self._on_spin_result)

    def _on_spin_result(self, fut):
        try:
            _ = fut.result()
        except Exception as e:
            self.get_logger().warn(f'[SPIN] result error: {e}')
        self._spin_goal_handle = None
        # tick()에서 정렬 재확인 → 정렬되었으면 BACKING_UP으로, 아니면 더 이상 시도하지 않음(최대 1회 보정)

    # -------------------- BACKUP(후진 1회) --------------------
    def _send_backup_once(self, dist: float, speed: float, timeout_sec: int):
        goal = BackUp.Goal()
        goal.target.x = -abs(dist); goal.target.y = 0.0; goal.target.z = 0.0
        goal.speed = float(abs(speed))
        goal.time_allowance = BiDuration(sec=max(5, int(timeout_sec)))
        self.get_logger().info(f'[BACKUP] send: {dist:.2f} m @ {speed:.2f} m/s')
        self._backup_client.send_goal_async(goal).add_done_callback(self._on_backup_goal_resp)

    def _on_backup_goal_resp(self, fut):
        try:
            gh = fut.result()
        except Exception as e:
            self.get_logger().warn(f'[BACKUP] send failed: {e}')
            self._backup_goal_handle = None
            return
        if not gh.accepted:
            self.get_logger().warn('[BACKUP] goal REJECTED')
            self._backup_goal_handle = None
            return
        self._backup_goal_handle = gh
        self.get_logger().info('[BACKUP] goal ACCEPTED')
        gh.get_result_async().add_done_callback(self._on_backup_result)

    def _on_backup_result(self, fut):
        try:
            _ = fut.result()
        except Exception as e:
            self.get_logger().warn(f'[BACKUP] result error: {e}')
        self._backup_goal_handle = None

        # 완료 후 퍼셉션 종료/리셋/완료 알림 → IDLE
        self._after_complete_cleanup()
        self._phase = Phase.IDLE
        self.get_logger().info('[DONE] sequence complete → IDLE')

    # -------------------- 공통 유틸 --------------------
    def _after_complete_cleanup(self):
        for (cli, name) in [(self.lift_stop_cli, 'lift_stop'),
                            (self.lift_reset_cli, 'reset_perception'),
                            (self.lift_done_cli, 'lift_done')]:
            if not cli.service_is_ready():
                cli.wait_for_service(timeout_sec=1.0)
            if cli.service_is_ready():
                fut = cli.call_async(Trigger.Request())
                def _cb(f):
                    try:
                        r = f.result()
                        self.get_logger().info(f'[{name}] success={r.success} msg="{r.message}"')
                    except Exception as e:
                        self.get_logger().warn(f'[{name}] call failed: {e}')
                fut.add_done_callback(_cb)
            else:
                self.get_logger().warn(f'[{name}] service not available.')

        # 내부 상태 리셋(다음 시퀀스 대기)
        self._cancel_all_actions()
        self._goal_pose = None
        self._target_yaw = None
        self._spin_attempts = 0
        self._stop_count = 0
        self._resume_count = 0

    def _cancel_nav_only(self):
        try:
            if self._nav_goal_handle is not None:
                self._nav_goal_handle.cancel_goal_async()
        except Exception:
            pass
        self._nav_goal_handle = None

    def _cancel_all_actions(self):
        self._cancel_nav_only()
        try:
            if self._spin_goal_handle is not None:
                self._spin_goal_handle.cancel_goal_async()
        except Exception:
            pass
        try:
            if self._backup_goal_handle is not None:
                self._backup_goal_handle.cancel_goal_async()
        except Exception:
            pass
        self._spin_goal_handle = None
        self._backup_goal_handle = None

    def _get_robot_yaw_in_map(self) -> Optional[float]:
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link',
                                                 rclpy.time.Time(),
                                                 timeout=Duration(seconds=0.2))
            q = tf.transform.rotation
            return quat_to_yaw(q.x, q.y, q.z, q.w)
        except Exception:
            return None

    def _min_human_distance(self, humans) -> Optional[float]:
        # 사람 배열에서 로봇 기준 최소 거리
        rxy = self._robot_xy_in_map()
        if rxy is None or not humans:
            return None
        rx, ry = rxy
        dmin = None
        for h in humans:
            try:
                d = math.hypot(float(h['x']) - rx, float(h['y']) - ry)
                dmin = d if dmin is None else min(dmin, d)
            except Exception:
                continue
        return dmin

    def _robot_xy_in_map(self) -> Optional[Tuple[float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link',
                                                 rclpy.time.Time(),
                                                 timeout=Duration(seconds=0.1))
            return (tf.transform.translation.x, tf.transform.translation.y)
        except Exception:
            return None

# -------------------- 엔트리포인트 --------------------
def main(args=None):
    rclpy.init(args=args)
    node = LiftNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
