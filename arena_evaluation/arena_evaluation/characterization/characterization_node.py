"""Open-loop characterization node: drives cmd_vel directly through the robot's
full operating envelope (linear sweep to the rated max, transient ramps,
angular pivot rates) and tags every maneuver with a phase marker on
``<ns>/characterization_phase`` so the offline Layer 3 calculator can map
energy/acoustic samples to exact working points.

The operating envelope and acoustic profile are resolved per robot from the
``arena_robots`` caps files (see :func:`maneuvers.resolve_envelope`), so this
node works for any robot in the arena. The robot model is taken from the
``robot_name`` parameter, or derived from the node's namespace (last segment,
e.g. ``.../jackal``) when the parameter is empty.

Safety: a watchdog zeroes ``cmd_vel`` and aborts the schedule if odometry goes
silent beyond ``odom_timeout_s``.

Run inside the robot's namespace:
    ros2 run arena_evaluation characterize --ros-args -r __ns:=<robot_ns> -p use_sim_time:=true
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .maneuvers import (
    CONTROL_RATE_HZ,
    MAX_SCHEDULE_DURATION_S,
    ODOM_STALL_TIMEOUT_S,
    Phase,
    build_schedule,
    resolve_envelope,
    schedule_duration,
)


class CharacterizationNode(Node):
    """Executes the maneuver schedule with a stall watchdog."""

    def __init__(self) -> None:
        super().__init__("characterization_node", automatically_declare_parameters_from_overrides=True)

        self._cmd_vel_topic = str(self.declare_parameter("cmd_vel_topic", "cmd_vel").value)
        self._odom_topic = str(self.declare_parameter("odom_topic", "odom").value)
        self._phase_topic = str(self.declare_parameter("phase_topic", "characterization_phase").value)
        self._robot_name = str(self.declare_parameter("robot_name", "").value)
        self._rate_hz = float(self.declare_parameter("control_rate_hz", CONTROL_RATE_HZ).value)
        self._odom_timeout_s = float(self.declare_parameter("odom_timeout_s", ODOM_STALL_TIMEOUT_S).value)
        self._max_duration_s = float(self.declare_parameter("max_schedule_duration_s", MAX_SCHEDULE_DURATION_S).value)
        self._idle_s = float(self.declare_parameter("idle_duration_s", 10.0).value)
        self._linear_dwell_s = float(self.declare_parameter("linear_dwell_s", 5.0).value)
        self._angular_dwell_s = float(self.declare_parameter("angular_dwell_s", 5.0).value)
        # Optional explicit envelope overrides (e.g. vx_max). None = resolve
        # from the robot's arena_robots caps file.
        self._vx_max = self.declare_parameter("vx_max", -1.0).value
        self._wz_max = self.declare_parameter("wz_max", -1.0).value

        # Robot model: parameter, else the last segment of the node namespace.
        if not self._robot_name:
            ns_parts = [p for p in self.get_namespace().strip("/").split("/") if p]
            self._robot_name = ns_parts[-1] if ns_parts else ""
        envelope = resolve_envelope(self._robot_name or None)
        vx_max = float(self._vx_max) if float(self._vx_max) > 0.0 else envelope["vx_max"]
        wz_max = float(self._wz_max) if float(self._wz_max) > 0.0 else envelope["wz_max"]

        self._schedule: list[Phase] = build_schedule(
            idle_s=self._idle_s,
            linear_dwell_s=self._linear_dwell_s,
            angular_dwell_s=self._angular_dwell_s,
            vx_max=vx_max,
            wz_max=wz_max,
        )
        self._schedule_duration = schedule_duration(self._schedule)
        n_linear = len([p for p in self._schedule if p.kind.value == "linear" and p.wz_target == 0.0])
        n_angular = len([p for p in self._schedule if p.kind.value == "angular"])
        self.get_logger().info(
            f"Characterization schedule (robot={self._robot_name or 'unknown'}): "
            f"{len(self._schedule)} phases, {self._schedule_duration:.0f}s total, "
            f"{n_linear} linear steps (vx up to {vx_max:.2f} m/s), "
            f"{n_angular} angular steps (wz up to {wz_max:.2f} rad/s)"
        )

        odom_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._twist_pub = self.create_publisher(Twist, self._cmd_vel_topic, odom_qos)
        self._phase_pub = self.create_publisher(String, self._phase_topic, odom_qos)
        self._odom_sub = self.create_subscription(Odometry, self._odom_topic, self._on_odom, odom_qos)

        # Sim-time state
        self._last_odom_receipt: float | None = None
        self._phase_idx: int = 0
        self._phase_start_time: float | None = None
        self._last_published_phase: str | None = None
        self._aborted: bool = False
        self._finished: bool = False

        self._timer = self.create_timer(1.0 / self._rate_hz, self._control_tick)

    # ── Odometry watchdog ─────────────────────────────────────────────────────

    def _on_odom(self, msg: Odometry) -> None:
        self._last_odom_receipt = self.get_clock().now().nanoseconds * 1e-9

    def _odom_fresh(self) -> bool:
        if self._last_odom_receipt is None:
            return False
        now = self.get_clock().now().nanoseconds * 1e-9
        return (now - self._last_odom_receipt) <= self._odom_timeout_s

    def _abort(self, reason: str) -> None:
        if self._aborted:
            return
        self._aborted = True
        self.get_logger().error(f"WATCHDOG: {reason} — zeroing cmd_vel and aborting schedule")
        self._twist_pub.publish(Twist())

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_tick(self) -> None:
        if self._finished:
            return
        if self._aborted:
            self._twist_pub.publish(Twist())
            return

        if not self._odom_fresh():
            self._abort(f"odometry silent for >{self._odom_timeout_s}s")
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        if self._phase_start_time is None:
            self._phase_start_time = now

        # Safety ceiling on the whole run
        if now - self._phase_start_time >= self._max_duration_s and self._phase_idx < len(self._schedule):
            self._abort(f"schedule exceeded {self._max_duration_s:.0f}s ceiling")
            return

        if self._phase_idx >= len(self._schedule):
            self._finish()
            return

        phase = self._schedule[self._phase_idx]
        elapsed = now - self._phase_start_time

        if elapsed >= phase.duration_s:
            self._phase_idx += 1
            self._phase_start_time = now
            if self._phase_idx >= len(self._schedule):
                self._finish()
                return
            phase = self._schedule[self._phase_idx]
            elapsed = 0.0

        self._publish_phase_marker(phase)
        self._twist_pub.publish(self._target_twist(phase, elapsed))

    def _publish_phase_marker(self, phase: Phase) -> None:
        if phase.name == self._last_published_phase:
            return
        self._last_published_phase = phase.name
        marker = String()
        marker.data = phase.name
        self._phase_pub.publish(marker)
        self.get_logger().info(f"phase → {phase.name} (kind={phase.kind.value} vx={phase.vx_target} wz={phase.wz_target} dt={phase.duration_s}s)")

    def _target_twist(self, phase: Phase, elapsed_s: float) -> Twist:
        twist = Twist()
        if phase.kind.value == "angular":
            twist.linear.x = 0.0
            twist.angular.z = phase.wz_target
        elif phase.ramp_s > 0.0:
            # Linear ramp: interpolate vx 0 → target over the ramp horizon.
            frac = min(max(elapsed_s / phase.ramp_s, 0.0), 1.0)
            twist.linear.x = phase.vx_target * frac
            if phase.kind.value == "ramp_down":
                twist.linear.x = phase.vx_target * (1.0 - frac)
        else:
            twist.linear.x = phase.vx_target
        return twist

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._twist_pub.publish(Twist())
        self.get_logger().info("Schedule complete — cmd_vel zeroed")
        # Keep the node alive (odom watchdog stays armed) until the runner
        # terminates it, so the final idle block is recorded in full.


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CharacterizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
