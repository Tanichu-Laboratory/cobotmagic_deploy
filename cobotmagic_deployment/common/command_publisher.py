"""High-rate interpolated command publishing (ROS independent).

``command_publish.mode: interpolated`` publishes between control steps: each
new step target is reached linearly over one control period from the
previous command, while the gripper jumps to its target immediately.
The publish callback and shutdown check are injected, so the same worker
can drive ROS topics or any other transport.
"""

import threading
import time

from cobotmagic_deployment.common.action_processing import interpolate_arm_command_keep_gripper
from cobotmagic_deployment.common.bridge_log import BridgeLog


def command_publish_settings(ros_cfg, rate_hz, eef_action_mode=False, log=None):
    """Resolve ``command_publish`` into (mode, publish_rate_hz, substeps, interpolation)."""
    log = log or BridgeLog()
    cfg = ros_cfg.get('command_publish', {})
    mode = cfg.get('mode', 'direct').lower()
    if mode not in ('direct', 'interpolated'):
        log.warn(f"Unsupported command_publish.mode={mode!r}; using ACT-style 'direct'.")
        mode = 'direct'
    if eef_action_mode and mode != 'direct':
        log.warn("eef_absolute only supports command_publish.mode='direct'; using direct.")
        mode = 'direct'
    if mode == 'direct':
        publish_rate_hz = float(rate_hz)
        substeps = 1
    else:
        publish_rate_hz = max(float(cfg.get('rate_hz', rate_hz)), float(rate_hz))
        substeps = max(int(round(publish_rate_hz / float(rate_hz))), 1)
        publish_rate_hz = float(substeps * rate_hz)
    interpolation = cfg.get('interpolation', 'linear').lower()
    if interpolation != 'linear':
        log.warn(f"Unsupported command_publish.interpolation={interpolation!r}; using 'linear'.")
        interpolation = 'linear'
    return mode, publish_rate_hz, substeps, interpolation


class InterpolatedCommandPublisher:
    """Background worker publishing interpolated commands at ``publish_rate_hz``.

    ``publish(left, right)`` is called from the worker thread;
    ``is_shutdown()`` stops the loop; ``on_publish()`` runs after each publish.
    """

    def __init__(self, publish, publish_rate_hz, step_duration_sec, is_shutdown,
                 on_publish=None, log=None):
        self.publish = publish
        self.publish_rate_hz = publish_rate_hz
        self.period_sec = 1.0 / publish_rate_hz
        self.step_duration_sec = step_duration_sec
        self.is_shutdown = is_shutdown
        self.on_publish = on_publish
        self.log = log or BridgeLog()
        self._lock = threading.Lock()
        self._state = None
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name='command_publish_worker', daemon=True)
        self._thread.start()
        return self

    def set_target(self, start_left, start_right, target_left, target_right, start_time=None):
        """Move from ``start_*`` to ``target_*`` over one control step."""
        with self._lock:
            self._state = {
                'start_time': time.monotonic() if start_time is None else start_time,
                'duration': self.step_duration_sec,
                'start_left': start_left.copy(),
                'start_right': start_right.copy(),
                'target_left': target_left.copy(),
                'target_right': target_right.copy(),
            }

    def _current(self):
        with self._lock:
            if self._state is None:
                return None
            return {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in self._state.items()}

    def _run(self):
        next_publish_time = time.monotonic()
        last_report_time = next_publish_time
        report_count = 0
        active_report_count = 0
        while not self.is_shutdown():
            state = self._current()
            if state is not None:
                elapsed = time.monotonic() - state['start_time']
                duration = float(state['duration'])
                alpha = 1.0 if duration <= 0.0 else min(max(elapsed / duration, 0.0), 1.0)
                self.publish(
                    interpolate_arm_command_keep_gripper(state['start_left'], state['target_left'], alpha),
                    interpolate_arm_command_keep_gripper(state['start_right'], state['target_right'], alpha),
                )
                if self.on_publish is not None:
                    self.on_publish()
                active_report_count += 1
            report_count += 1

            now_report = time.monotonic()
            if now_report - last_report_time >= 2.0:
                elapsed_report = now_report - last_report_time
                self.log.info(
                    "Command publish worker rate: "
                    f"loop_hz={report_count / elapsed_report:.1f} "
                    f"active_publish_hz={active_report_count / elapsed_report:.1f} "
                    f"target_hz={self.publish_rate_hz:.1f}"
                )
                last_report_time = now_report
                report_count = 0
                active_report_count = 0

            next_publish_time += self.period_sec
            sleep_sec = next_publish_time - time.monotonic()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
            else:
                next_publish_time = time.monotonic()
