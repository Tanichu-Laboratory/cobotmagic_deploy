"""Asynchronous chunk execution state (ROS independent).

The bridge keeps executing actions from received chunks while the next
policy request is in flight. :class:`ChunkScheduler` owns that timeline:

* ``step`` is the global control step; each chunk starts at the step at
  which its observation was requested.
* A chunk is *live* for ``steps_to_execute`` (+ skipped) steps from its start.
* ``temporal_ensemble`` blends overlapping live chunks, and
  ``latency_compensation`` skips actions whose time already passed.

:class:`PolicyRequestGate` decides when the next observation is sent
(``policy_request.when_live_chunk`` and the fresh-camera rule).
"""

from cobotmagic_deployment.common.action_processing import exponential_temporal_ensemble
from cobotmagic_deployment.common.bridge_log import BridgeLog


def chunk_live_steps(chunk):
    steps = int(chunk.get('steps_to_execute', chunk['left'].shape[0]))
    skip = int(chunk.get('action_skip_steps', 0))
    return min(steps + skip, int(chunk['left'].shape[0]))


class ChunkScheduler:
    def __init__(self, ros_cfg, rate_hz, log=None):
        self.log = log or BridgeLog()
        self.rate_hz = rate_hz
        ensemble = ros_cfg.get('temporal_ensemble', {})
        self.ensemble_enabled = bool(ensemble.get('enabled', False))
        self.exp_decay = max(float(ensemble.get('exp_decay', 0.7)), 0.0)
        self.max_history_chunks = max(int(ensemble.get('max_history_chunks', 4)), 1)
        max_age = ensemble.get('max_candidate_age')
        self.max_candidate_age = None if max_age is None else max(int(max_age), 0)
        self.min_action_index = max(int(ensemble.get('min_action_index', 0)), 0)
        overlap = ensemble.get('overlap_steps')
        self.overlap_steps = None if overlap is None else max(int(overlap), 0)
        latency = ensemble.get('latency_compensation', {})
        self.latency_enabled = bool(latency.get('enabled', False))
        self.latency_mode = latency.get('mode', 'measured').lower()
        self.latency_fixed_steps = max(int(latency.get('fixed_steps', 0)), 0)
        max_steps = latency.get('max_steps')
        self.latency_max_steps = None if max_steps is None else max(int(max_steps), 0)
        self.initial_action_skip_steps = max(int(ros_cfg.get('initial_action_skip_steps', 0)), 0)

        self.history = []
        self.step = 0
        self.request_id = 0  # number of chunks accepted so far

    def log_configuration(self):
        if self.ensemble_enabled:
            self.log.info(
                "Exponential temporal ensemble enabled: "
                f"exp_decay={self.exp_decay:.3f} "
                f"max_history_chunks={self.max_history_chunks} "
                f"max_candidate_age={self.max_candidate_age} "
                f"min_action_index={self.min_action_index} "
                f"overlap_steps={self.overlap_steps} "
                f"latency_compensation={self.latency_enabled} "
                f"latency_mode={self.latency_mode} "
                f"fixed_steps={self.latency_fixed_steps} "
                f"max_steps={self.latency_max_steps}"
            )

    # --- timeline queries -------------------------------------------------

    def live_chunks(self, step=None):
        step = self.step if step is None else step
        return [c for c in self.history if 0 <= step - c['start_step'] < chunk_live_steps(c)]

    def select_chunk(self):
        """Chunk whose action is executed at the current step (``None`` if idle)."""
        live = self.live_chunks()
        if not live:
            return None
        if not self.ensemble_enabled or self.min_action_index <= 0:
            return live[-1]
        mature = [c for c in live if self.step - c['start_step'] >= self.min_action_index]
        return mature[-1] if mature else live[0]

    def remaining_steps(self, live=None):
        live = self.live_chunks() if live is None else live
        if not live:
            return 0
        newest = live[-1]
        return newest['start_step'] + chunk_live_steps(newest) - self.step

    # --- response handling --------------------------------------------------

    def begin_response(self, request_step, latency_sec):
        """Apply latency compensation / initial skip when a reply arrives.

        Returns the latency in steps. Runs before the reply is validated, so
        the timeline advances even if the reply is later dropped.
        """
        if self.latency_enabled:
            if self.latency_mode == 'fixed':
                latency_steps = self.latency_fixed_steps
            else:
                latency_steps = int(round(latency_sec * self.rate_hz))
            latency_steps = max(latency_steps, 0)
            if self.latency_max_steps is not None:
                latency_steps = min(latency_steps, self.latency_max_steps)
            # Late responses contain actions for times that have already passed.
            self.step = max(self.step, request_step + latency_steps)
        else:
            latency_steps = 0
        if self.initial_action_skip_steps > 0 and self.request_id == 0:
            self.step = max(self.step, request_step + self.initial_action_skip_steps)
        return latency_steps

    def skip_chunk_start(self, request_step, skip_steps):
        if skip_steps > 0:
            self.step = max(self.step, request_step + skip_steps)

    def next_request_id(self):
        self.request_id += 1
        return self.request_id

    def add_chunk(self, chunk):
        """Register a processed chunk (needs ``start_step``, ``left``, ``right``)."""
        chunk.setdefault('executed_steps', 0)
        self.history.append(chunk)
        self.history = self.history[-self.max_history_chunks:]

    # --- per-step execution -------------------------------------------------

    def ensemble(self, chunk):
        """Temporal-ensemble target for the current step or ``None``."""
        if not self.ensemble_enabled:
            return None
        if chunk['overlap_steps'] is not None and chunk['executed_steps'] >= chunk['overlap_steps']:
            return None
        left, right, count, weights = exponential_temporal_ensemble(
            self.history,
            self.step,
            chunk['request_id'],
            self.exp_decay,
            self.max_candidate_age,
            self.min_action_index,
        )
        if left is None or right is None:
            return None
        return left, right, count, weights

    def advance(self, chunk):
        """Mark the current action of ``chunk`` as executed and move one step."""
        chunk['executed_steps'] += 1
        self.step += 1
        self.history = [
            c for c in self.history if self.step < c['start_step'] + chunk_live_steps(c)
        ][-self.max_history_chunks:]


def stale_observations(obs_time, now, max_image_age_sec, max_joint_age_sec,
                       camera_keys=('front', 'left', 'right'), joint_keys=('jl', 'jr')):
    """Return ``[(key, age_or_None), ...]`` for observations that are too old."""
    stale = []
    for keys, limit in ((camera_keys, max_image_age_sec), (joint_keys, max_joint_age_sec)):
        if limit > 0.0:
            for key in keys:
                age = None if obs_time.get(key) is None else now - obs_time[key]
                if age is None or age > limit:
                    stale.append((key, age))
    return stale


class PolicyRequestGate:
    """Decide whether to send the next observation.

    ``decide`` returns one of ``'send'`` (all cameras refreshed since the last
    request), ``'force'`` (some refreshed and the live chunk is nearly used
    up), ``'wait_chunk'`` (``when_live_chunk: wait``) or ``'wait_cameras'``.
    """

    def __init__(self, ros_cfg, fallback_steps=6, log=None):
        log = log or BridgeLog()
        mode = str(ros_cfg.get('policy_request', {}).get('when_live_chunk', 'allow')).lower()
        if mode not in ('allow', 'wait'):
            log.warn("policy_request.when_live_chunk must be 'allow' or 'wait'; using 'allow'.")
            mode = 'allow'
        self.when_live_chunk = mode
        self.fallback_steps = fallback_steps
        self.last_camera_seq = None

    def fresh_camera_count(self, camera_seq):
        if self.last_camera_seq is None:
            return len(camera_seq)
        return sum(1 for seq, last in zip(camera_seq, self.last_camera_seq) if seq > last)

    def decide(self, camera_seq, has_live_chunk, remaining_steps):
        fresh = self.fresh_camera_count(camera_seq)
        if self.when_live_chunk == 'wait' and has_live_chunk:
            return 'wait_chunk', fresh
        if fresh == len(camera_seq):
            return 'send', fresh
        if (
            self.last_camera_seq is not None
            and fresh > 0
            and (not has_live_chunk or remaining_steps <= max(6, self.fallback_steps))
        ):
            return 'force', fresh
        return 'wait_cameras', fresh

    def mark_sent(self, camera_seq):
        self.last_camera_seq = tuple(camera_seq)
