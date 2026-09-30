"""Per-arm binary gripper control with hysteresis and confirmed transitions."""
import copy
import numpy as np


class GripperHysteresis:
    def __init__(self, cfg):
        def pair(name):
            x = np.asarray(cfg[name], dtype=float)
            if x.shape != (2,) or not np.isfinite(x).all():
                raise ValueError(name + ' must contain two finite values')
            return x
        self.closed = pair('closed')
        self.opened = pair('open')
        self.close_threshold = pair('close_threshold_normalized')
        self.open_threshold = pair('open_threshold_normalized')
        if np.any(self.opened <= self.closed):
            raise ValueError('open must be greater than closed')
        if np.any(self.close_threshold < 0) or np.any(self.open_threshold > 1) or np.any(self.close_threshold >= self.open_threshold):
            raise ValueError('require 0 <= close threshold < open threshold <= 1')
        self.confirm_steps = cfg.get('confirm_steps', 2)
        if isinstance(self.confirm_steps, bool) or not isinstance(self.confirm_steps, int) or self.confirm_steps < 1:
            raise ValueError('confirm_steps must be a positive integer')
        relative = cfg.get('request_relative', {})
        self.request_relative = bool(relative.get('enabled', False))
        self.close_margin = float(relative.get('close_margin_normalized', .18))
        self.open_margin = float(relative.get('open_margin_normalized', .15))
        self.max_close = float(relative.get('max_close_threshold_normalized', .82))
        self.max_open = float(relative.get('max_open_threshold_normalized', .90))
        if self.request_relative:
            values = [self.close_margin, self.open_margin, self.max_close, self.max_open]
            if (not np.isfinite(values).all() or not 0 < self.close_margin < 1
                    or not 0 < self.open_margin < 1 or not 0 <= self.max_close < self.max_open <= 1
                    or np.any(self.close_threshold > self.max_close)
                    or np.any(self.open_threshold > self.max_open)):
                raise ValueError('invalid request-relative gripper thresholds')
        self.is_open = None
        self.count = np.zeros(2, dtype=int)

    def propose(self, commands, measured, request_opening=None):
        """Return a candidate state; caller adopts it ONLY after publishing.

        commands are physical openings with no server-side gain. State survives
        request/chunk boundaries. The dead band retains the last binary state.
        """
        candidate = copy.deepcopy(self)
        commands = np.asarray(commands, dtype=float)
        measured = np.asarray(measured, dtype=float)
        if commands.shape != (2,) or measured.shape != (2,) or not np.isfinite([commands, measured]).all():
            raise ValueError('gripper inputs must be finite pairs')
        if candidate.is_open is None:
            candidate.is_open = (measured - self.closed) / (self.opened - self.closed) >= 0.5
        normalized = np.clip((commands - self.closed) / (self.opened - self.closed), 0, 1)
        close_threshold = self.close_threshold
        open_threshold = self.open_threshold
        reference = None
        if self.request_relative:
            request_opening = np.asarray(request_opening, dtype=float)
            if request_opening.shape != (2,) or not np.isfinite(request_opening).all():
                raise ValueError('request-relative control requires finite request_opening pair')
            reference = np.clip((request_opening-self.closed)/(self.opened-self.closed), 0, 1)
            close_threshold = np.maximum(self.close_threshold, np.minimum(self.max_close, reference-self.close_margin))
            open_threshold = np.maximum(self.open_threshold, np.minimum(self.max_open, reference+self.open_margin))
        previous = candidate.is_open.copy()
        trigger = np.where(candidate.is_open, normalized <= close_threshold, normalized >= open_threshold)
        candidate.count = np.where(trigger, candidate.count + 1, 0)
        switch = candidate.count >= self.confirm_steps
        candidate.is_open[switch] = ~candidate.is_open[switch]
        candidate.count[switch] = 0
        values = np.where(candidate.is_open, self.opened, self.closed)
        diagnostics = {'normalized_input': normalized.tolist(), 'previous_open': previous.tolist(),
                       'output_open': candidate.is_open.tolist(), 'switched': switch.tolist(),
                       'pending_count': candidate.count.tolist(),
                       'request_normalized': None if reference is None else reference.tolist(),
                       'close_threshold_normalized': close_threshold.tolist(),
                       'open_threshold_normalized': open_threshold.tolist()}
        return values, candidate, diagnostics

    def hold(self, arm_index, published_value, diagnostics):
        """Re-sync one arm to a gripper value published instead of the proposal.

        Used when IK holds the last safe command: the proposed transition was
        not published, so the binary state follows ``published_value`` and any
        pending confirmation count is cleared. ``diagnostics`` is updated in place.
        """
        fraction = ((published_value - self.closed[arm_index]) /
                    (self.opened[arm_index] - self.closed[arm_index]))
        self.is_open[arm_index] = fraction >= 0.5
        self.count[arm_index] = 0
        diagnostics['output_open'][arm_index] = bool(fraction >= 0.5)
        diagnostics['switched'][arm_index] = False
        diagnostics['pending_count'][arm_index] = 0
