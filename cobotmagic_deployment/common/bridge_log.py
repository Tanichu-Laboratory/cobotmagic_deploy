"""Logging adapter shared by the ROS-independent bridge components.

Components log through a :class:`BridgeLog` so they can run under ROS
(``BridgeLog.for_rospy()``), plain Python logging (the default) or tests.
Throttling is keyed explicitly, so different messages never suppress each
other even though they are emitted from the same helper.
"""

import logging
import time


class BridgeLog:
    def __init__(self, info=None, warn=None, error=None, clock=time.monotonic):
        logger = logging.getLogger('cobotmagic_bridge')
        self._info = info or logger.info
        self._warn = warn or logger.warning
        self._error = error or logger.error
        self._clock = clock
        self._last = {}
        self._once = set()

    @classmethod
    def for_rospy(cls):
        import rospy

        return cls(rospy.loginfo, rospy.logwarn, rospy.logerr)

    def info(self, message):
        self._info(message)

    def warn(self, message):
        self._warn(message)

    def error(self, message):
        self._error(message)

    def _throttled(self, key, period):
        now = self._clock()
        last = self._last.get(key)
        if last is not None and now - last < period:
            return False
        self._last[key] = now
        return True

    def info_throttle(self, key, period, message):
        if self._throttled(key, period):
            self._info(message)

    def warn_throttle(self, key, period, message):
        if self._throttled(key, period):
            self._warn(message)

    def error_throttle(self, key, period, message):
        if self._throttled(key, period):
            self._error(message)

    def info_once(self, key, message):
        if key not in self._once:
            self._once.add(key)
            self._info(message)
