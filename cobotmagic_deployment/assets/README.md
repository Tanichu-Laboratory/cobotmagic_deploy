# Piper IK model

`piper_description_cobotmagic_joint5_pitch0.urdf` is an OpenWAM deployment copy of
`/workspace/project/piper_description_cobotmagic_local_20260529.urdf`.
The only change is joint5 origin rpy: `1.5708 -0.087266 0` -> `1.5708 0 0`.
The source is retained. Joint5 is ROS index 4; this correction is independent of
the ROS-to-URDF sign mapping for indices 3 and 5.

Validation and limitations: `../../docs/openwam/OPENWAM_JOINT_MAPPING_RECHECK.md`.
The deployment uses this file for numerical IK; mesh references are unchanged.
