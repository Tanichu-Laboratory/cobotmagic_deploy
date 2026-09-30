"""CSV action logs, rich request/chunk snapshots and rollout datasets for the bridge."""

import csv
import json
import os
import time

import numpy as np
import rospy


def vector_to_json(values):
    if values is None:
        return ''
    return json.dumps(np.asarray(values, dtype=np.float32).tolist(), separators=(',', ':'))



def norm_first_six(values):
    if values is None:
        return ''
    arr = np.asarray(values, dtype=np.float32)
    return f'{float(np.linalg.norm(arr[:6])):.6f}'



def make_action_logger(cfg, rate_hz):
    log_cfg = cfg['ros'].get('action_log', {})
    if not bool(log_cfg.get('enabled', False)):
        return None, None

    log_dir = os.path.expanduser(log_cfg.get('dir', 'logs/action_commands'))
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(log_dir, f'action_commands_{stamp}.csv')
    flush_every = max(int(log_cfg.get('flush_every_rows', 1)), 1)
    rich_cfg = log_cfg.get('rich', {})
    rich_enabled = bool(rich_cfg.get('enabled', False))
    save_request_images = rich_enabled and bool(rich_cfg.get('save_request_images', True))
    save_action_chunks = rich_enabled and bool(rich_cfg.get('save_action_chunks', True))
    request_snapshot_every_n = max(int(rich_cfg.get('request_snapshot_every_n', 1)), 1)
    request_snapshot_dir = None
    if save_request_images:
        request_snapshot_dir = os.path.join(log_dir, f'request_snapshots_{stamp}')
        os.makedirs(request_snapshot_dir, exist_ok=True)
    action_chunk_dir = None
    if save_action_chunks:
        action_chunk_dir = os.path.join(log_dir, f'action_chunks_{stamp}')
        os.makedirs(action_chunk_dir, exist_ok=True)

    columns = [
        'wall_time',
        'ros_time',
        'request_id',
        'tau',
        'global_action_step',
        'chunk_start_step',
        'action_index',
        'policy_latency_sec',
        'policy_latency_steps',
        'chunk_size',
        'received_chunk_size',
        'steps_to_execute',
        'discarded_steps',
        'rate_hz',
        'command_publish_rate_hz',
        'command_publish_substeps',
        'task_prompt',
        'action_mode',
        'chunk_interpolation_enabled',
        'chunk_interpolation_mode',
        'chunk_interpolation_factor',
        'request_obs_seq',
        'request_obs_age_sec',
        'request_snapshot_dir',
        'action_chunk_path',
        'rollout_sample_dir',
        'rollout_action_path',
        'request_fresh_camera_count',
        'request_fresh_camera_forced',
        'delta_clip_enabled',
        'delta_clip_reference',
        'command_delta_deadband_enabled',
        'first_action_delta_scale_enabled',
        'first_action_delta_scale_coefficient',
        'first_action_delta_scale_include_gripper',
        'first_action_delta_scale_applied',
        'initial_pose_delta_override_enabled',
        'initial_pose_delta_override_arms',
        'initial_pose_delta_override_include_gripper',
        'temporal_ensemble_count',
        'temporal_ensemble_weights',
        'current_left',
        'current_right',
        'request_measured_left',
        'request_measured_right',
        'publish_current_left',
        'publish_current_right',
        'clip_reference_left',
        'clip_reference_right',
        'command_left_before',
        'command_right_before',
        'raw_left',
        'raw_right',
        'ensembled_left',
        'ensembled_right',
        'filtered_left',
        'filtered_right',
        'target_left',
        'target_right',
        'ik_seed_joint_left',
        'ik_seed_joint_right',
        'ik_diagnostics_left',
        'ik_diagnostics_right',
        'ik_joint_left',
        'ik_joint_right',
        'ik_left_position_error_m',
        'ik_right_position_error_m',
        'ik_left_orientation_error_rad',
        'ik_right_orientation_error_rad',
        'raw_delta_left',
        'raw_delta_right',
        'applied_delta_left',
        'applied_delta_right',
        'reference_delta_left',
        'reference_delta_right',
        'tracking_error_before_left',
        'tracking_error_before_right',
        'tracking_error_after_left',
        'tracking_error_after_right',
        'clip_residual_left',
        'clip_residual_right',
        'command_delta_deadband_residual_left',
        'command_delta_deadband_residual_right',
        'raw_vs_publish_norm_left',
        'raw_vs_publish_norm_right',
        'ensembled_vs_publish_norm_left',
        'ensembled_vs_publish_norm_right',
        'target_vs_publish_norm_left',
        'target_vs_publish_norm_right',
        'reference_delta_norm_left',
        'reference_delta_norm_right',
        'applied_delta_norm_left',
        'applied_delta_norm_right',
        'tracking_error_before_norm_left',
        'tracking_error_before_norm_right',
        'tracking_error_after_norm_left',
        'tracking_error_after_norm_right',
        'clip_residual_norm_left',
        'clip_residual_norm_right',
        'command_delta_deadband_residual_norm_left',
        'command_delta_deadband_residual_norm_right',
        'publish_joint_age_left_sec',
        'publish_joint_age_right_sec',
        'policy_gripper_input_mode',
        'left_gripper_request_measured',
        'left_gripper_request_policy',
        'left_gripper_request_commanded',
        'right_gripper_request_measured',
        'right_gripper_request_policy',
        'right_gripper_request_commanded',
        'left_gripper_publish',
        'left_gripper_raw',
        'left_gripper_ensembled',
        'left_gripper_target',
        'right_gripper_publish',
        'right_gripper_raw',
        'right_gripper_ensembled',
        'right_gripper_target',
        'gripper_hysteresis',
        'base_vel',
    ]
    f = open(path, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    rospy.loginfo(f"Action command logging enabled: {path} (flush_every_rows={flush_every})")
    return {
        'file': f,
        'writer': writer,
        'path': path,
        'flush_every': flush_every,
        'rows_since_flush': 0,
        'rich_enabled': rich_enabled,
        'save_request_images': save_request_images,
        'save_action_chunks': save_action_chunks,
        'request_snapshot_every_n': request_snapshot_every_n,
        'request_snapshot_dir': request_snapshot_dir,
        'action_chunk_dir': action_chunk_dir,
        'request_snapshot_count': 0,
    }, path



def write_action_log(logger, row):
    if logger is None:
        return
    logger['writer'].writerow(row)
    logger['rows_since_flush'] += 1
    if logger['rows_since_flush'] >= logger['flush_every']:
        logger['file'].flush()
        logger['rows_since_flush'] = 0



def save_request_snapshot(logger, request_step, pkt, header, fresh_camera_count, forced_fresh_fallback):
    if logger is None or not logger.get('save_request_images'):
        return ''
    logger['request_snapshot_count'] += 1
    if (logger['request_snapshot_count'] - 1) % logger['request_snapshot_every_n'] != 0:
        return ''

    root = logger.get('request_snapshot_dir')
    if not root:
        return ''
    name = f"step_{int(request_step):06d}_req_{logger['request_snapshot_count']:06d}"
    out_dir = os.path.join(root, name)
    os.makedirs(out_dir, exist_ok=True)
    for key in ('front', 'left', 'right'):
        with open(os.path.join(out_dir, f'{key}.jpg'), 'wb') as f:
            f.write(pkt[key])
    now = time.monotonic()
    meta = {
        'wall_time': time.time(),
        'request_step': int(request_step),
        'fresh_camera_count': int(fresh_camera_count),
        'forced_fresh_fallback': bool(forced_fresh_fallback),
        'header': header,
        'obs_seq': dict(pkt.get('obs_seq', {})),
        'obs_age_sec': {
            key: None if value is None else now - value
            for key, value in pkt.get('obs_time', {}).items()
        },
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return out_dir



def save_action_chunk(logger, request_id, chunk):
    if logger is None or not logger.get('save_action_chunks'):
        return ''
    root = logger.get('action_chunk_dir')
    if not root:
        return ''
    path = os.path.join(root, f"request_{int(request_id):06d}.npz")
    meta = {
        'request_id': int(request_id),
        'start_step': int(chunk['start_step']),
        'received_chunk_size': int(chunk['received_chunk_size']),
        'processed_chunk_size': int(chunk['left'].shape[0]),
        'steps_to_execute': int(chunk['steps_to_execute']),
        'overlap_steps': None if chunk['overlap_steps'] is None else int(chunk['overlap_steps']),
        'policy_latency_sec': float(chunk['policy_latency_sec']),
        'policy_latency_steps': int(chunk['policy_latency_steps']),
        'request_fresh_camera_count': int(chunk['request_fresh_camera_count']),
        'request_fresh_camera_forced': bool(chunk['request_fresh_camera_forced']),
        'request_obs_seq': dict(chunk['request_obs_seq']),
        'request_snapshot_dir': chunk.get('request_snapshot_dir', ''),
        'policy_gripper_input': dict(chunk.get('policy_gripper_input', {})),
        'terminal_filter_enabled': bool(chunk.get('terminal_filter_enabled', False)),
        'terminal_endpoint_index': int(chunk.get('terminal_endpoint_index', -1)),
        'terminal_delta_left': np.asarray(chunk.get('terminal_delta_left', []), dtype=np.float32).tolist(),
        'terminal_delta_right': np.asarray(chunk.get('terminal_delta_right', []), dtype=np.float32).tolist(),
        'terminal_cancel_left': np.asarray(chunk.get('terminal_cancel_left', []), dtype=bool).tolist(),
        'terminal_cancel_right': np.asarray(chunk.get('terminal_cancel_right', []), dtype=bool).tolist(),
        'initial_pose_delta_override_enabled': bool(
            chunk.get('initial_pose_delta_override_enabled', False)
        ),
        'initial_pose_delta_override_arms': list(
            chunk.get('initial_pose_delta_override_arms', [])
        ),
        'initial_pose_delta_override_include_gripper': bool(
            chunk.get('initial_pose_delta_override_include_gripper', False)
        ),
        'first_action_delta_scale_enabled': bool(
            chunk.get('first_action_delta_scale_enabled', False)
        ),
        'first_action_delta_scale_coefficient': float(
            chunk.get('first_action_delta_scale_coefficient', 1.0)
        ),
        'first_action_delta_scale_include_gripper': bool(
            chunk.get('first_action_delta_scale_include_gripper', False)
        ),
        'initial_bridge_factor': int(chunk.get('initial_bridge_factor', 1)),
        'initial_bridge_added_steps': int(
            chunk.get('initial_bridge_added_steps', 0)
        ),
        'chunk_lowpass_enabled': bool(chunk.get('chunk_lowpass_enabled', False)),
        'chunk_lowpass_cutoff_hz': float(
            chunk.get('chunk_lowpass_cutoff_hz', 0.0)
        ),
        'chunk_lowpass_sample_rate_hz': float(
            chunk.get('chunk_lowpass_sample_rate_hz', 0.0)
        ),
        'chunk_lowpass_order': int(chunk.get('chunk_lowpass_order', 0)),
        'chunk_lowpass_preserve_endpoints': bool(
            chunk.get('chunk_lowpass_preserve_endpoints', False)
        ),
        'chunk_monotonic_enabled': bool(
            chunk.get('chunk_monotonic_enabled', False)
        ),
        'chunk_monotonic_arms': list(chunk.get('chunk_monotonic_arms', [])),
        'chunk_monotonic_strength': float(
            chunk.get('chunk_monotonic_strength', 0.0)
        ),
        'chunk_monotonic_min_terminal_delta': float(
            chunk.get('chunk_monotonic_min_terminal_delta', 0.0)
        ),
    }
    np.savez_compressed(
        path,
        processed_left=chunk['left'],
        processed_right=chunk['right'],
        received_left=chunk.get('received_left', chunk['left']),
        received_right=chunk.get('received_right', chunk['right']),
        lowpass_left=chunk.get('lowpass_left', chunk['left']),
        lowpass_right=chunk.get('lowpass_right', chunk['right']),
        monotonic_left=chunk.get('monotonic_left', chunk['left']),
        monotonic_right=chunk.get('monotonic_right', chunk['right']),
        model_raw_left=np.empty((0, 0), dtype=np.float32) if chunk.get('model_raw_left') is None else chunk['model_raw_left'],
        model_raw_right=np.empty((0, 0), dtype=np.float32) if chunk.get('model_raw_right') is None else chunk['model_raw_right'],
        vel=np.empty((0, 0), dtype=np.float32) if chunk['vel'] is None else chunk['vel'],
        request_current_left=chunk['request_current_left'],
        request_current_right=chunk['request_current_right'],
        request_measured_left=chunk.get('request_measured_left', chunk['request_current_left']),
        request_measured_right=chunk.get('request_measured_right', chunk['request_current_right']),
        terminal_delta_left=np.asarray(chunk.get('terminal_delta_left', []), dtype=np.float32),
        terminal_delta_right=np.asarray(chunk.get('terminal_delta_right', []), dtype=np.float32),
        terminal_cancel_left=np.asarray(chunk.get('terminal_cancel_left', []), dtype=bool),
        terminal_cancel_right=np.asarray(chunk.get('terminal_cancel_right', []), dtype=bool),
        meta=json.dumps(meta, separators=(',', ':')),
    )
    return path



def make_rollout_dataset_logger(cfg, rate_hz):
    rollout_cfg = cfg['ros'].get('rollout_dataset', {})
    if not bool(rollout_cfg.get('enabled', False)):
        return None

    root = os.path.expanduser(rollout_cfg.get('dir', '/workspace/project/cobotmagic_datasets'))
    os.makedirs(root, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    run_name = rollout_cfg.get('run_name') or f'rollout_{stamp}'
    episode_dir = os.path.join(root, run_name)
    suffix = 1
    while os.path.exists(episode_dir):
        episode_dir = os.path.join(root, f'{run_name}_{suffix:02d}')
        suffix += 1
    os.makedirs(episode_dir, exist_ok=False)
    dataset_format = str(rollout_cfg.get('format', 'hdf5')).lower()
    if dataset_format not in ('hdf5', 'directory'):
        rospy.logwarn(f"Unsupported rollout_dataset.format={dataset_format!r}; using 'hdf5'.")
        dataset_format = 'hdf5'
    samples_dir = os.path.join(episode_dir, 'samples')
    if dataset_format == 'directory':
        os.makedirs(samples_dir, exist_ok=False)

    meta = {
        'created_wall_time': time.time(),
        'created_stamp': stamp,
        'rate_hz': int(rate_hz),
        'task_prompt': cfg.get('task_prompt', ''),
        'policy_backend': cfg.get('policy_backend', ''),
        'openvla': cfg.get('openvla', {}),
        'ros': {
            'action_mode': cfg['ros'].get('action_mode'),
            'open_loop_steps': cfg['ros'].get('open_loop_steps'),
            'rate_hz': cfg['ros'].get('rate_hz'),
            'command_publish': cfg['ros'].get('command_publish', {}),
            'chunk_interpolation': cfg['ros'].get('chunk_interpolation', {}),
            'delta_clip': cfg['ros'].get('delta_clip', {}),
            'action_filter': cfg['ros'].get('action_filter', {}),
            'temporal_ensemble': cfg['ros'].get('temporal_ensemble', {}),
        },
        'format': {
            'type': dataset_format,
            'hdf5': 'episode.hdf5 with /samples/sample_XXXXXX groups',
            'directory': 'samples/sample_XXXXXX with jpg/json/npz files',
            'raw_action_definition': 'raw_left_chunk/raw_right_chunk are policy server outputs before bridge filters, clipping, thresholding, ensemble, and command blocking.',
        },
        'outcome': 'unknown',
    }
    with open(os.path.join(episode_dir, 'episode_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    h5_file = None
    h5_path = ''
    if dataset_format == 'hdf5':
        try:
            import h5py

            h5_path = os.path.join(episode_dir, 'episode.hdf5')
            h5_file = h5py.File(h5_path, 'w')
            h5_file.attrs['metadata_json'] = json.dumps(meta, separators=(',', ':'))
            h5_file.attrs['created_wall_time'] = meta['created_wall_time']
            h5_file.attrs['task_prompt'] = meta['task_prompt']
            h5_file.attrs['outcome'] = meta['outcome']
            h5_file.create_group('samples')
            h5_file.flush()
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn(f"Failed to create rollout HDF5 file; falling back to directory format: {exc}")
            dataset_format = 'directory'
            samples_dir = os.path.join(episode_dir, 'samples')
            os.makedirs(samples_dir, exist_ok=True)

    rospy.loginfo(f"Rollout dataset logging enabled: {episode_dir} format={dataset_format}")
    return {
        'episode_dir': episode_dir,
        'samples_dir': samples_dir,
        'format': dataset_format,
        'h5_file': h5_file,
        'h5_path': h5_path,
        'sample_count': 0,
        'save_images': bool(rollout_cfg.get('save_images', True)),
    }



def h5_write_dataset(group, name, data, **kwargs):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, **kwargs)



def h5_write_json(group, name, payload):
    h5_write_dataset(group, name, np.bytes_(json.dumps(payload, separators=(',', ':'))))



def h5_write_jpeg(group, name, payload):
    h5_write_dataset(group, name, np.frombuffer(payload, dtype=np.uint8), compression='gzip')



def save_rollout_observation(logger, request_step, pkt, header, fresh_camera_count, forced_fresh_fallback):
    if logger is None:
        return ''

    logger['sample_count'] += 1
    sample_name = f"sample_{logger['sample_count']:06d}"

    now = time.monotonic()
    obs_age_sec = {
        key: None if value is None else now - value
        for key, value in pkt.get('obs_time', {}).items()
    }
    meta = {
        'sample_id': logger['sample_count'],
        'wall_time': time.time(),
        'request_step': int(request_step),
        'task_prompt': pkt['task_prompt'],
        'fresh_camera_count': int(fresh_camera_count),
        'forced_fresh_fallback': bool(forced_fresh_fallback),
        'header': header,
        'obs_seq': dict(pkt.get('obs_seq', {})),
        'obs_age_sec': obs_age_sec,
    }

    if logger.get('format') == 'hdf5' and logger.get('h5_file') is not None:
        h5_file = logger['h5_file']
        group_path = f"samples/{sample_name}"
        sample_group = h5_file.create_group(group_path)
        sample_group.attrs['sample_id'] = logger['sample_count']
        sample_group.attrs['request_step'] = int(request_step)
        sample_group.attrs['wall_time'] = meta['wall_time']
        sample_group.attrs['task_prompt'] = pkt['task_prompt']
        sample_group.attrs['metadata_json'] = json.dumps(meta, separators=(',', ':'))
        obs_group = sample_group.create_group('observation')
        h5_write_dataset(obs_group, 'jleft', np.asarray(pkt['jleft'], dtype=np.float32))
        h5_write_dataset(obs_group, 'jright', np.asarray(pkt['jright'], dtype=np.float32))
        h5_write_dataset(
            obs_group,
            'odom',
            np.empty((0,), dtype=np.float32) if pkt.get('odom') is None else np.asarray(pkt['odom'], dtype=np.float32),
        )
        h5_write_json(obs_group, 'metadata_json', meta)
        h5_write_json(obs_group, 'obs_seq_json', pkt.get('obs_seq', {}))
        h5_write_json(obs_group, 'obs_age_sec_json', obs_age_sec)
        if logger.get('save_images', True):
            img_group = obs_group.create_group('images')
            for key in ('front', 'left', 'right'):
                h5_write_jpeg(img_group, f'{key}_jpg', pkt[key])
        h5_file.flush()
        return f"{logger['h5_path']}::/{group_path}"

    sample_dir = os.path.join(logger['samples_dir'], sample_name)
    os.makedirs(sample_dir, exist_ok=False)
    if logger.get('save_images', True):
        for key in ('front', 'left', 'right'):
            with open(os.path.join(sample_dir, f'{key}.jpg'), 'wb') as f:
                f.write(pkt[key])
    with open(os.path.join(sample_dir, 'observation.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    np.savez_compressed(
        os.path.join(sample_dir, 'observation.npz'),
        jleft=np.asarray(pkt['jleft'], dtype=np.float32),
        jright=np.asarray(pkt['jright'], dtype=np.float32),
        odom=np.empty((0,), dtype=np.float32) if pkt.get('odom') is None else np.asarray(pkt['odom'], dtype=np.float32),
        obs_seq=json.dumps(pkt.get('obs_seq', {}), separators=(',', ':')),
        obs_age_sec=json.dumps(obs_age_sec, separators=(',', ':')),
    )
    return sample_dir



def save_rollout_action(logger, sample_dir, request_id, chunk):
    if logger is None or not sample_dir:
        return ''

    meta = {
        'request_id': int(request_id),
        'start_step': int(chunk['start_step']),
        'received_chunk_size': int(chunk['received_chunk_size']),
        'processed_chunk_size': int(chunk['left'].shape[0]),
        'steps_to_execute': int(chunk['steps_to_execute']),
        'overlap_steps': None if chunk['overlap_steps'] is None else int(chunk['overlap_steps']),
        'policy_latency_sec': float(chunk['policy_latency_sec']),
        'policy_latency_steps': int(chunk['policy_latency_steps']),
        'request_fresh_camera_count': int(chunk['request_fresh_camera_count']),
        'request_fresh_camera_forced': bool(chunk['request_fresh_camera_forced']),
        'raw_action_definition': (
            'raw_left_chunk/raw_right_chunk are the policy server outputs as received by the bridge, '
            'before bridge-side interpolation, filters, clipping, thresholding, temporal ensemble, '
            'right-arm blocking, and command publish logic.'
        ),
        'model_raw_action_definition': (
            'model_raw_left_chunk/model_raw_right_chunk are the direct OpenVLA action-head outputs '
            'before server-side action_delta_gripper_abs postprocessing when the server provides them; '
            'empty arrays mean the server did not provide model raw action frames.'
        ),
    }

    if logger.get('format') == 'hdf5' and logger.get('h5_file') is not None:
        marker = '::/'
        if marker not in sample_dir:
            return ''
        group_path = sample_dir.split(marker, 1)[1].lstrip('/')
        h5_file = logger['h5_file']
        sample_group = h5_file[group_path]
        action_group = sample_group.create_group('actions')
        action_group.attrs['request_id'] = int(request_id)
        action_group.attrs['metadata_json'] = json.dumps(meta, separators=(',', ':'))
        h5_write_dataset(action_group, 'raw_left_chunk', chunk.get('received_left', chunk['left']), compression='gzip')
        h5_write_dataset(action_group, 'raw_right_chunk', chunk.get('received_right', chunk['right']), compression='gzip')
        h5_write_dataset(
            action_group,
            'model_raw_left_chunk',
            np.empty((0, 0), dtype=np.float32) if chunk.get('model_raw_left') is None else chunk['model_raw_left'],
            compression='gzip',
        )
        h5_write_dataset(
            action_group,
            'model_raw_right_chunk',
            np.empty((0, 0), dtype=np.float32) if chunk.get('model_raw_right') is None else chunk['model_raw_right'],
            compression='gzip',
        )
        h5_write_dataset(action_group, 'processed_left_chunk', chunk['left'], compression='gzip')
        h5_write_dataset(action_group, 'processed_right_chunk', chunk['right'], compression='gzip')
        h5_write_dataset(
            action_group,
            'vel',
            np.empty((0, 0), dtype=np.float32) if chunk['vel'] is None else chunk['vel'],
            compression='gzip',
        )
        h5_write_dataset(action_group, 'request_current_left', chunk['request_current_left'])
        h5_write_dataset(action_group, 'request_current_right', chunk['request_current_right'])
        h5_write_json(action_group, 'metadata_json', meta)
        h5_file.flush()
        return f"{logger['h5_path']}::/{group_path}/actions"

    action_path = os.path.join(sample_dir, 'actions.npz')
    with open(os.path.join(sample_dir, 'action_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    np.savez_compressed(
        action_path,
        raw_left_chunk=chunk.get('received_left', chunk['left']),
        raw_right_chunk=chunk.get('received_right', chunk['right']),
        model_raw_left_chunk=np.empty((0, 0), dtype=np.float32) if chunk.get('model_raw_left') is None else chunk['model_raw_left'],
        model_raw_right_chunk=np.empty((0, 0), dtype=np.float32) if chunk.get('model_raw_right') is None else chunk['model_raw_right'],
        processed_left_chunk=chunk['left'],
        processed_right_chunk=chunk['right'],
        vel=np.empty((0, 0), dtype=np.float32) if chunk['vel'] is None else chunk['vel'],
        request_current_left=chunk['request_current_left'],
        request_current_right=chunk['request_current_right'],
        meta=json.dumps(meta, separators=(',', ':')),
    )
    return action_path

