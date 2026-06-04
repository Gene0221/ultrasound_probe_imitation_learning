# visual_pose_tracking

This workspace performs dual-camera AprilTag pose-delta tracking under the
unified session controller.

## Main Entrypoint

```powershell
cd C:\Users\zhj80\OneDrive\Desktop\Master Course Material\research\data_collection\visual_pose_tracking
python main.py
```

## Control Logic

- startup initializes the visual tracker
- `Enter` starts a new `session_xxxx`
- `Enter` again pauses and closes the current session
- `Enter` again resumes into the next session
- `q` stops the whole run

## Output

Each session is written under:

```text
output/session_xxxx/
  tag_pose_deltas.jsonl
  tracking_summary.json
```

The main payload is the timestamped relative motion stream used by downstream
calibration and learning pipelines.

## Important Files

- [main.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/visual_pose_tracking/main.py)
- [config/session_collection.yaml](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/visual_pose_tracking/config/session_collection.yaml)
- [config/apriltag_tracking.yaml](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/visual_pose_tracking/config/apriltag_tracking.yaml)
- [scripts/track_apriltag_pose_deltas.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/visual_pose_tracking/scripts/track_apriltag_pose_deltas.py)
- [scripts/test_apriltag_pose_single_camera.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/visual_pose_tracking/scripts/test_apriltag_pose_single_camera.py)
- [scripts/test_apriltag_delta_consistency_dual_camera.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/visual_pose_tracking/scripts/test_apriltag_delta_consistency_dual_camera.py)

Use `main.py` for normal collection. The scripts under `scripts/` are the
tracking implementation and focused test utilities.
