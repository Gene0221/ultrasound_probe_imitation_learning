# Online Demo Capture

This folder contains scripts for recording the robot-side online inference demo
frames used in the paper table.

## Run With The Realtime Controller

Recommended one-command launch:

```bash
cd imitation_learning
./demo_capture/launch_demo_capture.bash --build --trial-id trial_001
```

This keeps the same startup order as `launch_realtime.bash`:

1. Run guided force initialization with `scripts/initialize_calibration.py`.
2. Start the Python policy sender with demo frame capture enabled.
3. Start the C++ Franka controller.
4. Wait for the contact force reference and initial EE/probe orientation records.
5. Press Enter once to save `frames/initial_frame.png`, send the ready signal,
   and start online inference.
6. After each executed action chunk, the controller asks for the next inference;
   the sender then reads the current ultrasound frame, saves it, runs inference
   on that same frame, and sends the next action chunk.

Stop the scan manually with `Ctrl+C`. The script then saves
`frames/final_frame.png` and closes the trial logs.

If you are debugging the sender and controller separately, run the C++ controller
as usual in one terminal, then run:

```bash
cd imitation_learning
python demo_capture/online_demo_capture.py --config config/infer.yaml --trial-id trial_001
```

In this manual mode the script pauses twice: once to save the initial frame, and
once to start inference.

## Output Layout

Each run creates a trial folder:

```text
demo_capture/trials/trial_001/
  frames/
    initial_frame.png
    step_0001.png
    step_0002.png
    final_frame.png
  frame_summary.csv
  inference_log.jsonl
  metadata.json
```

Use `--initial-position` and `--final-position` if you want text labels or pose
notes written into `metadata.json` for the paper table.

For a dry run using saved ultrasound images:

```bash
python demo_capture/online_demo_capture.py --image-dir /path/to/images --trial-id dry_run_001
```
