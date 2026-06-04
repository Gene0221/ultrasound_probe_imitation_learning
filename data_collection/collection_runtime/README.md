# collection_runtime

This workspace contains the shared session-based collection runtime used across
the `data_collection` tree.

It provides the neutral common layer for:

- single-module collection workspaces such as `imu_pose_grasping` and `6D_force_grasping`
- integrated collectors such as `hospital_data_collection`
- integrated collect-train flows such as `paxini26D_mapping`

## What Lives Here

- `launcher.py`: unified interactive control loop
- `config.py`: loader for launcher configs
- `state_machine.py`: runtime state transitions
- `session_manager.py`: session numbering and directory creation
- `metadata.py`: run/session metadata helpers
- `adapters/`: collector adapter implementations

## Design Role

This package exists so collection modules do not depend on the hospital-specific
workspace. Both single-module and integrated collection entrypoints import this
shared runtime instead.

When a single-module workspace starts, it uses that module's own
`config/default.yaml` directly. When an integrated workspace starts, any extra
fields declared under its top-level `modules.<module_name>` entry are merged
into the called submodule config before launch.
