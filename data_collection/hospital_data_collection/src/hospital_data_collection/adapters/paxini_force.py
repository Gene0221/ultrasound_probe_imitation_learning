from __future__ import annotations

from hospital_data_collection.adapters.controlled_process import ControlledProcessAdapter


class PaxiniForceAdapter(ControlledProcessAdapter):
    def initialize(self):
        status = super().initialize()
        if status.healthy and status.message:
            status.message = "Workspace detected; Paxini zero-calibration is configured."
        return status
