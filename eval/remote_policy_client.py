import json
import os
import tempfile

import robotmq
import torch
from eval_utils import obs_to_policy_input


class RemotePolicyClient:
    def __init__(
        self,
        policy_server_endpoint: str,
        timeout_s: int = 30,
        progress_file: str | None = None,
    ) -> None:
        self.client = robotmq.RMQClient("mikasa_eval_client", policy_server_endpoint)
        self.timeout_s = timeout_s
        # progress file for orchestrator to read eval step progress
        if progress_file is None:
            fd, self.progress_file = tempfile.mkstemp(prefix="mikasa_eval_progress_", suffix=".json")
            os.close(fd)
        else:
            self.progress_file = progress_file
        self.policy_config = self.get_policy_config()
        self.action_horizon = self.policy_config["workspace"]["model"]["action_length"]
        self.env_id = self.policy_config["workspace"]["train_dataset"]["name"]
        self.no_proprio: bool = int(self.policy_config["workspace"]["model"]["proprio_length"]) == 0

    def get_policy_config(self) -> dict:
        return robotmq.deserialize(
            self.client.request_with_data(
                "policy_config", 
                robotmq.serialize(True)
            )
        )

    def predict_action(self, obs: dict, num_envs: int, device: torch.device, no_proprio: bool | None = None) -> torch.Tensor:
        if no_proprio is None:
            no_proprio = self.no_proprio
        batch = obs_to_policy_input(obs, num_envs, device, no_proprio)
        send_dict = {}
        for k, v in batch.items():
            if k.endswith("camera"):
                send_dict[k] = (v * 255).byte().cpu().numpy()
            else:
                send_dict[k] = v.cpu().numpy()
        raw = self.client.request_with_data(
            "policy_inference", robotmq.serialize(send_dict), timeout_s=self.timeout_s
        )
        result = robotmq.deserialize(raw)
        if isinstance(result, str):
            raise RuntimeError(f"Policy server error: {result}")
        # action0_8d: delta/abs joint pos (action-space), robot0_8d: direct qpos (state-space)
        # which key the server returns depends on how the checkpoint was trained, not runtime mode
        output_key = "action0_8d" if "action0_8d" in result else "robot0_8d"
        return torch.from_numpy(result[output_key][:, :self.action_horizon, :]).to(device)

    def reset(self) -> None:
        self.client.request_with_data("policy_reset", robotmq.serialize(True))

    def report_progress(self, step: int, total: int) -> None:
        """Write eval progress to tmp file for orchestrator to read."""
        with open(self.progress_file, "w") as f:
            json.dump({"step": step, "total": total}, f)

    def report_results(self, results_dict: dict) -> None:
        self.client.put_data("rollout_results", robotmq.serialize(results_dict)) # log success rate
        self.client.request_with_data("done_rollout", robotmq.serialize(results_dict)) # tells server to unload

    def cleanup(self) -> None:
        """Remove tmp progress file."""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)