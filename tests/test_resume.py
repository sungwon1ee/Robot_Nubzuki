import json
import tempfile
import unittest
from pathlib import Path

from playground.nubzuki.ppo_config import eval_count, training_config
from playground.nubzuki.resume import (
    checkpoint_step,
    remaining_timesteps,
    resolve_restore,
)


def write_checkpoint(root: Path, step: int, with_metadata: bool = True) -> Path:
    artifact_dir = root / "checkpoints" / f"step_{step:012d}"
    params = artifact_dir / "params"
    params.mkdir(parents=True, exist_ok=True)
    if with_metadata:
        (artifact_dir / "policy.json").write_text(json.dumps({"checkpoint_step": step}))
    (root / "latest.json").write_text(
        json.dumps({"checkpoint_step": step, "checkpoint": str(params)})
    )
    return params


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "standing"

    def tearDown(self):
        self._tmp.cleanup()

    def test_step_comes_from_metadata(self):
        params = write_checkpoint(self.root, 15_000_000)
        self.assertEqual(checkpoint_step(params), 15_000_000)

    def test_step_falls_back_to_directory_name(self):
        params = write_checkpoint(self.root, 20_000_000, with_metadata=False)
        self.assertEqual(checkpoint_step(params), 20_000_000)

    def test_unlabelled_checkpoint_is_refused(self):
        params = self.root / "params"
        params.mkdir(parents=True)
        with self.assertRaises(ValueError):
            checkpoint_step(params)

    def test_auto_finds_the_latest_checkpoint(self):
        params = write_checkpoint(self.root, 25_000_000)
        path, offset = resolve_restore("auto", self.root)
        self.assertEqual(path, params)
        self.assertEqual(offset, 25_000_000)

    def test_auto_starts_from_scratch_without_a_previous_run(self):
        self.root.mkdir(parents=True)
        path, offset = resolve_restore("auto", self.root)
        self.assertIsNone(path)
        self.assertEqual(offset, 0)

    def test_fresh_run_ignores_existing_checkpoints(self):
        write_checkpoint(self.root, 25_000_000)
        path, offset = resolve_restore(None, self.root)
        self.assertIsNone(path)
        self.assertEqual(offset, 0)

    def test_missing_checkpoint_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            resolve_restore(str(self.root / "checkpoints" / "step_000000000001" / "params"), self.root)

    def test_explicit_offset_overrides_inference(self):
        write_checkpoint(self.root, 25_000_000)
        _, offset = resolve_restore("auto", self.root, step_offset=7)
        self.assertEqual(offset, 7)

    def test_remaining_completes_the_original_target(self):
        self.assertEqual(remaining_timesteps(150_000_000, 25_000_000), 125_000_000)

    def test_finished_run_does_not_restart(self):
        with self.assertRaises(SystemExit):
            remaining_timesteps(150_000_000, 150_000_000)


class CheckpointNamingTests(unittest.TestCase):
    """The regression this change exists for."""

    @staticmethod
    def names(step_offset, steps_this_run):
        # Mirrors BaseRunner.policy_params_fn.
        return [
            f"step_{step_offset + step:012d}"
            for step in range(5_000_000, steps_this_run + 1, 5_000_000)
        ]

    def test_resumed_run_does_not_rewrite_earlier_checkpoints(self):
        first = self.names(0, 15_000_000)
        resumed = self.names(15_000_000, 10_000_000)
        self.assertEqual(first[-1], "step_000015000000")
        self.assertEqual(resumed[0], "step_000020000000")
        self.assertFalse(set(first) & set(resumed))

    def test_without_the_offset_the_names_would_collide(self):
        first = self.names(0, 15_000_000)
        naive = self.names(0, 10_000_000)
        self.assertTrue(set(first) & set(naive))


class CheckpointCadenceTests(unittest.TestCase):
    def interval(self, num_timesteps, checkpoint_every):
        return num_timesteps // (eval_count(num_timesteps, checkpoint_every) - 1)

    def test_interval_never_exceeds_the_request(self):
        for every in (250_000, 1_000_000, 5_000_000):
            self.assertLessEqual(self.interval(150_000_000, every), every)

    def test_shorter_interval_means_more_checkpoints(self):
        default = eval_count(150_000_000, 5_000_000)
        frequent = eval_count(150_000_000, 1_000_000)
        self.assertEqual(default, 31)
        self.assertEqual(frequent, 151)

    def test_resumed_run_keeps_the_cadence(self):
        # 15M already done, 135M to go: still ~1M between checkpoints.
        self.assertLessEqual(self.interval(135_000_000, 1_000_000), 1_000_000)

    def test_non_positive_interval_is_refused(self):
        with self.assertRaises(ValueError):
            eval_count(150_000_000, 0)

    def test_config_carries_the_cadence_and_eval_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps({"backend": "cpu", "num_envs": 512}))
            config = training_config(
                "macbook", 150_000_000, path,
                checkpoint_every=1_000_000, num_eval_envs=32,
            )
            self.assertEqual(config["num_evals"], 151)
            self.assertEqual(config["num_eval_envs"], 32)
            self.assertEqual(config["num_envs"], 512)

    def test_smoke_preset_ignores_the_cadence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            config = training_config(
                "smoke", 1024, path, checkpoint_every=1_000_000, num_eval_envs=32
            )
            self.assertEqual(config["num_evals"], 1)
            self.assertNotIn("num_eval_envs", config)


if __name__ == "__main__":
    unittest.main()
