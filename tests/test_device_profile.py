import json
import tempfile
import unittest
from pathlib import Path

from playground.nubzuki.benchmark import (
    ACCELERATOR_CANDIDATES,
    CPU_CANDIDATES,
    SATURATION_TOLERANCE,
    select_candidate,
)
from playground.nubzuki.ppo_config import training_config


def candidate(num_envs, rate):
    return {"num_envs": num_envs, "environment_steps_per_second": rate}


class CandidateSelectionTests(unittest.TestCase):
    def test_saturated_cpu_picks_the_smallest_count_that_is_as_fast(self):
        """The measured Mac curve: flat from 512 up, so 512 is the right pick."""
        measured = [
            candidate(256, 2007.8), candidate(512, 2204.7),
            candidate(1024, 2219.7), candidate(2048, 2218.9),
        ]
        chosen, fastest = select_candidate(measured)
        self.assertEqual(chosen["num_envs"], 512)
        self.assertEqual(fastest["num_envs"], 1024)

    def test_still_scaling_picks_the_largest(self):
        """An accelerator that has not saturated should use every environment."""
        measured = [
            candidate(1024, 50_000), candidate(2048, 99_000),
            candidate(4096, 195_000), candidate(8192, 380_000),
        ]
        chosen, fastest = select_candidate(measured)
        self.assertEqual(chosen["num_envs"], 8192)
        self.assertEqual(chosen, fastest)

    def test_a_single_candidate_is_chosen(self):
        chosen, fastest = select_candidate([candidate(1024, 1234.0)])
        self.assertEqual(chosen["num_envs"], 1024)
        self.assertEqual(chosen, fastest)

    def test_tolerance_boundary_is_inclusive(self):
        best = 1000.0
        measured = [candidate(512, best * (1 - SATURATION_TOLERANCE)), candidate(1024, best)]
        self.assertEqual(select_candidate(measured)[0]["num_envs"], 512)

    def test_just_outside_the_tolerance_is_rejected(self):
        best = 1000.0
        measured = [candidate(512, best * (1 - SATURATION_TOLERANCE) - 1), candidate(1024, best)]
        self.assertEqual(select_candidate(measured)[0]["num_envs"], 1024)


class PresetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.profile = Path(self._tmp.name) / "profile.json"
        self.profile.write_text(json.dumps({"backend": "gpu", "num_envs": 8192}))

    def tearDown(self):
        self._tmp.cleanup()

    def test_macbook_is_an_alias_for_profile(self):
        target = 10_000_000
        self.assertEqual(
            training_config("macbook", target, self.profile),
            training_config("profile", target, self.profile),
        )

    def test_a_gpu_profile_is_accepted(self):
        """The profile used to be rejected unless its backend was cpu."""
        self.assertEqual(training_config("profile", 10_000_000, self.profile)["num_envs"], 8192)

    def test_official_matches_upstream_without_a_profile(self):
        missing = Path(self._tmp.name) / "absent.json"
        self.assertEqual(training_config("official", 10_000_000, missing)["num_envs"], 8192)

    def test_explicit_num_envs_overrides_the_preset(self):
        config = training_config("profile", 10_000_000, self.profile, num_envs=2048)
        self.assertEqual(config["num_envs"], 2048)

    def test_a_count_brax_cannot_split_is_refused(self):
        for bad in (3000, 5000, 16384):
            with self.subTest(num_envs=bad), self.assertRaises(ValueError):
                training_config("official", 10_000_000, self.profile, num_envs=bad)

    def test_every_benchmark_candidate_is_usable(self):
        for good in (256, 512, 1024, 2048, 4096, 8192):
            with self.subTest(num_envs=good):
                config = training_config("official", 10_000_000, self.profile, num_envs=good)
                self.assertEqual(config["num_envs"], good)

    def test_every_benchmark_candidate_is_one_training_can_accept(self):
        """A count the benchmark can select must not be refused by training."""
        for num_envs in CPU_CANDIDATES + ACCELERATOR_CANDIDATES:
            with self.subTest(num_envs=num_envs):
                config = training_config(
                    "official", 10_000_000, self.profile, num_envs=num_envs
                )
                self.assertEqual(config["num_envs"], num_envs)

    def test_missing_profile_names_the_command_to_run(self):
        missing = Path(self._tmp.name) / "absent.json"
        with self.assertRaises(FileNotFoundError) as caught:
            training_config("profile", 10_000_000, missing)
        self.assertIn("benchmark", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
