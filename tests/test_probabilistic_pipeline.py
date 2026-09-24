"""Unit and Functional Tests for Probabilistic Machine Learning Pipeline.

Verifies:
1. Directional Statistics & von Mises-Fisher (vMF) spherical attention zones on S^2.
2. Adaptive Conformal Prediction (ACI) for ocular state coverage and ambiguity.
3. Conformal-aware Bayesian cross-eye fusion.
4. GazeCalibrator spherical unit vector projection.
5. Probabilistic driver state monitoring with cockpit AOIs.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spherical_gaze import (
    SphericalAttentionEstimator,
    vmf_spherical_cap_probability,
    DEFAULT_COCKPIT_AOIS,
)
from conformal_prediction import ConformalEyePredictor
from models.eye_quality import EyeQualityResult
from eye_pair_fusion import fuse_eye_pair, eyes_closed_probability
from metrics import GazeCalibrator
from state_monitor import DriverStateMonitor


def test_vmf_spherical_cap_integration():
    """Verify vMF spherical cap probabilities are bounded, consistent, and monotonic."""
    mu = np.array([0.0, 0.0, 1.0])
    cap_center = np.array([0.0, 0.0, 1.0])
    radius = np.deg2rad(15.0)

    # High concentration should concentrate probability inside cap
    p_high = vmf_spherical_cap_probability(mu, cap_center, radius, kappa=30.0)
    assert 0.0 < p_high <= 1.0, f"Expected probability in (0, 1], got {p_high}"

    # Target far away from cap center should yield near-zero probability
    far_mu = np.array([1.0, 0.0, 0.0])
    p_far = vmf_spherical_cap_probability(far_mu, cap_center, radius, kappa=30.0)
    assert p_far < p_high, "Off-center sample should have strictly lower probability than on-center"
    assert p_far >= 0.0, f"Probability must be non-negative, got {p_far}"
    print("[PASS] test_vmf_spherical_cap_integration")


def test_spherical_attention_aois():
    """Verify that gaze vectors directed at specific cockpit zones are correctly classified."""
    estimator = SphericalAttentionEstimator(base_concentration=24.0)

    # 1. Straight ahead -> Windshield / Road Ahead
    road_vector = np.array([0.0, 0.0, 1.0])
    result_road = estimator.estimate(road_vector)
    assert result_road.active_aoi == "Windshield / Road Ahead", f"Expected Road, got {result_road.active_aoi}"
    assert result_road.is_safe is True
    assert result_road.active_probability > 0.60

    # 2. Downward pitch -> Lap / Mobile Phone
    lap_vector = np.array([0.0, 0.48, 0.87])
    result_lap = estimator.estimate(lap_vector)
    assert result_lap.active_aoi == "Lap / Mobile Phone", f"Expected Lap, got {result_lap.active_aoi}"
    assert result_lap.is_safe is False
    assert result_lap.active_aoi_category == "distraction"

    # 3. Glance towards left mirror
    mirror_l_vector = np.array([-0.42, -0.05, 0.90])
    result_mirror = estimator.estimate(mirror_l_vector)
    assert result_mirror.active_aoi == "Left Side Mirror", f"Expected Left Mirror, got {result_mirror.active_aoi}"
    assert result_mirror.is_safe is True
    assert result_mirror.active_aoi_category == "mirror"

    # 4. Probabilities sum to 1.0 (within float tolerance)
    total_p = sum(result_road.zone_probabilities.values())
    assert abs(total_p - 1.0) < 1e-4, f"Probabilities must sum to 1.0, got {total_p}"
    print("[PASS] test_spherical_attention_aois")


def test_conformal_prediction_coverage_and_ambiguity():
    """Verify conformal prediction set properties: singletons vs ambiguity."""
    predictor = ConformalEyePredictor(alpha=0.05, default_quantile=0.88)

    # 1. Clear, unambiguous eye open
    sharp_open = EyeQualityResult(probabilities=np.array([0.94, 0.03, 0.02, 0.01]))
    conf_open = predictor.predict(sharp_open)
    assert conf_open is not None
    assert conf_open.is_singleton is True
    assert conf_open.is_ambiguous is False
    assert conf_open.prediction_set == ("open_visible",)

    # 2. Ambiguous eye state (squinting / partial eyelid droop)
    ambiguous = EyeQualityResult(probabilities=np.array([0.48, 0.46, 0.04, 0.02]))
    conf_ambig = predictor.predict(ambiguous)
    assert conf_ambig is not None
    assert conf_ambig.is_ambiguous is True
    assert "open_visible" in conf_ambig.prediction_set
    assert "closed" in conf_ambig.prediction_set

    # 3. ACI Online Adapt
    q_before = predictor.q_hat
    predictor.adapt_online(was_correct=False)  # Coverage error -> quantile should decrease
    assert predictor.q_hat != q_before
    print("[PASS] test_conformal_prediction_coverage_and_ambiguity")


def test_conformal_aware_bayesian_fusion():
    """Verify Bayesian fusion uses the other eye when one eye is conformally ambiguous with occlusion."""
    predictor = ConformalEyePredictor(alpha=0.05, default_quantile=0.88)

    # Left eye: Ambiguous between open and occluded
    left_result = EyeQualityResult(probabilities=np.array([0.45, 0.10, 0.40, 0.05]))
    left_conf = predictor.predict(left_result)

    # Right eye: Confident singleton open
    right_result = EyeQualityResult(probabilities=np.array([0.92, 0.04, 0.02, 0.02]))
    right_conf = predictor.predict(right_result)

    left_fused, right_fused = fuse_eye_pair(
        left_result,
        right_result,
        left_conformal=left_conf,
        right_conformal=right_conf,
    )

    assert left_fused is not None
    assert left_fused.inferred_from_other_eye is True
    assert right_fused is not None
    assert right_fused.inferred_from_other_eye is False

    closed_p = eyes_closed_probability(left_fused, right_fused)
    assert closed_p is not None
    assert closed_p < 0.15  # Both inferred and direct reads confirm eye open
    print("[PASS] test_conformal_aware_bayesian_fusion")


def test_gaze_calibrator_spherical_vector():
    """Verify GazeCalibrator constructs normalized S^2 vectors."""
    calibrator = GazeCalibrator()
    dummy_baseline = np.zeros(7, dtype=np.float64)
    dummy_baseline[4] = 0.0  # forward_x
    dummy_baseline[5] = 0.0  # forward_y
    dummy_baseline[6] = 1.0  # forward_z
    calibrator.baseline_features = dummy_baseline
    calibrator.is_calibrated = True

    # At neutral, should produce [0, 0, 1]
    rel_neutral = np.zeros(7, dtype=np.float64)
    s2_vec = calibrator.get_spherical_gaze_vector(rel_neutral)
    norm = np.linalg.norm(s2_vec)
    assert abs(norm - 1.0) < 1e-6
    assert abs(s2_vec[2] - 1.0) < 1e-6

    # Deviated vector
    rel_deviated = np.zeros(7, dtype=np.float64)
    rel_deviated[4] = 0.3
    rel_deviated[5] = -0.2
    s2_dev = calibrator.get_spherical_gaze_vector(rel_deviated)
    assert abs(np.linalg.norm(s2_dev) - 1.0) < 1e-6
    assert s2_dev[0] > 0.0
    print("[PASS] test_gaze_calibrator_spherical_vector")


def test_state_monitor_with_aoi():
    """Verify DriverStateMonitor integrates AOI categories properly."""
    monitor = DriverStateMonitor()
    estimator = SphericalAttentionEstimator()

    # 1. Windshield
    road_s2 = np.array([0.0, 0.0, 1.0])
    road_attn = estimator.estimate(road_s2)
    res_road = monitor.update(0.0, 0.0, eyes_closed_probability=0.05, spherical_attention=road_attn)
    assert res_road.head_state == "Attentive"
    assert res_road.active_aoi == "Windshield / Road Ahead"

    # 2. Mirror check
    mirror_s2 = np.array([-0.42, -0.05, 0.90])
    mirror_attn = estimator.estimate(mirror_s2)
    obs_mirror = monitor.classify_head_direction(0.4, 0.0, spherical_attention=mirror_attn)
    assert "Checking Mirror" in obs_mirror

    # 3. Lap / Phone distraction
    lap_s2 = np.array([0.0, 0.48, 0.87])
    lap_attn = estimator.estimate(lap_s2)
    obs_lap = monitor.classify_head_direction(0.0, 0.5, spherical_attention=lap_attn)
    assert "Distracted" in obs_lap
    assert "Lap / Mobile Phone" in obs_lap
    print("[PASS] test_state_monitor_with_aoi")


if __name__ == "__main__":
    test_vmf_spherical_cap_integration()
    test_spherical_attention_aois()
    test_conformal_prediction_coverage_and_ambiguity()
    test_conformal_aware_bayesian_fusion()
    test_gaze_calibrator_spherical_vector()
    test_state_monitor_with_aoi()
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")
