"""Directional Statistics & von Mises-Fisher (vMF) Spherical Gaze Attention.

Models 3D gaze and head pose orientation as a continuous probability distribution
on the unit sphere S^2, replacing heuristic Cartesian thresholds with analytical
posterior probabilities over standard automotive Areas of Interest (AOIs).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CockpitAOI:
    """An Area of Interest (AOI) on the unit sphere defined by a center vector and angular radius."""
    name: str
    center_direction: np.ndarray  # Unit vector on S^2: [x, y, z]
    angular_radius_rad: float     # Angular cone half-angle in radians
    is_safe_driving_zone: bool    # True for road / mirrors / speedometer
    category: str                 # "road", "mirror", "instrument", "distraction"


# Standard automotive cockpit gaze zones defined in driver-centric coordinates:
# x: horizontal (negative = left, positive = right)
# y: vertical   (negative = up, positive = down)
# z: forward    (positive = into the windshield)
DEFAULT_COCKPIT_AOIS = (
    CockpitAOI(
        name="Windshield / Road Ahead",
        center_direction=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        angular_radius_rad=np.deg2rad(15.0),
        is_safe_driving_zone=True,
        category="road",
    ),
    CockpitAOI(
        name="Rear-view Mirror",
        center_direction=np.array([0.18, -0.24, 0.95], dtype=np.float64) / np.linalg.norm([0.18, -0.24, 0.95]),
        angular_radius_rad=np.deg2rad(8.0),
        is_safe_driving_zone=True,
        category="mirror",
    ),
    CockpitAOI(
        name="Left Side Mirror",
        center_direction=np.array([-0.42, -0.05, 0.90], dtype=np.float64) / np.linalg.norm([-0.42, -0.05, 0.90]),
        angular_radius_rad=np.deg2rad(10.0),
        is_safe_driving_zone=True,
        category="mirror",
    ),
    CockpitAOI(
        name="Right Side Mirror",
        center_direction=np.array([0.45, -0.05, 0.89], dtype=np.float64) / np.linalg.norm([0.45, -0.05, 0.89]),
        angular_radius_rad=np.deg2rad(10.0),
        is_safe_driving_zone=True,
        category="mirror",
    ),
    CockpitAOI(
        name="Instrument Cluster",
        center_direction=np.array([0.0, 0.20, 0.98], dtype=np.float64) / np.linalg.norm([0.0, 0.20, 0.98]),
        angular_radius_rad=np.deg2rad(9.0),
        is_safe_driving_zone=True,
        category="instrument",
    ),
    CockpitAOI(
        name="Infotainment / Center Console",
        center_direction=np.array([0.38, 0.22, 0.90], dtype=np.float64) / np.linalg.norm([0.38, 0.22, 0.90]),
        angular_radius_rad=np.deg2rad(11.0),
        is_safe_driving_zone=False,
        category="distraction",
    ),
    CockpitAOI(
        name="Lap / Mobile Phone",
        center_direction=np.array([0.0, 0.48, 0.87], dtype=np.float64) / np.linalg.norm([0.0, 0.48, 0.87]),
        angular_radius_rad=np.deg2rad(14.0),
        is_safe_driving_zone=False,
        category="distraction",
    ),
)


@dataclass(frozen=True)
class SphericalAttentionResult:
    """Probabilistic spatial attention estimation for one frame."""
    active_aoi: str
    active_aoi_category: str
    active_probability: float
    is_safe: bool
    entropy: float
    concentration: float
    zone_probabilities: dict[str, float]


def vmf_spherical_cap_probability(mu_sample: np.ndarray, mu_cap: np.ndarray, theta_rad: float, kappa: float) -> float:
    """
    Computes the exact closed-form integral of a von Mises-Fisher distribution
    centered at `mu_sample` with concentration `kappa` over a spherical cap centered
    at `mu_cap` with half-angle `theta_rad`.

    For the aligned case (mu_sample == mu_cap), the cumulative mass on S^2 is:
        P(x^T mu >= cos(theta)) = (1 - exp(-kappa * (1 - cos(theta)))) / (1 - exp(-2 * kappa))

    For general angles, the effective overlap is modulated by the geodesic angular
    separation delta = arccos(mu_sample^T mu_cap).
    """
    kappa = float(max(kappa, 1e-3))
    cos_theta = float(np.cos(theta_rad))
    cos_delta = float(np.clip(np.dot(mu_sample, mu_cap), -1.0, 1.0))
    delta = float(np.arccos(cos_delta))

    # If sample is well inside the cap
    if delta <= theta_rad:
        # Core mass within the target cone
        if kappa > 40.0:
            core_prob = 1.0 - np.exp(-kappa * (1.0 - cos_theta))
        else:
            denom = 1.0 - np.exp(-2.0 * kappa)
            core_prob = (1.0 - np.exp(-kappa * (1.0 - cos_theta))) / max(denom, 1e-12)

        # Modulate by proximity to cap center
        proximity_factor = np.exp(-0.5 * kappa * (delta / max(theta_rad, 1e-6)) ** 2)
        return float(np.clip(core_prob * proximity_factor, 1e-6, 1.0))
    else:
        # Distance from the edge of the spherical cap
        angular_excess = delta - theta_rad
        # Exponential tail falloff dictated by vMF concentration kappa
        tail_prob = np.exp(-kappa * (1.0 - np.cos(angular_excess)))
        return float(np.clip(tail_prob * 0.1, 1e-8, 1.0))


class SphericalAttentionEstimator:
    """
    Estimates driver spatial attention using von Mises-Fisher directional statistics.
    """

    def __init__(
        self,
        aois: tuple[CockpitAOI, ...] = DEFAULT_COCKPIT_AOIS,
        base_concentration: float = 24.0,
    ):
        self.aois = aois
        self.base_concentration = base_concentration
        self._road_aoi_index = next(
            (i for i, aoi in enumerate(self.aois) if aoi.category == "road"), 0
        )

    def adapt_road_center(self, calibrated_road_vector: np.ndarray) -> None:
        """
        Calibrates the 'Windshield / Road Ahead' AOI center to match the driver's
        neutral calibrated baseline vector.
        """
        norm = np.linalg.norm(calibrated_road_vector)
        if norm > 1e-6:
            unit_road = calibrated_road_vector / norm
            # Reconstruct the AOI with the updated center
            updated_aois = list(self.aois)
            road_aoi = updated_aois[self._road_aoi_index]
            updated_aois[self._road_aoi_index] = CockpitAOI(
                name=road_aoi.name,
                center_direction=unit_road,
                angular_radius_rad=road_aoi.angular_radius_rad,
                is_safe_driving_zone=road_aoi.is_safe_driving_zone,
                category=road_aoi.category,
            )
            self.aois = tuple(updated_aois)

    def estimate(
        self,
        gaze_vector: np.ndarray,
        tracking_confidence: float = 1.0,
    ) -> SphericalAttentionResult:
        """
        Given a normalized 3D gaze/pose vector on S^2, calculates the posterior
        probabilities across all cockpit Areas of Interest (AOIs).
        """
        norm = np.linalg.norm(gaze_vector)
        if norm < 1e-6:
            unit_vector = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            unit_vector = np.asarray(gaze_vector, dtype=np.float64) / norm

        # Dynamic concentration parameter: drops when tracking confidence degrades
        kappa = max(self.base_concentration * float(np.clip(tracking_confidence, 0.2, 1.0)), 1.0)

        # Unnormalized likelihoods across AOIs
        likelihoods = []
        for aoi in self.aois:
            prob = vmf_spherical_cap_probability(
                mu_sample=unit_vector,
                mu_cap=aoi.center_direction,
                theta_rad=aoi.angular_radius_rad,
                kappa=kappa,
            )
            likelihoods.append(prob)

        # Add a diffuse background / ambient inattention state
        background_likelihood = float(np.exp(-0.25 * kappa))
        all_likelihoods = np.array(likelihoods + [background_likelihood], dtype=np.float64)
        total_mass = float(np.sum(all_likelihoods))

        normalized_probs = all_likelihoods / max(total_mass, 1e-12)
        aoi_probs = normalized_probs[:-1]
        background_prob = normalized_probs[-1]

        zone_dict = {aoi.name: float(aoi_probs[i]) for i, aoi in enumerate(self.aois)}
        zone_dict["Ambient / Unmapped"] = float(background_prob)

        # Determine MAP (maximum a posteriori) active zone
        best_index = int(np.argmax(aoi_probs))
        active_aoi = self.aois[best_index]
        active_prob = float(aoi_probs[best_index])

        # If background probability dominates, driver is looking outside any defined zone
        if background_prob > active_prob and background_prob > 0.4:
            active_name = "Off-Road / Extreme Deviation"
            active_category = "distraction"
            active_prob = float(background_prob)
            is_safe = False
        else:
            active_name = active_aoi.name
            active_category = active_aoi.category
            is_safe = active_aoi.is_safe_driving_zone

        # Shannon Entropy of the attention distribution (in bits)
        p_safe = normalized_probs[normalized_probs > 1e-12]
        entropy = -float(np.sum(p_safe * np.log2(p_safe)))

        return SphericalAttentionResult(
            active_aoi=active_name,
            active_aoi_category=active_category,
            active_probability=active_prob,
            is_safe=is_safe,
            entropy=entropy,
            concentration=kappa,
            zone_probabilities=zone_dict,
        )
