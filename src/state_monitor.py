import time
from dataclasses import dataclass


@dataclass
class SingleStateResult:
    state: str
    color: tuple[int, int, int]
    duration: float
    confirmed: bool


@dataclass
class DriverStateResult:
    """Head-direction and eye-state are independent -- a driver can be
    looking sideways with eyes open, looking forward with eyes closed, or
    any other combination, so each gets its own state/color/duration rather
    than being collapsed into one label.
    """

    head_state: str
    head_color: tuple[int, int, int]
    head_duration: float
    head_confirmed: bool

    eye_state: str
    eye_color: tuple[int, int, int]
    eye_duration: float
    eye_confirmed: bool


class _DebouncedObservation:
    """Tracks one noisy per-frame signal, requiring it to persist for a
    configured duration before it's treated as confirmed. This reduces false
    alarms caused by blinks, mirror checks, brief head movements, and
    landmark jitter.
    """

    def __init__(self, required_durations: dict[str, float], default_state: str):
        self.required_durations = required_durations
        self.default_state = default_state
        self.current_observation = default_state
        self.observation_started_at = time.monotonic()
        self.confirmed_state = default_state

    def reset(self):
        self.current_observation = self.default_state
        self.observation_started_at = time.monotonic()
        self.confirmed_state = self.default_state

    def update(self, observation: str) -> tuple[str, float, bool]:
        now = time.monotonic()

        if observation != self.current_observation:
            self.current_observation = observation
            self.observation_started_at = now

        duration = now - self.observation_started_at
        required_duration = self.required_durations[observation]
        confirmed = duration >= required_duration

        if confirmed:
            self.confirmed_state = observation

        # Return to the default state immediately once the signal clears.
        if observation == self.default_state:
            self.confirmed_state = self.default_state
            confirmed = True

        return self.confirmed_state, duration, confirmed


class DriverStateMonitor:
    """
    Converts noisy frame-by-frame observations into stable driver states.

    Head-direction (sideways/down) and eye-state (open/closed) are tracked
    as two independent debounced signals -- see `_DebouncedObservation` --
    since a driver's head direction and eye state can combine in any way.
    """

    def __init__(
        self,
        sideways_threshold=0.08,
        downward_threshold=0.12,
        upward_threshold=-0.12,
        eyes_closed_probability_threshold=0.5,
        eye_closed_threshold=0.20,
        sideways_duration=1.5,
        downward_duration=1.5,
        upward_duration=1.5,
        eyes_closed_duration=1.2,
    ):
        self.sideways_threshold = sideways_threshold
        # vertical_deviation is read from FORWARD_Y_INDEX (see metrics.py):
        # positive = looking down, negative = looking up.
        self.downward_threshold = downward_threshold
        self.upward_threshold = upward_threshold
        self.eyes_closed_probability_threshold = eyes_closed_probability_threshold
        # Geometric EAR fallback, used only when no ML probability is available
        # (eye-quality model off, or no eye crop this frame).
        self.eye_closed_threshold = eye_closed_threshold

        self._head_tracker = _DebouncedObservation(
            required_durations={
                "Distracted (Looking Sideways)": sideways_duration,
                "Distracted (Looking Down)": downward_duration,
                "Distracted (Looking Up)": upward_duration,
                "Attentive": 0.0,
            },
            default_state="Attentive",
        )
        self._eye_tracker = _DebouncedObservation(
            required_durations={
                "Drowsy / Eyes Closed": eyes_closed_duration,
                "Eyes Open": 0.0,
            },
            default_state="Eyes Open",
        )

        self.face_missing_started_at = None
        self.face_missing_duration = 1.0
        self._face_missing_confirmed_state = "Attentive"

    @property
    def head_observation(self) -> str:
        return self._head_tracker.current_observation

    @property
    def eye_observation(self) -> str:
        return self._eye_tracker.current_observation

    def reset(self):
        """
        Clears all temporal state.
        """
        self._head_tracker.reset()
        self._eye_tracker.reset()

    def _eyes_closed(self, eyes_closed_probability, avg_ear):
        """
        Decides eye-closure from the ML classifier's probability when it's
        available, falling back to the geometric EAR threshold only when it
        isn't (model off, or no eye crop this frame).
        """
        if eyes_closed_probability is not None:
            return eyes_closed_probability >= self.eyes_closed_probability_threshold

        if avg_ear is not None:
            return avg_ear < self.eye_closed_threshold

        return False

    def is_horizontal_neutral(self, horizontal_deviation) -> bool:
        """True when this frame's sideways reading is within the attentive range."""
        return abs(horizontal_deviation) <= self.sideways_threshold

    def is_vertical_neutral(self, vertical_deviation) -> bool:
        """True when this frame's up/down reading is within the attentive range."""
        return self.upward_threshold <= vertical_deviation <= self.downward_threshold

    def classify_head_direction(self, horizontal_deviation, vertical_deviation):
        """Produces the raw head-direction observation for one frame.

        vertical_deviation is positive when looking down, negative when
        looking up (see FORWARD_Y_INDEX in metrics.py).
        """
        if abs(horizontal_deviation) > self.sideways_threshold:
            return "Distracted (Looking Sideways)"

        if vertical_deviation > self.downward_threshold:
            return "Distracted (Looking Down)"

        if vertical_deviation < self.upward_threshold:
            return "Distracted (Looking Up)"

        return "Attentive"

    def classify_eye_state(self, eyes_closed_probability=None, avg_ear=None):
        """Produces the raw eye-state observation for one frame."""
        if self._eyes_closed(eyes_closed_probability, avg_ear):
            return "Drowsy / Eyes Closed"

        return "Eyes Open"

    def update(
        self,
        horizontal_deviation,
        vertical_deviation,
        eyes_closed_probability=None,
        avg_ear=None,
    ):
        """
        Updates both temporal tracks and returns their independent results.
        """
        head_observation = self.classify_head_direction(horizontal_deviation, vertical_deviation)
        eye_observation = self.classify_eye_state(eyes_closed_probability, avg_ear)

        head_state, head_duration, head_confirmed = self._head_tracker.update(head_observation)
        eye_state, eye_duration, eye_confirmed = self._eye_tracker.update(eye_observation)

        return DriverStateResult(
            head_state=head_state,
            head_color=self.get_state_color(head_state),
            head_duration=head_duration,
            head_confirmed=head_confirmed,
            eye_state=eye_state,
            eye_color=self.get_state_color(eye_state),
            eye_duration=eye_duration,
            eye_confirmed=eye_confirmed,
        )

    def update_face_missing(self):
        now = time.monotonic()

        if self.face_missing_started_at is None:
            self.face_missing_started_at = now

        duration = now - self.face_missing_started_at

        if duration >= self.face_missing_duration:
            self._face_missing_confirmed_state = "Driver Face Unavailable"

        return SingleStateResult(
            state=self._face_missing_confirmed_state,
            color=(0, 0, 255),
            duration=duration,
            confirmed=duration >= self.face_missing_duration,
        )

    def mark_face_detected(self):
        self.face_missing_started_at = None
        self._face_missing_confirmed_state = "Attentive"

    @staticmethod
    def get_state_color(state):
        if state == "Drowsy / Eyes Closed":
            return 0, 165, 255

        if state.startswith("Distracted"):
            return 0, 0, 255

        return 0, 255, 0
