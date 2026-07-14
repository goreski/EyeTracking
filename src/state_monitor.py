import time
from dataclasses import dataclass


@dataclass
class DriverStateResult:
    state: str
    color: tuple[int, int, int]
    duration: float
    confirmed: bool


class DriverStateMonitor:
    """
    Converts noisy frame-by-frame observations into stable driver states.

    A state must persist for a configured amount of time before it becomes
    confirmed. This reduces false alarms caused by blinks, mirror checks,
    brief head movements, and landmark jitter.
    """

    def __init__(
        self,
        sideways_threshold=0.08,
        downward_threshold=-0.05,
        eye_closed_threshold=0.20,
        sideways_duration=1.5,
        downward_duration=1.5,
        eyes_closed_duration=1.2,
    ):
        self.sideways_threshold = sideways_threshold
        self.downward_threshold = downward_threshold
        self.eye_closed_threshold = eye_closed_threshold

        self.required_durations = {
            "Distracted (Looking Sideways)": sideways_duration,
            "Distracted (Looking Down)": downward_duration,
            "Drowsy / Eyes Closed": eyes_closed_duration,
            "Attentive": 0.0,
        }

        self.current_observation = "Attentive"
        self.observation_started_at = time.monotonic()
        self.confirmed_state = "Attentive"
        
        self.face_missing_started_at = None
        self.face_missing_duration = 1.0

    def reset(self):
        """
        Clears all temporal state.
        """
        self.current_observation = "Attentive"
        self.observation_started_at = time.monotonic()
        self.confirmed_state = "Attentive"

    def classify_frame(
        self,
        avg_ear,
        horizontal_deviation,
        vertical_deviation,
    ):
        """
        Produces the raw observation for one frame.

        Drowsiness is checked first because closed eyes should take priority
        over head-direction measurements.
        """
        if avg_ear < self.eye_closed_threshold:
            return "Drowsy / Eyes Closed"

        if abs(horizontal_deviation) > self.sideways_threshold:
            return "Distracted (Looking Sideways)"

        if vertical_deviation < self.downward_threshold:
            return "Distracted (Looking Down)"

        return "Attentive"

    def update(
        self,
        avg_ear,
        horizontal_deviation,
        vertical_deviation,
    ):
        """
        Updates the temporal state and returns a stable result.
        """
        now = time.monotonic()

        observation = self.classify_frame(
            avg_ear=avg_ear,
            horizontal_deviation=horizontal_deviation,
            vertical_deviation=vertical_deviation,
        )

        if observation != self.current_observation:
            self.current_observation = observation
            self.observation_started_at = now

        observation_duration = now - self.observation_started_at
        required_duration = self.required_durations[observation]

        confirmed = observation_duration >= required_duration

        if confirmed:
            self.confirmed_state = observation

        # Return to attentive immediately once the driver is attentive again.
        if observation == "Attentive":
            self.confirmed_state = "Attentive"
            confirmed = True

        color = self.get_state_color(self.confirmed_state)

        return DriverStateResult(
            state=self.confirmed_state,
            color=color,
            duration=observation_duration,
            confirmed=confirmed,
        )

    def update_face_missing(self):
        now = time.monotonic()

        if self.face_missing_started_at is None:
            self.face_missing_started_at = now

        duration = now - self.face_missing_started_at

        if duration >= self.face_missing_duration:
            self.confirmed_state = "Driver Face Unavailable"

        return DriverStateResult(
            state=self.confirmed_state,
            color=(0, 0, 255),
            duration=duration,
            confirmed=duration >= self.face_missing_duration,
        )


    def mark_face_detected(self):
        self.face_missing_started_at = None

    @staticmethod
    def get_state_color(state):
        if state == "Drowsy / Eyes Closed":
            return 0, 165, 255

        if state.startswith("Distracted"):
            return 0, 0, 255

        return 0, 255, 0