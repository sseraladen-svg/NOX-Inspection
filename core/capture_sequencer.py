"""
core/capture_sequencer.py

Software-side sequencing of the multi-station, multi-camera capture flow.
Station count, cameras-per-station, and reorientation steps all come from
config — adding a station or changing the reorientation sequence for a new
product needs no code change here.

This module defines the sequence and aggregates per-cell results; it does
not talk to physical camera/actuator hardware (that integration point is
left as a hook — see CaptureSequencer.capture_station).
"""


class CaptureSequencer:
    def __init__(self, config: dict, preprocessor, detector, decision_engine):
        self.stations = config["stations"]
        self.cameras_per_station = config["cameras_per_station"]
        self.reorientation_steps = config["reorientation_steps"]

        self.preprocessor = preprocessor
        self.detector = detector
        self.decision_engine = decision_engine

    def capture_station(self, station_id: int, camera_id: int):
        """
        Hook for hardware integration. Must return a raw frame (np.ndarray)
        from the given station/camera. Raise NotImplementedError until wired
        to real camera hardware — do not silently return fake data.
        """
        raise NotImplementedError(
            f"capture_station not wired to hardware yet "
            f"(station={station_id}, camera={camera_id})"
        )

    def trigger_reorientation(self, step: str):
        """
        Hook for the flip/rotate actuator. Raise until wired to hardware.
        """
        raise NotImplementedError(f"trigger_reorientation not wired to hardware yet (step={step})")

    def run_full_cycle(self, cell_id: str):
        """
        Runs the full station-1 -> reorientation -> station-2 -> ... sequence
        for one product unit, aggregating a single overall PASS/FAIL.

        Returns:
            {
                "cell_id": str,
                "overall_result": "PASS" | "FAIL",
                "station_results": [ {station_id, camera_id, ...decision_engine output} ],
            }
        """
        station_results = []

        for station_id in range(1, self.stations + 1):
            for camera_id in range(1, self.cameras_per_station + 1):
                frame = self.capture_station(station_id, camera_id)
                processed = self.preprocessor.process(frame)
                _, detections = self.detector.detect(processed)
                result = self.decision_engine.evaluate(
                    detections,
                    station_id=station_id,
                    camera_id=camera_id,
                )
                station_results.append(result)

            # reorientation happens between stations, not after the last one
            if station_id < self.stations and self.reorientation_steps:
                for step in self.reorientation_steps:
                    self.trigger_reorientation(step)

        overall_result = "FAIL" if any(r["result"] == "FAIL" for r in station_results) else "PASS"

        return {
            "cell_id": cell_id,
            "overall_result": overall_result,
            "station_results": station_results,
        }
