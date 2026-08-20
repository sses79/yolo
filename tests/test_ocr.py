from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roundabout_ai.capture import Frame
from roundabout_ai.ocr import RapidOcrRecognizer


@dataclass
class FakeResult:
    txts: tuple[str, ...]
    scores: tuple[float, ...]


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        img_content: Frame,
        use_det: bool | None = None,
        use_cls: bool | None = None,
        use_rec: bool | None = None,
    ) -> FakeResult:
        self.calls.append({"use_det": use_det, "use_cls": use_cls, "use_rec": use_rec})
        return FakeResult((" AB12 CDE ",), (0.91,))


def test_recognizer_constructs_engine_once_and_uses_recognition_only() -> None:
    engine = FakeEngine()
    factory_calls = 0

    def factory() -> FakeEngine:
        nonlocal factory_calls
        factory_calls += 1
        return engine

    recognizer = RapidOcrRecognizer(engine_factory=factory)
    image = np.zeros((32, 128, 3), dtype=np.uint8)
    first = recognizer.recognize(image, image_id="one")
    recognizer.recognize(image, image_id="two")

    assert factory_calls == 1
    assert first.normalized_text == "AB12CDE"
    assert first.format_valid
    assert engine.calls == [
        {"use_det": False, "use_cls": False, "use_rec": True},
        {"use_det": False, "use_cls": False, "use_rec": True},
    ]
