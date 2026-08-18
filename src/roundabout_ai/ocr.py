"""Long-lived, recognition-only RapidOCR adapter for plate crops."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, cast

from roundabout_ai.anpr import (
    OcrObservation,
    matches_uk_plate_format,
    normalize_plate_text,
)
from roundabout_ai.capture import Frame


class RapidResultLike(Protocol):
    @property
    def txts(self) -> Sequence[str] | None: ...

    @property
    def scores(self) -> Sequence[float] | None: ...


class RapidEngineLike(Protocol):
    def __call__(
        self,
        img_content: Frame,
        use_det: bool | None = None,
        use_cls: bool | None = None,
        use_rec: bool | None = None,
    ) -> RapidResultLike: ...


EngineFactory = Callable[[], RapidEngineLike]


def _default_engine_factory() -> RapidEngineLike:
    from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

    return cast(
        RapidEngineLike,
        RapidOCR(
            params={
                "Global.use_det": False,
                "Global.use_cls": False,
                "Global.use_rec": True,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.ocr_version": OCRVersion.PPOCRV6,
                "Rec.model_type": ModelType.SMALL,
            }
        ),
    )


class RapidOcrRecognizer:
    """Own one RapidOCR engine and run only its recognition stage."""

    def __init__(self, *, engine_factory: EngineFactory | None = None) -> None:
        self._engine = (engine_factory or _default_engine_factory)()

    def recognize(self, image: Frame, *, image_id: str) -> OcrObservation:
        result = self._engine(
            image,
            use_det=False,
            use_cls=False,
            use_rec=True,
        )
        texts = tuple(result.txts or ())
        scores = tuple(float(score) for score in (result.scores or ()))
        pairs = tuple(zip(texts, scores, strict=False))
        raw_text, confidence = max(pairs, key=lambda item: item[1], default=("", 0.0))
        normalized = normalize_plate_text(raw_text)
        return OcrObservation(
            image_id=image_id,
            raw_text=raw_text,
            normalized_text=normalized,
            confidence=confidence,
            format_valid=matches_uk_plate_format(normalized),
        )
