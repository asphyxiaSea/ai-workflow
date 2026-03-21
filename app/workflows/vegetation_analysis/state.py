from __future__ import annotations

from typing import Any, TypedDict
from typing_extensions import NotRequired

import numpy as np

from app.api.models import FileItem


class VegetationAnalysisState(TypedDict):
    origin_file_item: FileItem
    ndvi_file_item: FileItem
    gndvi_file_item: FileItem
    lci_file_item: FileItem
    config: dict[str, Any]
    geojson: NotRequired[dict[str, Any]]
    mask: NotRequired[np.ndarray]
    ndvi_stats: NotRequired[dict[str, Any]]
    gndvi_stats: NotRequired[dict[str, Any]]
    lci_stats: NotRequired[dict[str, Any]]
