from __future__ import annotations

from typing import Any

from app.api.models import FileItem
from app.core.errors import InvalidRequestError
from app.workflows.vegetation_analysis.graph import build_vegetation_analysis_graph
from app.workflows.vegetation_analysis.state import VegetationAnalysisState


async def run_vegetation_analysis_pipeline(
    *,
    origin_file_item: FileItem,
    ndvi_file_item: FileItem,
    gndvi_file_item: FileItem,
    lci_file_item: FileItem,
    config: dict[str, Any],
) -> dict[str, Any]:
    for field_name, file_item in (
        ("origin_file", origin_file_item),
        ("ndvi_file", ndvi_file_item),
        ("gndvi_file", gndvi_file_item),
        ("lci_file", lci_file_item),
    ):
        if not file_item.path:
            raise InvalidRequestError(message="临时文件路径缺失", detail={"field": field_name})

    graph = build_vegetation_analysis_graph()

    state: VegetationAnalysisState = {
        "origin_file_item": origin_file_item,
        "ndvi_file_item": ndvi_file_item,
        "gndvi_file_item": gndvi_file_item,
        "lci_file_item": lci_file_item,
        "config": config,
    }

    result = await graph.ainvoke(state)
    return {
        "geojson": result.get("geojson", {}),
        "index_stats": {
            "NDVI": result.get("ndvi_stats", {}),
            "GNDVI": result.get("gndvi_stats", {}),
            "LCI": result.get("lci_stats", {}),
        },
    }
