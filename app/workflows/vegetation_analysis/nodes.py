from __future__ import annotations

from typing import Any

import cv2
import httpx
import json
import numpy as np

from app.api.models import FileItem
from app.core.errors import ExternalServiceError, InvalidRequestError
from app.workflows.vegetation_analysis.state import VegetationAnalysisState


SAM3_ENDPOINT = "http://localhost:8002/sam3/image/segment/semantic/texts"


def _draw_polygon(mask: np.ndarray, coordinates: list[Any]) -> None:
    if not coordinates:
        return
    exterior = np.array(coordinates[0], dtype=np.int32)
    if exterior.size == 0:
        return
    cv2.fillPoly(mask, [exterior], color=255)
    for hole in coordinates[1:]:
        hole_points = np.array(hole, dtype=np.int32)
        if hole_points.size > 0:
            cv2.fillPoly(mask, [hole_points], color=0)


def _geojson_to_mask(geojson: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for feature in geojson.get("features", []):
        geom = feature.get("geometry", {})
        geom_type = geom.get("type")
        coordinates = geom.get("coordinates", [])
        if geom_type == "Polygon":
            _draw_polygon(mask, coordinates)
        elif geom_type == "MultiPolygon":
            for poly_coords in coordinates:
                _draw_polygon(mask, poly_coords)

    return mask


def _read_image(file_item: FileItem, field_name: str) -> np.ndarray:
    if not file_item.path:
        raise InvalidRequestError(message="临时文件路径缺失", detail={"field": field_name})

    image = cv2.imread(file_item.path)
    if image is None:
        raise InvalidRequestError(
            message="图片读取失败",
            detail={"field": field_name, "path": file_item.path},
        )
    return image


def _analyze_index(image_bgr: np.ndarray, mask: np.ndarray, index_name: str) -> dict[str, Any]:
    r_channel = image_bgr[:, :, 2]
    plant_values = r_channel[mask > 0].astype(float) / 255.0

    if len(plant_values) == 0:
        return {
            "index": index_name,
            "pixel_count": 0,
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "std": 0.0,
        }

    return {
        "index": index_name,
        "pixel_count": int(len(plant_values)),
        "mean": float(plant_values.mean()),
        "max": float(plant_values.max()),
        "min": float(plant_values.min()),
        "std": float(plant_values.std()),
    }


async def sam_segment_node(state: VegetationAnalysisState) -> dict[str, Any]:
    origin_file_item = state["origin_file_item"]
    content_type = origin_file_item.content_type or ""

    if not content_type.startswith("image/"):
        raise InvalidRequestError(message="不支持的图片类型", detail=content_type)
    if not origin_file_item.data:
        raise InvalidRequestError(message="origin_file 内容为空")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                SAM3_ENDPOINT,
                files={
                    "image_file": (
                        origin_file_item.filename,
                        origin_file_item.data,
                        content_type,
                    )
                },
                data={"config": json.dumps(state["config"], ensure_ascii=False)},
            )
            resp.raise_for_status()
            payload = resp.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise ExternalServiceError(message="Sam3 服务异常", detail=str(exc)) from exc

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise InvalidRequestError(message="Sam3 返回结果为空", detail=payload)

    first_item = results[0] if isinstance(results[0], dict) else {}
    geojson = first_item.get("geojson")
    if not isinstance(geojson, dict):
        raise InvalidRequestError(message="Sam3 返回缺少 geojson", detail=payload)

    return {
        "sam3_raw": payload,
        "geojson": geojson,
    }


async def mask_build_node(state: VegetationAnalysisState) -> dict[str, Any]:
    origin_img = _read_image(state["origin_file_item"], "origin_file")
    ndvi_img = _read_image(state["ndvi_file_item"], "ndvi_file")
    gndvi_img = _read_image(state["gndvi_file_item"], "gndvi_file")
    lci_img = _read_image(state["lci_file_item"], "lci_file")

    expected_shape = origin_img.shape[:2]
    for field_name, image in (
        ("ndvi_file", ndvi_img),
        ("gndvi_file", gndvi_img),
        ("lci_file", lci_img),
    ):
        if image.shape[:2] != expected_shape:
            raise InvalidRequestError(
                message="图片尺寸不一致",
                detail={
                    "field": field_name,
                    "expected": expected_shape,
                    "actual": image.shape[:2],
                },
            )

    geojson = state.get("geojson")
    if not isinstance(geojson, dict):
        raise InvalidRequestError(message="缺少 sam3 分割结果")

    mask = _geojson_to_mask(geojson, expected_shape)
    return {"mask": mask}


async def index_metrics_node(state: VegetationAnalysisState) -> dict[str, Any]:
    mask = state.get("mask")
    if not isinstance(mask, np.ndarray):
        raise InvalidRequestError(message="掩膜未生成")

    ndvi_img = _read_image(state["ndvi_file_item"], "ndvi_file")
    gndvi_img = _read_image(state["gndvi_file_item"], "gndvi_file")
    lci_img = _read_image(state["lci_file_item"], "lci_file")

    ndvi_stats = _analyze_index(ndvi_img, mask, "NDVI")
    gndvi_stats = _analyze_index(gndvi_img, mask, "GNDVI")
    lci_stats = _analyze_index(lci_img, mask, "LCI")

    return {
        "ndvi_stats": ndvi_stats,
        "gndvi_stats": gndvi_stats,
        "lci_stats": lci_stats,
        "ndvi_mean": float(ndvi_stats["mean"]),
        "gndvi_mean": float(gndvi_stats["mean"]),
        "lci_mean": float(lci_stats["mean"]),
    }
