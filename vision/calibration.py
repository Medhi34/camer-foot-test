"""Homography calibration utilities for football pitch coordinate mapping.

This module provides a production-oriented `HomographyProvider` to estimate and
apply a planar homography between image pixel coordinates and metric field
coordinates (in meters, default 105x68).

The provider supports:
1) Automatic estimation from matched keypoints.
2) Manual fallback estimation from the 4 corners of one penalty area.

Coordinate system (metric plane):
- x axis: pitch length direction, [0, pitch_length]
- y axis: pitch width direction, [0, pitch_width]
- origin: top-left corner of the pitch in the canonical tactical view
  (convention can be changed upstream if needed).

Notes:
- Homography assumes points belong to the same planar surface (the pitch).
- Broadcasting camera changes may require re-estimation over time.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, Literal

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

PointArray = npt.NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class PitchDimensions:
    """Football pitch dimensions in meters."""

    length_m: float = 105.0
    width_m: float = 68.0


@dataclass(slots=True, frozen=True)
class HomographyResult:
    """Container for homography estimation output.

    Attributes:
        matrix: 3x3 homography matrix mapping source -> destination.
        inlier_mask: RANSAC inlier mask returned by OpenCV (N x 1) or None.
        reprojection_error: Mean reprojection error in destination space.
    """

    matrix: PointArray
    inlier_mask: npt.NDArray[np.uint8] | None
    reprojection_error: float | None


class HomographyProvider:
    """Estimate and apply homography between pixel and metric pitch coordinates.

    The class keeps an internal homography matrix `H_pixel_to_metric` and its
    inverse `H_metric_to_pixel` once calibration succeeds.

    Typical usage:
        provider = HomographyProvider()
        provider.fit_from_correspondences(pixel_points, metric_points)
        xy_m = provider.pixel_to_metric(np.array([[512, 370]], dtype=np.float64))
    """

    # Canonical penalty-area dimensions (IFAB):
    # depth = 16.5m from goal line, width = 40.32m centered on goal.
    PENALTY_AREA_DEPTH_M: float = 16.5
    PENALTY_AREA_WIDTH_M: float = 40.32

    def __init__(
        self,
        pitch_dimensions: PitchDimensions = PitchDimensions(),
        *,
        ransac_reproj_threshold: float = 3.0,
        confidence: float = 0.995,
        max_iters: int = 5000,
    ) -> None:
        """Initialize homography provider.

        Args:
            pitch_dimensions: Field dimensions in meters.
            ransac_reproj_threshold: Max reprojection error to classify inliers
                in RANSAC (OpenCV units of destination plane).
            confidence: RANSAC confidence (0..1).
            max_iters: Max RANSAC iterations.
        """
        self.pitch_dimensions = pitch_dimensions
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.confidence = confidence
        self.max_iters = max_iters

        self._h_pixel_to_metric: PointArray | None = None
        self._h_metric_to_pixel: PointArray | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether a valid homography has been estimated."""
        return self._h_pixel_to_metric is not None and self._h_metric_to_pixel is not None

    @property
    def h_pixel_to_metric(self) -> PointArray:
        """Return pixel->metric homography matrix.

        Raises:
            RuntimeError: If provider is not calibrated.
        """
        if self._h_pixel_to_metric is None:
            raise RuntimeError("Homography not fitted. Call a fit_* method first.")
        return self._h_pixel_to_metric

    @property
    def h_metric_to_pixel(self) -> PointArray:
        """Return metric->pixel homography matrix.

        Raises:
            RuntimeError: If provider is not calibrated.
        """
        if self._h_metric_to_pixel is None:
            raise RuntimeError("Homography not fitted. Call a fit_* method first.")
        return self._h_metric_to_pixel

    def fit_from_correspondences(
        self,
        pixel_points: npt.ArrayLike,
        metric_points: npt.ArrayLike,
    ) -> HomographyResult:
        """Estimate homography from matched pixel and metric points.

        Args:
            pixel_points: Source points in image, shape (N, 2).
            metric_points: Destination points on pitch plane in meters, shape (N, 2).

        Returns:
            HomographyResult containing matrix, inlier mask and reprojection error.

        Raises:
            ValueError: If inputs are invalid or insufficient.
            RuntimeError: If homography estimation fails.
        """
        src = self._to_points_array(pixel_points, name="pixel_points")
        dst = self._to_points_array(metric_points, name="metric_points")

        if src.shape[0] != dst.shape[0]:
            raise ValueError("pixel_points and metric_points must have same number of points.")
        if src.shape[0] < 4:
            raise ValueError("At least 4 point correspondences are required.")

        logger.info("Estimating homography from %d correspondences.", src.shape[0])

        h, inlier_mask = cv2.findHomography(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
            maxIters=self.max_iters,
            confidence=self.confidence,
        )

        if h is None:
            raise RuntimeError("Homography estimation failed (cv2.findHomography returned None).")

        h = self._normalize_h(h)
        h_inv = np.linalg.inv(h)
        h_inv = self._normalize_h(h_inv)

        self._h_pixel_to_metric = h
        self._h_metric_to_pixel = h_inv

        reproj_error = self._compute_reprojection_error(src, dst, h, inlier_mask)
        logger.info("Homography fitted successfully. mean_reprojection_error=%.4f", reproj_error or -1.0)

        return HomographyResult(
            matrix=h,
            inlier_mask=inlier_mask,
            reprojection_error=reproj_error,
        )

    def fit_from_penalty_area_fallback(
        self,
        pixel_penalty_corners: npt.ArrayLike,
        side: Literal["left", "right"] = "left",
    ) -> HomographyResult:
        """Fallback calibration using 4 corners of one penalty area.

        The user provides the 4 detected/manual image corners corresponding to
        one penalty area rectangle. We map these points to canonical metric
        coordinates and estimate homography.

        Corner ordering expected (clockwise):
            [top_outer, top_inner, bottom_inner, bottom_outer]

        Definitions:
        - "outer" lies on goal line (x=0 for left side, x=105 for right side)
        - "inner" lies toward the center of the field by 16.5m

        Args:
            pixel_penalty_corners: Image points, shape (4, 2).
            side: Penalty area side ("left" or "right").

        Returns:
            HomographyResult as in `fit_from_correspondences`.

        Raises:
            ValueError: If side invalid or points malformed.
        """
        src = self._to_points_array(pixel_penalty_corners, name="pixel_penalty_corners")
        if src.shape != (4, 2):
            raise ValueError("pixel_penalty_corners must have shape (4, 2).")

        dst = self._build_penalty_area_metric_corners(side=side)

        logger.warning(
            "Using fallback homography from manual penalty-area corners (side=%s).",
            side,
        )
        return self.fit_from_correspondences(src, dst)

    def pixel_to_metric(self, points_xy: npt.ArrayLike) -> PointArray:
        """Project pixel points to metric pitch coordinates.

        Args:
            points_xy: Pixel points shape (N, 2) or (2,).

        Returns:
            Metric points shape (N, 2), in meters.
        """
        pts = self._to_points_array(points_xy, name="points_xy")
        return self._apply_homography(pts, self.h_pixel_to_metric)

    def metric_to_pixel(self, points_xy: npt.ArrayLike) -> PointArray:
        """Project metric pitch points to pixel coordinates.

        Args:
            points_xy: Metric points shape (N, 2) or (2,).

        Returns:
            Pixel points shape (N, 2).
        """
        pts = self._to_points_array(points_xy, name="points_xy")
        return self._apply_homography(pts, self.h_metric_to_pixel)

    def clip_metric_to_pitch(self, metric_points: npt.ArrayLike) -> PointArray:
        """Clip metric coordinates to valid pitch bounds [0, L] x [0, W].

        Useful to handle noisy detections near boundaries.

        Args:
            metric_points: Points shape (N, 2).

        Returns:
            Clipped points shape (N, 2).
        """
        pts = self._to_points_array(metric_points, name="metric_points")
        pts[:, 0] = np.clip(pts[:, 0], 0.0, self.pitch_dimensions.length_m)
        pts[:, 1] = np.clip(pts[:, 1], 0.0, self.pitch_dimensions.width_m)
        return pts

    def _build_penalty_area_metric_corners(self, side: Literal["left", "right"]) -> PointArray:
        """Create canonical metric points for one penalty area rectangle."""
        if side not in {"left", "right"}:
            raise ValueError("side must be either 'left' or 'right'.")

        w = self.pitch_dimensions.width_m
        y_top = (w - self.PENALTY_AREA_WIDTH_M) / 2.0
        y_bottom = y_top + self.PENALTY_AREA_WIDTH_M

        if side == "left":
            x_outer = 0.0
            x_inner = self.PENALTY_AREA_DEPTH_M
            dst = np.array(
                [
                    [x_outer, y_top],     # top_outer
                    [x_inner, y_top],     # top_inner
                    [x_inner, y_bottom],  # bottom_inner
                    [x_outer, y_bottom],  # bottom_outer
                ],
                dtype=np.float64,
            )
        else:
            x_outer = self.pitch_dimensions.length_m
            x_inner = self.pitch_dimensions.length_m - self.PENALTY_AREA_DEPTH_M
            dst = np.array(
                [
                    [x_outer, y_top],     # top_outer
                    [x_inner, y_top],     # top_inner
                    [x_inner, y_bottom],  # bottom_inner
                    [x_outer, y_bottom],  # bottom_outer
                ],
                dtype=np.float64,
            )

        return dst

    @staticmethod
    def _apply_homography(points_xy: PointArray, h: PointArray) -> PointArray:
        """Apply homography to 2D points using homogeneous coordinates."""
        ones = np.ones((points_xy.shape[0], 1), dtype=np.float64)
        hom = np.hstack([points_xy, ones])  # (N, 3)
        transformed = hom @ h.T             # (N, 3)

        z = transformed[:, 2:3]
        eps = 1e-12
        z = np.where(np.abs(z) < eps, eps, z)

        xy = transformed[:, :2] / z
        return xy.astype(np.float64)

    @staticmethod
    def _to_points_array(points: npt.ArrayLike, name: str) -> PointArray:
        """Validate/normalize input points into float64 array of shape (N, 2)."""
        arr = np.asarray(points, dtype=np.float64)

        if arr.ndim == 1:
            if arr.shape[0] != 2:
                raise ValueError(f"{name} 1D input must have shape (2,), got {arr.shape}.")
            arr = arr.reshape(1, 2)

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{name} must have shape (N, 2), got {arr.shape}.")

        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values.")

        return arr

    @staticmethod
    def _normalize_h(h: npt.ArrayLike) -> PointArray:
        """Normalize homography so H[2,2] == 1 when possible."""
        mat = np.asarray(h, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {mat.shape}.")
        if abs(mat[2, 2]) > 1e-12:
            mat = mat / mat[2, 2]
        return mat

    def _compute_reprojection_error(
        self,
        src: PointArray,
        dst: PointArray,
        h: PointArray,
        inlier_mask: npt.NDArray[np.uint8] | None,
    ) -> float | None:
        """Compute mean reprojection error in destination (metric) space.

        Formula:
            e_i = || x_i' - H x_i ||_2
            mean_error = (1/N) * sum_i e_i
        """
        proj = self._apply_homography(src, h)
        errors = np.linalg.norm(dst - proj, axis=1)

        if inlier_mask is not None and inlier_mask.size == errors.size:
            inliers = inlier_mask.reshape(-1).astype(bool)
            if np.any(inliers):
                return float(np.mean(errors[inliers]))

        if errors.size == 0:
            return None
        return float(np.mean(errors))


def order_points_clockwise(points_xy: npt.ArrayLike) -> PointArray:
    """Order 4 points clockwise as [top-left, top-right, bottom-right, bottom-left].

    Useful helper when manual points are provided in uncertain order.

    Args:
        points_xy: Array shape (4, 2).

    Returns:
        Ordered points shape (4, 2).

    Raises:
        ValueError: If input shape is not (4, 2).
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected shape (4, 2), got {pts.shape}.")

    # Sum and diff heuristic
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float64)


def reorder_penalty_corners_from_tl_tr_br_bl(
    points_tl_tr_br_bl: npt.ArrayLike,
    *,
    side: Literal["left", "right"] = "left",
) -> PointArray:
    """Convert [TL, TR, BR, BL] to fallback expected order:
    [top_outer, top_inner, bottom_inner, bottom_outer].

    Args:
        points_tl_tr_br_bl: points in canonical rectangle order.
        side: penalty side. Determines which vertical edge is 'outer'.

    Returns:
        Reordered points shape (4, 2).
    """
    pts = np.asarray(points_tl_tr_br_bl, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected shape (4, 2), got {pts.shape}.")

    tl, tr, br, bl = pts

    if side == "left":
        # outer edge is left vertical => TL/BL outer, TR/BR inner
        return np.array([tl, tr, br, bl], dtype=np.float64)

    if side == "right":
        # outer edge is right vertical => TR/BR outer, TL/BL inner
        return np.array([tr, tl, bl, br], dtype=np.float64)

    raise ValueError("side must be either 'left' or 'right'.")