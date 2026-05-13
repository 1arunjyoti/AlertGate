from typing import Optional, Tuple

import cv2


def get_include_crop_rect(roi_manager, frame_shape) -> Optional[Tuple[int, int, int, int]]:
    """Compute a tight bounding box around all include zones."""
    try:
        if not roi_manager.config.get('enabled', False):
            return None

        contours_map = getattr(roi_manager, 'include_contours', None)
        if not contours_map:
            return None

        contours = []
        for zone_contours in contours_map.values():
            if zone_contours:
                contours.extend(zone_contours)

        if not contours:
            return None

        height, width = frame_shape[:2]
        min_x, min_y, max_x, max_y = width, height, 0, 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        margin_x = int(0.05 * width)
        margin_y = int(0.05 * height)
        x1 = max(0, min_x - margin_x)
        y1 = max(0, min_y - margin_y)
        x2 = min(width, max_x + margin_x)
        y2 = min(height, max_y + margin_y)

        if (x2 - x1) < 64 or (y2 - y1) < 64:
            return None

        return (x1, y1, x2, y2)
    except Exception:
        return None
