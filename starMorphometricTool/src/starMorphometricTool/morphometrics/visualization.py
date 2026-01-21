import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from io import BytesIO
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2


def calculate_max_diameter(arm_data):
    """
    Calculate maximum tip-to-tip diameter from arm data.
    
    Args:
        arm_data: List of [arm_number, x_vec, y_vec, length_mm]
    
    Returns:
        Maximum diameter in mm, or None if insufficient data
    """
    if len(arm_data) < 2:
        return None
    
    # Calculate angle of each arm vector
    arm_angles = []
    for arm in arm_data:
        arm_number, x_vec, y_vec, length_mm = arm
        angle = math.atan2(y_vec, x_vec)
        arm_angles.append((arm_number, angle, x_vec, y_vec, length_mm))
    
    max_diameter = 0.0
    
    # For each arm, find the most opposite arm
    for i, (_, arm1_angle, x1, y1, _) in enumerate(arm_angles):
        opposite_angle = (arm1_angle + math.pi) % (2 * math.pi)
        
        min_diff = float('inf')
        opposite_idx = -1
        
        for j, (_, arm2_angle, _, _, _) in enumerate(arm_angles):
            if j == i:
                continue
            
            # Angular difference accounting for circular nature
            diff = abs((arm2_angle - opposite_angle + math.pi) % (2 * math.pi) - math.pi)
            if diff < min_diff:
                min_diff = diff
                opposite_idx = j
        
        if opposite_idx >= 0:
            _, _, x2, y2, _ = arm_angles[opposite_idx]
            # Tip-to-tip distance (vectors are from center)
            diameter = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            max_diameter = max(max_diameter, diameter)
    
    return max_diameter if max_diameter > 0 else None


def create_morphometrics_visualization(ax, corrected_object_rgb, center, arm_data,
                                       rotation, ellipse_data, morphometrics_data):
    """
    Create a visualization of the specimen with numbered arms.

    Args:
        ax: Matplotlib axis to draw on
        corrected_object_rgb: RGB image of the corrected object
        center: (x, y) center point
        arm_data: List of [arm_number, x_vec, y_vec, length_mm]
        rotation: Rotation value for arm numbering
        ellipse_data: Tuple of ellipse parameters (x0, y0, major_len, minor_len, angle)
        morphometrics_data: Dictionary of morphometric measurements

    Returns:
        arm_angles, arm_dists, arm_labels, arm_colors for polar plot
    """
    ax.clear()
    ax.axis('off')

    # Draw the specimen image
    ax.imshow(corrected_object_rgb)

    cx, cy = center
    num_arms = len(arm_data)
    new_order = [(i - rotation) % num_arms + 1 for i in range(num_arms)]

    # Mark the center (smaller marker, no legend entry)
    ax.plot(cx, cy, 'yo', markersize=5)

    # Polar plot data
    polar_angles = []
    polar_dists = []
    polar_labels = []
    polar_colors = []

    # Draw each arm
    for i, arm_info in enumerate(arm_data):
        arm_number, x_vec, y_vec, length_mm = arm_info
        tip_x = cx + x_vec
        tip_y = cy + y_vec
        new_num = new_order[i]
        color = 'red' if new_num == 1 else 'blue'

        # Line from center to tip
        ax.plot([cx, tip_x], [cy, tip_y], color=color, linewidth=1.5)
        ax.plot(tip_x, tip_y, 'o', color=color, markersize=4)

        # Number label (smaller font)
        text_x = (cx + tip_x) / 2
        text_y = (cy + tip_y) / 2
        ax.text(text_x, text_y, str(new_num),
                color='white', fontweight='bold', fontsize=7, ha='center', va='center',
                bbox=dict(facecolor=color, edgecolor='none', alpha=0.7, pad=1))

        # Store polar plot data
        tip_angle = np.arctan2(y_vec, x_vec)
        tip_dist = np.hypot(x_vec, y_vec)

        polar_angles.append(tip_angle)
        polar_dists.append(tip_dist)
        polar_labels.append(new_num)
        polar_colors.append(color)

    # Draw ellipse if available (thinner line)
    if ellipse_data:
        x0, y0, major_len, minor_len, angle = ellipse_data
        ellipse_patch = Ellipse((x0, y0), major_len, minor_len, angle=angle,
                                edgecolor='yellow', facecolor='none', linewidth=1)
        ax.add_patch(ellipse_patch)

    # Show measurements as compact text overlay (bottom-left, smaller font)
    area_val = morphometrics_data.get("area_mm2", 0)
    num_arms_val = len(arm_data)
    major_mm = morphometrics_data.get("major_axis_mm", 0)
    minor_mm = morphometrics_data.get("minor_axis_mm", 0)
    
    # Calculate max diameter from arm data
    max_diam = calculate_max_diameter(arm_data)

    # Format values more compactly
    major_str = f'{major_mm:.1f}' if major_mm else 'N/A'
    minor_str = f'{minor_mm:.1f}' if minor_mm else 'N/A'
    max_diam_str = f'{max_diam:.1f}' if max_diam else 'N/A'
    
    meas_text = (
        f'Area: {area_val:.1f}mm² | Arms: {num_arms_val}\n'
        f'Axes: {major_str}/{minor_str}mm\n'
        f'Max Ø: {max_diam_str}mm'
    )
    props = dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6)
    ax.text(0.02, 0.02, meas_text, transform=ax.transAxes,
            fontsize=6, verticalalignment='bottom', bbox=props, color='white')

    # No legend or title to keep visualization clean
    return polar_angles, polar_dists, polar_labels, polar_colors


def render_figure_to_pixmap(fig, target_widget):
    """
    Render a matplotlib figure to a QPixmap for display.

    Args:
        fig: Matplotlib figure to render
        target_widget: QWidget for determining target size

    Returns:
        QPixmap ready for display
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    qimg = QImage.fromData(buf.getvalue())
    pixmap = QPixmap.fromImage(qimg)

    return pixmap.scaled(
        target_widget.size(),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )