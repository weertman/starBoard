# src/data/color_config.py
"""
Color configuration management for perceptual color comparison.

Manages color definitions with RGB/LAB coordinates, supporting:
- Initialization from Python's named colors (matplotlib CSS4)
- User-defined colors via color picker or eyedropper
- Persistence to color_config.yaml
- LAB color space conversion for perceptual distance calculations
"""
from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from threading import Lock

_log = logging.getLogger("starBoard.data.color_config")

# Try to import yaml, fall back to json if not available
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    import json
    _HAS_YAML = False
    _log.warning("PyYAML not installed, using JSON fallback for color config")


# =============================================================================
# COLOR SPACE CONVERSIONS (RGB <-> LAB)
# =============================================================================

def rgb_to_xyz(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Convert RGB (0-255) to CIE XYZ color space.
    Uses sRGB color space with D65 illuminant.
    """
    # Normalize to 0-1
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    
    # Apply gamma correction (sRGB)
    def gamma(c):
        if c > 0.04045:
            return ((c + 0.055) / 1.055) ** 2.4
        return c / 12.92
    
    r, g, b = gamma(r), gamma(g), gamma(b)
    
    # Convert to XYZ using sRGB matrix (D65)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    return (x * 100, y * 100, z * 100)


def xyz_to_lab(xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Convert CIE XYZ to CIE LAB color space.
    Uses D65 illuminant reference white.
    """
    # D65 reference white
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    
    x, y, z = xyz[0] / ref_x, xyz[1] / ref_y, xyz[2] / ref_z
    
    def f(t):
        if t > 0.008856:
            return t ** (1/3)
        return (7.787 * t) + (16 / 116)
    
    fx, fy, fz = f(x), f(y), f(z)
    
    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return (L, a, b)


def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB (0-255) directly to CIE LAB."""
    return xyz_to_lab(rgb_to_xyz(rgb))


def lab_to_xyz(lab: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert CIE LAB to CIE XYZ color space."""
    L, a, b = lab
    
    # D65 reference white
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    def f_inv(t):
        if t ** 3 > 0.008856:
            return t ** 3
        return (t - 16 / 116) / 7.787
    
    x = ref_x * f_inv(fx)
    y = ref_y * f_inv(fy)
    z = ref_z * f_inv(fz)
    
    return (x, y, z)


def xyz_to_rgb(xyz: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Convert CIE XYZ to RGB (0-255)."""
    x, y, z = xyz[0] / 100, xyz[1] / 100, xyz[2] / 100
    
    # XYZ to linear RGB (sRGB matrix inverse)
    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252
    
    # Apply gamma correction
    def gamma_inv(c):
        if c > 0.0031308:
            return 1.055 * (c ** (1 / 2.4)) - 0.055
        return 12.92 * c
    
    r, g, b = gamma_inv(r), gamma_inv(g), gamma_inv(b)
    
    # Clamp and convert to 0-255
    def clamp(c):
        return max(0, min(255, int(round(c * 255))))
    
    return (clamp(r), clamp(g), clamp(b))


def lab_to_rgb(lab: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Convert CIE LAB to RGB (0-255)."""
    return xyz_to_rgb(lab_to_xyz(lab))


# =============================================================================
# DELTA E COLOR DISTANCE (CIE2000)
# =============================================================================

def delta_e_cie2000(
    lab1: Tuple[float, float, float],
    lab2: Tuple[float, float, float],
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> float:
    """
    Calculate Delta E (CIE2000) perceptual color difference.
    
    This is the industry standard for measuring how different two colors
    appear to the human eye. Lower values = more similar.
    
    Reference thresholds:
        0-1:   Not perceptible by human eye
        1-2:   Perceptible through close observation
        2-10:  Perceptible at a glance
        11-49: Colors are more similar than opposite
        100:   Colors are exact opposite
    
    Args:
        lab1, lab2: Colors in LAB space
        kL, kC, kH: Weighting factors (usually 1.0)
    
    Returns:
        Delta E value (0 = identical, higher = more different)
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # Calculate C' and h'
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2
    
    G = 0.5 * (1 - math.sqrt(C_avg**7 / (C_avg**7 + 25**7)))
    
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)
    
    def calc_h_prime(a_prime, b):
        if a_prime == 0 and b == 0:
            return 0
        h = math.degrees(math.atan2(b, a_prime))
        if h < 0:
            h += 360
        return h
    
    h1_prime = calc_h_prime(a1_prime, b1)
    h2_prime = calc_h_prime(a2_prime, b2)
    
    # Calculate deltas
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    if C1_prime * C2_prime == 0:
        delta_h_prime = 0
    else:
        diff = h2_prime - h1_prime
        if abs(diff) <= 180:
            delta_h_prime = diff
        elif diff > 180:
            delta_h_prime = diff - 360
        else:
            delta_h_prime = diff + 360
    
    delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2))
    
    # Calculate averages
    L_avg_prime = (L1 + L2) / 2
    C_avg_prime = (C1_prime + C2_prime) / 2
    
    if C1_prime * C2_prime == 0:
        h_avg_prime = h1_prime + h2_prime
    else:
        if abs(h1_prime - h2_prime) <= 180:
            h_avg_prime = (h1_prime + h2_prime) / 2
        elif h1_prime + h2_prime < 360:
            h_avg_prime = (h1_prime + h2_prime + 360) / 2
        else:
            h_avg_prime = (h1_prime + h2_prime - 360) / 2
    
    # Calculate T
    T = (1 
         - 0.17 * math.cos(math.radians(h_avg_prime - 30))
         + 0.24 * math.cos(math.radians(2 * h_avg_prime))
         + 0.32 * math.cos(math.radians(3 * h_avg_prime + 6))
         - 0.20 * math.cos(math.radians(4 * h_avg_prime - 63)))
    
    # Calculate weighting functions
    SL = 1 + (0.015 * (L_avg_prime - 50)**2) / math.sqrt(20 + (L_avg_prime - 50)**2)
    SC = 1 + 0.045 * C_avg_prime
    SH = 1 + 0.015 * C_avg_prime * T
    
    # Calculate RT
    delta_theta = 30 * math.exp(-((h_avg_prime - 275) / 25)**2)
    RC = 2 * math.sqrt(C_avg_prime**7 / (C_avg_prime**7 + 25**7))
    RT = -RC * math.sin(math.radians(2 * delta_theta))
    
    # Final calculation
    term1 = (delta_L_prime / (kL * SL))**2
    term2 = (delta_C_prime / (kC * SC))**2
    term3 = (delta_H_prime / (kH * SH))**2
    term4 = RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
    
    delta_e = math.sqrt(term1 + term2 + term3 + term4)
    return delta_e


# =============================================================================
# COLOR DEFINITION DATA CLASS
# =============================================================================

@dataclass
class ColorDefinition:
    """
    A color with its name and coordinates in multiple color spaces.
    """
    name: str
    rgb: Tuple[int, int, int]
    lab: Tuple[float, float, float] = field(default=None)
    source: str = "python"  # "python", "user", "eyedropper"
    
    def __post_init__(self):
        # Compute LAB if not provided
        if self.lab is None:
            self.lab = rgb_to_lab(self.rgb)
        # Ensure tuples
        self.rgb = tuple(self.rgb)
        self.lab = tuple(self.lab)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rgb": list(self.rgb),
            "lab": [round(v, 3) for v in self.lab],
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ColorDefinition":
        """Create from dictionary."""
        rgb = tuple(data.get("rgb", [128, 128, 128]))
        lab = tuple(data["lab"]) if "lab" in data else None
        source = data.get("source", "user")
        return cls(name=name, rgb=rgb, lab=lab, source=source)
    
    def hex_code(self) -> str:
        """Return hex color code (#RRGGBB)."""
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)


# =============================================================================
# PYTHON COLOR DATABASE (from matplotlib CSS4 colors)
# =============================================================================

# CSS4 named colors - a comprehensive set of standard web colors
# These serve as the initial color database
CSS4_COLORS: Dict[str, Tuple[int, int, int]] = {
    # Whites
    "white": (255, 255, 255),
    "snow": (255, 250, 250),
    "ivory": (255, 255, 240),
    "floralwhite": (255, 250, 240),
    "ghostwhite": (248, 248, 255),
    "whitesmoke": (245, 245, 245),
    "seashell": (255, 245, 238),
    "beige": (245, 245, 220),
    "oldlace": (253, 245, 230),
    "linen": (250, 240, 230),
    "antiquewhite": (250, 235, 215),
    "papayawhip": (255, 239, 213),
    "blanchedalmond": (255, 235, 205),
    "bisque": (255, 228, 196),
    "moccasin": (255, 228, 181),
    "navajowhite": (255, 222, 173),
    "cornsilk": (255, 248, 220),
    "lemonchiffon": (255, 250, 205),
    
    # Yellows
    "yellow": (255, 255, 0),
    "lightyellow": (255, 255, 224),
    "lightgoldenrodyellow": (250, 250, 210),
    "gold": (255, 215, 0),
    "khaki": (240, 230, 140),
    "darkkhaki": (189, 183, 107),
    "palegoldenrod": (238, 232, 170),
    
    # Oranges
    "orange": (255, 165, 0),
    "darkorange": (255, 140, 0),
    "orangered": (255, 69, 0),
    "coral": (255, 127, 80),
    "tomato": (255, 99, 71),
    "peachpuff": (255, 218, 185),
    
    # Reds
    "red": (255, 0, 0),
    "darkred": (139, 0, 0),
    "crimson": (220, 20, 60),
    "firebrick": (178, 34, 34),
    "indianred": (205, 92, 92),
    "lightcoral": (240, 128, 128),
    "salmon": (250, 128, 114),
    "lightsalmon": (255, 160, 122),
    "darksalmon": (233, 150, 122),
    
    # Pinks
    "pink": (255, 192, 203),
    "lightpink": (255, 182, 193),
    "hotpink": (255, 105, 180),
    "deeppink": (255, 20, 147),
    "mediumvioletred": (199, 21, 133),
    "palevioletred": (219, 112, 147),
    
    # Purples/Violets
    "purple": (128, 0, 128),
    "magenta": (255, 0, 255),
    "fuchsia": (255, 0, 255),
    "violet": (238, 130, 238),
    "plum": (221, 160, 221),
    "orchid": (218, 112, 214),
    "mediumorchid": (186, 85, 211),
    "darkorchid": (153, 50, 204),
    "darkviolet": (148, 0, 211),
    "blueviolet": (138, 43, 226),
    "darkmagenta": (139, 0, 139),
    "mediumpurple": (147, 112, 219),
    "thistle": (216, 191, 216),
    "lavender": (230, 230, 250),
    "lavenderblush": (255, 240, 245),
    
    # Mauve/Maroon
    "maroon": (128, 0, 0),
    "rosybrown": (188, 143, 143),
    "mistyrose": (255, 228, 225),
    
    # Browns/Tans
    "brown": (165, 42, 42),
    "saddlebrown": (139, 69, 19),
    "sienna": (160, 82, 45),
    "chocolate": (210, 105, 30),
    "peru": (205, 133, 63),
    "sandybrown": (244, 164, 96),
    "burlywood": (222, 184, 135),
    "tan": (210, 180, 140),
    "wheat": (245, 222, 179),
    
    # Greens
    "green": (0, 128, 0),
    "lime": (0, 255, 0),
    "limegreen": (50, 205, 50),
    "forestgreen": (34, 139, 34),
    "darkgreen": (0, 100, 0),
    "seagreen": (46, 139, 87),
    "mediumseagreen": (60, 179, 113),
    "springgreen": (0, 255, 127),
    "mediumspringgreen": (0, 250, 154),
    "lightgreen": (144, 238, 144),
    "palegreen": (152, 251, 152),
    "darkseagreen": (143, 188, 143),
    "greenyellow": (173, 255, 47),
    "yellowgreen": (154, 205, 50),
    "chartreuse": (127, 255, 0),
    "lawngreen": (124, 252, 0),
    "olivedrab": (107, 142, 35),
    "olive": (128, 128, 0),
    "darkolivegreen": (85, 107, 47),
    
    # Cyans/Teals
    "cyan": (0, 255, 255),
    "aqua": (0, 255, 255),
    "lightcyan": (224, 255, 255),
    "paleturquoise": (175, 238, 238),
    "aquamarine": (127, 255, 212),
    "turquoise": (64, 224, 208),
    "mediumturquoise": (72, 209, 204),
    "darkturquoise": (0, 206, 209),
    "teal": (0, 128, 128),
    "darkcyan": (0, 139, 139),
    "cadetblue": (95, 158, 160),
    
    # Blues
    "blue": (0, 0, 255),
    "navy": (0, 0, 128),
    "darkblue": (0, 0, 139),
    "mediumblue": (0, 0, 205),
    "royalblue": (65, 105, 225),
    "steelblue": (70, 130, 180),
    "dodgerblue": (30, 144, 255),
    "deepskyblue": (0, 191, 255),
    "cornflowerblue": (100, 149, 237),
    "skyblue": (135, 206, 235),
    "lightskyblue": (135, 206, 250),
    "lightsteelblue": (176, 196, 222),
    "lightblue": (173, 216, 230),
    "powderblue": (176, 224, 230),
    "aliceblue": (240, 248, 255),
    "midnightblue": (25, 25, 112),
    "slateblue": (106, 90, 205),
    "darkslateblue": (72, 61, 139),
    "mediumslateblue": (123, 104, 238),
    
    # Grays
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "darkgray": (169, 169, 169),
    "darkgrey": (169, 169, 169),
    "silver": (192, 192, 192),
    "lightgray": (211, 211, 211),
    "lightgrey": (211, 211, 211),
    "gainsboro": (220, 220, 220),
    "dimgray": (105, 105, 105),
    "dimgrey": (105, 105, 105),
    "slategray": (112, 128, 144),
    "slategrey": (112, 128, 144),
    "lightslategray": (119, 136, 153),
    "lightslategrey": (119, 136, 153),
    "darkslategray": (47, 79, 79),
    "darkslategrey": (47, 79, 79),
    
    # Additional sea star relevant colors
    "peach": (255, 218, 185),  # Same as peachpuff
    "burgundy": (128, 0, 32),
    "mauve": (224, 176, 255),
}


# =============================================================================
# COLOR CONFIG STORE
# =============================================================================

class ColorConfigStore:
    """
    Thread-safe singleton store for color configurations.
    
    Manages color definitions with RGB/LAB coordinates for perceptual
    distance calculations. Colors are persisted to color_config.yaml.
    """
    _instance: Optional["ColorConfigStore"] = None
    _lock = Lock()
    
    def __new__(cls) -> "ColorConfigStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        from src.data.archive_paths import archive_root
        self._config_path: Path = archive_root() / "color_config.yaml"
        self._colors: Dict[str, ColorDefinition] = {}
        self._dirty = False
        
        self._load_or_initialize()
    
    def _load_or_initialize(self) -> None:
        """Load config from file, or initialize from Python colors if not exists."""
        if self._config_path.exists():
            self._load_from_file()
        else:
            self._initialize_from_python_colors()
            self._save_to_file()
    
    def _initialize_from_python_colors(self) -> None:
        """Initialize color database from CSS4 named colors."""
        _log.info("Initializing color config from Python CSS4 colors")
        for name, rgb in CSS4_COLORS.items():
            self._colors[name.lower()] = ColorDefinition(
                name=name.lower(),
                rgb=rgb,
                source="python"
            )
        _log.info("Initialized %d colors from CSS4 database", len(self._colors))
    
    def _load_from_file(self) -> None:
        """Load color definitions from YAML/JSON file."""
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                if _HAS_YAML:
                    data = yaml.safe_load(f) or {}
                else:
                    import json
                    data = json.load(f)
            
            colors_data = data.get("colors", {})
            for name, cdata in colors_data.items():
                try:
                    self._colors[name.lower()] = ColorDefinition.from_dict(name.lower(), cdata)
                except Exception as e:
                    _log.warning("Failed to load color '%s': %s", name, e)
            
            _log.info("Loaded %d colors from %s", len(self._colors), self._config_path)
            
        except Exception as e:
            _log.warning("Failed to load color config: %s. Initializing from defaults.", e)
            self._initialize_from_python_colors()
    
    def _save_to_file(self) -> None:
        """Save color definitions to YAML/JSON file."""
        try:
            # Sort colors for consistent output
            colors_dict = {}
            for name in sorted(self._colors.keys()):
                colors_dict[name] = self._colors[name].to_dict()
            
            data = {
                "color_space": "LAB",
                "distance_metric": "delta_e_cie2000",
                "colors": colors_dict,
            }
            
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self._config_path, "w", encoding="utf-8") as f:
                if _HAS_YAML:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                else:
                    import json
                    json.dump(data, f, indent=2)
            
            self._dirty = False
            _log.info("Saved %d colors to %s", len(self._colors), self._config_path)
            
        except Exception as e:
            _log.error("Failed to save color config: %s", e)
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def get_color(self, name: str) -> Optional[ColorDefinition]:
        """Get a color definition by name."""
        return self._colors.get(name.lower().strip())
    
    def has_color(self, name: str) -> bool:
        """Check if a color exists in the database."""
        return name.lower().strip() in self._colors
    
    def get_all_colors(self) -> List[ColorDefinition]:
        """Get all color definitions, sorted by name."""
        return [self._colors[k] for k in sorted(self._colors.keys())]
    
    def get_color_names(self) -> List[str]:
        """Get all color names, sorted."""
        return sorted(self._colors.keys())
    
    def add_color(
        self,
        name: str,
        rgb: Tuple[int, int, int],
        source: str = "user",
        save: bool = True,
    ) -> ColorDefinition:
        """
        Add a new color or update an existing one.
        
        Args:
            name: Color name (will be lowercased)
            rgb: RGB tuple (0-255 each)
            source: Origin of the color ("user", "eyedropper", "python")
            save: Whether to save to file immediately
        
        Returns:
            The created/updated ColorDefinition
        """
        name = name.lower().strip()
        color = ColorDefinition(name=name, rgb=rgb, source=source)
        self._colors[name] = color
        self._dirty = True
        
        if save:
            self._save_to_file()
        
        _log.info("Added color '%s' rgb=%s lab=(%.1f, %.1f, %.1f) source=%s",
                  name, rgb, *color.lab, source)
        return color
    
    def remove_color(self, name: str, save: bool = True) -> bool:
        """Remove a color from the database."""
        name = name.lower().strip()
        if name in self._colors:
            del self._colors[name]
            self._dirty = True
            if save:
                self._save_to_file()
            return True
        return False
    
    def get_distance(self, name1: str, name2: str) -> Optional[float]:
        """
        Get perceptual distance (Delta E CIE2000) between two named colors.
        
        Returns None if either color is not in the database.
        """
        c1 = self.get_color(name1)
        c2 = self.get_color(name2)
        if c1 is None or c2 is None:
            return None
        return delta_e_cie2000(c1.lab, c2.lab)
    
    def get_similarity(self, name1: str, name2: str, threshold: float = 50.0) -> float:
        """
        Get similarity score (0-1) between two named colors.
        
        Uses exponential decay on Delta E distance:
            similarity = exp(-deltaE / threshold)
        
        With threshold=50 (default):
            deltaE=0  -> 1.00 (identical)
            deltaE=5  -> 0.90 (very similar)
            deltaE=10 -> 0.82 (similar)
            deltaE=25 -> 0.61 (somewhat similar)
            deltaE=50 -> 0.37 (different)
            deltaE=100 -> 0.14 (very different)
        
        Returns 0.0 if either color is unknown.
        """
        dist = self.get_distance(name1, name2)
        if dist is None:
            return 0.0
        return math.exp(-dist / threshold)
    
    def find_closest(self, rgb: Tuple[int, int, int], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the closest named colors to a given RGB value.
        
        Returns:
            List of (color_name, delta_e_distance) tuples, sorted by distance
        """
        target_lab = rgb_to_lab(rgb)
        distances = []
        
        for name, color in self._colors.items():
            dist = delta_e_cie2000(target_lab, color.lab)
            distances.append((name, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:top_k]
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._colors.clear()
        self._load_or_initialize()
    
    def save(self) -> None:
        """Save current state to file."""
        if self._dirty or True:  # Always save to ensure consistency
            self._save_to_file()
    
    def merge_existing_vocabularies(self) -> int:
        """
        Import colors from existing vocabulary JSON files that aren't
        already in the color config.
        
        Returns number of colors added.
        """
        from src.data.archive_paths import archive_root
        vocab_dir = archive_root() / "vocabularies"
        
        added = 0
        for path in vocab_dir.glob("colors_*.json"):
            try:
                import json
                with open(path, "r", encoding="utf-8") as f:
                    colors = json.load(f)
                
                for color_name in colors:
                    name = color_name.lower().strip()
                    if name and not self.has_color(name):
                        # Try to find a close match in CSS4 colors
                        closest = None
                        for css_name, rgb in CSS4_COLORS.items():
                            if css_name.lower() == name or name in css_name.lower() or css_name.lower() in name:
                                closest = rgb
                                break
                        
                        if closest:
                            self.add_color(name, closest, source="vocabulary", save=False)
                        else:
                            # Use a neutral gray as placeholder
                            self.add_color(name, (128, 128, 128), source="vocabulary", save=False)
                            _log.warning("Color '%s' from vocabulary has no RGB - using gray placeholder", name)
                        added += 1
                        
            except Exception as e:
                _log.warning("Failed to read vocabulary %s: %s", path, e)
        
        if added > 0:
            self._save_to_file()
            _log.info("Merged %d colors from existing vocabularies", added)
        
        return added


def get_color_config() -> ColorConfigStore:
    """Get the singleton color config store instance."""
    return ColorConfigStore()



