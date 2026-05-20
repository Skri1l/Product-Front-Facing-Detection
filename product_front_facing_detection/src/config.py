from dataclasses import dataclass


@dataclass
class Config:
    resize_width: int = 1000
    min_area: int = 1200
    max_area: int = 500000
    min_width: int = 20
    min_height: int = 30
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 1.4
    rectangularity_threshold: float = 0.62
    edge_density_threshold: float = 0.06
    symmetry_threshold: float = 0.45
    front_texture_threshold: float = 0.10
    pass_threshold: float = 0.7
    gamma: float = 1.0
    segmentation_mode: str = "combined"
    debug: bool = False
