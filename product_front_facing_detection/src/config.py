from dataclasses import dataclass


@dataclass
class Config:
    resize_width: int = 1000
    min_area: int = 1200
    max_area: int = 800000
    min_component_area: int = 350
    min_width: int = 18
    min_height: int = 22
    min_aspect_ratio: float = 0.15
    max_aspect_ratio: float = 3.2
    rectangularity_threshold: float = 0.55
    edge_density_threshold: float = 0.04
    symmetry_threshold: float = 0.40
    front_texture_threshold: float = 0.08
    pass_threshold: float = 0.7
    gamma: float = 1.0
    segmentation_mode: str = "combined"
    debug: bool = False
