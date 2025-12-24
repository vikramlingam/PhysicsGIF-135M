"""
PhysicsGIF Scene DSL (Domain Specific Language)

Defines structured scene specifications for text-to-video generation.
This is the bridge between natural language and physics simulation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import json


class ObjectType(Enum):
    """Supported object types."""
    BALL = "ball"
    SQUARE = "square"
    TRIANGLE = "triangle"


@dataclass
class ObjectSpec:
    """Specification for a single object in the scene."""
    type: ObjectType
    color: str = "#FF0000"  # Hex color
    size: float = 10.0       # Radius for ball, side length for others
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "color": self.color,
            "size": self.size,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObjectSpec":
        return cls(
            type=ObjectType(d["type"]),
            color=d.get("color", "#FF0000"),
            size=d.get("size", 10.0),
        )


@dataclass
class MotionSpec:
    """Physics/motion specification."""
    start_pos: Tuple[float, float] = (32.0, 32.0)  # Initial (x, y)
    velocity: Tuple[float, float] = (2.0, -3.0)    # Initial (vx, vy)
    gravity: float = 0.3                            # Downward acceleration
    bounce: float = 0.9                             # Coefficient of restitution
    friction: float = 0.0                           # Velocity damping
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_pos": list(self.start_pos),
            "velocity": list(self.velocity),
            "gravity": self.gravity,
            "bounce": self.bounce,
            "friction": self.friction,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MotionSpec":
        return cls(
            start_pos=tuple(d.get("start_pos", [32, 32])),
            velocity=tuple(d.get("velocity", [2, -3])),
            gravity=d.get("gravity", 0.3),
            bounce=d.get("bounce", 0.9),
            friction=d.get("friction", 0.0),
        )


@dataclass
class CanvasSpec:
    """Canvas/output specification."""
    size: int = 128          # Canvas size (square)
    frames: int = 40         # Number of frames
    fps: int = 12            # Frames per second
    background: str = "#000000"  # Background color
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "frames": self.frames,
            "fps": self.fps,
            "background": self.background,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanvasSpec":
        return cls(
            size=d.get("size", 64),
            frames=d.get("frames", 20),
            fps=d.get("fps", 10),
            background=d.get("background", "#000000"),
        )


@dataclass
class SceneSpec:
    """
    Complete scene specification.
    
    This is the structured representation that bridges
    natural language and physics simulation.
    """
    objects: List[ObjectSpec] = field(default_factory=lambda: [ObjectSpec(ObjectType.BALL)])
    motion: MotionSpec = field(default_factory=MotionSpec)
    canvas: CanvasSpec = field(default_factory=CanvasSpec)
    enable_blast: bool = False  # Enable particle effects on collision
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "motion": self.motion.to_dict(),
            "canvas": self.canvas.to_dict(),
            "enable_blast": self.enable_blast,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneSpec":
        return cls(
            objects=[ObjectSpec.from_dict(o) for o in d.get("objects", [{"type": "ball"}])],
            motion=MotionSpec.from_dict(d.get("motion", {})),
            canvas=CanvasSpec.from_dict(d.get("canvas", {})),
            enable_blast=d.get("enable_blast", False),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "SceneSpec":
        return cls.from_dict(json.loads(json_str))


# Color name to hex mapping
COLOR_MAP = {
    "red": "#FF0000",
    "blue": "#0000FF",
    "green": "#00FF00",
    "yellow": "#FFFF00",
    "orange": "#FFA500",
    "purple": "#800080",
    "pink": "#FFC0CB",
    "white": "#FFFFFF",
    "black": "#000000",
    "cyan": "#00FFFF",
    "magenta": "#FF00FF",
    "gray": "#808080",
    "grey": "#808080",
}


def color_to_hex(color: str) -> str:
    """Convert color name to hex, or return as-is if already hex."""
    if color.startswith("#"):
        return color
    return COLOR_MAP.get(color.lower(), "#FF0000")


if __name__ == "__main__":
    # Test DSL
    print("Testing Scene DSL...")
    
    scene = SceneSpec(
        objects=[ObjectSpec(ObjectType.BALL, color="#FF0000", size=8)],
        motion=MotionSpec(start_pos=(10, 50), velocity=(3, -2), gravity=0.3),
        canvas=CanvasSpec(size=64, frames=20, fps=10),
    )
    
    json_str = scene.to_json()
    print(f"Scene JSON:\n{json_str}\n")
    
    # Roundtrip test
    scene2 = SceneSpec.from_json(json_str)
    assert scene.to_dict() == scene2.to_dict()
    print("✓ DSL roundtrip test passed!")
