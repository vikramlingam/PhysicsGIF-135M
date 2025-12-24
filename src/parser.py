"""
PhysicsGIF Text Parser

Parses natural language prompts into structured SceneSpec.
Uses SmolLM2-135M with LoRA fine-tuning (or rule-based fallback).
"""

import json
import re
import os
from typing import Optional, Dict, Any

from .dsl import (
    SceneSpec, ObjectSpec, ObjectType, MotionSpec, CanvasSpec,
    COLOR_MAP, color_to_hex
)


# Rule-based patterns for fallback parsing
OBJECT_PATTERNS = {
    r'\b(balls?|circles?|spheres?|orbs?)\b': ObjectType.BALL,
    r'\b(squares?|box|boxes|rectangles?|cubes?)\b': ObjectType.SQUARE,
    r'\b(triangles?|pyramids?)\b': ObjectType.TRIANGLE,
}

NUMBER_PATTERNS = {
    r'\b(two|2)\b': 2,
    r'\b(three|3)\b': 3,
    r'\b(four|4)\b': 4,
    r'\b(five|5)\b': 5,
}

# Secondary colors for multiple objects
MULTI_COLORS = ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#FF00FF"]

DIRECTION_PATTERNS = {
    r'\b(left)\b': (-3, 0),
    r'\b(right)\b': (3, 0),
    r'\b(up)\b': (0, -3),
    r'\b(down)\b': (0, 3),
    r'\b(left.{0,10}right|right.{0,10}left)\b': (3, 0),
    r'\b(up.{0,10}down|down.{0,10}up)\b': (0, 3),
}

MOTION_PATTERNS = {
    r'\b(bounc\w*)\b': {"bounce": 0.9, "gravity": 0.3},
    r'\b(fall\w*|drop\w*)\b': {"gravity": 0.5, "bounce": 0.7, "start_y": 8},
    r'\b(roll\w*)\b': {"gravity": 0.0, "bounce": 0.95, "friction": 0.02},
    r'\b(float\w*|hover\w*)\b': {"gravity": 0.0, "bounce": 1.0},
    r'\b(collid\w*|crash\w*|hit\w*)\b': {"gravity": 0.0, "bounce": 1.0, "velocity": (4, 0)},
    r'\b(ricochet\w*)\b': {"gravity": 0.0, "bounce": 1.0, "velocity": (3, 2)},
    r'\b(slide\w*|slid\w*)\b': {"gravity": 0.0, "bounce": 0.9, "friction": 0.03},
    r'\b(blast\w*|explod\w*|burst\w*|shatter\w*)\b': {"gravity": 0.5, "bounce": 0.8, "start_y": 15, "enable_blast": True},
}

SIZE_PATTERNS = {
    r'\b(tiny|small)\b': 6,
    r'\b(big|large|huge)\b': 18,
}


def parse_with_rules(text: str) -> SceneSpec:
    """
    Rule-based parser for natural language -> SceneSpec.
    
    Supports multiple objects (e.g., "two balls colliding").
    """
    text_lower = text.lower()
    
    # Parse number of objects
    num_objects = 1
    for pattern, num in NUMBER_PATTERNS.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            num_objects = num
            break
    
    # Parse object type
    obj_type = ObjectType.BALL  # Default
    for pattern, otype in OBJECT_PATTERNS.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            obj_type = otype
            break
    
    # Parse primary color
    primary_color = "#FF0000"  # Default red
    for color_name, hex_code in COLOR_MAP.items():
        if color_name in text_lower:
            primary_color = hex_code
            break
    
    # Parse size
    size = 10.0  # Default
    for pattern, sz in SIZE_PATTERNS.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            size = sz
            break
    
    # Parse direction/velocity
    velocity = (3.0, 0.0)  # Default: moving right
    for pattern, vel in DIRECTION_PATTERNS.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            velocity = vel
            break
    
    # Parse motion type
    gravity = 0.0  # Default for collision scenarios
    bounce = 0.95
    friction = 0.0
    start_pos = (20, 64)
    velocity_override = None
    enable_blast = False
    
    for pattern, params in MOTION_PATTERNS.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            gravity = params.get("gravity", gravity)
            bounce = params.get("bounce", bounce)
            friction = params.get("friction", friction)
            if "start_y" in params:
                start_pos = (64, params["start_y"])
            if "velocity" in params:
                velocity_override = params["velocity"]
            # Don't break for enable_blast - check it separately
            if not params.get("enable_blast"):
                break
    
    # SEPARATE CHECK for explosion keywords (works with ANY motion)
    blast_keywords = r'\b(explod\w*|blast\w*|burst\w*|shatter\w*|destro\w*|smash\w*)\b'
    if re.search(blast_keywords, text_lower, re.IGNORECASE):
        enable_blast = True
        # If no gravity set and explosion requested, add some gravity
        if gravity == 0.0:
            gravity = 0.3
    if velocity_override:
        velocity = velocity_override
    
    # Create objects with different colors for multi-object scenes
    objects = []
    for i in range(num_objects):
        if i == 0:
            color = primary_color
        else:
            # Use a different color from MULTI_COLORS
            color = MULTI_COLORS[i % len(MULTI_COLORS)]
            # Skip if same as primary
            if color == primary_color and len(MULTI_COLORS) > 1:
                color = MULTI_COLORS[(i + 1) % len(MULTI_COLORS)]
        
        objects.append(ObjectSpec(type=obj_type, color=color, size=size))
    
    # Build spec
    return SceneSpec(
        objects=objects,
        motion=MotionSpec(
            start_pos=start_pos,
            velocity=velocity,
            gravity=gravity,
            bounce=bounce,
            friction=friction,
        ),
        canvas=CanvasSpec(size=128, frames=40, fps=12, background="#000000"),
        enable_blast=enable_blast,
    )


class TextParser:
    """
    Text-to-SceneSpec parser.
    
    Uses a fine-tuned LLM if available, otherwise falls back to rules.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the parser.
        
        Args:
            model_path: Path to fine-tuned model directory
        """
        self.model = None
        self.tokenizer = None
        self.use_llm = False
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> None:
        """Load the fine-tuned LLM."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            print(f"Loading parser model from {model_path}...")
            
            # Check if this is a LoRA adapter or full model
            adapter_config = os.path.join(model_path, "adapter_config.json")
            
            if os.path.exists(adapter_config):
                # Load base model from config
                with open(adapter_config) as f:
                    config = json.load(f)
                base_model_name = config.get("base_model_name_or_path", "HuggingFaceTB/SmolLM2-135M-Instruct")
                
                # Load base + adapter
                base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            else:
                # Full model
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            self.use_llm = True
            print("Loaded LLM parser successfully!")
            
        except Exception as e:
            print(f"Could not load LLM model: {e}")
            print("Falling back to rule-based parser.")
            self.use_llm = False
    
    def _parse_with_llm(self, text: str) -> SceneSpec:
        """Parse using the fine-tuned LLM."""
        prompt = f"""Convert this text description to a scene specification JSON.

Text: {text}

JSON:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                spec_dict = json.loads(json_match.group())
                return SceneSpec.from_dict(spec_dict)
            except json.JSONDecodeError:
                pass
        
        # Fall back to rules if LLM output is invalid
        return parse_with_rules(text)
    
    def parse(self, text: str) -> SceneSpec:
        """
        Parse natural language to SceneSpec.
        
        Args:
            text: Natural language description
        
        Returns:
            SceneSpec for the described scene
        """
        if self.use_llm:
            return self._parse_with_llm(text)
        else:
            return parse_with_rules(text)


if __name__ == "__main__":
    # Test parser
    print("Testing Text Parser (Rule-based)...")
    
    parser = TextParser()
    
    test_cases = [
        "A red ball bouncing from left to right",
        "Blue square falling down",
        "Small green triangle floating",
        "Big orange ball rolling to the left",
        "Purple circle bouncing up and down",
    ]
    
    for text in test_cases:
        spec = parser.parse(text)
        print(f"\nInput: '{text}'")
        print(f"Output: {spec.to_json()}")
    
    print("\n✓ Parser test passed!")
