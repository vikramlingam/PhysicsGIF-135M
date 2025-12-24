"""
PhysicsGIF Pipeline

Full text-to-GIF pipeline combining parser, physics, and renderer.
"""

import os
from typing import Optional

from .dsl import SceneSpec
from .parser import TextParser
from .physics import simulate_scene
from .renderer import render_gif


class TextToGifPipeline:
    """
    Complete text-to-GIF generation pipeline.
    
    Usage:
        pipeline = TextToGifPipeline()
        pipeline.generate("A red ball bouncing", "output.gif")
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            model_path: Path to fine-tuned parser model (optional)
        """
        self.parser = TextParser(model_path)
    
    def generate(
        self,
        text: str,
        output_path: str = "output.gif",
        verbose: bool = True,
    ) -> SceneSpec:
        """
        Generate a GIF from text description.
        
        Args:
            text: Natural language description
            output_path: Path to save the GIF
            verbose: Print progress messages
        
        Returns:
            The SceneSpec that was generated
        """
        if verbose:
            print(f"Generating GIF for: '{text}'")
        
        # Step 1: Parse text to scene spec
        if verbose:
            print("  [1/3] Parsing text...")
        scene = self.parser.parse(text)
        
        if verbose:
            print(f"        Object: {scene.objects[0].type.value}, Color: {scene.objects[0].color}")
            print(f"        Motion: velocity={scene.motion.velocity}, gravity={scene.motion.gravity}")
            if scene.enable_blast:
                print("        Effects: Particle blast enabled")
        
        # Step 2: Simulate physics (with optional particle effects)
        if verbose:
            print("  [2/3] Simulating physics...")
        frames, particles = simulate_scene(scene, enable_blast=scene.enable_blast)
        
        if verbose:
            total_particles = sum(len(p) for p in particles)
            print(f"        Simulated {len(frames)} frames, {total_particles} particles")
        
        # Step 3: Render to GIF
        if verbose:
            print("  [3/3] Rendering GIF...")
        render_gif(frames, scene, output_path, particles_data=particles)
        
        if verbose:
            file_size = os.path.getsize(output_path)
            print(f"  Done! Output: {output_path} ({file_size} bytes)")
        
        return scene
    
    def generate_from_spec(
        self,
        scene: SceneSpec,
        output_path: str = "output.gif",
    ) -> None:
        """
        Generate GIF from an existing SceneSpec (skip parsing).
        
        Args:
            scene: Scene specification
            output_path: Path to save the GIF
        """
        frames = simulate_scene(scene)
        render_gif(frames, scene, output_path)


if __name__ == "__main__":
    # Test pipeline
    print("Testing Text-to-GIF Pipeline...")
    
    pipeline = TextToGifPipeline()
    
    test_prompts = [
        "A red ball bouncing to the right",
        "Blue square falling down",
        "Green triangle floating left",
    ]
    
    for i, prompt in enumerate(test_prompts):
        output = f"test_pipeline_{i}.gif"
        pipeline.generate(prompt, output)
        
        # Cleanup
        if os.path.exists(output):
            os.remove(output)
    
    print("\n✓ Pipeline test passed!")
