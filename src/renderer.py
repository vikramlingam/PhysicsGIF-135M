"""
PhysicsGIF Renderer (Enhanced with Dramatic Particle Effects)

Renders physics simulation frames to GIF with:
- Large, visible particles
- Glow effects
- Multiple particle types (sparks, waves, trails)
"""

from typing import List, Tuple, Optional
from PIL import Image, ImageDraw
import os

from .dsl import SceneSpec, ObjectSpec, ObjectType
from .physics import Particle


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def brighten_color(color: Tuple[int, int, int], factor: float = 1.5) -> Tuple[int, int, int]:
    """Brighten a color for glow effect."""
    return tuple(min(255, int(c * factor)) for c in color)


def draw_ball(draw: ImageDraw.Draw, x: float, y: float, radius: float, color: Tuple[int, int, int]) -> None:
    """Draw a filled circle with glow."""
    # Draw outer glow
    glow_color = brighten_color(color, 0.3)
    for i in range(3, 0, -1):
        r = radius + i * 2
        x0, y0 = x - r, y - r
        x1, y1 = x + r, y + r
        draw.ellipse([x0, y0, x1, y1], fill=glow_color)
    
    # Draw main ball
    x0, y0 = x - radius, y - radius
    x1, y1 = x + radius, y + radius
    draw.ellipse([x0, y0, x1, y1], fill=color)
    
    # Draw highlight
    highlight_color = brighten_color(color, 1.5)
    hr = radius * 0.3
    hx, hy = x - radius * 0.3, y - radius * 0.3
    draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=highlight_color)


def draw_square(draw: ImageDraw.Draw, x: float, y: float, size: float, color: Tuple[int, int, int]) -> None:
    """Draw a filled square."""
    half = size / 2
    draw.rectangle([x - half, y - half, x + half, y + half], fill=color)


def draw_triangle(draw: ImageDraw.Draw, x: float, y: float, size: float, color: Tuple[int, int, int]) -> None:
    """Draw a filled triangle."""
    half = size / 2
    points = [(x, y - half), (x - half, y + half), (x + half, y + half)]
    draw.polygon(points, fill=color)


def draw_particle(draw: ImageDraw.Draw, particle: Particle, canvas_size: int) -> None:
    """Draw a single particle with brightness based on life."""
    # Skip particles outside canvas
    if particle.x < 0 or particle.x > canvas_size or particle.y < 0 or particle.y > canvas_size:
        return
    
    # Calculate brightness based on remaining life
    life_ratio = particle.life / max(1, particle.max_life)
    
    # Base color
    color = hex_to_rgb(particle.color)
    
    # Brighten particles based on life
    brightness = 1.0 + life_ratio * 0.5
    color = tuple(min(255, int(c * brightness)) for c in color)
    
    # Size based on particle type and life
    size = max(2, int(particle.size * (0.5 + life_ratio * 0.5)))
    
    x, y = int(particle.x), int(particle.y)
    
    if particle.particle_type == "spark":
        # Draw bright spark with small glow
        glow_color = tuple(min(255, c + 50) for c in color)
        # Outer glow
        draw.ellipse([x - size - 2, y - size - 2, x + size + 2, y + size + 2], fill=(color[0]//3, color[1]//3, color[2]//3))
        # Inner bright
        draw.ellipse([x - size, y - size, x + size, y + size], fill=color)
        # Center hot spot
        if size > 2:
            draw.ellipse([x - size//2, y - size//2, x + size//2, y + size//2], fill=glow_color)
        
    elif particle.particle_type == "wave":
        # Draw wave particle (ring segment)
        draw.ellipse([x - size, y - size, x + size, y + size], fill=color)
        
    elif particle.particle_type == "trail":
        # Draw trail as fading dot
        alpha_color = tuple(int(c * life_ratio) for c in color)
        draw.ellipse([x - size, y - size, x + size, y + size], fill=alpha_color)
        
    else:
        # Default particle
        draw.ellipse([x - size, y - size, x + size, y + size], fill=color)


def render_frame_with_particles(
    positions: List[Tuple[float, float]],
    objects: List[ObjectSpec],
    particles: List[Particle],
    canvas_size: int,
    background_color: str = "#000000",
) -> Image.Image:
    """Render a single frame with objects and dramatic particles."""
    bg_rgb = hex_to_rgb(background_color)
    img = Image.new("RGB", (canvas_size, canvas_size), bg_rgb)
    draw = ImageDraw.Draw(img)
    
    # Draw particles BEHIND objects (trails first)
    for particle in particles:
        if particle.life > 0 and particle.particle_type == "trail":
            draw_particle(draw, particle, canvas_size)
    
    # Draw objects
    for (x, y), obj in zip(positions, objects):
        color = hex_to_rgb(obj.color)
        if obj.type == ObjectType.BALL:
            draw_ball(draw, x, y, obj.size, color)
        elif obj.type == ObjectType.SQUARE:
            draw_square(draw, x, y, obj.size, color)
        elif obj.type == ObjectType.TRIANGLE:
            draw_triangle(draw, x, y, obj.size, color)
    
    # Draw particles ON TOP of objects (sparks and waves)
    for particle in particles:
        if particle.life > 0 and particle.particle_type != "trail":
            draw_particle(draw, particle, canvas_size)
    
    return img


def render_frame(
    positions: List[Tuple[float, float]],
    objects: List[ObjectSpec],
    canvas_size: int,
    background_color: str = "#000000",
) -> Image.Image:
    """Render a single frame (backwards compatible)."""
    return render_frame_with_particles(positions, objects, [], canvas_size, background_color)


def render_gif(
    frames_data: List[List[Tuple[float, float]]],
    scene: SceneSpec,
    output_path: str,
    particles_data: Optional[List[List[Particle]]] = None,
) -> None:
    """Render all frames and save as GIF."""
    images = []
    
    for i, positions in enumerate(frames_data):
        particles = particles_data[i] if particles_data else []
        img = render_frame_with_particles(
            positions=positions,
            objects=scene.objects,
            particles=particles,
            canvas_size=scene.canvas.size,
            background_color=scene.canvas.background,
        )
        images.append(img)
    
    duration = int(1000 / scene.canvas.fps)
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )
    print(f"Saved GIF to: {output_path}")


if __name__ == "__main__":
    from .dsl import SceneSpec, ObjectSpec, ObjectType, MotionSpec, CanvasSpec
    from .physics import simulate_scene
    
    print("Testing Renderer with Particles...")
    
    scene = SceneSpec(
        objects=[ObjectSpec(ObjectType.BALL, color="#FF0000", size=10)],
        motion=MotionSpec(start_pos=(64, 20), velocity=(3, 0), gravity=0.5, bounce=0.8),
        canvas=CanvasSpec(size=128, frames=40, fps=12, background="#000000"),
        enable_blast=True,
    )
    
    frames, particles = simulate_scene(scene, enable_blast=True)
    render_gif(frames, scene, "test_particles.gif", particles)
    
    if os.path.exists("test_particles.gif"):
        size = os.path.getsize("test_particles.gif")
        print(f"Created test_particles.gif ({size} bytes)")
        os.remove("test_particles.gif")
    
    print("✓ Particle renderer test passed!")
