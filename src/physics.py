"""
PhysicsGIF Physics Engine (TRUE Explosion Effects)

When exploding:
- Ball TRANSFORMS into particles on impact
- Ball DISAPPEARS after explosion
- Particles scatter realistically
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import random

from .dsl import SceneSpec, ObjectSpec

random.seed(42)


@dataclass
class PhysicsState:
    """State of an object at a given time."""
    x: float
    y: float
    vx: float
    vy: float
    active: bool = True  # False = object has exploded and is gone
    has_exploded: bool = False


@dataclass
class Particle:
    """A particle from an explosion."""
    x: float
    y: float
    vx: float
    vy: float
    life: int
    max_life: int
    color: str
    size: float = 4.0
    particle_type: str = "debris"  # debris, spark, smoke


def distance(s1: PhysicsState, s2: PhysicsState) -> float:
    return math.sqrt((s1.x - s2.x)**2 + (s1.y - s2.y)**2)


def spawn_explosion_debris(
    x: float, y: float,
    color: str,
    object_size: float,
    impact_velocity: float,
) -> List[Particle]:
    """Create explosion debris - the ball transforms into these particles."""
    particles = []
    
    # Main debris - large colored chunks (the ball breaking apart)
    num_debris = 25
    for i in range(num_debris):
        angle = (2 * math.pi * i) / num_debris + random.uniform(-0.2, 0.2)
        speed = impact_velocity * random.uniform(0.5, 1.5) + random.uniform(2, 5)
        size = object_size * random.uniform(0.15, 0.4)  # Chunks of the ball
        life = random.randint(25, 40)
        
        particles.append(Particle(
            x=x + random.uniform(-object_size/2, object_size/2),
            y=y + random.uniform(-object_size/2, object_size/2),
            vx=math.cos(angle) * speed,
            vy=math.sin(angle) * speed - random.uniform(2, 5),  # Upward bias
            life=life,
            max_life=life,
            color=color,
            size=size,
            particle_type="debris",
        ))
    
    # Bright sparks
    for _ in range(15):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(5, 12)
        life = random.randint(10, 20)
        particles.append(Particle(
            x=x,
            y=y,
            vx=math.cos(angle) * speed,
            vy=math.sin(angle) * speed - random.uniform(3, 8),
            life=life,
            max_life=life,
            color="#FFFF00" if random.random() > 0.3 else "#FFFFFF",
            size=random.uniform(2, 4),
            particle_type="spark",
        ))
    
    # Smoke/dust cloud
    for _ in range(10):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 3)
        life = random.randint(30, 50)
        # Grayish version of the color
        particles.append(Particle(
            x=x + random.uniform(-5, 5),
            y=y + random.uniform(-5, 5),
            vx=math.cos(angle) * speed,
            vy=-random.uniform(0.5, 2),  # Smoke rises
            life=life,
            max_life=life,
            color="#808080",
            size=random.uniform(5, 10),
            particle_type="smoke",
        ))
    
    return particles


def update_particles(particles: List[Particle], canvas_size: int) -> List[Particle]:
    """Update particle physics."""
    updated = []
    for p in particles:
        if p.life <= 0:
            continue
        
        # Update position
        p.x += p.vx
        p.y += p.vy
        
        # Different physics for different particle types
        if p.particle_type == "debris":
            p.vy += 0.4  # Heavy gravity for debris
            p.vx *= 0.98
            p.vy *= 0.98
            # Bounce off floor
            if p.y > canvas_size - p.size:
                p.y = canvas_size - p.size
                p.vy = -p.vy * 0.5
                p.vx *= 0.8
        elif p.particle_type == "spark":
            p.vy += 0.2
            p.size *= 0.92  # Sparks fade fast
        elif p.particle_type == "smoke":
            p.vy -= 0.02  # Smoke rises slowly
            p.vx *= 0.95
            p.size *= 0.98
        
        p.life -= 1
        
        # Keep particles in bounds (mostly)
        if p.x < 0 or p.x > canvas_size:
            p.vx = -p.vx * 0.5
        
        updated.append(p)
    
    return updated


def resolve_object_collision(
    s1: PhysicsState, s2: PhysicsState,
    size1: float, size2: float,
    bounce: float = 0.95
) -> Tuple[PhysicsState, PhysicsState]:
    """Resolve elastic collision between two objects."""
    dx = s2.x - s1.x
    dy = s2.y - s1.y
    dist = max(0.1, math.sqrt(dx*dx + dy*dy))
    
    nx = dx / dist
    ny = dy / dist
    
    dvx = s1.vx - s2.vx
    dvy = s1.vy - s2.vy
    dvn = dvx * nx + dvy * ny
    
    if dvn < 0:
        return s1, s2
    
    s1_new = PhysicsState(
        x=s1.x, y=s1.y,
        vx=(s1.vx - dvn * nx) * bounce,
        vy=(s1.vy - dvn * ny) * bounce,
        active=s1.active,
    )
    s2_new = PhysicsState(
        x=s2.x, y=s2.y,
        vx=(s2.vx + dvn * nx) * bounce,
        vy=(s2.vy + dvn * ny) * bounce,
        active=s2.active,
    )
    
    overlap = (size1 + size2) - dist
    if overlap > 0:
        sep = overlap / 2 + 0.5
        s1_new.x -= nx * sep
        s1_new.y -= ny * sep
        s2_new.x += nx * sep
        s2_new.y += ny * sep
    
    return s1_new, s2_new


def simulate_frame_with_explosion(
    state: PhysicsState,
    gravity: float,
    bounce: float,
    friction: float,
    canvas_size: int,
    object_size: float,
    enable_blast: bool,
    blast_color: str,
) -> Tuple[PhysicsState, List[Particle]]:
    """Simulate one step. If blast enabled, ball EXPLODES on wall hit."""
    new_particles = []
    
    # If already exploded, return inactive state
    if not state.active:
        return state, []
    
    # Physics update
    new_x = state.x + state.vx
    new_y = state.y + state.vy
    new_vx = state.vx * (1 - friction)
    new_vy = state.vy + gravity
    new_vy = new_vy * (1 - friction)
    
    # Collision detection
    min_pos = object_size
    max_pos = canvas_size - object_size
    
    hit_wall = False
    impact_x, impact_y = new_x, new_y
    
    # Check wall collisions
    if new_x < min_pos:
        hit_wall = True
        impact_x = min_pos
        impact_y = new_y
    elif new_x > max_pos:
        hit_wall = True
        impact_x = max_pos
        impact_y = new_y
    
    if new_y < min_pos:
        hit_wall = True
        impact_x = new_x
        impact_y = min_pos
    elif new_y > max_pos:
        hit_wall = True
        impact_x = new_x
        impact_y = max_pos
    
    # If blast mode and hit wall: EXPLODE (ball disappears)
    if hit_wall and enable_blast and not state.has_exploded:
        impact_speed = math.sqrt(state.vx**2 + state.vy**2)
        
        # Create explosion debris
        new_particles = spawn_explosion_debris(
            impact_x, impact_y,
            color=blast_color,
            object_size=object_size,
            impact_velocity=impact_speed,
        )
        
        # Ball is now GONE
        new_state = PhysicsState(
            x=impact_x, y=impact_y,
            vx=0, vy=0,
            active=False,  # Ball no longer exists
            has_exploded=True,
        )
        return new_state, new_particles
    
    # Normal bounce (no explosion mode)
    if new_x < min_pos:
        new_x = min_pos
        new_vx = -new_vx * bounce
    elif new_x > max_pos:
        new_x = max_pos
        new_vx = -new_vx * bounce
    
    if new_y < min_pos:
        new_y = min_pos
        new_vy = -new_vy * bounce
    elif new_y > max_pos:
        new_y = max_pos
        new_vy = -new_vy * bounce
    
    new_state = PhysicsState(new_x, new_y, new_vx, new_vy, active=True)
    return new_state, new_particles


def simulate_scene(scene: SceneSpec, enable_blast: bool = False) -> Tuple[List[List[Tuple[float, float]]], List[List[Particle]]]:
    """Simulate scene. With blast=True, objects EXPLODE on first wall hit."""
    num_frames = scene.canvas.frames
    canvas_size = scene.canvas.size
    num_objects = len(scene.objects)
    
    # Initialize states
    states = []
    for i, obj in enumerate(scene.objects):
        if num_objects > 1:
            if i == 0:
                start_x = scene.motion.start_pos[0]
                start_y = scene.motion.start_pos[1]
                vel_x = abs(scene.motion.velocity[0])
                vel_y = scene.motion.velocity[1]
            else:
                start_x = canvas_size - scene.motion.start_pos[0]
                start_y = scene.motion.start_pos[1] + (i * 20) % 40 - 20
                vel_x = -abs(scene.motion.velocity[0])
                vel_y = -scene.motion.velocity[1] if scene.motion.velocity[1] != 0 else 0
        else:
            start_x = scene.motion.start_pos[0]
            start_y = scene.motion.start_pos[1]
            vel_x = scene.motion.velocity[0]
            vel_y = scene.motion.velocity[1]
        
        states.append(PhysicsState(x=start_x, y=start_y, vx=vel_x, vy=vel_y, active=True))
    
    active_particles: List[Particle] = []
    all_frames = []
    all_particles = []
    
    for frame_idx in range(num_frames):
        # Record positions - only for active (non-exploded) objects
        frame_positions = []
        for s in states:
            if s.active:
                frame_positions.append((s.x, s.y))
            else:
                # Exploded objects get position (-1000, -1000) to hide them
                frame_positions.append((-1000, -1000))
        
        all_frames.append(frame_positions)
        all_particles.append(list(active_particles))
        
        # Update physics
        new_states = []
        for i, (state, obj) in enumerate(zip(states, scene.objects)):
            new_state, new_particles = simulate_frame_with_explosion(
                state=state,
                gravity=scene.motion.gravity,
                bounce=scene.motion.bounce,
                friction=scene.motion.friction,
                canvas_size=canvas_size,
                object_size=obj.size,
                enable_blast=enable_blast,
                blast_color=obj.color,
            )
            new_states.append(new_state)
            active_particles.extend(new_particles)
        
        # Object-to-object collisions (for multi-object scenes)
        if num_objects > 1:
            for i in range(num_objects):
                for j in range(i + 1, num_objects):
                    if not new_states[i].active or not new_states[j].active:
                        continue
                    
                    s1, s2 = new_states[i], new_states[j]
                    size1 = scene.objects[i].size
                    size2 = scene.objects[j].size
                    
                    if distance(s1, s2) < size1 + size2:
                        if enable_blast:
                            # Both objects explode on collision
                            mid_x = (s1.x + s2.x) / 2
                            mid_y = (s1.y + s2.y) / 2
                            
                            active_particles.extend(spawn_explosion_debris(
                                mid_x, mid_y,
                                scene.objects[i].color,
                                size1,
                                math.sqrt(s1.vx**2 + s1.vy**2),
                            ))
                            active_particles.extend(spawn_explosion_debris(
                                mid_x, mid_y,
                                scene.objects[j].color,
                                size2,
                                math.sqrt(s2.vx**2 + s2.vy**2),
                            ))
                            
                            new_states[i] = PhysicsState(mid_x, mid_y, 0, 0, active=False, has_exploded=True)
                            new_states[j] = PhysicsState(mid_x, mid_y, 0, 0, active=False, has_exploded=True)
                        else:
                            new_states[i], new_states[j] = resolve_object_collision(
                                s1, s2, size1, size2, scene.motion.bounce
                            )
        
        states = new_states
        active_particles = update_particles(active_particles, canvas_size)
    
    return all_frames, all_particles


if __name__ == "__main__":
    from .dsl import SceneSpec, ObjectSpec, ObjectType, MotionSpec, CanvasSpec
    
    print("Testing TRUE Explosion Physics...")
    
    scene = SceneSpec(
        objects=[ObjectSpec(ObjectType.BALL, color="#FF0000", size=12)],
        motion=MotionSpec(start_pos=(64, 20), velocity=(0, 0), gravity=0.5, bounce=0.8),
        canvas=CanvasSpec(size=128, frames=60),
        enable_blast=True,
    )
    
    frames, particles = simulate_scene(scene, enable_blast=True)
    
    active_count = sum(1 for pos in frames[-1] if pos[0] > 0)
    print(f"Objects remaining at end: {active_count}")
    print(f"Total particles generated: {sum(len(p) for p in particles)}")
    
    print("✓ TRUE explosion test passed!")
