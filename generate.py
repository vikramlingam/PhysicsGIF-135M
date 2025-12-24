#!/usr/bin/env python3
"""
PhysicsGIF Text-to-GIF Generator

Generate animated GIFs from natural language descriptions.

Usage:
    python generate.py                           # Interactive mode
    python generate.py "A red ball bouncing"     # Single prompt
    python generate.py "Blue square" -o out.gif  # Custom output
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import TextToGifPipeline

# Default model path (auto-load fine-tuned model)
DEFAULT_MODEL = "models/PhysicsGIF-135M"


def interactive_mode(pipeline):
    """Run in interactive mode - prompt for input repeatedly."""
    print("\n" + "="*50)
    print("🎬 PhysicsGIF Text-to-GIF Generator")
    print("="*50)
    print("Type a description to generate a GIF.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    count = 1
    while True:
        try:
            prompt = input("Enter prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break
        
        if not prompt:
            continue
        
        if prompt.lower() in ('quit', 'exit', 'q'):
            print("\nGoodbye! 👋")
            break
        
        # Generate unique filename
        output_path = f"output_{count}.gif"
        
        try:
            scene = pipeline.generate(
                text=prompt,
                output_path=output_path,
                verbose=True,
            )
            print(f"\n✓ Generated: {output_path}")
            print(f"  Scene: {scene.objects[0].type.value} ({scene.objects[0].color})")
            print(f"  Frames: {scene.canvas.frames} @ {scene.canvas.fps}fps\n")
            count += 1
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GIFs from text descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate.py                                    # Interactive mode
    python generate.py "A red ball bouncing to the right"
    python generate.py "Blue square falling" -o falling.gif
        """
    )
    parser.add_argument(
        "text",
        type=str,
        nargs="?",  # Make optional for interactive mode
        default=None,
        help="Text description of the animation (optional - runs interactive mode if not provided)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.gif",
        help="Output GIF path (default: output.gif)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to fine-tuned model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Use rule-based parser instead of LLM"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    args = parser.parse_args()
    
    # Determine model path
    model_path = None if args.no_model else args.model
    
    # Check if model exists
    if model_path and not os.path.exists(model_path):
        print(f"Note: Model not found at {model_path}, using rule-based parser.")
        model_path = None
    
    # Initialize pipeline (loads model once)
    print("Loading PhysicsGIF...")
    pipeline = TextToGifPipeline(model_path=model_path)
    
    # Interactive mode if no text provided
    if args.text is None:
        interactive_mode(pipeline)
        return
    
    # Single generation mode
    scene = pipeline.generate(
        text=args.text,
        output_path=args.output,
        verbose=not args.quiet,
    )
    
    if not args.quiet:
        print(f"\n✓ Generated: {args.output}")
        print(f"  Scene: {scene.objects[0].type.value} ({scene.objects[0].color})")
        print(f"  Frames: {scene.canvas.frames} @ {scene.canvas.fps}fps")


if __name__ == "__main__":
    main()
