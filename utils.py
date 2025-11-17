"""UNI-Vaasa, Applied ML / Group20 Project work: RL-Mario utilities."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
from math import ceil

# --- Frame Preprocessing ---
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    return frame / 255.0

# --- Reward Plotting ---
def plot_rewards(metrics, save_path="assets/reward_plot.png"):
    """Plot multiple training metrics for richer diagnostics."""
    if not metrics:
        return

    plot_meta = {
        "reward": ("Episode Reward", "Reward"),
        "forward_progress": ("Forward Progress", "Δx per Episode"),
        "avg_speed": ("Average Speed", "x per Step"),
        "air_steps": ("Air Time", "Steps in Air"),
        "high_jumps": ("High Jumps", "Count"),
        "max_air_chain": ("Max Air Chain", "Consecutive Air Steps"),
        "avg_loss": ("Average Loss", "Loss"),
        "epsilon": ("Exploration Rate", "Epsilon"),
        "stuck_termination": ("Stuck Terminations", "Episodes"),
    }

    keys = list(metrics.keys())
    num_plots = len(keys)
    cols = 2
    rows = ceil(num_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3.5))
    axes = axes.flatten().tolist() if isinstance(axes, np.ndarray) else [axes]

    for idx, key in enumerate(keys):
        ax = axes[idx]
        data = metrics.get(key)
        if not data:
            ax.set_visible(False)
            continue
        title, ylabel = plot_meta.get(
            key, (key.replace("_", " ").title(), key)
        )
        data_episodes = range(1, len(data) + 1)
        ax.plot(data_episodes, data, label=title, color="#1f77b4")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    # Hide any unused axes
    for ax in axes[num_plots:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- Video Writer Setup ---
def create_video_writer(path="assets/mario_training.mp4", fps=30, resolution=(256, 240)):
    # Use H.264 codec for better macOS compatibility
    # Try 'avc1' first (H.264), fallback to 'mp4v' if not available
    import sys
    if sys.platform == 'darwin':  # macOS
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    writer = cv2.VideoWriter(path, fourcc, fps, resolution)
    if not writer.isOpened():
        # Fallback to mp4v if avc1 doesn't work
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, resolution)
    return writer

# --- Visual Display Setup ---
class MarioViewer:
    """Display Mario game using pygame."""
    def __init__(self, scale=3, caption="Mario RL Agent"):
        """
        Initialize the viewer.
        
        Args:
            scale: Scale factor for the display (default 3 means 3x larger)
            caption: Window caption
        """
        self.scale = scale
        self.caption = caption
        self.screen = None
        self.clock = None
        self.initialized = False
        
    def init(self, frame_shape):
        """Initialize pygame display."""
        try:
            pygame.init()
            # Frame shape is (height, width, channels)
            height, width = frame_shape[:2]
            self.screen = pygame.display.set_mode((width * self.scale, height * self.scale))
            pygame.display.set_caption(self.caption)
            self.clock = pygame.time.Clock()
            self.initialized = True
            return True
        except Exception as e:
            print(f"⚠️  Could not initialize pygame display: {e}")
            return False
    
    def display_frame(self, frame, fps=30):
        """
        Display a frame.
        
        Args:
            frame: RGB frame as numpy array (height, width, 3)
            fps: Target frames per second
        """
        if not self.initialized:
            if not self.init(frame.shape):
                return False
        
        try:
            # Handle pygame events (prevents window from freezing)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # Convert numpy array to pygame surface
            # Frame is (height, width, 3) RGB
            # Convert to uint8 if needed
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            
            # pygame.surfarray expects (width, height, 3) format
            # Our frame is (height, width, 3), so we need to transpose
            # Swap axes: (height, width, 3) -> (width, height, 3)
            frame_transposed = np.transpose(frame, (1, 0, 2))
            
            # Create surface from array
            # Note: pygame uses RGB format which matches our frame
            frame_surface = pygame.surfarray.make_surface(frame_transposed)
            
            # Scale the frame if needed
            if self.scale != 1:
                new_width = frame.shape[1] * self.scale
                new_height = frame.shape[0] * self.scale
                frame_surface = pygame.transform.scale(frame_surface, (new_width, new_height))
            
            # Display the frame
            self.screen.blit(frame_surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(fps)
            return True
        except Exception as e:
            # If surfarray doesn't work, try alternative method
            try:
                # Alternative: use PIL/Pillow as intermediary
                from PIL import Image
                img = Image.fromarray(frame, 'RGB')
                img = img.resize((frame.shape[1] * self.scale, frame.shape[0] * self.scale))
                frame_surface = pygame.image.fromstring(img.tobytes(), img.size, img.mode)
                self.screen.blit(frame_surface, (0, 0))
                pygame.display.flip()
                self.clock.tick(fps)
                return True
            except Exception as e2:
                # Final fallback: create surface and set pixels manually
                try:
                    height, width = frame.shape[:2]
                    frame_surface = pygame.Surface((width, height))
                    # Convert RGB to pygame format
                    frame_rgb = np.flipud(frame)  # Flip vertically for pygame coordinates
                    pygame.surfarray.blit_array(frame_surface, np.transpose(frame_rgb, (1, 0, 2)))
                    
                    if self.scale != 1:
                        new_width = width * self.scale
                        new_height = height * self.scale
                        frame_surface = pygame.transform.scale(frame_surface, (new_width, new_height))
                    
                    self.screen.blit(frame_surface, (0, 0))
                    pygame.display.flip()
                    self.clock.tick(fps)
                    return True
                except Exception as e3:
                    if not hasattr(self, '_display_warning_shown'):
                        print(f"⚠️  Error displaying frame: {e}")
                        print(f"   All display methods failed. Visual display disabled.")
                        self._display_warning_shown = True
                    return False
    
    def close(self):
        """Close the display."""
        if self.initialized:
            pygame.quit()
            self.initialized = False
