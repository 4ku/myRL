import jax
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def log_info(writer: SummaryWriter, info: dict, global_step: int, prefix: str = ""):
    """
    Recursively log any data from info dictionary.
    - For arrays: log statistics (min, max, mean, std, shape)
    - For scalars: log directly
    - For nested dicts: recurse with updated prefix
    """
    for key, value in info.items():
        log_key = f"{prefix}/{key}" if prefix else key
        
        # Handle nested dictionaries
        if isinstance(value, dict):
            log_info(writer, value, global_step, prefix=log_key)
            continue
        
        # Convert to numpy/jax array if needed
        try:
            value = jax.device_get(value)
        except:
            pass
        
        # Handle arrays/tensors
        if isinstance(value, (np.ndarray, jax.Array)) or hasattr(value, 'shape'):
            value = np.asarray(value)
            
            # Log array statistics
            if value.size > 1:
                writer.add_scalar(f"{log_key}/mean", float(np.mean(value)), global_step)
                writer.add_scalar(f"{log_key}/min", float(np.min(value)), global_step)
                writer.add_scalar(f"{log_key}/max", float(np.max(value)), global_step)
                writer.add_scalar(f"{log_key}/std", float(np.std(value)), global_step)
                # Log shape as text occasionally (every 100 steps)
                if global_step % 100 == 0:
                    writer.add_text(f"{log_key}/shape", str(value.shape), global_step)
            else:
                # Single element array - log as scalar
                writer.add_scalar(log_key, float(value.item()), global_step)
        
        # Handle scalar values (int, float, bool)
        elif isinstance(value, (int, float, bool, np.number)):
            writer.add_scalar(log_key, float(value), global_step)
        
        # Handle other types (log as text)
        else:
            try:
                writer.add_text(log_key, str(value), global_step)
            except:
                pass  # Skip if we can't log it

def compute_returns(episode_transitions, gamma):
    """Compute discounted returns for each transition in the episode."""
    returns = []
    G = 0.0
    # Compute returns backwards from end of episode
    for transition in reversed(episode_transitions):
        G = transition["reward"] + gamma * G
        returns.append(G)
    returns.reverse()
    
    # Add returns to transitions
    for i, transition in enumerate(episode_transitions):
        transition["return"] = returns[i]
    
    return episode_transitions