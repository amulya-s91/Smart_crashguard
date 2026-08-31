try:
    from .velocity import VelocityTracker, is_sudden_deceleration
except ImportError:
    from velocity import VelocityTracker, is_sudden_deceleration
