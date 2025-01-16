__version__ = "0.0.3"


class MechirConfig:
    """Configuration manager for the mechir package."""

    _instance = None
    _config = {
        "ignore-official": False,  # default value
        # Add other default config options here
    }

    def __new__(cls):
        # Singleton pattern to ensure only one config instance exists
        if cls._instance is None:
            cls._instance = super(MechirConfig, cls).__new__(cls)
        return cls._instance

    def __call__(self, key, value=None):
        """
        Get or set a configuration value using function-style access.

        Args:
            key (str): The configuration key
            value (Any, optional): If provided, sets the configuration value

        Returns:
            The configuration value if value=None, otherwise None

        Raises:
            KeyError: If the configuration key doesn't exist
        """
        if key not in self._config:
            raise KeyError(f"Unknown configuration option: {key}")

        if value is not None:
            self._config[key] = value
            return None

        return self._config[key]

    def __getitem__(self, key):
        """Enable dict-style access for getting values: config['key']"""
        if key not in self._config:
            raise KeyError(f"Unknown configuration option: {key}")
        return self._config[key]

    def __setitem__(self, key, value):
        """Enable dict-style access for setting values: config['key'] = value"""
        if key not in self._config:
            raise KeyError(f"Unknown configuration option: {key}")
        self._config[key] = value

    def __contains__(self, key):
        """Enable 'in' operator: 'key' in config"""
        return key in self._config

    def reset(self):
        """Reset all configurations to their default values."""
        self._config = {
            "ignore-official": False,
            # Add other default config options here
        }


config = MechirConfig()

from .modelling import Cat, Dot, MonoT5, PatchedMixin, SAEMixin
from .modelling.hooked import conversion
from .modelling.hooked import states
from . import perturb as perturb
from . import data as data
