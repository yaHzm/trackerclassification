from __future__ import annotations
from abc import ABCMeta
import logging

from ._registry import Registry


class RegistryMeta(ABCMeta):
    """
    Metaclass that enforces the ``NameBase`` / ``SomethingName`` naming convention
    and automatically registers subclasses in :class:`Registry`.

    Convention
    ------------
    1. **Base (interface) classes** must be named ``NameBase`` (e.g., ``ExampleBase``).
    2. **Implementations** must subclass the base and **end with the same suffix
       as the base minus "Base"**.  
       For ``ExampleBase``, the suffix is ``"Example"``.  
       Therefore, valid implementation names include:
       - ``FastExample``
       - ``RobustExample``
       - ``AsyncExample``

       If a subclass does **not** end with the required suffix, a `ValueError` is raised
       immediately at class definition time.

    Behavior
    -----------
    When a new class using this metaclass is **defined** (the `class` statement executes),
    :meth:`__init__` of this metaclass is invoked. This happens *before* the module
    is fully imported.

    During that moment:
        1. The metaclass verifies that the subclass name ends with the correct suffix.
        2. It derives a short registration key by removing the suffix
           (e.g., ``"FastExample" → "Fast"``).
        3. It registers the class with :class:`Registry` under the group
           corresponding to the base (e.g., ``"ExampleBase"``).

    Notes
    -------
    - The base class itself (e.g., ``ExampleBase``) is **not registered**.
    - Registration occurs at **class definition time**, not merely at import time
      (though defining the class usually happens as the module is imported).
    - This system works automatically — no decorators or explicit registration calls
      are needed.

    Attributes:
        _interface_name (str): The configured base class name (e.g., ``"ExampleBase"``).
        _suffix (str): The required suffix for implementations (e.g., ``"Example"``).
    """

    _interface_name: str = ""
    _suffix: str = ""

    def __class_getitem__(cls, interface_name: str):
        """
        Configure the metaclass for a specific base interface.

        Usage:
            ``metaclass = RegistryMeta["ExampleBase"]``

        Args:
            interface_name (str): The **base class name**, which must end with ``"Base"``.

        Returns:
            Type[RegistryMeta]: A specialized metaclass bound to this base.

        Raises:
            ValueError: If ``interface_name`` does not end with ``"Base"``.
        """
        logging.info(f"Configuring RegistryMeta for {interface_name}")
        if not interface_name.endswith("Base"):
            raise ValueError(
                f"Base interface name must end with 'Base', got {interface_name!r}."
            )
        suffix = interface_name[: -len("Base")]  # e.g., "ExampleBase" → "Example"

        class CustomMeta(RegistryMeta):
            _interface_name = interface_name
            _suffix = suffix

        CustomMeta.__name__ = f"RegistryMeta[{interface_name}]"
        logging.info(f"Created custom metaclass: {CustomMeta.__name__}")
        return CustomMeta

    def __init__(cls, name, bases, namespace):
        """
        Invoked automatically when a new class using this metaclass is **defined**.

        Behavior:
            - If the metaclass is unconfigured (no ``_interface_name/_suffix``), do nothing.
            - If the class being created **is the base** itself (``name == _interface_name``),
              do nothing.
            - Otherwise (it's an implementation):
                1. Verify that the class name ends with the required suffix.
                2. Compute the short key (``name`` without the suffix).
                3. Register the class with :class:`Registry`.

        Raises:
            ValueError: If an implementation class does not end with the required suffix.
        """
        # Delegate to ABCMeta first
        super().__init__(name, bases, namespace)

        # Unconfigured base metaclass — nothing to do
        if not cls._suffix or not cls._interface_name:
            return

        # Base interface class itself — not registered
        if name == cls._interface_name:
            return
        
        if name.endswith("Meta"):
            # Avoid registering metaclass helper classes
            return
        
        if name.endswith("Base"):
            # Avoid registering other base classes
            return

        # Enforce naming rule
        if not name.endswith(cls._suffix):
            raise ValueError(
                f"Class '{name}' must end with '{cls._suffix}' "
                f"because it subclasses '{cls._interface_name}'."
            )

        # Derive short key and register
        short_name = name[: -len(cls._suffix)]
        logging.info(f"Registering {name} as {short_name} in {cls._interface_name}")
        Registry.register(cls._interface_name, short_name, cls)