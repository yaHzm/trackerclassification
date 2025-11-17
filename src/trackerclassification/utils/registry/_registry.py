from __future__ import annotations
import logging
from threading import RLock
from typing import Any, Dict, Type


class Registry:
    """
    Central registry for mapping base interfaces (e.g., ``ExampleBase``) to their implementations.

    Concept
    -------
    Implementations are grouped by the **base class name** (e.g., ``"ExampleBase"``).
    Within each group, classes are stored under a short, human-friendly key derived
    from the class name by removing the base suffix. For an interface named
    ``ExampleBase``, the required suffix for implementations is ``"Example"``:

        - ``FastExample``   → key ``"Fast"``
        - ``RobustExample`` → key ``"Robust"``

    The registry is populated automatically by :class:`RegistryMeta` when each
    implementation class is **defined** (class creation time).

    Attributes:
        _registries (Dict[str, Dict[str, Type[Any]]]):
            Internal mapping with the structure
            ``{ "<BaseName>": { "<short_name>": <class> } }``, e.g.
            ``{ "ExampleBase": { "Fast": FastExample } }``.
        _lock (RLock): A re-entrant lock to ensure thread-safe registration.
    """

    _registries: Dict[str, Dict[str, Type[Any]]] = {}
    _lock: RLock = RLock()

    @classmethod
    def register(cls, interface_name: str, name: str, item: Type[Any]) -> None:
        """
        Register a class under a given interface group.

        This is typically invoked automatically by :class:`RegistryMeta`
        when an implementation class is defined.

        Args:
            interface_name (str): The base class name, e.g., ``"ExampleBase"``.
            name (str): The short key under which to store the class, e.g., ``"Fast"``.
            item (Type[Any]): The class object to register.

        Raises:
            ValueError: If ``name`` is already registered under ``interface_name``.
        """
        with cls._lock:
            cls._registries.setdefault(interface_name, {})
            if name in cls._registries[interface_name]:
                raise ValueError(f"{name} already registered under {interface_name}.")
            cls._registries[interface_name][name] = item
            logging.info(f"Registered {name} under {interface_name}.")

    @classmethod
    def get(cls, interface_name: str, name: str) -> Type[Any]:
        """
        Retrieve a registered class by interface and short name.

        Args:
            interface_name (str): The base class name, e.g., ``"ExampleBase"``.
            name (str): The short key, e.g., ``"Fast"``.

        Returns:
            Type[Any]: The registered class.

        Raises:
            ValueError: If the interface group or key does not exist.
        """
        with cls._lock:
            if interface_name not in cls._registries:
                raise ValueError(f"No registry found for {interface_name}.")
            reg = cls._registries[interface_name]
            if name not in reg:
                raise ValueError(
                    f"{name} not registered under {interface_name}. "
                    f"Registered classes are {list(reg.keys())}."
                )
            return reg[name]

    @classmethod
    def get_all(cls, interface_name: str) -> Dict[str, Type[Any]]:
        """
        Get a shallow copy of the registry mapping for a given interface group.

        Args:
            interface_name (str): The base class name, e.g., ``"ExampleBase"``.

        Returns:
            Dict[str, Type[Any]]: A copy of the mapping
            ``{ "<short_name>": <class> }`` for that interface.

        Raises:
            ValueError: If the interface group does not exist.
        """
        with cls._lock:
            if interface_name not in cls._registries:
                raise ValueError(f"No registry found for {interface_name}.")
            return dict(cls._registries[interface_name])

    @classmethod
    def unregister(cls, interface_name: str, name: str) -> None:
        """
        Remove a class from the registry if present.

        Args:
            interface_name (str): The base class name, e.g., ``"ExampleBase"``.
            name (str): The short key, e.g., ``"Fast"``.
        """
        with cls._lock:
            if interface_name in cls._registries:
                cls._registries[interface_name].pop(name, None)

    @classmethod
    def clear(cls, interface_name: str | None = None) -> None:
        """
        Clear one interface registry or **all** registries.

        Args:
            interface_name (str | None): If ``None``, clear all registries.
                Otherwise, clear only the specified interface group.
        """
        with cls._lock:
            if interface_name is None:
                cls._registries.clear()
            else:
                cls._registries.pop(interface_name, None)
