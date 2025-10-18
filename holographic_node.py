# file: holographic.py
"""
Pragmatic implementation of the "holographic" processing concept.

- GlobalSystemState: shared dict + change notifications (thread-safe).
- EntanglementField: builds a Superposition from transforms in either
  "chain" (cumulative) or "branch" (independent) mode.
- Superposition: scored selection with error-tolerant collapse.
- HolographicNode: orchestrates pipeline vs. superposition collapse.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Literal,
)

Observer = Callable[[Any], float]
Transform = Callable[[Any], Any]


# ----------------------------- Global state -----------------------------------

class GlobalSystemState:
    """Store shared state and notify interested observers when it changes.

    Thread-safe; observers are called outside the lock to avoid reentrancy deadlocks.
    """

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}
        self._observers: List[Callable[[Mapping[str, Any]], None]] = []
        self._lock = RLock()

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key* and broadcast the updated state."""
        with self._lock:
            self._state[key] = value
            snapshot = dict(self._state)  # avoid exposing internal dict
            observers = list(self._observers)
        # Notify outside lock (why: user callbacks may call back into us).
        for observer in observers:
            try:
                observer(snapshot)
            except Exception:
                # Notifications must not break state updates.
                pass

    def get(self, key: str, default: Any | None = None) -> Any:
        """Return the stored value for *key* or *default* when absent."""
        with self._lock:
            return self._state.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy of the current state."""
        with self._lock:
            return dict(self._state)

    def subscribe(self, callback: Callable[[Mapping[str, Any]], None]) -> "Subscription":
        """Register *callback* to be invoked whenever the state changes."""
        with self._lock:
            if callback not in self._observers:
                self._observers.append(callback)
        return Subscription(self, callback)

    def unsubscribe(self, callback: Callable[[Mapping[str, Any]], None]) -> None:
        """Remove *callback* if it was previously registered."""
        with self._lock:
            try:
                self._observers.remove(callback)
            except ValueError:
                pass

    @contextmanager
    def subscribed(self, callback: Callable[[Mapping[str, Any]], None]):
        """Context manager that subscribes for the duration of the 'with' block."""
        sub = self.subscribe(callback)
        try:
            yield sub
        finally:
            sub.cancel()


@dataclass(frozen=True)
class Subscription:
    """Disposable subscription handle."""
    state: GlobalSystemState
    callback: Callable[[Mapping[str, Any]], None]

    def cancel(self) -> None:
        self.state.unsubscribe(self.callback)


# ----------------------------- Superposition ----------------------------------

@dataclass(frozen=True)
class Outcome:
    """Result of applying a transform to a value."""
    transform: Transform
    value: Any | None
    error: BaseException | None = None


@dataclass(frozen=True)
class Superposition:
    """Collection of candidate outcomes produced by multiple transforms."""

    source: Any
    outcomes: Tuple[Outcome, ...]  # immutable ordering

    def _iter_valid(self) -> Iterator[Outcome]:
        for oc in self.outcomes:
            if oc.error is None:
                yield oc

    def collapse(self, observer: Optional[Observer] = None) -> Any:
        """Select and return the best candidate according to *observer*.

        If *observer* is None the first valid outcome is returned.
        When no valid outcomes exist the original ``source`` is returned unchanged.
        """
        valid = list(self._iter_valid())
        if not valid:
            return self.source
        if observer is None:
            return valid[0].value
        best_score = float("-inf")
        best_val: Any = valid[0].value
        for oc in valid:
            score = observer(oc.value)
            if score > best_score:
                best_score = score
                best_val = oc.value
        return best_val

    def rank(self, observer: Observer) -> List[Tuple[Outcome, float]]:
        """Return outcomes paired with scores, sorted descending by score."""
        scored: List[Tuple[Outcome, float]] = []
        for oc in self._iter_valid():
            scored.append((oc, observer(oc.value)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored


# ----------------------------- Entanglement -----------------------------------

class EntanglementField:
    """Generate :class:`Superposition` objects from candidate transforms.

    mode='chain': outcomes are cumulative (x->t1(x)->t2(t1(x))->...).
    mode='branch': outcomes are independent (each transform sees original input).
    """

    def __init__(self, *, mode: Literal["chain", "branch"] = "chain") -> None:
        self.mode = mode

    def create_superposition(
        self,
        input_tensor: Any,
        transforms: Iterable[Transform],
    ) -> Superposition:
        outcomes: List[Outcome] = []
        if self.mode == "chain":
            current = input_tensor
            for transform in transforms:
                try:
                    current = transform(current)
                    outcomes.append(Outcome(transform, current, None))
                except BaseException as exc:  # keep other candidates alive
                    outcomes.append(Outcome(transform, None, exc))
        else:  # branch
            for transform in transforms:
                try:
                    value = transform(input_tensor)
                    outcomes.append(Outcome(transform, value, None))
                except BaseException as exc:
                    outcomes.append(Outcome(transform, None, exc))
        return Superposition(source=input_tensor, outcomes=tuple(outcomes))


# ----------------------------- Node -------------------------------------------

class HolographicNode:
    """Process inputs using a pool of transforms and a scoring function.

    First `process()` call runs the full pipeline (sequential transforms),
    marks the node as expanded, and updates global state. Subsequent calls
    create a superposition and collapse it with the configured observer.
    """

    def __init__(
        self,
        transforms: Optional[Iterable[Transform]] = None,
        *,
        client_aesthetic_profile: Optional[Observer] = None,
        superposition_mode: Literal["chain", "branch"] = "chain",
    ) -> None:
        self.local_state: Dict[str, Any] = {}
        self.global_consciousness = GlobalSystemState()
        self.quantum_field = EntanglementField(mode=superposition_mode)
        self._transforms: List[Transform] = list(transforms or [])
        self.client_aesthetic_profile: Observer = (
            client_aesthetic_profile if client_aesthetic_profile is not None else (lambda _: 0.0)
        )

    # --- transform management

    def add_transform(self, transform: Transform) -> None:
        """Register an additional candidate transform."""
        self._transforms.append(transform)
        self.local_state["expanded"] = False  # why: transform set changed

    def set_transforms(self, transforms: Sequence[Transform]) -> None:
        """Replace the entire transform list."""
        self._transforms = list(transforms)
        self.local_state["expanded"] = False

    def clear_transforms(self) -> None:
        """Remove all transforms."""
        self._transforms.clear()
        self.local_state["expanded"] = False

    def enumerate_possible_transforms(self) -> Iterator[Transform]:
        """Yield the currently registered transforms."""
        yield from self._transforms

    # --- lifecycle

    def requires_expansion(self) -> bool:
        """Return True when the node should run the full pipeline."""
        return not self.local_state.get("expanded", False)

    def spawn_full_pipeline(self, input_tensor: Any) -> Any:
        """Run the transforms sequentially and mark the node as expanded."""
        result = input_tensor
        for transform in self._transforms:
            result = transform(result)
        self.local_state["expanded"] = True
        self.global_consciousness.set("last_result", result)
        return result

    def process(self, input_tensor: Any) -> Any:
        """Process *input_tensor* using either the full pipeline or scoring."""
        if self.requires_expansion():
            return self.spawn_full_pipeline(input_tensor)
        superposition = self.quantum_field.create_superposition(
            input_tensor, self.enumerate_possible_transforms()
        )
        return superposition.collapse(observer=self.client_aesthetic_profile)


__all__ = [
    "EntanglementField",
    "GlobalSystemState",
    "HolographicNode",
    "Superposition",
    "Outcome",
    "Subscription",
]