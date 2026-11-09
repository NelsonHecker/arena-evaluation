class ArenaEvaluationError(Exception):
    """Base exception for arena_evaluation pipeline."""

    pass


class MetricCalculationError(ArenaEvaluationError):
    """Raised when a metric calculator fails."""

    pass


class CircularDependencyError(ArenaEvaluationError):
    """Raised when metric calculators have a circular dependency."""

    pass


class SchemaViolationError(ArenaEvaluationError):
    """Raised when data schema validation fails."""

    pass


class RobotNotFoundError(ArenaEvaluationError):
    """Raised when robot parameters cannot be resolved."""

    pass


class ManifestGenerationError(ArenaEvaluationError):
    """Raised when generating or validating a manifest fails."""

    pass


class MetricInputError(ArenaEvaluationError):
    """Raised when a metric calculator's required input is missing; the registry re-raises it instead of recording a null."""

    pass


class AcousticsError(MetricInputError):
    """Raised when the acoustic metric cannot run because its world or barrier map is missing."""

    pass
