"""
Conditional Execution for TAgent Pipeline System.

This module implements the logic for conditional execution of pipeline steps,
allowing steps to run based on the results of previous steps.
"""

import logging
from typing import Any, Callable, Dict, List, Literal, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Condition Models ---

class ValueReference(BaseModel):
    """Represents a reference to a value in the pipeline context."""
    ref: str = Field(description="Reference to a value, e.g., 'step_name.result.key'")

    def __str__(self) -> str:
        return f"${self.ref}"


ConditionValue = Union[ValueReference, str, int, float, bool, list, dict, None]


class BaseCondition(BaseModel):
    """Base model for a condition."""
    operator: str


class EqualsCondition(BaseCondition):
    """Condition to check for equality."""
    operator: Literal['equals'] = 'equals'
    left: ConditionValue
    right: ConditionValue


class NotEqualsCondition(BaseCondition):
    """Condition to check for inequality."""
    operator: Literal['not_equals'] = 'not_equals'
    left: ConditionValue
    right: ConditionValue


class ContainsCondition(BaseCondition):
    """Condition to check if a container has an item."""
    operator: Literal['contains'] = 'contains'
    container: ConditionValue
    item: ConditionValue


class GreaterThanCondition(BaseCondition):
    """Condition for numerical greater than comparison."""
    operator: Literal['greater_than'] = 'greater_than'
    left: ConditionValue
    right: ConditionValue


class LessThanCondition(BaseCondition):
    """Condition for numerical less than comparison."""
    operator: Literal['less_than'] = 'less_than'
    left: ConditionValue
    right: ConditionValue


class ExistsCondition(BaseCondition):
    """Condition to check if a key exists in the context."""
    operator: Literal['exists'] = 'exists'
    key: str


class NotExistsCondition(BaseCondition):
    """Condition to check if a key does not exist in the context."""
    operator: Literal['not_exists'] = 'not_exists'
    key: str


class AndCondition(BaseCondition):
    """Logical AND for multiple conditions."""
    operator: Literal['and'] = 'and'
    conditions: List['AnyCondition']


class OrCondition(BaseCondition):
    """Logical OR for multiple conditions."""
    operator: Literal['or'] = 'or'
    conditions: List['AnyCondition']


AnyCondition = Union[
    EqualsCondition,
    NotEqualsCondition,
    ContainsCondition,
    GreaterThanCondition,
    LessThanCondition,
    ExistsCondition,
    NotExistsCondition,
    AndCondition,
    OrCondition,
]

# For recursive models in Pydantic
AndCondition.model_rebuild()
OrCondition.model_rebuild()


# --- Condition Evaluator ---

class ConditionEvaluator:
    """Evaluates conditions for conditional step execution."""

    def __init__(self) -> None:
        self.operators: Dict[str, Callable[..., bool]] = {
            'equals': self._equals,
            'not_equals': self._not_equals,
            'contains': self._contains,
            'greater_than': self._greater_than,
            'less_than': self._less_than,
            'exists': self._exists,
            'not_exists': self._not_exists,
            'and': self._and,
            'or': self._or,
        }

    def evaluate(self, condition: AnyCondition, context: Dict[str, Any]) -> bool:
        """Evaluate a condition against the current context."""
        if not isinstance(condition, BaseCondition):
            return bool(condition)

        eval_func = self.operators.get(condition.operator)
        if not eval_func:
            raise ValueError(f"Unknown operator: {condition.operator}")

        return eval_func(condition, context)

    def _resolve_value(self, value: ConditionValue, context: Dict[str, Any]) -> Any:
        """Resolve a value that might be a reference to context data."""
        if isinstance(value, ValueReference):
            return self._get_nested_value(context, value.ref)
        if isinstance(value, str) and value.startswith('$'):
            key = value[1:]
            return self._get_nested_value(context, key)
        return value

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key.split('.')
        current = data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            elif hasattr(current, k):
                current = getattr(current, k)
            else:
                return None
        return current

    def _equals(self, condition: EqualsCondition, context: Dict[str, Any]) -> bool:
        left = self._resolve_value(condition.left, context)
        right = self._resolve_value(condition.right, context)
        return left == right

    def _not_equals(self, condition: NotEqualsCondition, context: Dict[str, Any]) -> bool:
        left = self._resolve_value(condition.left, context)
        right = self._resolve_value(condition.right, context)
        return left != right

    def _contains(self, condition: ContainsCondition, context: Dict[str, Any]) -> bool:
        container = self._resolve_value(condition.container, context)
        item = self._resolve_value(condition.item, context)
        if container is None:
            return False
        return item in container

    def _greater_than(self, condition: GreaterThanCondition, context: Dict[str, Any]) -> bool:
        left = self._resolve_value(condition.left, context)
        right = self._resolve_value(condition.right, context)
        try:
            return float(left) > float(right)
        except (ValueError, TypeError):
            return False

    def _less_than(self, condition: LessThanCondition, context: Dict[str, Any]) -> bool:
        left = self._resolve_value(condition.left, context)
        right = self._resolve_value(condition.right, context)
        try:
            return float(left) < float(right)
        except (ValueError, TypeError):
            return False

    def _exists(self, condition: ExistsCondition, context: Dict[str, Any]) -> bool:
        return self._get_nested_value(context, condition.key) is not None

    def _not_exists(self, condition: NotExistsCondition, context: Dict[str, Any]) -> bool:
        return self._get_nested_value(context, condition.key) is None

    def _and(self, condition: AndCondition, context: Dict[str, Any]) -> bool:
        return all(self.evaluate(cond, context) for cond in condition.conditions)

    def _or(self, condition: OrCondition, context: Dict[str, Any]) -> bool:
        return any(self.evaluate(cond, context) for cond in condition.conditions)


# --- Condition DSL ---

class ConditionDSL:
    """Domain Specific Language for creating pipeline conditions."""

    @staticmethod
    def step_result_contains(step_name: str, value: Any) -> ContainsCondition:
        """Create condition: step result contains value."""
        return ContainsCondition(
            container=ValueReference(ref=f'{step_name}.result'),
            item=value,
        )

    @staticmethod
    def step_succeeded(step_name: str) -> EqualsCondition:
        """Create condition: step succeeded."""
        return EqualsCondition(
            left=ValueReference(ref=f'{step_name}.status'),
            right='completed',
        )

    @staticmethod
    def data_exists(key: str) -> ExistsCondition:
        """Create condition: data exists in context."""
        return ExistsCondition(key=key)

    @staticmethod
    def combine_and(*conditions: AnyCondition) -> AndCondition:
        """Combine conditions with AND logic."""
        return AndCondition(conditions=list(conditions))

    @staticmethod
    def combine_or(*conditions: AnyCondition) -> OrCondition:
        """Combine conditions with OR logic."""
        return OrCondition(conditions=list(conditions))

    @staticmethod
    def equals(left: ConditionValue, right: ConditionValue) -> EqualsCondition:
        """Create an equals condition."""
        return EqualsCondition(left=left, right=right)
