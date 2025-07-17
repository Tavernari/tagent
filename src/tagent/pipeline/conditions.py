"""
Advanced condition classes for TAgent Pipeline system.

This module provides expressive, type-safe condition classes for conditional
step execution in pipelines. These classes offer a more intuitive and
maintainable way to define execution conditions.
"""

import logging
from typing import Any, Callable, Dict, List, Literal, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- New Expressive Condition Classes ---

class BaseCondition(ABC):
    """Base class for all pipeline conditions."""
    operator: str
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate the condition against the given context.
        
        Args:
            context: Dictionary containing step results and execution context
            
        Returns:
            bool: True if condition is met, False otherwise
        """
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """String representation of the condition."""
        pass


@dataclass
class DataExists(BaseCondition):
    """Condition that checks if data exists at a given path."""
    
    path: str
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if data exists at the specified path."""
        if '.' in self.path:
            step_name, _, attr_name = self.path.partition('.')
            step_result = context.get(step_name)
            if step_result is None:
                return False
            return hasattr(step_result, attr_name) and getattr(step_result, attr_name) is not None
        else:
            return self.path in context and context[self.path] is not None
    
    def __str__(self) -> str:
        return f"DataExists({self.path})"


@dataclass
class IsGreaterThan(BaseCondition):
    """Condition that checks if a numeric value is greater than a threshold."""
    
    path: str
    value: Union[int, float]
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if the value at path is greater than the threshold."""
        actual_value = self._get_value(context)
        if actual_value is None:
            return False
        
        try:
            return float(actual_value) > float(self.value)
        except (ValueError, TypeError):
            return False
    
    def _get_value(self, context: Dict[str, Any]) -> Any:
        """Extract value from context using the path."""
        if '.' in self.path:
            step_name, _, attr_name = self.path.partition('.')
            step_result = context.get(step_name)
            if step_result is None:
                return None
            return getattr(step_result, attr_name, None)
        else:
            return context.get(self.path)
    
    def __str__(self) -> str:
        return f"IsGreaterThan({self.path} > {self.value})"


@dataclass
class IsLessThan(BaseCondition):
    """Condition that checks if a numeric value is less than a threshold."""
    
    path: str
    value: Union[int, float]
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if the value at path is less than the threshold."""
        actual_value = self._get_value(context)
        if actual_value is None:
            return False
        
        try:
            return float(actual_value) < float(self.value)
        except (ValueError, TypeError):
            return False
    
    def _get_value(self, context: Dict[str, Any]) -> Any:
        """Extract value from context using the path."""
        if '.' in self.path:
            step_name, _, attr_name = self.path.partition('.')
            step_result = context.get(step_name)
            if step_result is None:
                return None
            return getattr(step_result, attr_name, None)
        else:
            return context.get(self.path)
    
    def __str__(self) -> str:
        return f"IsLessThan({self.path} < {self.value})"


@dataclass
class IsEqualTo(BaseCondition):
    """Condition that checks if a value equals a specific value."""
    
    path: str
    value: Any
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if the value at path equals the expected value."""
        actual_value = self._get_value(context)
        return actual_value == self.value
    
    def _get_value(self, context: Dict[str, Any]) -> Any:
        """Extract value from context using the path."""
        if '.' in self.path:
            step_name, _, attr_name = self.path.partition('.')
            step_result = context.get(step_name)
            if step_result is None:
                return None
            return getattr(step_result, attr_name, None)
        else:
            return context.get(self.path)
    
    def __str__(self) -> str:
        return f"IsEqualTo({self.path} == {self.value})"


@dataclass
class Contains(BaseCondition):
    """Condition that checks if a string or list contains a specific value."""
    
    path: str
    value: Any
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if the value at path contains the expected value."""
        actual_value = self._get_value(context)
        if actual_value is None:
            return False
        
        try:
            return self.value in actual_value
        except TypeError:
            return False
    
    def _get_value(self, context: Dict[str, Any]) -> Any:
        """Extract value from context using the path."""
        if '.' in self.path:
            step_name, _, attr_name = self.path.partition('.')
            step_result = context.get(step_name)
            if step_result is None:
                return None
            return getattr(step_result, attr_name, None)
        else:
            return context.get(self.path)
    
    def __str__(self) -> str:
        return f"Contains({self.path} contains {self.value})"


@dataclass
class IsEmpty(BaseCondition):
    """Condition that checks if a value is empty (None, empty string, empty list, etc.)."""
    
    path: str
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if the value at path is empty."""
        actual_value = self._get_value(context)
        
        if actual_value is None:
            return True
        
        if isinstance(actual_value, str):
            return len(actual_value.strip()) == 0
        
        if isinstance(actual_value, (list, dict, tuple)):
            return len(actual_value) == 0
        
        return False
    
    def _get_value(self, context: Dict[str, Any]) -> Any:
        """Extract value from context using the path."""
        if '.' in self.path:
            step_name, _, attr_name = self.path.partition('.')
            step_result = context.get(step_name)
            if step_result is None:
                return None
            return getattr(step_result, attr_name, None)
        else:
            return context.get(self.path)
    
    def __str__(self) -> str:
        return f"IsEmpty({self.path})"


class CombinedCondition(BaseCondition):
    """Base class for combining multiple conditions."""
    
    def __init__(self, *conditions: BaseCondition):
        self.conditions = conditions
    
    def __str__(self) -> str:
        condition_strs = [str(c) for c in self.conditions]
        return f"{self.__class__.__name__}({', '.join(condition_strs)})"


class And(CombinedCondition):
    """Condition that requires ALL sub-conditions to be true."""
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """All conditions must be true."""
        return all(condition.evaluate(context) for condition in self.conditions)


class Or(CombinedCondition):
    """Condition that requires ANY sub-condition to be true."""
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Any condition must be true."""
        return any(condition.evaluate(context) for condition in self.conditions)


class Not(BaseCondition):
    """Condition that negates another condition."""
    
    def __init__(self, condition: BaseCondition):
        self.condition = condition
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Negate the wrapped condition."""
        return not self.condition.evaluate(context)
    
    def __str__(self) -> str:
        return f"Not({self.condition})"


# --- Legacy Condition Models (for backward compatibility) ---

class ValueReference(BaseModel):
    """Represents a reference to a value in the pipeline context."""
    ref: str = Field(description="Reference to a value, e.g., 'step_name.result.key'")

    def __str__(self) -> str:
        return f"${self.ref}"


ConditionValue = Union[ValueReference, str, int, float, bool, list, dict, None]

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
        """Get value from nested dictionary using dot notation with array indexing support."""
        keys = key.split('.')
        current = data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            elif hasattr(current, k):
                current = getattr(current, k)
            elif isinstance(current, list) and k.isdigit():
                # Handle array indexing for lists
                index = int(k)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
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