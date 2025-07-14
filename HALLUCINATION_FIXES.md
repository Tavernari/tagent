# TAgent Hallucination Prevention Improvements

## Critical Issues Found

### 1. JSON Recovery System Corruption
**File**: `src/tagent/llm_adapter.py:129-255`
**Issue**: Complex regex-based JSON repair causing corrupted output (visible as Japanese characters)
**Impact**: High - causes visible corruption in output

### 2. LLM Fallback Without Validation
**File**: `src/tagent/agent.py:128-226`
**Issue**: Automatic tool substitution with LLM responses without validation
**Impact**: Medium - may provide incorrect information

### 3. Automatic Context Loss
**File**: `src/tagent/actions.py:19-69`
**Issue**: Auto-summarization triggers without user awareness
**Impact**: Medium - may lose critical context

### 4. State Machine Override Confusion
**File**: `src/tagent/state_machine.py:208-227`
**Issue**: Forces actions without clear feedback to user
**Impact**: Low - may confuse LLM reasoning

## Recommended Fixes

### Fix 1: Simplify JSON Recovery
```python
def parse_structured_response(json_str: str, verbose: bool = False) -> StructuredResponse:
    """Simplified JSON parsing with minimal recovery attempts."""
    try:
        # Try direct parsing first
        return StructuredResponse.model_validate_json(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        if verbose:
            print(f"[JSON_ERROR] Failed to parse: {e}")
        
        # Only attempt basic cleanup - no complex regex
        try:
            # Remove common escaping issues
            cleaned = json_str.replace('\\"', '"').replace('\\n', '\n')
            return StructuredResponse.model_validate_json(cleaned)
        except:
            # If cleanup fails, return a safe default
            return StructuredResponse(
                action="plan",
                params={},
                reasoning="JSON parsing failed - returning safe default"
            )
```

### Fix 2: Validate LLM Fallback Responses
```python
def _execute_llm_fallback(state, tool_name, tool_args, model_selector, verbose=False):
    """Execute LLM fallback with validation."""
    try:
        # ... existing code ...
        
        # Validate LLM response before storing
        if not llm_result or len(llm_result.strip()) < 10:
            if verbose:
                print(f"[LLM_FALLBACK] Response too short or empty, rejecting")
            return None
            
        # Check for common hallucination patterns
        hallucination_patterns = [
            "I cannot", "I don't have access", "I would need", 
            "as an AI", "I'm not able", "I cannot access"
        ]
        
        if any(pattern in llm_result.lower() for pattern in hallucination_patterns):
            if verbose:
                print(f"[LLM_FALLBACK] Detected hallucination pattern, rejecting")
            return None
            
        # ... rest of existing code ...
```

### Fix 3: Explicit Auto-Summarization Control
```python
def enhanced_goal_evaluation_action(state, model, api_key, **kwargs):
    """Enhanced evaluator with explicit auto-summarization control."""
    # Only auto-summarize if explicitly configured
    auto_summarize = kwargs.get('auto_summarize', False)  # Default to False
    
    if auto_summarize and conversation_history:
        # ... existing auto-summarization logic ...
        if verbose:
            print("[AUTO-SUMMARIZE] Auto-summarization is enabled and triggered")
    else:
        if verbose:
            print("[AUTO-SUMMARIZE] Auto-summarization is disabled")
```

### Fix 4: Transparent State Machine Actions
```python
def get_forced_action(self, rejected_action, agent_data=None):
    """Get forced action with transparency."""
    allowed = self.get_allowed_actions(agent_data)
    
    if allowed:
        forced_action = list(allowed)[0]
        # Log the override for transparency
        print(f"[STATE_MACHINE] Overriding '{rejected_action}' with '{forced_action}'")
        return forced_action
    else:
        print(f"[STATE_MACHINE] Emergency fallback to 'plan' action")
        return ActionType.PLAN
```

### Fix 5: Response Validation Layer
```python
def validate_llm_response(response: StructuredResponse, context: Dict[str, Any]) -> bool:
    """Validate LLM response for hallucination indicators."""
    
    # Check for empty or minimal responses
    if not response.reasoning or len(response.reasoning.strip()) < 20:
        return False
        
    # Check for action consistency
    if response.action not in ["plan", "execute", "summarize", "evaluate", "finalize"]:
        return False
        
    # Check for tool existence in execute actions
    if response.action == "execute":
        tool_name = response.params.get("tool")
        if tool_name and tool_name not in context.get("available_tools", []):
            return False
            
    # Check for repetitive patterns
    words = response.reasoning.lower().split()
    if len(words) > 10:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        # Flag if any word appears more than 20% of the time
        max_frequency = max(word_counts.values()) / len(words)
        if max_frequency > 0.2:
            return False
            
    return True
```

## Implementation Priority

1. **High Priority**: Fix JSON recovery system (Fix 1)
2. **Medium Priority**: Add LLM fallback validation (Fix 2)
3. **Medium Priority**: Add response validation layer (Fix 5)
4. **Low Priority**: Explicit auto-summarization control (Fix 3)
5. **Low Priority**: Transparent state machine actions (Fix 4)

## Testing Strategy

1. Create unit tests for each fix
2. Test with malformed JSON inputs
3. Test with various LLM models
4. Monitor for reduction in corrupted outputs
5. Validate that essential functionality still works

## Monitoring

Add metrics to track:
- JSON parsing failure rates
- LLM fallback usage frequency
- Response validation failure rates
- User-reported hallucination incidents