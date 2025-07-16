# Template Resolution Issue Validation

This example reproduces and validates the template resolution issue identified in the TAgent logs.

## Issue Description

The issue occurs when the TAgent tries to use template placeholders like `{{task_1.output.0.url}}` to reference data from previous task outputs, but the template is not properly resolved and is passed literally to the next tool.

### Log Evidence

From the celery worker log:
```
[TASK_EXECUTE] Args: {'url': '{{task_1.output.0.url}}', 'search_goal': "Extract the company's main profile information, including its score, number of complaints, and general reputation."}
URL still contains unresolved template: {{task_1.output.0.url}}
Failed to fetch https://{{task_1.output.0.url}}: HTTPSConnectionPool(host='%7b%7btask_1.output.0.url%7d%7d', port=443): Max retries exceeded...
```

## Root Cause

The template resolution mechanism is not properly replacing `{{task_1.output.0.url}}` with the actual URL value from the previous task's output before passing it to the `load_web_page_tool`.

## Validation Test

Run the validation script to reproduce the issue:

```bash
cd examples/template_resolution_issue
python template_resolution_validation.py
```

### Expected Behavior

The agent should:
1. Execute the search tool and get results with URLs
2. Resolve `{{task_1.output.0.url}}` to the actual URL from step 1
3. Pass the resolved URL to the web page loading tool
4. Successfully load the page content

### Actual Behavior (Issue)

The agent:
1. Executes the search tool and gets results with URLs
2. **FAILS** to resolve `{{task_1.output.0.url}}`
3. Passes the literal template string to the web page loading tool
4. The web page tool fails with a DNS resolution error

## Technical Details

### Task Flow
1. `task_1`: Search tool returns `SearchResult` with URL
2. `task_2`: Uses `{{task_1.output.0.url}}` to reference the first result's URL
3. **BUG**: Template is not resolved, literal string is passed to tool
4. `load_web_page_tool` receives malformed URL and fails

### Fix Requirements

The template resolution system needs to:
1. Parse template placeholders like `{{task_1.output.0.url}}`
2. Navigate the task output structure to find the referenced value
3. Replace the template with the actual value before tool execution
4. Handle nested object access (e.g., `output.0.url` for first result's URL)

## Files

- `template_resolution_validation.py`: Main validation script
- `README.md`: This documentation file

## Usage

This validation case helps verify that template resolution fixes work correctly and prevents regressions in the future.