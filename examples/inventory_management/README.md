# Inventory Management Pipeline Example

This example demonstrates a multi-step inventory management pipeline using TAgent's pipeline features.

## Objective

The goal of this pipeline is to automate the process of checking stock levels, getting quotes from suppliers for low-stock items, and creating purchase orders.

## Pipeline Steps

The pipeline is defined in `inventory_management_example.py` and consists of the following steps:

1.  **`check_stock`**: Checks the stock levels for a predefined list of products.
2.  **`get_quotes_for_<product_id>`**: For each product identified as "low_stock" in the previous step, this step runs in parallel to get quotes from different suppliers. This step is conditional and only runs if the product's stock is low.
3.  **`create_po_for_<product_id>`**: If a supplier quote is below a certain price, this step creates a purchase order for the product. This step is also conditional.
4.  **`summarize_results`**: After all other steps are complete, this step generates a summary of the actions taken, including which products were checked, which were low on stock, and which purchase orders were created.

## How to Run

To run the example, simply execute the python script:

```bash
python examples/inventory_management/inventory_management_example.py
```

## Key Features Demonstrated

This example showcases several key features of the TAgent pipeline system:

*   **Dynamic Pipeline Creation**: The pipeline is created dynamically based on a list of products.
*   **Conditional Execution**: Steps to get quotes and create purchase orders are only executed if certain conditions are met (e.g., low stock, good price).
*   **Parallel Execution**: The `get_quotes_for_<product_id>` steps are configured to run in parallel to speed up the process.
*   **Strongly-Typed Inputs and Outputs**: The tools use Pydantic models to define their input and output schemas, ensuring data integrity throughout the pipeline.
*   **`run_agent` with a Pipeline**: The example shows how to use the `run_agent` function to execute a `Pipeline` object.
