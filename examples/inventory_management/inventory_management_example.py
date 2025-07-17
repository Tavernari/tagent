
Inventory Management Pipeline Example

This example demonstrates a multi-step inventory management pipeline that:
1. Checks stock levels for a list of products.
2. For products with low stock, it gets quotes from different suppliers in parallel.
3. If a good quote is found, it creates a purchase order.
4. Finally, it summarizes the actions taken.


from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import random

from tagent import run_agent
from tagent.pipeline import PipelineBuilder, ConditionDSL, ExecutionMode

# --- Tool-Specific Pydantic Models (Input/Output Contracts) ---

class StockStatus(BaseModel):
    """Represents the stock status of a single product."""
    product_id: str
    quantity: int
    status: str  # "in_stock", "low_stock", "out_of_stock"

class CheckStockLevelsOutput(BaseModel):
    """Output for the check_stock_levels tool."""
    stock_levels: List[StockStatus]

class SupplierQuote(BaseModel):
    """Represents a quote from a supplier."""
    supplier_name: str
    price: float
    delivery_days: int

class GetSupplierQuotesOutput(BaseModel):
    """Output for the get_supplier_quotes tool."""
    product_id: str
    quotes: List[SupplierQuote]

class PurchaseOrder(BaseModel):
    """Represents a created purchase order."""
    order_id: str
    product_id: str
    quantity: int
    supplier: str
    total_cost: float

class CreatePurchaseOrderOutput(BaseModel):
    """Output for the create_purchase_order tool."""
    purchase_order: Optional[PurchaseOrder] = None
    error: Optional[str] = None

# --- Tools (Fake implementations for demonstration) ---

def check_stock_levels(product_ids: List[str]) -> CheckStockLevelsOutput:
    """Checks the stock levels for a list of product IDs."""
    stock_levels = []
    for product_id in product_ids:
        quantity = random.randint(0, 100)
        status = "in_stock"
        if quantity < 20:
            status = "low_stock"
        if quantity == 0:
            status = "out_of_stock"
        stock_levels.append(StockStatus(product_id=product_id, quantity=quantity, status=status))
    return CheckStockLevelsOutput(stock_levels=stock_levels)

def get_supplier_quotes(product_id: str) -> GetSupplierQuotesOutput:
    """Gets quotes from different suppliers for a given product."""
    quotes = [
        SupplierQuote(supplier_name="SupplierA", price=random.uniform(10.0, 15.0), delivery_days=random.randint(1, 3)),
        SupplierQuote(supplier_name="SupplierB", price=random.uniform(11.0, 16.0), delivery_days=random.randint(2, 4)),
    ]
    return GetSupplierQuotesOutput(product_id=product_id, quotes=quotes)

def create_purchase_order(product_id: str, quantity: int, supplier: str, price: float) -> CreatePurchaseOrderOutput:
    """Creates a purchase order for a product."""
    if quantity <= 0:
        return CreatePurchaseOrderOutput(error="Quantity must be positive.")
    
    order_id = f"PO-{random.randint(1000, 9999)}"
    total_cost = quantity * price
    
    po = PurchaseOrder(
        order_id=order_id,
        product_id=product_id,
        quantity=quantity,
        supplier=supplier,
        total_cost=total_cost,
    )
    return CreatePurchaseOrderOutput(purchase_order=po)

# --- Final Output Model ---

class InventorySummary(BaseModel):
    """The final, structured output of the inventory management pipeline."""
    products_checked: List[str]
    low_stock_products: List[str]
    purchase_orders_created: List[PurchaseOrder]
    summary: str

# --- Pipeline Definition ---

def create_inventory_pipeline(products_to_check: List[str]) -> PipelineBuilder:
    """Creates the inventory management pipeline."""
    
    pipeline_builder = PipelineBuilder(
        "inventory_management_pipeline",
        "A pipeline to check stock, get quotes, and create purchase orders."
    )

    pipeline_builder.step(
        name="check_stock",
        goal=f"Check stock levels for the following products: {', '.join(products_to_check)}",
        output_schema=CheckStockLevelsOutput,
    )

    # For each product, add parallel steps to get quotes if stock is low
    for i, product_id in enumerate(products_to_check):
        pipeline_builder.step(
            name=f"get_quotes_for_{product_id}",
            goal=f"Get supplier quotes for {product_id}",
            depends_on=["check_stock"],
            execution_mode=ExecutionMode.CONCURRENT,
            output_schema=GetSupplierQuotesOutput,
            condition=ConditionDSL.contains(
                left=f"$check_stock.result.stock_levels.{i}.status",
                right="low_stock"
            )
        )

    # For each product, add a step to create a purchase order if a good quote was found
    for i, product_id in enumerate(products_to_check):
        pipeline_builder.step(
            name=f"create_po_for_{product_id}",
            goal=f"Create a purchase order for {product_id} if a good quote is available.",
            depends_on=[f"get_quotes_for_{product_id}"],
            output_schema=CreatePurchaseOrderOutput,
            condition=ConditionDSL.less_than(
                left=f"$get_quotes_for_{product_id}.result.quotes.0.price", # Simplified: checks first quote
                right=12.0
            )
        )

    pipeline_builder.step(
        name="summarize_results",
        goal="Summarize the inventory check and purchase orders created.",
        depends_on=[f"create_po_for_{p}" for p in products_to_check],
        output_schema=InventorySummary,
    )

    return pipeline_builder

# --- Agent Execution ---

if __name__ == "__main__":
    products = ["prod_a", "prod_b", "prod_c"]
    
    inventory_pipeline = create_inventory_pipeline(products).build()

    # The 'goal' for a pipeline execution is a high-level description of the overall objective.
    # The agent will use this to understand the context and orchestrate the pipeline steps.
    pipeline_goal = "Manage the inventory for the given products by checking stock, getting quotes for low-stock items, and creating purchase orders for the best quotes."

    result = run_agent(
        goal_or_pipeline=inventory_pipeline,
        goal=pipeline_goal,
        tools={
            "check_stock_levels": check_stock_levels,
            "get_supplier_quotes": get_supplier_quotes,
            "create_purchase_order": create_purchase_order,
        },
        output_format=InventorySummary,
        max_iterations=10, # Increased for potentially more complex pipeline execution
        verbose=True,
    )

    print("\n--- Inventory Management Pipeline Results ---")
    if result.success:
        print("Pipeline executed successfully!")
        if result.final_output:
            summary = result.final_output
            print(f"\nSummary: {summary.summary}")
            print(f"Products Checked: {summary.products_checked}")
            print(f"Low Stock Products: {summary.low_stock_products}")
            print("Purchase Orders Created:")
            for po in summary.purchase_orders_created:
                print(f"  - Order ID: {po.order_id}, Product: {po.product_id}, Quantity: {po.quantity}, Supplier: {po.supplier}, Cost: ${po.total_cost:.2f}")
    else:
        print("Pipeline execution failed.")
        print(f"Error details: {result.error_details}")

    print("\n--- Execution Stats ---")
    print(f"Total execution time: {result.execution_time:.2f} seconds")
    print(f"Steps completed: {result.steps_completed}")
    print(f"Steps failed: {result.steps_failed}")
    print(f"Steps skipped: {result.steps_skipped}")
