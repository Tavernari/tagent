
# Inventory Management Pipeline Example

# This example demonstrates a realistic multi-step inventory management pipeline that:
# 1. Checks stock levels for a list of products (prod_a=15, prod_b=8, prod_c=45)
# 2. For products with low stock (<20), it gets quotes from different suppliers in parallel
# 3. If a cost-effective quote is found (<$12.00), it creates a purchase order
# 4. Finally, it summarizes all actions taken across the pipeline
#
# Expected behavior:
# - prod_a: Low stock (15) → Get quotes → Good quote ($11.50) → Create PO
# - prod_b: Low stock (8) → Get quotes → Expensive quotes → Skip PO
# - prod_c: Good stock (45) → Skip quotes → Skip PO
# 
# This demonstrates conditional pipeline execution where steps are intelligently
# skipped based on business logic, while dependent steps continue to execute.


from typing import Optional, List
from pydantic import BaseModel
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
    
    # Create a more realistic scenario that guarantees we see the full pipeline
    # We'll ensure at least one product is low stock and one has good stock
    predefined_stocks = {
        "prod_a": {"quantity": 15, "status": "low_stock"},  # Will trigger quotes
        "prod_b": {"quantity": 8, "status": "low_stock"},   # Will trigger quotes
        "prod_c": {"quantity": 45, "status": "in_stock"},   # Will be skipped
    }
    
    for product_id in product_ids:
        if product_id in predefined_stocks:
            data = predefined_stocks[product_id]
            quantity = data["quantity"]
            status = data["status"]
        else:
            # For any other products, use random generation
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
    
    # Create more realistic quotes that will demonstrate both purchase order creation and skipping
    # prod_a will get a good quote (< 12.0), prod_b will get expensive quotes (> 12.0)
    if product_id == "prod_a":
        quotes = [
            SupplierQuote(supplier_name="SupplierA", price=11.50, delivery_days=2),  # Good quote
            SupplierQuote(supplier_name="SupplierB", price=13.20, delivery_days=3),  # Expensive
        ]
    elif product_id == "prod_b":
        quotes = [
            SupplierQuote(supplier_name="SupplierA", price=14.80, delivery_days=1),  # Too expensive
            SupplierQuote(supplier_name="SupplierB", price=15.50, delivery_days=2),  # Too expensive
        ]
    else:
        # For other products, use random generation
        quotes = [
            SupplierQuote(supplier_name="SupplierA", price=random.uniform(10.0, 15.0), delivery_days=random.randint(1, 3)),
            SupplierQuote(supplier_name="SupplierB", price=random.uniform(11.0, 16.0), delivery_days=random.randint(2, 4)),
        ]
    
    return GetSupplierQuotesOutput(product_id=product_id, quotes=quotes)

def create_purchase_order(product_id: str, quantity: int, supplier: str, price: float) -> CreatePurchaseOrderOutput:
    """Creates a purchase order for a product."""
    if quantity <= 0:
        return CreatePurchaseOrderOutput(error="Quantity must be positive.")
    
    # Generate a realistic order ID with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d")
    order_id = f"PO-{timestamp}-{random.randint(1000, 9999)}"
    
    # Calculate reorder quantity based on low stock situation
    # For low stock products, order enough to reach a reasonable level
    reorder_quantity = max(50, quantity * 3)  # Reorder to 50+ units
    total_cost = reorder_quantity * price
    
    po = PurchaseOrder(
        order_id=order_id,
        product_id=product_id,
        quantity=reorder_quantity,
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
        tools=[check_stock_levels],  # Step-specific tool
    )

    # For each product, add parallel steps to get quotes if stock is low
    for i, product_id in enumerate(products_to_check):
        pipeline_builder.step(
            name=f"get_quotes_for_{product_id}",
            goal=f"Get supplier quotes for {product_id}",
            depends_on=["check_stock"],
            execution_mode=ExecutionMode.CONCURRENT,
            output_schema=GetSupplierQuotesOutput,
            condition=ConditionDSL.equals(
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
    
    print("\n🏭 INVENTORY MANAGEMENT PIPELINE DEMO")
    print("=====================================\n")
    print("This example demonstrates:")
    print("• Conditional pipeline execution based on stock levels")
    print("• Parallel quote gathering for low-stock products")
    print("• Purchase order creation for cost-effective quotes")
    print("• Comprehensive result summarization\n")
    
    inventory_pipeline = create_inventory_pipeline(products).build()

    # The 'goal' for a pipeline execution is a high-level description of the overall objective.
    # The agent will use this to understand the context and orchestrate the pipeline steps.
    pipeline_goal = "Manage the inventory for the given products by checking stock, getting quotes for low-stock items, and creating purchase orders for the best quotes."

    result = run_agent(
        goal_or_pipeline=inventory_pipeline,
        # Global tools available to all steps unless overridden
        tools=[
            get_supplier_quotes,
            create_purchase_order,
        ],
        output_format=InventorySummary,
        max_iterations=10, # Increased for potentially more complex pipeline execution
        verbose=False,
        model="openrouter/google/gemini-2.5-flash",
    )

    print("\n📊 INVENTORY MANAGEMENT RESULTS")
    print("===============================")
    if result.success:
        print("✅ Pipeline executed successfully!")
        if result.final_output:
            summary = result.final_output
            print(f"\n📋 Summary: {summary.summary}")
            print(f"\n🔍 Products Checked: {', '.join(summary.products_checked)}")
            print(f"⚠️  Low Stock Products: {', '.join(summary.low_stock_products) if summary.low_stock_products else 'None'}")
            
            if summary.purchase_orders_created:
                print("\n🛒 Purchase Orders Created:")
                for po in summary.purchase_orders_created:
                    print(f"  • Order ID: {po.order_id}")
                    print(f"    Product: {po.product_id} | Quantity: {po.quantity} units")
                    print(f"    Supplier: {po.supplier} | Total Cost: ${po.total_cost:.2f}")
                    print(f"    Unit Price: ${po.total_cost/po.quantity:.2f}")
            else:
                print("\n🛒 Purchase Orders Created: None (no cost-effective quotes found)")
        
        # Show detailed pipeline step results
        print("\n🔍 DETAILED PIPELINE STEP RESULTS")
        print("=================================")
        if hasattr(result, 'step_outputs') and result.step_outputs:
            for step_name, step_result in result.step_outputs.items():
                print(f"\n📋 Step: {step_name}")
                print(f"   Status: ✅ Completed")
                
                # Access the actual step result data
                if hasattr(step_result, 'data'):
                    data = step_result.data
                elif hasattr(step_result, '__dict__'):
                    data = step_result.__dict__
                else:
                    data = step_result
                
                # Show condensed results for readability
                if isinstance(data, dict):
                    if 'stock_levels' in data:
                        print(f"   Result: Found {len(data['stock_levels'])} products")
                        for stock in data['stock_levels']:
                            if isinstance(stock, dict):
                                print(f"     • {stock['product_id']}: {stock['quantity']} units ({stock['status']})")
                            else:
                                print(f"     • {stock.product_id}: {stock.quantity} units ({stock.status})")
                    elif 'quotes' in data:
                        print(f"   Result: {len(data['quotes'])} quotes for {data.get('product_id', 'unknown')}")
                        for quote in data['quotes']:
                            if isinstance(quote, dict):
                                print(f"     • {quote['supplier_name']}: ${quote['price']:.2f} ({quote['delivery_days']} days)")
                            else:
                                print(f"     • {quote.supplier_name}: ${quote.price:.2f} ({quote.delivery_days} days)")
                    elif 'purchase_order' in data:
                        po = data['purchase_order']
                        if po:
                            if isinstance(po, dict):
                                print(f"   Result: Created PO {po['order_id']} for {po['product_id']}")
                                print(f"     • Quantity: {po['quantity']} units from {po['supplier']}")
                                print(f"     • Total Cost: ${po['total_cost']:.2f}")
                            else:
                                print(f"   Result: Created PO {po.order_id} for {po.product_id}")
                                print(f"     • Quantity: {po.quantity} units from {po.supplier}")
                                print(f"     • Total Cost: ${po.total_cost:.2f}")
                        else:
                            print(f"   Result: No purchase order created")
                    else:
                        print(f"   Result: {str(data)[:100]}...")
                elif hasattr(data, '__dict__'):
                    # Handle Pydantic models
                    data_dict = data.__dict__
                    if 'stock_levels' in data_dict:
                        print(f"   Result: Found {len(data_dict['stock_levels'])} products")
                        for stock in data_dict['stock_levels']:
                            if isinstance(stock, dict):
                                print(f"     • {stock['product_id']}: {stock['quantity']} units ({stock['status']})")
                            else:
                                print(f"     • {stock.product_id}: {stock.quantity} units ({stock.status})")
                    elif 'quotes' in data_dict:
                        print(f"   Result: {len(data_dict['quotes'])} quotes for {data_dict.get('product_id', 'unknown')}")
                        for quote in data_dict['quotes']:
                            if isinstance(quote, dict):
                                print(f"     • {quote['supplier_name']}: ${quote['price']:.2f} ({quote['delivery_days']} days)")
                            else:
                                print(f"     • {quote.supplier_name}: ${quote.price:.2f} ({quote.delivery_days} days)")
                    elif 'purchase_order' in data_dict:
                        po = data_dict['purchase_order']
                        if po:
                            if isinstance(po, dict):
                                print(f"   Result: Created PO {po['order_id']} for {po['product_id']}")
                                print(f"     • Quantity: {po['quantity']} units from {po['supplier']}")
                                print(f"     • Total Cost: ${po['total_cost']:.2f}")
                            else:
                                print(f"   Result: Created PO {po.order_id} for {po.product_id}")
                                print(f"     • Quantity: {po.quantity} units from {po.supplier}")
                                print(f"     • Total Cost: ${po.total_cost:.2f}")
                        else:
                            print(f"   Result: No purchase order created")
                    else:
                        print(f"   Result: {str(data_dict)[:100]}...")
                else:
                    print(f"   Result: {str(data)[:100]}...")
        
        # Show information about skipped steps
        if hasattr(result, 'steps_skipped') and result.steps_skipped > 0:
            print(f"\n⏭️  {result.steps_skipped} steps were skipped due to conditions")
            print("   (Steps that didn't meet their execution conditions)")
            
        # Show step metadata if available
        if hasattr(result, 'step_metadata') and result.step_metadata:
            print(f"\n📊 Step Execution Metadata:")
            for step_name, metadata in result.step_metadata.items():
                if metadata:
                    print(f"   {step_name}: {metadata}")
                    
        # Show failed steps if any
        if hasattr(result, 'failed_steps') and result.failed_steps:
            print(f"\n❌ Failed Steps: {', '.join(result.failed_steps)}")
            for step_name in result.failed_steps:
                if hasattr(result, 'error_details') and step_name in result.error_details:
                    print(f"   {step_name}: {result.error_details[step_name]}")
    else:
        print("❌ Pipeline execution failed.")
        print(f"Error details: {result.error_details}")

    print("\n📈 EXECUTION STATISTICS")
    print("=======================")
    print(f"⏱️  Total execution time: {result.execution_time:.2f} seconds")
    print(f"✅ Steps completed: {result.steps_completed}")
    print(f"❌ Steps failed: {result.steps_failed}")
    print(f"⏭️  Steps skipped: {result.steps_skipped}")
    
    print("\n💡 PIPELINE FLOW EXPLANATION")
    print("=============================")
    print("1. ✅ check_stock - Always executed to check all product stock levels")
    print("2. 🔄 get_quotes_for_* - Only executed for products with low stock")
    print("3. 🛒 create_po_for_* - Only executed if quotes are cost-effective (< $12.00)")
    print("4. 📋 summarize_results - Always executed to provide final summary")
    print("\nThis demonstrates conditional execution where steps are intelligently")
    print("skipped based on business logic, while dependent steps still execute.")
