"""
Pipeline Templates for TAgent Pipeline System.

This module provides a registry for reusable pipeline templates, making it easy
to create and share common pipeline structures.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .models import Pipeline, PipelineTemplate, ExecutionMode
from .conditions import ConditionDSL


logger = logging.getLogger(__name__)


class TemplateNotFoundError(Exception):
    """Raised when a pipeline template is not found in the registry."""
    pass


class PipelineTemplateRegistry:
    """Registry for reusable pipeline templates."""

    def __init__(self):
        self.templates: Dict[str, PipelineTemplate] = {}
        self._load_builtin_templates()

    def register_template(self, template: PipelineTemplate):
        """Register a new template."""
        if template.name in self.templates:
            logger.warning(f"Overwriting existing template: {template.name}")
        self.templates[template.name] = template
        logger.info(f"Registered pipeline template: {template.name}")

    def get_template(self, name: str) -> Optional[PipelineTemplate]:
        """Get template by name."""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List all available templates."""
        return list(self.templates.keys())

    def create_pipeline_from_template(
        self,
        template_name: str,
        parameters: Dict[str, Any]
    ) -> Pipeline:
        """Create a pipeline instance from a template."""
        template = self.get_template(template_name)
        if not template:
            raise TemplateNotFoundError(f"Template '{template_name}' not found")

        return template.create_pipeline(parameters)

    def _load_builtin_templates(self):
        """Load built-in pipeline templates."""
        self.register_template(COMPANY_RESEARCH_TEMPLATE)
        self.register_template(ECOMMERCE_ANALYSIS_TEMPLATE)


# --- Built-in Templates ---

def _company_research_factory(params: Dict[str, Any]) -> Pipeline:
    return (
        Pipeline("company_research", f"Company research for {params['company_name']}")
        .step(
            name="web_search",
            goal=f"Search for information about {params['company_name']}",
            constraints=["Use reliable sources", "Focus on recent information"]
        )
        .step(
            name="review_analysis",
            goal=f"Analyze customer reviews for {params['company_name']}",
            depends_on=["web_search"],
            condition=ConditionDSL.equals(
                left=params.get("include_reviews", True),
                right=True
            )
        )
        .step(
            name="social_media_analysis",
            goal=f"Analyze social media presence of {params['company_name']}",
            depends_on=["web_search"],
            condition=ConditionDSL.equals(
                left=params.get("include_social_media", True),
                right=True
            ),
            execution_mode=ExecutionMode.CONCURRENT
        )
        .step(
            name="final_report",
            goal=f"Generate comprehensive report in {params['report_format']} format",
            depends_on=["web_search", "review_analysis", "social_media_analysis"]
        )
    )

COMPANY_RESEARCH_TEMPLATE = PipelineTemplate(
    name="company_research",
    description="A comprehensive company analysis pipeline.",
    parameters={
        "company_name": {"type": "string", "required": True},
        "include_social_media": {"type": "boolean", "default": True},
        "include_reviews": {"type": "boolean", "default": True},
        "report_format": {"type": "string", "default": "markdown"}
    },
    pipeline_factory=_company_research_factory
)


def _ecommerce_analysis_factory(params: Dict[str, Any]) -> Pipeline:
    return (
        Pipeline("ecommerce_analysis", f"E-commerce analysis for {params['product_category']}")
        .step(
            name="product_research",
            goal=f"Research {params['product_category']} products in {params['target_market']}",
            tools_filter=["web_search", "product_search"]
        )
        .step(
            name="competitor_analysis",
            goal=f"Analyze competitors in {params['product_category']} market",
            depends_on=["product_research"],
            execution_mode=ExecutionMode.CONCURRENT
        )
        .step(
            name="market_analysis",
            goal=f"Analyze market trends for {params['product_category']}",
            depends_on=["product_research"],
            execution_mode=ExecutionMode.CONCURRENT
        )
        .step(
            name="opportunity_analysis",
            goal="Identify market opportunities and gaps",
            depends_on=["competitor_analysis", "market_analysis"],
            condition=ConditionDSL.combine_and(
                ConditionDSL.step_succeeded("competitor_analysis"),
                ConditionDSL.step_succeeded("market_analysis")
            )
        )
    )

ECOMMERCE_ANALYSIS_TEMPLATE = PipelineTemplate(
    name="ecommerce_analysis",
    description="An e-commerce product and market analysis pipeline.",
    parameters={
        "product_category": {"type": "string", "required": True},
        "target_market": {"type": "string", "required": True},
        "budget_range": {"type": "string", "default": "all"}
    },
    pipeline_factory=_ecommerce_analysis_factory
)
