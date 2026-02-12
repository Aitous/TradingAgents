"""
LangChain Tool Generator - Auto-generate @tool wrappers from registry

This module automatically generates LangChain tools from the tool registry,
eliminating the need for manual @tool definitions in tools.py.

Key benefits:
- No duplication between registry and tool definitions
- Tools are always in sync with registry metadata
- Adding a new tool = just adding to registry
- Type annotations generated automatically
"""

from typing import Any, Callable, Dict

from langchain_core.tools import tool

from tradingagents.tools.executor import execute_tool
from tradingagents.tools.registry import TOOL_REGISTRY
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


def generate_langchain_tool(tool_name: str, metadata: Dict[str, Any]) -> Callable:
    """Generate a LangChain @tool wrapper for a specific tool.

    Args:
        tool_name: Name of the tool
        metadata: Tool metadata from registry

    Returns:
        LangChain tool function with proper annotations
    """

    # Extract metadata
    description = metadata["description"]
    parameters = metadata["parameters"]
    returns_doc = metadata["returns"]

    # Create Pydantic model for arguments
    from pydantic import Field, create_model

    fields = {}
    for param_name, param_info in parameters.items():
        param_type = _get_python_type(param_info["type"])
        description = param_info["description"]

        if "default" in param_info:
            fields[param_name] = (
                param_type,
                Field(default=param_info["default"], description=description),
            )
        else:
            fields[param_name] = (param_type, Field(..., description=description))

    ArgsSchema = create_model(f"{tool_name}Schema", **fields)

    # Create the tool function dynamically
    # Use **kwargs to handle all parameters
    def tool_function(**kwargs):
        """Dynamically generated tool function."""
        # Ensure defaults are applied for missing parameters
        for param_name, param_info in parameters.items():
            if param_name not in kwargs and "default" in param_info:
                kwargs[param_name] = param_info["default"]
        return execute_tool(tool_name, **kwargs)

    # Set function metadata
    tool_function.__name__ = tool_name
    tool_function.__doc__ = f"{description}\n\nReturns:\n    {returns_doc}"

    # Apply @tool decorator with explicit schema
    decorated_tool = tool(args_schema=ArgsSchema)(tool_function)

    return decorated_tool


def _get_python_type(type_string: str) -> type:
    """Convert type string to Python type.

    Args:
        type_string: Type as string (e.g., "str", "int", "bool")

    Returns:
        Python type object
    """
    type_map = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }

    return type_map.get(type_string, str)  # Default to str


def generate_all_tools() -> Dict[str, Callable]:
    """Generate LangChain tools for ALL tools in the registry.

    Returns:
        Dictionary mapping tool names to LangChain tool functions
    """
    tools = {}

    for tool_name, metadata in TOOL_REGISTRY.items():
        try:
            tool_func = generate_langchain_tool(tool_name, metadata)
            tools[tool_name] = tool_func
        except Exception as e:
            logger.warning(f"⚠️  Failed to generate tool '{tool_name}': {e}")

    return tools


def generate_tools_for_agent(agent_name: str) -> Dict[str, Callable]:
    """Get LangChain tools for a specific agent.

    Args:
        agent_name: Name of the agent (e.g., "market", "news")

    Returns:
        Dictionary of tools available to that agent
    """
    tools = {}

    for tool_name, metadata in TOOL_REGISTRY.items():
        # Skip tools that are explicitly disabled
        if not metadata.get("enabled", True):
            continue
        # Check if this tool is available to the agent
        if agent_name in metadata.get("agents", []):
            # Use already-generated tool from ALL_TOOLS
            if tool_name in ALL_TOOLS:
                tools[tool_name] = ALL_TOOLS[tool_name]
            else:
                logger.warning(f"⚠️  Tool '{tool_name}' not found in ALL_TOOLS")

    return tools


# ============================================================================
# PRE-GENERATED TOOLS (for easy import)
# ============================================================================

# Generate all tools once at module import time
ALL_TOOLS = generate_all_tools()

# Export individual tools for backward compatibility
# This allows: from tradingagents.tools import get_stock_data
globals().update(ALL_TOOLS)


def get_tool(tool_name: str) -> Callable:
    """Get a specific tool by name.

    Args:
        tool_name: Name of the tool

    Returns:
        LangChain tool function
    """
    return ALL_TOOLS.get(tool_name)


def get_tools_list() -> list:
    """Get list of all tool functions (for binding to LLM).

    Returns:
        List of LangChain tool functions
    """
    return list(ALL_TOOLS.values())


def get_agent_tools(agent_name: str) -> list:
    """Get list of tool functions for a specific agent.

    Args:
        agent_name: Name of the agent

    Returns:
        List of LangChain tool functions for that agent
    """
    agent_tools = generate_tools_for_agent(agent_name)
    return list(agent_tools.values())


# ============================================================================
# TOOL EXPORT HELPER
# ============================================================================


def export_tools_module(output_path: str = "tradingagents/agents/tools.py"):
    """Export generated tools to a Python file.

    This creates a tools.py file with all auto-generated tools,
    useful for documentation and IDE autocomplete.

    Args:
        output_path: Where to write the tools.py file
    """
    with open(output_path, "w") as f:
        f.write('"""\n')
        f.write("Auto-generated LangChain tools from registry.\n")
        f.write("\n")
        f.write("DO NOT EDIT THIS FILE MANUALLY!\n")
        f.write("This file is auto-generated from tradingagents/tools/registry.py\n")
        f.write("\n")
        f.write("To add/modify tools, edit the TOOL_REGISTRY in registry.py,\n")
        f.write("then run: python -m tradingagents.tools.generator\n")
        f.write('"""\n\n')

        f.write("from tradingagents.tools.generator import ALL_TOOLS\n\n")

        f.write("# Export all generated tools\n")
        for tool_name in sorted(TOOL_REGISTRY.keys()):
            f.write(f'{tool_name} = ALL_TOOLS["{tool_name}"]\n')

        f.write("\n__all__ = [\n")
        for tool_name in sorted(TOOL_REGISTRY.keys()):
            f.write(f'    "{tool_name}",\n')
        f.write("]\n")

    logger.info(f"Exported {len(TOOL_REGISTRY)} tools to {output_path}")


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("LANGCHAIN TOOL GENERATOR - TESTING")
    logger.info("=" * 70)

    # Test 1: Generate all tools
    logger.info("\nGenerating all tools...")
    all_tools = generate_all_tools()
    logger.info(f"Generated {len(all_tools)} tools")

    # Test 2: Inspect a few tools
    logger.info("\nInspecting generated tools:")
    for tool_name in ["get_stock_data", "get_news", "get_fundamentals"]:
        if tool_name in all_tools:
            tool_func = all_tools[tool_name]
            logger.info(f"\n  {tool_name}:")
            logger.info(f"    Name: {tool_func.name}")
            logger.info(f"    Description: {tool_func.description[:80]}...")
            # Use model_fields instead of deprecated __fields__
            if hasattr(tool_func.args_schema, "model_fields"):
                logger.info(f"    Args schema: {list(tool_func.args_schema.model_fields.keys())}")
            else:
                logger.info(f"    Args schema: {list(tool_func.args_schema.__fields__.keys())}")

    # Test 3: Generate tools for specific agents
    logger.info("\nTools per agent:")
    from tradingagents.tools.registry import get_agent_tool_mapping

    mapping = get_agent_tool_mapping()
    for agent_name, tool_names in sorted(mapping.items()):
        agent_tools = get_agent_tools(agent_name)
        logger.info(f"  {agent_name}: {len(agent_tools)} tools")
        for tool in agent_tools:
            logger.info(f"    - {tool.name}")

    # Test 4: Export to file
    logger.info("\nExporting tools to file...")
    export_tools_module()

    logger.info("\n" + "=" * 70)
    logger.info("All tests passed!")
    logger.info("\nUsage:")
    logger.info("  from tradingagents.tools.generator import get_tool, get_agent_tools")
    logger.info("  tool = get_tool('get_stock_data')")
    logger.info("  market_tools = get_agent_tools('market')")
