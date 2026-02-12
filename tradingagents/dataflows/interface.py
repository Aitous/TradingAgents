# Import from vendor-specific modules

# ============================================================================
# LEGACY COMPATIBILITY LAYER
# ============================================================================
# This module now only provides backward compatibility.
# All new code should use tradingagents.tools.executor.execute_tool() directly.
# ============================================================================


def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support.

    DEPRECATED: This function now delegates to the new execute_tool() from the registry system.
    Use tradingagents.tools.executor.execute_tool() directly in new code.

    This function is kept for backward compatibility only.
    """
    from tradingagents.tools.executor import execute_tool

    # Delegate to new system
    return execute_tool(method, *args, **kwargs)
