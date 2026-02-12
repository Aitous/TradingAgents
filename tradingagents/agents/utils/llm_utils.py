from typing import Any, Dict, List, Union

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def parse_llm_response(response_content: Union[str, List[Union[str, Dict[str, Any]]]]) -> str:
    """
    Parse content from an LLM response, handling both string and list formats.

    This function standardizes extraction of text from various LLM provider response formats
    (e.g., standard strings vs Anthropic's block format).

    Args:
        response_content: The raw content field from an LLM response object.

    Returns:
        The extracted text content as a string.
    """
    if isinstance(response_content, list):
        return "\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in response_content
        )

    return str(response_content) if response_content is not None else ""


def create_and_invoke_chain(
    llm: Any, tools: List[Any], system_message: str, messages: List[BaseMessage]
) -> Any:
    """
    Create and invoke a standard agent chain with tools.

    Args:
        llm: The Language Model to use
        tools: List of tools to bind to the LLM
        system_message: The system prompt content
        messages: The chat history messages

    Returns:
        The LLM response (AIMessage)
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Ensure at least one non-system message for Gemini compatibility
    # Gemini API requires at least one HumanMessage in addition to SystemMessage
    if not messages:
        messages = [
            HumanMessage(content="Please provide your analysis based on the context above.")
        ]

    chain = prompt | llm.bind_tools(tools)
    return chain.invoke({"messages": messages})
