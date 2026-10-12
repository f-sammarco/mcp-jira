import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

import requests
from fastmcp import Context
from fastmcp.exceptions import ToolError
from requests.exceptions import HTTPError

from mcp_atlassian.exceptions import MCPAtlassianAuthenticationError

logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Awaitable[Any]])

HTTP_ERROR_BODY_MAX_CHARS = 500
HTTP_ERROR_PREFIX_FORMAT = "HTTP {status} {reason}"
HTTP_ERROR_BODY_SEPARATOR = ": "
HTTP_ERROR_RETRY_AFTER_FORMAT = "; Retry-After={value}"


def _describe_error(error: Exception) -> str:
    """Build the error text; HTTPError gets status, body snippet, Retry-After.

    Format is parsed by clients: "HTTP <3-digit status>" and "Retry-After=<seconds>".
    """
    response = getattr(error, "response", None)
    if not isinstance(error, HTTPError) or response is None:
        return str(error).strip() or type(error).__name__
    reason = response.reason or ""
    detail = HTTP_ERROR_PREFIX_FORMAT.format(
        status=response.status_code, reason=reason
    ).strip()
    body = " ".join((response.text or "").split())[:HTTP_ERROR_BODY_MAX_CHARS]
    if body:
        detail += HTTP_ERROR_BODY_SEPARATOR + body
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        detail += HTTP_ERROR_RETRY_AFTER_FORMAT.format(value=retry_after)
    return detail


def handle_tool_errors(func: F) -> F:
    """
    Convert tool handler exceptions to ToolError with details preserved.

    Raising ToolError from inside the tool preserves actionable Atlassian API
    errors for MCP clients even when the FastMCP server masks non-ToolError
    exceptions.
    """
    tool_name = func.__name__

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except ToolError:
            raise
        except Exception as e:
            detail = _describe_error(e)
            logger.error(f"Error in tool '{tool_name}': {detail}", exc_info=True)
            message = f"Error calling tool '{tool_name}': {detail}"
            raise ToolError(message) from e

    return wrapper  # type: ignore


def check_write_access(func: F) -> F:
    """
    Decorator for FastMCP tools to check if the application is in read-only mode.
    If in read-only mode, it raises a ToolError.
    Also catches unhandled exceptions and re-raises them as ToolError with
    descriptive error messages.
    Assumes the decorated function is async and has `ctx: Context` as its
    first argument.
    """
    tool_name = func.__name__

    @handle_tool_errors
    @wraps(func)
    async def wrapper(ctx: Context, *args: Any, **kwargs: Any) -> Any:
        lifespan_ctx_dict = ctx.request_context.lifespan_context
        app_lifespan_ctx = (
            lifespan_ctx_dict.get("app_lifespan_context")
            if isinstance(lifespan_ctx_dict, dict)
            else None
        )  # type: ignore

        if app_lifespan_ctx is not None and app_lifespan_ctx.read_only:
            action_description = tool_name.replace(
                "_", " "
            )  # e.g., "create_issue" -> "create issue"
            logger.warning(f"Attempted to call tool '{tool_name}' in read-only mode.")
            message = f"Cannot {action_description} in read-only mode."
            raise ValueError(message)

        return await func(ctx, *args, **kwargs)

    return wrapper  # type: ignore


def handle_auth_errors(
    service_name: str = "Atlassian API",
) -> Callable:
    """Decorator to handle 401/403 HTTPError as auth errors.

    Only catches HTTPError with 401/403 status codes and raises
    MCPAtlassianAuthenticationError. All other exceptions pass
    through unmodified.

    Args:
        service_name: Name of the service for error messages.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return func(self, *args, **kwargs)
            except HTTPError as http_err:
                if http_err.response is not None and http_err.response.status_code in [
                    401,
                    403,
                ]:
                    error_msg = (
                        f"Authentication failed for "
                        f"{service_name} "
                        f"({http_err.response.status_code}). "
                        "Token may be expired or invalid. "
                        "Please verify credentials."
                    )
                    logger.error(error_msg)
                    raise MCPAtlassianAuthenticationError(error_msg) from http_err
                raise  # re-raise non-auth HTTPError

        return wrapper

    return decorator


def handle_atlassian_api_errors(service_name: str = "Atlassian API") -> Callable:
    """
    Decorator to handle common Atlassian API exceptions (Jira, Confluence, etc.).

    Args:
        service_name: Name of the service for error logging (e.g., "Jira API").
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return func(self, *args, **kwargs)
            except HTTPError as http_err:
                if http_err.response is not None and http_err.response.status_code in [
                    401,
                    403,
                ]:
                    error_msg = (
                        f"Authentication failed for {service_name} "
                        f"({http_err.response.status_code}). "
                        "Token may be expired or invalid. Please verify credentials."
                    )
                    logger.error(error_msg)
                    raise MCPAtlassianAuthenticationError(error_msg) from http_err
                else:
                    operation_name = getattr(func, "__name__", "API operation")
                    logger.error(
                        f"HTTP error during {operation_name}: {http_err}",
                        exc_info=False,
                    )
                    raise http_err
            except KeyError as e:
                operation_name = getattr(func, "__name__", "API operation")
                logger.error(f"Missing key in {operation_name} results: {str(e)}")
                message = (
                    f"{operation_name} returned an unexpected response from "
                    f"{service_name}: missing key {e}"
                )
                raise ValueError(message) from e
            except requests.RequestException as e:
                operation_name = getattr(func, "__name__", "API operation")
                logger.error(f"Network error during {operation_name}: {str(e)}")
                message = f"Network error during {operation_name}: {e}"
                raise ValueError(message) from e
            except (ValueError, TypeError) as e:
                operation_name = getattr(func, "__name__", "API operation")
                logger.error(f"Error processing {operation_name} results: {str(e)}")
                message = f"Error processing {operation_name} results: {e}"
                raise ValueError(message) from e
            except Exception as e:  # noqa: BLE001 - Intentional fallback with logging
                operation_name = getattr(func, "__name__", "API operation")
                logger.error(f"Unexpected error during {operation_name}: {str(e)}")
                logger.debug(
                    f"Full exception details for {operation_name}:", exc_info=True
                )
                message = f"Unexpected error during {operation_name}: {e}"
                raise RuntimeError(message) from e

        return wrapper

    return decorator
