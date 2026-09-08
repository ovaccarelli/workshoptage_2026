"""Error recovery for the Part 03 multi-tool agent."""

from __future__ import annotations

from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import (
    RawToolArgs,
    ValidatedToolArgs,
    WrapModelRequestHandler,
    WrapToolExecuteHandler,
    WrapToolValidateHandler,
)
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext, ToolDefinition


class RetryAndRecoverCapability(AbstractCapability[None]):
    """Retry failed model/tool calls once and prevent uncaught run failures."""

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        try:
            return await handler(request_context)
        except Exception:
            try:
                return await handler(request_context)
            except Exception as second_error:
                return ModelResponse(
                    parts=[
                        TextPart(
                            content=(
                                "I could not contact the language model after two "
                                "attempts. The request was not completed. "
                                f"Last error: {type(second_error).__name__}: "
                                f"{second_error}"
                            )
                        )
                    ],
                    model_name="local-error-recovery",
                )

    async def wrap_tool_validate(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
        handler: WrapToolValidateHandler,
    ) -> ValidatedToolArgs:
        """Repair empty-tool calls and retry other malformed arguments once."""
        try:
            return await handler(args)
        except Exception as error:
            required_arguments = tool_def.parameters_json_schema.get("required", [])
            if not required_arguments:
                return await handler({})

            if ctx.retry < 1:
                raise ModelRetry(
                    f"Invalid arguments for {tool_def.name}: {error}. Call the "
                    "tool again with one complete valid JSON object."
                ) from error

            # Let execution reach the global error boundary after one malformed
            # retry, rather than terminating the complete agent run.
            return {}

    async def wrap_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        try:
            return await handler(args)
        except Exception:
            try:
                return await handler(args)
            except Exception as second_error:
                return (
                    f"TOOL_ERROR: {tool_def.name} failed twice. "
                    f"Last error: {type(second_error).__name__}: {second_error}. "
                    "Explain the failure briefly and continue with another "
                    "available tool or the context already collected."
                )
