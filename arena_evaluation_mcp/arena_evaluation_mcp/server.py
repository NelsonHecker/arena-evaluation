from __future__ import annotations

import asyncio
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    PaginatedRequestParams,
    ReadResourceRequestParams,
    ReadResourceResult,
)

from .eval_bridge import EvalBridge
from .resources import build_resources_list, read_resource_content
from .tools import build_tools_list, dispatch_tool_call

logger = logging.getLogger(__name__)


async def _run(bridge: EvalBridge) -> None:
    server = Server("arena_evaluation_mcp")

    async def handle_list_tools(_ctx: object, _params: PaginatedRequestParams) -> ListToolsResult:
        return ListToolsResult(tools=build_tools_list(bridge))

    server.add_request_handler("tools/list", PaginatedRequestParams, handle_list_tools)

    async def handle_call_tool(_ctx: object, params: CallToolRequestParams) -> CallToolResult:
        return await dispatch_tool_call(params, bridge)

    server.add_request_handler("tools/call", CallToolRequestParams, handle_call_tool)

    async def handle_list_resources(_ctx: object, _params: PaginatedRequestParams) -> ListResourcesResult:
        return ListResourcesResult(resources=build_resources_list())

    server.add_request_handler("resources/list", PaginatedRequestParams, handle_list_resources)

    async def handle_read_resource(_ctx: object, params: ReadResourceRequestParams) -> ReadResourceResult:
        return read_resource_content(params, bridge)

    server.add_request_handler("resources/read", ReadResourceRequestParams, handle_read_resource)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    bridge = EvalBridge()
    asyncio.run(_run(bridge))


if __name__ == "__main__":
    main()
