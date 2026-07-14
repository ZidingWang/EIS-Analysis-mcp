"""Compatibility entry point for the MCP server."""

from .mcp_server import create_mcp_server, main

__all__ = ["create_mcp_server", "main"]


if __name__ == "__main__":
    main()
