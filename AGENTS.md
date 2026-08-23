# AGENTS.md

## NES-LTER data via MCP

For any question about NES-LTER stations, cruises, casts, CTD data, or oceanographic metadata, use the `nes-lter-mcp` MCP server via the **mcp-pair** skill:

- `mcporter list nes-lter-mcp --schema` — see available tools and argument schemas
- `mcporter call nes-lter-mcp.<tool> [key=value ...]` — invoke a tool

Do not fetch this data manually (curl/API) when the MCP server can serve it. If the server errors, check `~/.mcporter/mcporter.json` and the repo at `/Users/brett/Projects/lter/nes-lter-mcp`.
