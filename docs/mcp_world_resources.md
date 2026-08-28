# MCP world resources

The MCP server exposes the shared `WorldState` as live, read-only JSON resources.

- `world://snapshot` returns the complete current world snapshot.
- `world://entities/<id>` returns one current entity, with URI-encoded entity IDs.

Resource reads are resolved at request time. They are intentionally read-only: trusted perception and localization code owns entity mutation, while MCP robot tools remain the action surface.
