# star_browser

Isolated browser interface for selected starBoard workflows.

Current implemented slice
- FastAPI app scaffold
- health route
- auth dependency using Cloudflare Access-style header auth
- session route with browser capability map
- gallery entity read route
- gallery media full/preview routes
- batch-upload discover/preview route
- batch-upload execute route
- minimal first-order search route
- frontend placeholder scaffold

Isolation guarantees
- no imports from `src/ui/*`
- no morphometric code
- core reuse only through explicit adapters/services

Current non-scope
- no desktop UI reuse
- no morphometric workflows
- no browser undo/redo parity
- no rich frontend implementation yet

Test command
- `./scripts/test -q star_browser/tests`
