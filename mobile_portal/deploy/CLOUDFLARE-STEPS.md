# Cloudflare steps for mobile.fhl-star-board.com

Current status:
- `upload.fhl-star-board.com` resolves and is protected by Cloudflare Access.
- `mobile.fhl-star-board.com` does not currently resolve publicly.
- The existing tunnel service on this machine is `cloudflared-starboard.service` for tunnel `fhl-star-board`.

Target origin:
- `http://127.0.0.1:8091`

Add this hostname to the existing tunnel ingress:

1. Open the Cloudflare Zero Trust / Tunnels UI for tunnel `fhl-star-board`.
2. Add a public hostname:
   - Hostname: `mobile.fhl-star-board.com`
   - Service type: `HTTP`
   - URL: `http://127.0.0.1:8091`
3. Save the tunnel config.
4. Ensure the application is protected by the same Cloudflare Access policy style as `upload.fhl-star-board.com`.
5. If Access policy is separate, allow the desired email identities/groups.
6. Wait for propagation, then verify:
   - `curl -I https://mobile.fhl-star-board.com`
   - expected result: redirect to Cloudflare Access login or authenticated 200 depending on session state

If using config-as-file instead of dashboard-managed ingress, the relevant rule is:

```yaml
ingress:
  - hostname: mobile.fhl-star-board.com
    service: http://127.0.0.1:8091
  - service: http_status:404
```

Local verification after systemd install:
- `curl http://127.0.0.1:8091/api/health`
- `curl -H 'cf-access-authenticated-user-email: field@example.org' http://127.0.0.1:8091/api/session`
