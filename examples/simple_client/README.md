# Simple Client

A minimal client that sends observations to the server and prints the inference rate.

You can specify which runtime environment to use using the `--env` flag. You can see the available options by running:

```bash
uv run examples/simple_client/main.py --help
```

## Without Docker

Terminal window 1:

```bash
uv run examples/simple_client/main.py --env DROID
```

Terminal window 2:

```bash
cd /workspace/project/openpi && uv run scripts/serve_policy.py --env DROID
```
