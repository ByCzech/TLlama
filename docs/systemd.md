# Running TLlama with systemd

[tllamad.service](../systemd/tllamad.service) runs the foreground daemon as the
dedicated `tllama` account. It reads `/etc/tllama/server.toml` explicitly, so
user config discovery cannot change the system service. Packaging is a later,
separate branch; the unit is supplied here for manual installation.

## Installation

Install TLlama and the native runtime first. The unit assumes the Python
daemon is `/usr/bin/tllamad`. Check `command -v tllamad`; for a virtualenv or
`/usr/local/bin` installation use the executable override below. Native
`tllama-server` and its backend libraries are separate runtime dependencies.

Create the service account if it does not already exist:

```sh
sudo useradd --system --user-group --home-dir /var/lib/tllama \
  --no-create-home --shell /usr/sbin/nologin tllama
```

From the repository root, install the unit and an initial configuration:

```sh
sudo install -d -m 0755 /etc/tllama
# On first installation only; retain and edit an existing configuration.
sudo install -m 0644 examples/server.toml /etc/tllama/server.toml
sudo install -m 0644 systemd/tllamad.service /etc/systemd/system/tllamad.service
sudo systemctl daemon-reload
sudo systemctl enable --now tllamad.service
systemctl status tllamad.service
journalctl -u tllamad.service -f
```

Systemd creates `/var/lib/tllama` and `/var/cache/tllama`, owned by `tllama`.
The daemon creates the default `models` store under its state directory. The
Hugging Face cache uses `/var/cache/tllama/huggingface`. Pre-existing models and
custom storage paths must be readable/writable as required by the service
account; ensure parent directories can be traversed. Relative server paths
use the unit's `WorkingDirectory=/var/lib/tllama`.

GPU access follows host device permissions. Add the service account to the
appropriate existing device groups (commonly `render` and/or `video`) for the
chosen driver, then restart the service. Do not assume every backend uses the
same groups or device nodes; verify access on the target host. CPU-only operation
requires no GPU group. The unit does not use `PrivateDevices`, which would hide
GPU devices.

## Configuration and overrides

Edit `/etc/tllama/server.toml`, then restart:

```sh
sudo systemctl restart tllamad.service
```

The pre-start configuration check runs as the service account. Invalid
configuration exits 2 and is not retried automatically. Configuration is also
validated by the actual daemon, before the listener opens. There is no live
configuration reload or `ExecReload`; Uvicorn's development `--reload` is not
intended for this service.

Use `sudo systemctl edit tllamad.service` for local overrides. For an alternate
Python installation, reset both executable commands:

```ini
[Service]
ExecStartPre=
ExecStartPre=/opt/tllama/venv/bin/tllamad --config /etc/tllama/server.toml --check-config
ExecStart=
ExecStart=/opt/tllama/venv/bin/tllamad --config /etc/tllama/server.toml
```

That installation must be accessible to `tllama`; a virtualenv inside a private
home directory generally is not. Keep native executable selection in
`[engine].executable` in server TOML.

Environment overrides remain available in a drop-in:

```ini
[Service]
Environment=TLLAMA_LOG_LEVEL=debug
Environment=TLLAMA_BACKEND_PRIORITY=Vulkan,CPU
```

To use CLI overrides, replace both `ExecStartPre` and `ExecStart` with matching
options. Global precedence is CLI > environment > server TOML > defaults;
model/request precedence is described in [daemon.md](daemon.md). The unit
does not read `/etc/default/tllama` automatically. An administrator can add an
`EnvironmentFile` in a drop-in when needed.

On shutdown, SIGTERM first reaches the daemon so it can drain and close its
managed workers. `KillMode=mixed` lets systemd kill any remaining processes in
the unit's cgroup after the stop timeout. A failed daemon restarts after five
seconds; configuration exit 2 is excluded. The 120-second stop timeout can be
adjusted for the host's workloads. For `mlock`, configure an appropriate
`LimitMEMLOCK` in a drop-in when required by the selected runtime settings.

## Validation

Run `systemd-analyze verify /etc/systemd/system/tllamad.service` after installing
the referenced executable. Check real service startup, GPU access and shutdown
on the target host; syntax validation alone does not exercise GPU drivers or
prove native model compatibility.
