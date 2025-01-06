import wandb

try:
    from wandb_workspaces.reports.v2 import *  # noqa: F403
except ImportError:
    wandb.termerror(
        "Failed to import wandb_workspaces.  To edit reports programmatically, please install it using `pip install wandb[workspaces]`."
    )
