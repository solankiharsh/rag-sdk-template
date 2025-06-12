import os

import typer  # type: ignore

from ragsdk.core.audit.traces import set_trace_handlers
from ragsdk.core.config import import_modules_from_config

if os.getenv("RAGSDK_VERBOSE", "0") == "1":
    typer.echo('Verbose mode is enabled with environment variable "RAGSDK_VERBOSE".')
    set_trace_handlers("cli")

import_modules_from_config()
