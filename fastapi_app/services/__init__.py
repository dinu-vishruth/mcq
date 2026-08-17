"""Shared read-model builders used by both HTML routes and the JSON API.

Extracting these means the server-rendered page and ``/api/*`` return the same
shape from the same code, which is what lets the dashboard render immediately
from its embedded bootstrap payload instead of re-fetching after load.
"""
