"""
Independent, single-responsibility agents.

Each agent does one job and exposes a `run(...)` method. They are composed by
services (ingestion_service, mcq_pipeline, learning_service) rather than calling
each other directly, which keeps them independently testable and swappable.
"""
