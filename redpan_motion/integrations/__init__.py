"""Adapters that expose RED-PAN through other projects' interfaces.

Each one is optional and imports its host library lazily, so the base package
never depends on them. Install what you need:

    pip install "redpan-motion[seisbench]"

    from redpan_motion.integrations.seisbench import RedpanSB60s, RedpanSB90s
"""
