from __future__ import annotations

import pandas as pd


def get_ai_tool_preference_table() -> pd.DataFrame:
    """Return a small synthetic contingency table for CA."""
    return pd.DataFrame(
        {
            "ChatGPT": [42, 28, 35],
            "Gemini": [18, 12, 20],
            "Copilot": [25, 14, 19],
            "Claude": [10, 16, 13],
        },
        index=["Campus", "Distans", "Hybrid"],
    )