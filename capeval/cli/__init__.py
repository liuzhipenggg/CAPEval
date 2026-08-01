"""Root launcher implementations (``caption.py`` / ``score.py`` shims)."""
from capeval.cli.caption_launcher import main as caption_main
from capeval.cli.score_launcher import main as score_main

__all__ = ["caption_main", "score_main"]
