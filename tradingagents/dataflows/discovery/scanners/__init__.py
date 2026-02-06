"""Discovery scanners for modular pipeline architecture."""

# Import all scanners to trigger registration
from . import insider_buying  # noqa: F401
from . import options_flow  # noqa: F401
from . import reddit_trending  # noqa: F401
from . import market_movers  # noqa: F401
from . import volume_accumulation  # noqa: F401
from . import semantic_news  # noqa: F401
from . import reddit_dd  # noqa: F401
from . import earnings_calendar  # noqa: F401
