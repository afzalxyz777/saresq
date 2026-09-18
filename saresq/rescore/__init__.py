"""Ground-station re-scoring: the cascade's last, cheapest-to-afford stage.

The payload spends 28 ms on a 160 px crop with a 3 MB nano detector because
that is what 1 GB of RAM and a battery allow. The ground station is a laptop
with none of those limits, and by the time a crop reaches it the hard part --
deciding that this patch of ground was worth looking at -- has already been
done. So the crop is scored again, by a model an order of magnitude larger, at
an input size four times bigger.

Both numbers are kept. The payload's belief is never overwritten.
"""
from saresq.rescore.engine import Rescorer, RescoreResult
from saresq.rescore.worker import RescoreWorker

__all__ = ["Rescorer", "RescoreResult", "RescoreWorker"]
