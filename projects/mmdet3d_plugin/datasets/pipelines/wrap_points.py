import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class WrapPointsToDataContainer(object):
    """Wrap 'points' (and optionally other keys) into DataContainer to avoid stacking.

    Insert this module near the end of pipeline (right before Collect).
    """

    def __init__(self, keys=['points'], cpu_only=True):
        self.keys = keys
        self.cpu_only = cpu_only

    def __call__(self, results):
        for k in self.keys:
            if k in results:
                # only wrap numpy / tensor / list-like types
                results[k] = DC(results[k], cpu_only=self.cpu_only)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys}, cpu_only={self.cpu_only})'
