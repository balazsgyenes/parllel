import cProfile
import datetime

from .process import ProcessCage
from .synchronized import SynchronizedProcessCage


class ProfilingCageMixin:

    def run(self):

        with cProfile.Profile() as profiler:
            super().run()

        datetimestr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        profiler.dump_stats(datetimestr + f"_ParallelCage-{self.pid}.profile")

class ProfilingProcessCage(ProfilingCageMixin, ProcessCage):
    pass


class ProfilingSynchronizedProcessCage(ProfilingCageMixin,
    SynchronizedProcessCage):
    pass
