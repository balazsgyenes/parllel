import cProfile
import datetime

from .process import ProcessCage


class ProfilingProcessCage(ProcessCage):

    def run(self):

        with cProfile.Profile() as profiler:
            super().run()

        datetimestr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        profiler.dump_stats(datetimestr + f"_ParallelCage-{self.pid}.profile")
