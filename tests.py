#!/usr/bin/env python3

import unittest
import sys
from colour_runner import runner

if __name__ == '__main__':
    suite = unittest.TestLoader().discover('tests')
    result = runner.ColourTextTestRunner(verbosity=2).run(suite)
    ret = not result.wasSuccessful()
    sys.exit(ret)
