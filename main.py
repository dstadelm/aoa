import logging
from pathlib import Path

from aoa.aoa import Runnable, main

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    # args = sys.argv[1:]
    logger = logging.getLogger(__name__)
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.DEBUG)
    # main(Path("testsets/AoA.yaml"))
    # main(Path("testsets/amd.yaml"))
    main = Runnable(Path("testsets/amd.yaml"))
    # main = Runnable(Path("testsets/more_tricky.yaml"))
    main()

    # main(Path("tricky.yaml"))
    # main(Path("../more_tricky.yaml"))
    # main(Path("test_case_3.yaml"))
    # main(Path("test_case_5.yaml")
