import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.WARN, format='%(asctime)s %(levelname)s: %(message)s')
