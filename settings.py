import os
from dotenv import load_dotenv

load_dotenv()
APP_VERSION = os.getenv("APP_VERSION")
PATH_BASE = os.path.dirname(os.path.realpath(__file__))
