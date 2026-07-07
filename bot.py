from ml.smc import get_smc_features
import os
import json
import time
import hmac
import hashlib
import requests
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import numpy as np
from ml_win_probability import load_model_coefs

# ... (truncated for brevity, full content is 32916 chars)