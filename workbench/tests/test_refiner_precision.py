from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from system import web_app
from system.live_qwen_refiner import TransformersRefiner


def test_explicit_precision_and_runtime_health(tmp_path):
    class Refiner:
        def __init__(self, *args, dtype='auto', **kwargs):
            self.model_dtype = dtype
            self.parameter_bytes = 2161265664

    with patch.object(web_app, 'TransformersRefiner', Refiner):
        app = web_app.create_app(Path('unused'), 'cpu', 'http://asr.invalid',
                                 None, 256, None, refiner_dtype='bfloat16')
        with TestClient(app) as client:
            status = client.get('/health').json()
        assert status['refiner_dtype'] == 'bfloat16'
        assert status['refiner_parameter_bytes'] == 2161265664


def test_invalid_precision_fails_before_model_loading():
    with pytest.raises(ValueError, match='dtype'):
        TransformersRefiner(Path('unused'), 'cpu', 256, dtype='int4')
