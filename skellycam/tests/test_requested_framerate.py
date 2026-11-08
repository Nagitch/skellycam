"""Requested FPS must survive config updates and format renegotiation."""
from unittest.mock import MagicMock, patch
import cv2
import pytest
from skellycam.core.camera.config.camera_config import CameraConfig
from skellycam.core.camera.opencv.opencv_helpers.opencv_apply_config import apply_camera_configuration

MODULE = 'skellycam.core.camera.opencv.opencv_helpers.opencv_apply_config'

@pytest.mark.parametrize('prior,requested,expected', [
    (None, CameraConfig(framerate=60), True),
    (CameraConfig(framerate=30), CameraConfig(framerate=60), True),
    (CameraConfig(framerate=60), CameraConfig(framerate=60), False),
    (None, CameraConfig(framerate=-1), False),
    (None, CameraConfig(framerate=0), False),
    (CameraConfig(framerate=60), CameraConfig(framerate=60, resolution={'width':640,'height':480}), True),
    (CameraConfig(framerate=60), CameraConfig(framerate=60, capture_fourcc='YUY2'), True),
])
def test_fps_applied_after_format(prior, requested, expected):
    capture = MagicMock(spec=cv2.VideoCapture)
    capture.isOpened.return_value = True
    extracted = requested.model_copy(update={'framerate':59.94})
    with patch(MODULE+'._apply_exposure'), patch(MODULE+'.extract_config_from_cv2_capture', return_value=extracted):
        result = apply_camera_configuration(capture, prior, requested)
    calls = [c.args for c in capture.set.call_args_list]
    fps = [(i,args) for i,args in enumerate(calls) if args[0] == cv2.CAP_PROP_FPS]
    assert bool(fps) == expected
    if expected:
        assert fps == [(len(calls)-1, (cv2.CAP_PROP_FPS, 60))]
    assert result.framerate == 59.94  # do not report the request as the actual rate


def test_setting_update_keeps_requested_fps():
    config = CameraConfig(framerate=30)
    config.accept_settable_parameters(CameraConfig(framerate=60).to_settable_parameters())
    assert config.framerate == 60


def test_rejected_request_returns_device_config(caplog):
    capture = MagicMock(spec=cv2.VideoCapture)
    capture.isOpened.return_value = True
    capture.set.return_value = False
    with patch(MODULE+'._apply_exposure'), patch(MODULE+'.extract_config_from_cv2_capture', return_value=CameraConfig(framerate=30)):
        result = apply_camera_configuration(capture, None, CameraConfig(framerate=60))
    assert result.framerate == 30
    assert 'rejected requested FPS' in caplog.text
