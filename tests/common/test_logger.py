# tests/common/test_logger.py

from src.common.logger import check_log_file

def test_check_log_file():
    assert check_log_file("test_log") == "test_log.1"
