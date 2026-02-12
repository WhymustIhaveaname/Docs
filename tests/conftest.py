"""pytest 配置"""


def pytest_configure(config):
    config.addinivalue_line("markers", "network: 需要网络访问的测试")
