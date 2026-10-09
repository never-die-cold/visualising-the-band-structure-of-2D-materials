"""Keep GUI state inside pytest's isolated workspace directories."""
import os
import pytest


@pytest.fixture(scope='session', autouse=True)
def isolated_application_state(tmp_path_factory):
    previous = os.environ.get('BANDVIZ_DATA_DIR')
    os.environ['BANDVIZ_DATA_DIR'] = str(tmp_path_factory.mktemp('bandviz-app-state'))
    yield
    if previous is None:
        os.environ.pop('BANDVIZ_DATA_DIR', None)
    else:
        os.environ['BANDVIZ_DATA_DIR'] = previous
