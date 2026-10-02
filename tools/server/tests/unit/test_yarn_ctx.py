import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _n_ctx_train():
    global server
    server.start()
    n_ctx_train = None
    res = server.make_request("GET", "/models")
    if res.status_code == 200 and res.body.get("data"):
        n_ctx_train = res.body["data"][0].get("meta", {}).get("n_ctx_train")
    if n_ctx_train is None:
        res = server.make_request("GET", "/props")
        assert res.status_code == 200
        n_ctx_train = res.body.get("n_ctx_train")
        if n_ctx_train is None:
            n_ctx_train = res.body.get("default_generation_settings", {}).get("n_ctx_train")
    server.stop()
    assert n_ctx_train is not None and n_ctx_train > 0
    if n_ctx_train > 8192:
        pytest.skip(f"tiny model n_ctx_train={n_ctx_train} is already large")
    return n_ctx_train


def test_slot_ctx_capped_without_rope():
    global server
    n_ctx_train = _n_ctx_train()
    server.n_ctx = n_ctx_train * 2
    server.n_slots = 1
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["default_generation_settings"]["n_ctx"] == n_ctx_train


def test_slot_ctx_not_capped_with_yarn():
    global server
    n_ctx_train = _n_ctx_train()
    server.n_ctx = n_ctx_train * 2
    server.n_slots = 1
    server.extra_args = [
        "--rope-scaling", "yarn",
        "--rope-scale", "2",
        "--yarn-orig-ctx", str(n_ctx_train),
    ]
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["default_generation_settings"]["n_ctx"] == n_ctx_train * 2
