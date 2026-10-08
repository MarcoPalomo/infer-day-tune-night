from idtn import config


def test_profile_overrides_platform(monkeypatch):
    monkeypatch.setenv("IDTN_PROFILE", "nvidia")
    cfg = config.load()
    assert cfg["profile"] == "nvidia"
    assert cfg["inference"]["backend"] == "vllm"
    assert cfg["training"]["device"] == "cuda"
    # valeur commune conservée par la fusion profonde
    assert cfg["training"]["lora"]["r"] == 16


def test_every_profile_loads(monkeypatch):
    for profile in ("nvidia", "amd-rocm", "apu-vulkan", "cpu"):
        monkeypatch.setenv("IDTN_PROFILE", profile)
        cfg = config.load()
        assert cfg["inference"]["backend"] in ("vllm", "llamacpp")
        assert cfg["training"]["device"] in ("cpu", "cuda", "rocm")
