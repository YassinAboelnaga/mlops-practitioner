from prodml.train import main
from prodml.config import settings


def test_train_main_produces_model(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "model_path", str(tmp_path / "model.pkl"))
    monkeypatch.setattr(settings, "report_path", str(tmp_path / "report.md"))

    main()

    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "report.md").exists()
