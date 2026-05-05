import json

import pytest

from tools import google_drive_boundary_tool as gdb


class _Exec:
    def __init__(self, value):
        self.value = value

    def execute(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class _FakeFiles:
    def __init__(self, files):
        self.files = files
        self.export_calls = []
        self.download_calls = []

    def get(self, fileId, **_kwargs):
        value = self.files.get(fileId)
        if value is None:
            value = RuntimeError("not found")
        return _Exec(value)

    def export_media(self, fileId, mimeType):
        self.export_calls.append((fileId, mimeType))
        return _Exec(b"export")

    def get_media(self, fileId, **_kwargs):
        self.download_calls.append(fileId)
        return _Exec(b"download")


class _FakeService:
    def __init__(self, files):
        self._files = _FakeFiles(files)

    def files(self):
        return self._files

    @property
    def fake_files(self):
        return self._files


def test_ancestry_allows_child_under_allowed_root():
    service = _FakeService(
        {
            "child": {"id": "child", "name": "Child", "mimeType": "text/plain", "parents": ["folder"]},
            "folder": {"id": "folder", "name": "Folder", "mimeType": gdb.FOLDER_MIME, "parents": ["root"]},
        }
    )

    ok, reason, meta, ancestry = gdb._ancestry_allowed(service, "child", {"root"})

    assert ok is True
    assert reason == "inside_allowed_root"
    assert meta["id"] == "child"
    assert ancestry == ["child", "folder", "root"]


def test_ancestry_rejects_shortcuts_by_default():
    service = _FakeService(
        {
            "shortcut": {
                "id": "shortcut",
                "name": "Shortcut",
                "mimeType": gdb.SHORTCUT_MIME,
                "parents": ["root"],
                "shortcutDetails": {"targetId": "target"},
            }
        }
    )

    ok, reason, meta, ancestry = gdb._ancestry_allowed(service, "shortcut", {"root"})

    assert ok is False
    assert reason == "shortcut_rejected_by_policy"
    assert meta["id"] == "shortcut"
    assert ancestry == ["shortcut"]


def test_restaurant_profile_cannot_request_another_boundary(monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "cantaritos")

    with pytest.raises(PermissionError):
        gdb._resolve_restaurant("pga")


def test_check_handler_does_not_return_metadata_for_denied_id(monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "pga")
    monkeypatch.setattr(
        gdb,
        "_load_config",
        lambda _restaurant: {
            "allowed_roots": ["root"],
            "token_path": "/tmp/token.json",
        },
    )
    monkeypatch.setattr(
        gdb,
        "_drive_service",
        lambda _token_path: _FakeService(
            {
                "outside": {
                    "id": "outside",
                    "name": "Outside Secret",
                    "mimeType": "text/plain",
                    "parents": [],
                }
            }
        ),
    )

    result = json.loads(gdb._handle_check({"file_id": "outside"}))

    assert result["allowed"] is False
    assert result["result"] == "no_allowed_root_in_ancestry"
    assert "metadata" not in result


def test_download_rejects_output_path_outside_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_PROFILE", "pga")
    monkeypatch.setattr(
        gdb,
        "_load_config",
        lambda _restaurant: {
            "workspace": str(tmp_path),
            "allowed_roots": ["root"],
            "token_path": "/tmp/token.json",
            "originals_path": "/workspace/drive/originals",
        },
    )
    monkeypatch.setattr(
        gdb,
        "_drive_service",
        lambda _token_path: _FakeService(
            {
                "child": {
                    "id": "child",
                    "name": "Child.pdf",
                    "mimeType": "application/pdf",
                    "parents": ["root"],
                },
                "root": {
                    "id": "root",
                    "name": "Recipes",
                    "mimeType": gdb.FOLDER_MIME,
                    "parents": [],
                },
            }
        ),
    )

    result = json.loads(
        gdb._handle_download(
            {
                "file_id": "child",
                "output_path": "/home/clockwork/.hermes/secret.pdf",
            }
        )
    )

    assert "error" in result
    assert "inside restaurant workspace" in result["error"]


def test_download_binary_file_writes_inside_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_PROFILE", "pga")
    service = _FakeService(
        {
            "child": {
                "id": "child",
                "name": "Child.pdf",
                "mimeType": "application/pdf",
                "parents": ["root"],
                "modifiedTime": "2026-05-04T00:00:00Z",
                "md5Checksum": "abc123",
            },
            "root": {
                "id": "root",
                "name": "Recipes",
                "mimeType": gdb.FOLDER_MIME,
                "parents": [],
            },
        }
    )
    monkeypatch.setattr(
        gdb,
        "_load_config",
        lambda _restaurant: {
            "workspace": str(tmp_path),
            "allowed_roots": ["root"],
            "token_path": "/tmp/token.json",
            "originals_path": "/workspace/drive/originals",
        },
    )
    monkeypatch.setattr(gdb, "_drive_service", lambda _token_path: service)

    def fake_download(_request, output_path, _max_bytes):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"pdf")
        return 3

    monkeypatch.setattr(gdb, "_download_request_to_path", fake_download)

    result = json.loads(
        gdb._handle_download(
            {
                "file_id": "child",
                "output_path": "/workspace/drive/originals/Child.pdf",
            }
        )
    )

    assert result["ok"] is True
    assert result["operation"] == "download"
    assert result["workspace_path"] == "/workspace/drive/originals/Child.pdf"
    assert result["change_key"]["md5_checksum"] == "abc123"
    assert service.fake_files.download_calls == ["child"]


def test_download_google_native_file_uses_export(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_PROFILE", "pga")
    service = _FakeService(
        {
            "doc": {
                "id": "doc",
                "name": "Prep",
                "mimeType": "application/vnd.google-apps.document",
                "parents": ["root"],
                "modifiedTime": "2026-05-04T00:00:00Z",
            },
            "root": {
                "id": "root",
                "name": "Recipes",
                "mimeType": gdb.FOLDER_MIME,
                "parents": [],
            },
        }
    )
    monkeypatch.setattr(
        gdb,
        "_load_config",
        lambda _restaurant: {
            "workspace": str(tmp_path),
            "allowed_roots": ["root"],
            "token_path": "/tmp/token.json",
            "originals_path": "/workspace/drive/originals",
        },
    )
    monkeypatch.setattr(gdb, "_drive_service", lambda _token_path: service)
    monkeypatch.setattr(gdb, "_download_request_to_path", lambda _request, _path, _max: 7)

    result = json.loads(
        gdb._handle_download(
            {
                "file_id": "doc",
                "export_mime_type": "text/plain",
            }
        )
    )

    assert result["ok"] is True
    assert result["operation"] == "export"
    assert result["export_mime_type"] == "text/plain"
    assert result["workspace_path"] == "/workspace/drive/originals/Prep__doc.txt"
    assert result["change_key"]["export_mime_type"] == "text/plain"
    assert service.fake_files.export_calls == [("doc", "text/plain")]
