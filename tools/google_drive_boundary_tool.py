"""Allowlist-enforcing Google Drive tools for restaurant workers.

The Google OAuth token stays in the host Hermes home. Restaurant agents call
these tools instead of importing Google clients or receiving token files in
their Docker workspace.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
GOOGLE_NATIVE_MIME_PREFIX = "application/vnd.google-apps."
DEFAULT_EXPORT_MIME_TYPE = "application/pdf"

EXPORT_EXTENSIONS = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}

RESTAURANT_WORKSPACES = {
    "pga": Path("/home/clockwork/restaurants/pga"),
    "cantaritos": Path("/home/clockwork/restaurants/cantaritos"),
}

FILE_FIELDS = (
    "id,name,mimeType,parents,modifiedTime,webViewLink,size,md5Checksum,"
    "shortcutDetails(targetId,targetMimeType)"
)


def _current_profile() -> str:
    profile = (os.getenv("HERMES_PROFILE") or "").strip().lower()
    if profile:
        return profile

    home = (os.getenv("HERMES_HOME") or "").strip()
    if home:
        name = Path(home).name.lower()
        if name in RESTAURANT_WORKSPACES:
            return name

    return ""


def _resolve_restaurant(requested: str | None) -> str:
    restaurant = (requested or "").strip().lower()
    profile = _current_profile()

    if not restaurant:
        restaurant = profile
    if restaurant not in RESTAURANT_WORKSPACES:
        raise ValueError(
            "restaurant is required and must be one of: "
            + ", ".join(sorted(RESTAURANT_WORKSPACES))
        )

    # A restaurant profile may only use its own boundary. The main coordinator
    # normally has no HERMES_PROFILE and can pass an explicit restaurant.
    if profile in RESTAURANT_WORKSPACES and restaurant != profile:
        raise PermissionError(
            f"profile {profile!r} cannot use the {restaurant!r} Google Drive boundary"
        )

    return restaurant


def _load_config(restaurant: str) -> dict[str, Any]:
    workspace = RESTAURANT_WORKSPACES[restaurant]
    config_path = workspace / "config" / "google-drive.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing Google Drive boundary config: {config_path}")

    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency check handles this
        raise RuntimeError("PyYAML is required for google_drive_boundary tools") from exc

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if data.get("restaurant_id") != restaurant:
        raise ValueError(
            f"boundary config restaurant_id mismatch: expected {restaurant!r}, "
            f"got {data.get('restaurant_id')!r}"
        )

    drive_cfg = data.get("drive", {}) or {}
    allowed_roots = drive_cfg.get("allowed_roots") or []
    allowed_roots = [str(root).strip() for root in allowed_roots if str(root).strip()]
    if not allowed_roots:
        raise ValueError(f"no allowed Drive roots configured for {restaurant}")

    token_path = str(
        data.get("google_credentials", {}).get("token_path")
        or "/home/clockwork/.hermes/google_token.json"
    )
    if not Path(token_path).exists():
        raise FileNotFoundError(f"missing Google token file: {token_path}")

    boundary = data.get("api_boundary", {})
    if boundary.get("selected") != "host-side-wrapper":
        raise ValueError("Google API boundary must be selected: host-side-wrapper")
    if boundary.get("profile_container_token_access") is not False:
        raise ValueError("profile_container_token_access must be false")

    return {
        "workspace": str(workspace),
        "config_path": str(config_path),
        "allowed_roots": allowed_roots,
        "token_path": token_path,
        "mirror_path": str(drive_cfg.get("mirror_path") or "/workspace/drive"),
        "originals_path": str(drive_cfg.get("originals_path") or "/workspace/drive/originals"),
        "text_path": str(drive_cfg.get("text_path") or "/workspace/drive/text"),
    }


def _check_google_drive_boundary_requirements() -> bool:
    try:
        import googleapiclient.discovery  # noqa: F401
        import google.oauth2.credentials  # noqa: F401
        import yaml  # noqa: F401
    except Exception:
        return False

    profile = _current_profile()
    if profile in RESTAURANT_WORKSPACES:
        try:
            _load_config(profile)
            return True
        except Exception:
            return False

    # Do not expose restaurant Drive tools in unrelated profiles.
    return False


def _drive_service(token_path: str):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds.valid:
        raise RuntimeError("Google credentials are not valid; re-run Google OAuth setup")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _files_resource(service):
    return service.files()


def _get_file(service, file_id: str) -> dict[str, Any]:
    return (
        _files_resource(service)
        .get(
            fileId=file_id,
            fields=FILE_FIELDS,
            supportsAllDrives=True,
        )
        .execute()
    )


def _list_children(service, folder_id: str) -> list[dict[str, Any]]:
    files = []
    page_token = None
    while True:
        result = (
            _files_resource(service)
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields=f"nextPageToken,files({FILE_FIELDS})",
                pageToken=page_token,
                pageSize=1000,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                corpora="allDrives",
            )
            .execute()
        )
        files.extend(result.get("files", []))
        page_token = result.get("nextPageToken")
        if not page_token:
            return files


def _shortcut_target_id(meta: dict[str, Any]) -> str:
    details = meta.get("shortcutDetails") or {}
    return str(details.get("targetId") or "").strip()


def _is_google_native_file(mime_type: str | None) -> bool:
    mime = str(mime_type or "")
    return mime.startswith(GOOGLE_NATIVE_MIME_PREFIX) and mime not in {
        FOLDER_MIME,
        SHORTCUT_MIME,
    }


def _export_extension(mime_type: str | None) -> str:
    mime = str(mime_type or "").strip()
    if mime in EXPORT_EXTENSIONS:
        return EXPORT_EXTENSIONS[mime]
    return mimetypes.guess_extension(mime) or ""


def _safe_path_segment(value: str | None, fallback: str) -> str:
    text = str(value or "").strip().replace("\x00", "")
    text = text.replace("/", "_").replace("\\", "_")
    text = text.strip(". ")
    return text or fallback


def _join_drive_path(parts: list[str]) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def _drive_paths_from_ancestry(
    service,
    ancestry: list[str],
    allowed_roots: set[str],
) -> tuple[str, str]:
    """Return (drive_path, relative_drive_path) from file->root ancestry ids."""
    if not ancestry:
        return "", ""

    ordered_ids = list(reversed(ancestry))
    names: list[str] = []
    relative_names: list[str] = []
    seen_allowed_root = False

    for file_id in ordered_ids:
        meta = _get_file(service, file_id)
        name = str(meta.get("name") or file_id)
        names.append(name)
        if seen_allowed_root:
            relative_names.append(name)
        elif file_id in allowed_roots:
            seen_allowed_root = True

    return _join_drive_path(names), _join_drive_path(relative_names)


def _workspace_output_path(cfg: dict[str, Any], path_text: str) -> Path:
    """Map an output path to the host workspace and reject path escapes."""
    raw = str(path_text or "").strip()
    if not raw:
        raise ValueError("output_path is required")

    workspace = Path(cfg["workspace"]).resolve()
    if raw == "/workspace":
        candidate = workspace
    elif raw.startswith("/workspace/"):
        candidate = workspace / raw.removeprefix("/workspace/")
    else:
        path = Path(raw)
        candidate = path if path.is_absolute() else workspace / path

    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise PermissionError(
            f"output_path must resolve inside restaurant workspace {workspace}"
        ) from exc
    return resolved


def _host_to_workspace_path(cfg: dict[str, Any], host_path: Path) -> str:
    workspace = Path(cfg["workspace"]).resolve()
    rel = host_path.resolve(strict=False).relative_to(workspace)
    return "/workspace" if str(rel) == "." else "/workspace/" + rel.as_posix()


def _default_download_output_path(
    cfg: dict[str, Any],
    meta: dict[str, Any],
    relative_drive_path: str,
    export_mime_type: str | None,
) -> Path:
    """Build a duplicate-safe default path under drive/originals."""
    originals = _workspace_output_path(cfg, str(cfg["originals_path"]))
    rel_parts = [
        _safe_path_segment(part, "untitled")
        for part in relative_drive_path.split("/")
        if part
    ]
    if not rel_parts:
        rel_parts = [_safe_path_segment(meta.get("name"), str(meta.get("id") or "file"))]

    filename = rel_parts[-1]
    suffix = _export_extension(export_mime_type) if export_mime_type else Path(filename).suffix
    if export_mime_type:
        filename = f"{Path(filename).stem}__{meta.get('id')}{suffix}"
    else:
        stem = Path(filename).stem or filename
        existing_suffix = Path(filename).suffix
        filename = f"{stem}__{meta.get('id')}{existing_suffix}"
    rel_parts[-1] = _safe_path_segment(filename, f"{meta.get('id')}{suffix}")
    return originals.joinpath(*rel_parts)


def _download_request_to_path(request, output_path: Path, max_bytes: int) -> int:
    from googleapiclient.http import MediaIoBaseDownload

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
                if handle.tell() > max_bytes:
                    raise RuntimeError(
                        f"download exceeded max_bytes={max_bytes}"
                    )
        os.replace(tmp_path, output_path)
        return output_path.stat().st_size
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _ancestry_allowed(
    service,
    file_id: str,
    allowed_roots: set[str],
    *,
    shortcut_policy: str = "reject",
    max_depth: int = 50,
) -> tuple[bool, str, dict[str, Any] | None, list[str]]:
    """Return (allowed, reason, metadata, ancestry_ids)."""
    current_id = str(file_id or "").strip()
    if not current_id:
        return False, "file_id is required", None, []

    ancestry: list[str] = []
    seen: set[str] = set()
    first_meta: dict[str, Any] | None = None

    for _ in range(max_depth):
        if current_id in allowed_roots:
            if first_meta is None:
                try:
                    first_meta = _get_file(service, current_id)
                except Exception:
                    first_meta = None
            return True, "inside_allowed_root", first_meta, ancestry + [current_id]
        if current_id in seen:
            return False, "cycle_detected_in_drive_parents", first_meta, ancestry
        seen.add(current_id)

        try:
            meta = _get_file(service, current_id)
        except Exception as exc:
            return False, f"inaccessible_or_not_found: {exc}", first_meta, ancestry

        if first_meta is None:
            first_meta = meta
            if meta.get("mimeType") == SHORTCUT_MIME:
                if shortcut_policy == "reject":
                    return False, "shortcut_rejected_by_policy", meta, [current_id]
                if shortcut_policy != "resolve":
                    return False, f"unknown_shortcut_policy:{shortcut_policy}", meta, [current_id]
                target_id = _shortcut_target_id(meta)
                if not target_id:
                    return False, "shortcut_missing_target", meta, [current_id]
                current_id = target_id
                ancestry.append(str(meta.get("id") or file_id))
                continue

        ancestry.append(str(meta.get("id") or current_id))
        parents = [str(p).strip() for p in (meta.get("parents") or []) if str(p).strip()]
        if not parents:
            return False, "no_allowed_root_in_ancestry", first_meta, ancestry

        # Google Drive v3 files usually have one parent. If there are multiple,
        # try each parent and accept the first path that reaches an allowed root.
        blocked_reasons: list[str] = []
        for parent_id in parents:
            ok, reason, _, parent_path = _ancestry_allowed(
                service,
                parent_id,
                allowed_roots,
                shortcut_policy=shortcut_policy,
                max_depth=max_depth - len(ancestry),
            )
            if ok:
                return True, reason, first_meta, ancestry + parent_path
            blocked_reasons.append(reason)
        return False, "; ".join(blocked_reasons) or "no_allowed_root_in_ancestry", first_meta, ancestry

    return False, "max_parent_depth_exceeded", first_meta, ancestry


def _safe_meta(
    meta: dict[str, Any],
    *,
    drive_path: str | None = None,
    relative_drive_path: str | None = None,
) -> dict[str, Any]:
    result = {
        "id": meta.get("id"),
        "name": meta.get("name"),
        "mimeType": meta.get("mimeType"),
        "modifiedTime": meta.get("modifiedTime"),
        "webViewLink": meta.get("webViewLink"),
    }
    if meta.get("md5Checksum") is not None:
        result["md5Checksum"] = meta.get("md5Checksum")
    if meta.get("size") is not None:
        result["size"] = meta.get("size")
    if drive_path is not None:
        result["drivePath"] = drive_path
    if relative_drive_path is not None:
        result["relativeDrivePath"] = relative_drive_path
    if meta.get("mimeType") == SHORTCUT_MIME:
        result["shortcut"] = {
            "targetMimeType": (meta.get("shortcutDetails") or {}).get("targetMimeType"),
            "policy": "rejected_by_default",
        }
    return result


def _handle_list(args: dict, **kw) -> str:
    try:
        restaurant = _resolve_restaurant(args.get("restaurant"))
        cfg = _load_config(restaurant)
        service = _drive_service(cfg["token_path"])
        allowed_roots = set(cfg["allowed_roots"])
        max_items = max(1, min(int(args.get("max_items") or 500), 5000))
        shortcut_policy = str(args.get("shortcut_policy") or "reject").strip().lower()

        roots = []
        folders = []
        files = []
        inaccessible = []
        shortcuts = []
        queue = [(root_id, [], []) for root_id in cfg["allowed_roots"]]
        seen = set()

        while queue and (len(folders) + len(files) + len(shortcuts)) < max_items:
            folder_id, queued_drive_parts, queued_relative_parts = queue.pop(0)
            if folder_id in seen:
                continue
            seen.add(folder_id)

            try:
                meta = _get_file(service, folder_id)
                folder_name = str(meta.get("name") or folder_id)
                if folder_id in allowed_roots:
                    current_drive_parts = [folder_name]
                    current_relative_parts: list[str] = []
                    roots.append(
                        _safe_meta(
                            meta,
                            drive_path=_join_drive_path(current_drive_parts),
                            relative_drive_path="",
                        )
                    )
                else:
                    current_drive_parts = queued_drive_parts or [folder_name]
                    current_relative_parts = queued_relative_parts
                if folder_id in allowed_roots:
                    pass
                elif meta.get("mimeType") == FOLDER_MIME:
                    folders.append(
                        _safe_meta(
                            meta,
                            drive_path=_join_drive_path(current_drive_parts),
                            relative_drive_path=_join_drive_path(current_relative_parts),
                        )
                    )
                children = _list_children(service, folder_id)
            except Exception as exc:
                inaccessible.append({"id": folder_id, "reason": str(exc)})
                continue

            for child in children:
                if len(folders) + len(files) + len(shortcuts) >= max_items:
                    queue.clear()
                    break
                child_id = str(child.get("id") or "")
                if not child_id:
                    continue
                child_name = str(child.get("name") or child_id)
                child_drive_parts = current_drive_parts + [child_name]
                child_relative_parts = current_relative_parts + [child_name]
                child_drive_path = _join_drive_path(child_drive_parts)
                child_relative_path = _join_drive_path(child_relative_parts)
                mime = child.get("mimeType")
                if mime == SHORTCUT_MIME:
                    shortcuts.append({
                        "id": child_id,
                        "name": child.get("name"),
                        "policy": "rejected" if shortcut_policy == "reject" else shortcut_policy,
                        "drivePath": child_drive_path,
                        "relativeDrivePath": child_relative_path,
                    })
                    if shortcut_policy == "resolve":
                        target_id = _shortcut_target_id(child)
                        if target_id:
                            ok, reason, target_meta, _ = _ancestry_allowed(
                                service,
                                target_id,
                                allowed_roots,
                                shortcut_policy="reject",
                            )
                            shortcuts[-1]["target_allowed"] = ok
                            shortcuts[-1]["target_result"] = reason
                            if ok and target_meta and target_meta.get("mimeType") == FOLDER_MIME:
                                queue.append(target_id)
                    continue
                if mime == FOLDER_MIME:
                    folders.append(
                        _safe_meta(
                            child,
                            drive_path=child_drive_path,
                            relative_drive_path=child_relative_path,
                        )
                    )
                    queue.append((child_id, child_drive_parts, child_relative_parts))
                else:
                    files.append(
                        _safe_meta(
                            child,
                            drive_path=child_drive_path,
                            relative_drive_path=child_relative_path,
                        )
                    )

        return tool_result(
            ok=True,
            restaurant=restaurant,
            allowed_roots=cfg["allowed_roots"],
            root_count=len(roots),
            folder_count=len(folders),
            file_count=len(files),
            shortcut_count=len(shortcuts),
            truncated=bool(queue),
            roots=roots,
            folders=folders,
            files=files,
            shortcuts=shortcuts,
            inaccessible=inaccessible,
            boundary="host-side-wrapper",
            credential_exposure="host-side-only",
        )
    except Exception as exc:
        logger.exception("google_drive_boundary_list failed")
        return tool_error(exc)


def _handle_check(args: dict, **kw) -> str:
    try:
        restaurant = _resolve_restaurant(args.get("restaurant"))
        file_id = str(args.get("file_id") or "").strip()
        shortcut_policy = str(args.get("shortcut_policy") or "reject").strip().lower()
        cfg = _load_config(restaurant)
        service = _drive_service(cfg["token_path"])
        ok, reason, meta, ancestry = _ancestry_allowed(
            service,
            file_id,
            set(cfg["allowed_roots"]),
            shortcut_policy=shortcut_policy,
        )
        result: dict[str, Any] = {
            "ok": True,
            "restaurant": restaurant,
            "file_id": file_id,
            "allowed": ok,
            "result": reason,
            "allowed_roots": cfg["allowed_roots"],
            "boundary": "host-side-wrapper",
            "credential_exposure": "host-side-only",
        }
        if ok and meta:
            drive_path, relative_drive_path = _drive_paths_from_ancestry(
                service,
                ancestry,
                set(cfg["allowed_roots"]),
            )
            result["metadata"] = _safe_meta(meta)
            if drive_path:
                result["metadata"]["drivePath"] = drive_path
                result["metadata"]["relativeDrivePath"] = relative_drive_path
            result["ancestry_ids"] = ancestry
        return tool_result(result)
    except Exception as exc:
        logger.exception("google_drive_boundary_check failed")
        return tool_error(exc)


def _handle_metadata(args: dict, **kw) -> str:
    try:
        restaurant = _resolve_restaurant(args.get("restaurant"))
        file_id = str(args.get("file_id") or "").strip()
        shortcut_policy = str(args.get("shortcut_policy") or "reject").strip().lower()
        cfg = _load_config(restaurant)
        service = _drive_service(cfg["token_path"])
        ok, reason, meta, ancestry = _ancestry_allowed(
            service,
            file_id,
            set(cfg["allowed_roots"]),
            shortcut_policy=shortcut_policy,
        )
        if not ok:
            return tool_result(
                ok=True,
                restaurant=restaurant,
                file_id=file_id,
                allowed=False,
                result=reason,
                boundary="host-side-wrapper",
            )
        drive_path, relative_drive_path = _drive_paths_from_ancestry(
            service,
            ancestry,
            set(cfg["allowed_roots"]),
        )
        return tool_result(
            ok=True,
            restaurant=restaurant,
            file_id=file_id,
            allowed=True,
            result=reason,
            metadata=_safe_meta(
                meta or {},
                drive_path=drive_path or None,
                relative_drive_path=relative_drive_path,
            ),
            ancestry_ids=ancestry,
            boundary="host-side-wrapper",
        )
    except Exception as exc:
        logger.exception("google_drive_boundary_metadata failed")
        return tool_error(exc)


def _handle_download(args: dict, **kw) -> str:
    try:
        restaurant = _resolve_restaurant(args.get("restaurant"))
        file_id = str(args.get("file_id") or "").strip()
        if not file_id:
            raise ValueError("file_id is required")

        shortcut_policy = str(args.get("shortcut_policy") or "reject").strip().lower()
        export_mime_type = str(
            args.get("export_mime_type")
            or args.get("export_format")
            or DEFAULT_EXPORT_MIME_TYPE
        ).strip()
        max_bytes = max(1, min(int(args.get("max_bytes") or 104_857_600), 524_288_000))

        cfg = _load_config(restaurant)
        service = _drive_service(cfg["token_path"])
        allowed_roots = set(cfg["allowed_roots"])
        ok, reason, meta, ancestry = _ancestry_allowed(
            service,
            file_id,
            allowed_roots,
            shortcut_policy=shortcut_policy,
        )
        if not ok:
            return tool_result(
                ok=True,
                restaurant=restaurant,
                file_id=file_id,
                allowed=False,
                result=reason,
                boundary="host-side-wrapper",
                credential_exposure="host-side-only",
            )
        if not meta:
            raise RuntimeError("Drive metadata unavailable after boundary check")

        mime_type = str(meta.get("mimeType") or "")
        if mime_type == FOLDER_MIME:
            raise ValueError("file_id points to a folder; download requires a file")
        if mime_type == SHORTCUT_MIME:
            raise ValueError("shortcuts are not downloadable with shortcut_policy=reject")

        drive_path, relative_drive_path = _drive_paths_from_ancestry(
            service,
            ancestry,
            allowed_roots,
        )

        native_export = _is_google_native_file(mime_type)
        if native_export:
            request = _files_resource(service).export_media(
                fileId=file_id,
                mimeType=export_mime_type,
            )
            operation = "export"
        else:
            export_mime_type = ""
            request = _files_resource(service).get_media(
                fileId=file_id,
                supportsAllDrives=True,
            )
            operation = "download"

        output_arg = str(args.get("output_path") or "").strip()
        if output_arg:
            output_path = _workspace_output_path(cfg, output_arg)
        else:
            output_path = _default_download_output_path(
                cfg,
                meta,
                relative_drive_path,
                export_mime_type or None,
            )

        bytes_written = _download_request_to_path(request, output_path, max_bytes)
        workspace_path = _host_to_workspace_path(cfg, output_path)

        change_key = {
            "google_file_id": file_id,
            "modified_time": meta.get("modifiedTime"),
        }
        if native_export:
            change_key["export_mime_type"] = export_mime_type
        elif meta.get("md5Checksum"):
            change_key["md5_checksum"] = meta.get("md5Checksum")

        return tool_result(
            ok=True,
            restaurant=restaurant,
            file_id=file_id,
            allowed=True,
            result="downloaded",
            operation=operation,
            export_mime_type=export_mime_type or None,
            metadata=_safe_meta(
                meta,
                drive_path=drive_path or None,
                relative_drive_path=relative_drive_path,
            ),
            ancestry_ids=ancestry,
            drive_path=drive_path,
            relative_drive_path=relative_drive_path,
            host_path=str(output_path),
            workspace_path=workspace_path,
            bytes_written=bytes_written,
            change_key=change_key,
            boundary="host-side-wrapper",
            credential_exposure="host-side-only",
        )
    except Exception as exc:
        logger.exception("google_drive_boundary_download failed")
        return tool_error(exc)


GOOGLE_DRIVE_BOUNDARY_LIST_SCHEMA = {
    "name": "google_drive_boundary_list",
    "description": (
        "List files and folders under the restaurant's configured Google Drive "
        "allowed roots. Host-side wrapper; does not expose Google token/client "
        "files to the profile container."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "restaurant": {
                "type": "string",
                "description": "Restaurant id. Defaults to the active profile.",
                "enum": sorted(RESTAURANT_WORKSPACES),
            },
            "max_items": {
                "type": "integer",
                "description": "Maximum folders/files/shortcuts to return.",
                "default": 500,
                "minimum": 1,
                "maximum": 5000,
            },
            "shortcut_policy": {
                "type": "string",
                "description": "Reject shortcuts by default, or resolve and verify target ancestry.",
                "enum": ["reject", "resolve"],
                "default": "reject",
            },
        },
    },
}


GOOGLE_DRIVE_BOUNDARY_CHECK_SCHEMA = {
    "name": "google_drive_boundary_check",
    "description": (
        "Check whether a Google Drive file/folder id is inside the restaurant's "
        "configured allowed roots. Returns allowed=false for out-of-boundary ids."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_id": {"type": "string", "description": "Google Drive file or folder id to check."},
            "restaurant": {
                "type": "string",
                "description": "Restaurant id. Defaults to the active profile.",
                "enum": sorted(RESTAURANT_WORKSPACES),
            },
            "shortcut_policy": {
                "type": "string",
                "enum": ["reject", "resolve"],
                "default": "reject",
            },
        },
        "required": ["file_id"],
    },
}


GOOGLE_DRIVE_BOUNDARY_METADATA_SCHEMA = {
    "name": "google_drive_boundary_metadata",
    "description": (
        "Return metadata for a Google Drive id only after the id passes the "
        "restaurant allowed-root ancestry check."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_id": {"type": "string", "description": "Google Drive file or folder id."},
            "restaurant": {
                "type": "string",
                "description": "Restaurant id. Defaults to the active profile.",
                "enum": sorted(RESTAURANT_WORKSPACES),
            },
            "shortcut_policy": {
                "type": "string",
                "enum": ["reject", "resolve"],
                "default": "reject",
            },
        },
        "required": ["file_id"],
    },
}


GOOGLE_DRIVE_BOUNDARY_DOWNLOAD_SCHEMA = {
    "name": "google_drive_boundary_download",
    "description": (
        "Download or export Google Drive file content after the file passes the "
        "restaurant allowed-root ancestry check. Writes only inside the restaurant "
        "workspace and keeps Google token/client files host-side."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_id": {"type": "string", "description": "Google Drive file id to download/export."},
            "restaurant": {
                "type": "string",
                "description": "Restaurant id. Defaults to the active profile.",
                "enum": sorted(RESTAURANT_WORKSPACES),
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Destination inside the restaurant workspace. Accepts /workspace/... "
                    "or a relative path. If omitted, writes under /workspace/drive/originals "
                    "using a duplicate-safe filename containing the Google file id."
                ),
            },
            "export_mime_type": {
                "type": "string",
                "description": (
                    "Export MIME type for Google-native Docs/Sheets/Slides. "
                    "Defaults to application/pdf. Ignored for binary Drive files."
                ),
                "default": DEFAULT_EXPORT_MIME_TYPE,
            },
            "shortcut_policy": {
                "type": "string",
                "enum": ["reject", "resolve"],
                "default": "reject",
            },
            "max_bytes": {
                "type": "integer",
                "description": "Maximum downloaded/exported byte count.",
                "default": 104857600,
                "minimum": 1,
                "maximum": 524288000,
            },
        },
        "required": ["file_id"],
    },
}


registry.register(
    name="google_drive_boundary_list",
    toolset="google_drive_boundary",
    schema=GOOGLE_DRIVE_BOUNDARY_LIST_SCHEMA,
    handler=_handle_list,
    check_fn=_check_google_drive_boundary_requirements,
    emoji="G",
    max_result_size_chars=100_000,
)
registry.register(
    name="google_drive_boundary_check",
    toolset="google_drive_boundary",
    schema=GOOGLE_DRIVE_BOUNDARY_CHECK_SCHEMA,
    handler=_handle_check,
    check_fn=_check_google_drive_boundary_requirements,
    emoji="G",
    max_result_size_chars=50_000,
)
registry.register(
    name="google_drive_boundary_metadata",
    toolset="google_drive_boundary",
    schema=GOOGLE_DRIVE_BOUNDARY_METADATA_SCHEMA,
    handler=_handle_metadata,
    check_fn=_check_google_drive_boundary_requirements,
    emoji="G",
    max_result_size_chars=50_000,
)
registry.register(
    name="google_drive_boundary_download",
    toolset="google_drive_boundary",
    schema=GOOGLE_DRIVE_BOUNDARY_DOWNLOAD_SCHEMA,
    handler=_handle_download,
    check_fn=_check_google_drive_boundary_requirements,
    emoji="G",
    max_result_size_chars=50_000,
)
