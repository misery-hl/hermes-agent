from pathlib import Path
import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/pga/order/catalog", adapter._handle_pga_order_catalog)
    app.router.add_post("/v1/pga/order/submissions", adapter._handle_pga_order_submission)
    app.router.add_get("/v1/pga/order/submissions/{submission_id}", adapter._handle_pga_order_submission_status)
    app.router.add_post(
        "/v1/pga/order/submissions/{submission_id}/amazon-cart",
        adapter._handle_pga_order_amazon_cart,
    )
    return app


def _install_fake_intake(profile_home: Path) -> None:
    plugin_dir = profile_home / "plugins" / "restaurant-output"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
    (plugin_dir / "intake.py").write_text(
        """
CART_STATUS = None
SUBMISSIONS = {}


def catalog_projection():
    return {"vendorGroups": [{"vendorId": "ven-amazon", "vendorName": "Amazon", "items": []}]}


def submit_order_payload(payload, *, actor=None, idempotency_key=None):
    submission_id = "order-test"
    SUBMISSIONS[submission_id] = payload
    lines = payload.get("lines") or payload.get("items") or []
    if any(isinstance(line, dict) and line.get("itemId") == "item-bagel" for line in lines):
        return {
            "ok": True,
            "submissionId": submission_id,
            "status": "dry_run_dispatch_planned",
            "locationCode": payload.get("locationCode"),
            "lineCount": 1,
            "vendorCount": 1,
            "unitCount": 1,
            "actor": actor,
            "idempotencyKey": idempotency_key,
            "vendorOutputs": [
                {
                    "vendor_id": "ven-better-brands",
                    "output_type": "sms",
                    "status": "sms_send_planned",
                }
            ],
        }
    return {
        "ok": True,
        "submissionId": submission_id,
        "status": "dry_run_dispatch_planned",
        "locationCode": payload.get("locationCode"),
        "lineCount": 1,
        "vendorCount": 1,
        "unitCount": 1,
        "actor": actor,
        "idempotencyKey": idempotency_key,
        "vendorOutputs": [
            {
                "vendor_id": "ven-amazon",
                "output_type": "amazon_cart",
                "mcp_tool": "prepare_cart",
                "mcp_request": {"lines": [{"item_id": "item-ginger-juice", "quantity": 1}]},
            }
        ],
    }


def get_submission_status(submission_id):
    result = {
        "ok": True,
        "submissionId": submission_id,
        "status": "dry_run_dispatch_planned",
        "dispatch": {"dryRun": True, "status": "dry_run_dispatch_planned"},
    }
    if CART_STATUS is not None:
        result["amazonCart"] = CART_STATUS
    return result


def prepare_amazon_cart_from_submission(submission_id, payload):
    return {
        "ok": True,
        "submissionId": submission_id,
        "status": "amazon_cart_mcp_request_ready",
        "vendorId": "ven-amazon",
        "mcpServer": "clockwork_amazon",
        "mcpTool": "prepare_cart",
        "mcpRequest": {"lines": [{"item_id": "item-ginger-juice", "quantity": 1}]},
        "lineCount": 1,
        "unitCount": 1,
        "confirmed": payload.get("confirm_cart_mutation") is True,
    }


def record_amazon_cart_prepare_result(submission_id, cart_result, *, mcp_request=None):
    global CART_STATUS
    CART_STATUS = {
        "ok": cart_result.get("ok") is True,
        "status": "amazon_cart_prepared" if cart_result.get("ok") is True else "amazon_cart_prepare_failed",
        "cartStatus": cart_result.get("status"),
        "vendorId": "ven-amazon",
        "mode": "cart_only_live",
        "cartMutationPerformed": cart_result.get("status") == "cart_prepared",
        "statusPersisted": True,
        "noCheckout": True,
        "noPurchaseMade": True,
        "noPaymentSet": True,
        "noAddressSet": True,
    }
    return {
        "ok": True,
        "submissionId": submission_id,
        "status": CART_STATUS["status"],
        "cartStatus": cart_result.get("status"),
        "mcpRequest": mcp_request,
        "statusPersisted": True,
    }
""",
        encoding="utf-8",
    )


@pytest.fixture()
def pga_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_home = tmp_path / ".hermes" / "profiles" / "pga"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    _install_fake_intake(profile_home)
    return profile_home


@pytest.mark.asyncio
async def test_pga_order_intake_routes_require_bearer_auth(pga_profile):
    adapter = _make_adapter()
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/v1/pga/order/catalog")

    assert resp.status == 401


@pytest.mark.asyncio
async def test_pga_order_intake_routes_load_profile_module(pga_profile):
    adapter = _make_adapter()
    cart_calls = []

    async def fake_prepare_cart(mcp_request):
        cart_calls.append(mcp_request)
        return {
            "ok": True,
            "status": "cart_prepared",
            "no_purchase_made": True,
            "cart_item_count": 1,
            "cart_unit_count": 1,
        }

    adapter._execute_pga_amazon_prepare_cart = fake_prepare_cart
    app = _create_app(adapter)
    headers = {"Authorization": "Bearer sk-test"}

    async with TestClient(TestServer(app)) as client:
        catalog_resp = await client.get("/v1/pga/order/catalog", headers=headers)
        catalog = await catalog_resp.json()
        submit_resp = await client.post(
            "/v1/pga/order/submissions",
            headers={**headers, "Idempotency-Key": "idem-1"},
            json={"locationCode": "pga", "lines": [{"itemId": "item-a", "quantity": 1}]},
        )
        submitted = await submit_resp.json()
        if adapter._pga_order_cart_tasks:
            await asyncio.gather(*list(adapter._pga_order_cart_tasks))
        status_resp = await client.get("/v1/pga/order/submissions/order-test", headers=headers)
        status = await status_resp.json()
        cart_resp = await client.post(
            "/v1/pga/order/submissions/order-test/amazon-cart",
            headers=headers,
            json={"confirm_cart_mutation": True},
        )
        cart = await cart_resp.json()

    assert catalog_resp.status == 200
    assert catalog["vendorGroups"][0]["vendorId"] == "ven-amazon"
    assert submit_resp.status == 202
    assert submitted["actor"] == "watch"
    assert submitted["idempotencyKey"] == "idem-1"
    assert submitted["amazonCart"]["status"] == "amazon_cart_prepare_queued"
    assert submitted["amazonCart"]["cartOnlyLive"] is True
    assert submitted["amazonCart"]["dispatchDryRun"] is True
    assert status_resp.status == 200
    assert status["submissionId"] == "order-test"
    assert cart_resp.status == 202
    assert cart["ok"] is True
    assert cart["status"] == "amazon_cart_prepared"
    assert cart["cartOnlyLive"] is True
    assert cart_calls == [
        {"lines": [{"item_id": "item-ginger-juice", "quantity": 1}]},
    ]


@pytest.mark.asyncio
async def test_pga_order_submission_does_not_queue_amazon_cart_without_amazon_plan(pga_profile):
    adapter = _make_adapter()

    async def fail_prepare_cart(_mcp_request):
        raise AssertionError("non-Amazon order must not call Amazon MCP")

    adapter._execute_pga_amazon_prepare_cart = fail_prepare_cart
    app = _create_app(adapter)
    headers = {"Authorization": "Bearer sk-test"}

    async with TestClient(TestServer(app)) as client:
        submit_resp = await client.post(
            "/v1/pga/order/submissions",
            headers={**headers, "Idempotency-Key": "idem-no-amazon"},
            json={"locationCode": "pga", "lines": [{"itemId": "item-bagel", "quantity": 1}]},
        )
        submitted = await submit_resp.json()

    assert submit_resp.status == 202
    assert "amazonCart" not in submitted
    assert adapter._pga_order_cart_tasks == set()
    assert adapter._pga_order_cart_tasks_by_submission == {}


@pytest.mark.asyncio
async def test_pga_order_submission_coalesces_duplicate_amazon_cart_prepare_tasks(pga_profile):
    adapter = _make_adapter()
    cart_calls = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_prepare_cart(mcp_request):
        cart_calls.append(mcp_request)
        started.set()
        await release.wait()
        return {
            "ok": True,
            "status": "cart_prepared",
            "no_purchase_made": True,
            "cart_item_count": 1,
            "cart_unit_count": 1,
        }

    adapter._execute_pga_amazon_prepare_cart = slow_prepare_cart
    app = _create_app(adapter)
    headers = {"Authorization": "Bearer sk-test", "Idempotency-Key": "idem-repeat"}

    async with TestClient(TestServer(app)) as client:
        first_resp = await client.post(
            "/v1/pga/order/submissions",
            headers=headers,
            json={"locationCode": "pga", "lines": [{"itemId": "item-a", "quantity": 1}]},
        )
        first = await first_resp.json()
        await asyncio.wait_for(started.wait(), timeout=1)

        second_resp = await client.post(
            "/v1/pga/order/submissions",
            headers=headers,
            json={"locationCode": "pga", "lines": [{"itemId": "item-a", "quantity": 1}]},
        )
        second = await second_resp.json()
        release.set()
        if adapter._pga_order_cart_tasks:
            await asyncio.gather(*list(adapter._pga_order_cart_tasks))

    assert first_resp.status == 202
    assert first["amazonCart"]["status"] == "amazon_cart_prepare_queued"
    assert first["amazonCart"]["queued"] is True
    assert second_resp.status == 202
    assert second["amazonCart"]["status"] == "amazon_cart_prepare_in_progress"
    assert second["amazonCart"]["queued"] is False
    assert cart_calls == [{"lines": [{"item_id": "item-ginger-juice", "quantity": 1}]}]


@pytest.mark.asyncio
async def test_pga_order_intake_routes_fail_closed_outside_pga_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    adapter = _make_adapter()
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/v1/pga/order/catalog", headers={"Authorization": "Bearer sk-test"})
        payload = await resp.json()

    assert resp.status == 404
    assert payload == {
        "ok": False,
        "error": "pga_order_intake_unavailable",
        "message": "PGA order intake is unavailable.",
    }
