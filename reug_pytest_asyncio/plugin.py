"""Minimal asyncio support plugin for pytest."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from ._compat import ensure_fixturedef_alias

ensure_fixturedef_alias(pytest)

_VALID_MODES = {"auto", "strict", "legacy"}


def _get_mode(config: pytest.Config) -> str:
    cli_mode = config.getoption("asyncio_mode", default=None)
    ini_mode = config.getini("asyncio_mode") if hasattr(config, "getini") else None
    mode = cli_mode or ini_mode or "auto"
    if mode not in _VALID_MODES:
        return "auto"
    return mode


def pytest_addoption(parser: pytest.Parser) -> None:  # pragma: no cover - exercised via integration
    group = parser.getgroup("asyncio")
    try:
        group.addoption(
            "--asyncio-mode",
            action="store",
            dest="asyncio_mode",
            choices=sorted(_VALID_MODES),
            help="Set asyncio plugin mode (auto, strict, legacy).",
        )
    except ValueError:  # pragma: no cover - option already registered.
        pass

    try:
        parser.addini("asyncio_mode", "Asyncio handling mode.", default="auto")
    except ValueError:  # pragma: no cover - ini option already registered.
        pass


def pytest_configure(config: pytest.Config) -> None:
    mode = _get_mode(config)
    setattr(config, "_asyncio_mode", mode)
    config.addinivalue_line(
        "markers",
        "asyncio: execute the test inside an asyncio event loop.",
    )


@pytest.fixture
def event_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except RuntimeError:
            pass
        loop.close()


@pytest.fixture
def asyncio_event_loop(event_loop: asyncio.AbstractEventLoop) -> asyncio.AbstractEventLoop:
    return event_loop


def _should_run_async(pyfuncitem: pytest.Function) -> bool:
    obj = pyfuncitem.obj
    marker_present = pyfuncitem.get_closest_marker("asyncio") is not None
    is_coroutine = asyncio.iscoroutinefunction(obj)
    mode = getattr(pyfuncitem.config, "_asyncio_mode", "auto")

    if mode == "strict":
        return marker_present
    if mode == "legacy":
        return marker_present or is_coroutine
    # auto
    return marker_present or is_coroutine


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    if not _should_run_async(pyfuncitem):
        return None

    func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(func):
        return None

    loop = pyfuncitem.funcargs.get("event_loop")
    created_loop = False
    if loop is None:
        loop = asyncio.new_event_loop()
        created_loop = True

    call_kwargs = {
        name: pyfuncitem.funcargs[name]
        for name in getattr(pyfuncitem._fixtureinfo, "argnames", ())
        if name in pyfuncitem.funcargs
    }

    try:
        coroutine = func(**call_kwargs)
        if not inspect.isawaitable(coroutine):
            return None
        loop.run_until_complete(coroutine)
    finally:
        if created_loop:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except RuntimeError:
                pass
            loop.close()
    return True
