"""``kuhaku-mcp --config``: resolving the configured server. Needs neither extra -- the
MCP SDK is only imported when the server actually runs."""

from __future__ import annotations

import pytest

from kuhaku.core.exceptions import ConfigError
from kuhaku.serving import KuhakuMCPServer
from kuhaku.serving.cli import load_server, main

CONFIG = """
from pydantic import BaseModel
from kuhaku import Extractor
from kuhaku.serving import KuhakuMCPServer


class Order(BaseModel):
    product: str


class _LLM:
    name = "fake:model"

    def generate(self, system, user):
        return '{"data": {"product": "tea"}}'


def build():
    return KuhakuMCPServer(extractors={"order": Extractor(Order, llm=_LLM())})


server = build()
not_a_server = 42
"""


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "my_server.py"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def test_loads_an_instance_or_a_factory(config_file):
    assert isinstance(load_server(f"{config_file}:server"), KuhakuMCPServer)
    built = load_server(f"{config_file}:build")
    assert built.tool_names == ["extract_order"]


@pytest.mark.parametrize("suffix", [":not_a_server", ":missing", ""])
def test_bad_specs_are_config_errors(config_file, suffix):
    with pytest.raises(ConfigError):
        load_server(f"{config_file}{suffix}")


def test_main_reports_config_errors_on_stderr(config_file, capsys):
    assert main(["--config", f"{config_file}:not_a_server"]) == 2
    captured = capsys.readouterr()
    assert "kuhaku-mcp:" in captured.err
    assert captured.out == ""


def test_stdio_end_to_end_with_stray_prints(config_file, tmp_path):
    """A real subprocess over stdio: a config that prints while loading must not corrupt
    the protocol stream."""

    pytest.importorskip("mcp")
    import os
    import sys
    from pathlib import Path

    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    config_file.write_text("print('loading...')\n" + CONFIG, encoding="utf-8")
    src = str(Path(__file__).resolve().parents[2] / "src")
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kuhaku.serving.cli", "--config", f"{config_file}:server"],
        env=env,
        cwd=str(tmp_path),
    )

    async def run():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool("extract_order", {"text": "tea"})

    result = anyio.run(run)
    assert not result.isError
    assert result.structuredContent["value"] == {"product": "tea"}
