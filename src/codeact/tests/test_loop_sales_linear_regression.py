import textwrap

import pytest

from codeact.actions import AgentFinish
from codeact.bus_handlers import CodeActStartRequest, handle_start
from codeact.runner import CodeActRunner
from codeact.sandbox import PythonSandbox
from tests.runtime.fakes import FakeEventBus


class FinishPolicy:
    def __call__(self, _):
        return AgentFinish()


@pytest.mark.asyncio
async def test_sales_regression(tmp_path):
    data = "ads,sales\n1,3\n2,5\n3,7\n"
    (tmp_path / "sales.csv").write_text(data)
    code = textwrap.dedent(
        """
        import csv, statistics
        xs, ys = [], []
        with open('sales.csv') as f:
            for row in csv.DictReader(f):
                xs.append(float(row['ads']))
                ys.append(float(row['sales']))
        n = len(xs)
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((x - x_mean)*(y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean)**2 for x in xs)
        m = num / den
        b = y_mean - m * x_mean
        print(f'y={m:.1f}x+{b:.1f}')
        """
    ).strip()
    sandbox = PythonSandbox(workdir=str(tmp_path))
    runner = CodeActRunner(sandbox, FinishPolicy())
    bus = FakeEventBus()
    event = CodeActStartRequest(code=code)
    await handle_start(event, bus, runner)
    ui_event = [e for e in bus.events if e["event_type"] == "ui_notification"][0]
    assert "y=2.0x+1.0" in ui_event["message"]
