import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

P = Path(__file__).resolve().parents[1] / 'broker_multigpu_v1.py'
spec = importlib.util.spec_from_file_location('multi_broker', P)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class BudgetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.broker = object.__new__(b.Broker)
        self.broker.local = self.root

    def write(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))

    def prior(self, calls=10, inputs=200000, outputs=7000):
        self.write('binding.json', {'prior_usage': dict(cli_calls=calls, input_tokens=inputs, output_tokens=outputs)})

    def test_inherits_prior_consumption(self):
        self.prior()
        self.write('batches/b1/cli_started.json', {})
        self.write('batches/b1/receipt.json', {'usage_totals': dict(input_tokens=100, output_tokens=20)})
        self.assertEqual(self.broker.budget_state(), dict(cli_calls=11, input_tokens=200100, output_tokens=7020, unknown_usage=False))

    def test_prior_exhaustion_blocks_before_new_cli(self):
        self.prior(calls=1500)
        self.assertIn('exhausted', self.broker.cap_reason()[0])

    def test_unknown_prior_is_rejected(self):
        self.prior(inputs=None)
        with self.assertRaises(ValueError): self.broker.budget_state()

    def test_interrupted_new_call_blocks_followup(self):
        self.prior()
        self.write('batches/b1/cli_started.json', {})
        self.assertIn('Unresolved', self.broker.cap_reason()[0])

    def test_scope_and_frozen_model(self):
        self.assertEqual(b.OUTPUT_NAMES, ('originx-confirmatory-multigpu-20261009-v1',))
        self.assertEqual((b.MODEL, b.EFFORT), ('gpt-6-astra', 'high'))
        compile(b.remote_guard_source(), '<remote guard>', 'exec')
        self.assertIn('originx-confirmatory-multigpu-20261009-v1', b.REMOTE_PUBLISH)


if __name__ == '__main__': unittest.main()
