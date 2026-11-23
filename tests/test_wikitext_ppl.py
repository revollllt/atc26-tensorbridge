"""Offline checks of WikiText prompt construction and GPTQ loss arithmetic."""

import io
import json
import math
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import eval_wikitext_ppl as evaluator  # noqa: E402


class WikiTextTest(unittest.TestCase):
    def test_teacher_forcing_and_windows(self) -> None:
        # Two complete windows plus a tail. Give each window independent BF16 head outputs.
        seqlen, vocab = 7, 29
        generator = torch.Generator().manual_seed(91)
        logits = torch.randn(2, seqlen, vocab, generator=generator).bfloat16()
        labels = torch.arange(2 * seqlen + 3)
        windows = labels[: 2 * seqlen].view(2, seqlen)
        scores = logits[:, :-1].float().log_softmax(-1).gather(-1, windows[:, 1:, None]).squeeze(-1)
        ce = torch.nn.CrossEntropyLoss()
        losses = torch.stack([ce(pred[:-1].float(), ids[1:]) for pred, ids in zip(logits, windows, strict=True)])
        self.assertLess((losses + scores.mean(1)).abs().max().item(), 1e-6)

        def respond(request: urllib.request.Request, timeout: int) -> io.BytesIO:
            body = json.loads(request.data)
            prompt = body.pop("prompt")
            self.assertEqual(
                body,
                dict(
                    model="glm53",
                    echo=True,
                    add_special_tokens=False,
                    logprobs=1,
                    temperature=0,
                    return_token_ids=True,
                    max_tokens=0,
                ),
            )
            index = windows.tolist().index(prompt)
            choice = dict(prompt_token_ids=prompt, logprobs=dict(token_logprobs=[None, *scores[index].tolist()]))
            return io.BytesIO(json.dumps(dict(choices=[choice], usage=dict(prompt_tokens=seqlen))).encode())

        texts = ["", " Heading\n", "", "最后一行\n"]
        joined = "\n\n".join(texts)
        tokenizer = Mock(return_value=SimpleNamespace(input_ids=labels[None, :]))
        factory = Mock()
        factory.from_pretrained.return_value = tokenizer
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "test.json")
            path.write_text(json.dumps([{"text": row} for row in texts]))
            plain = path.with_suffix(".txt")
            plain.write_bytes(joined.encode())
            self.assertEqual(evaluator.read_wikitext(path), evaluator.read_wikitext(plain))
            path.write_text("\n".join(json.dumps({"text": row}) for row in texts))
            self.assertEqual(evaluator.read_wikitext(path), joined)
            with (
                patch.dict("sys.modules", transformers=SimpleNamespace(AutoTokenizer=factory)),
                patch.object(evaluator.urllib.request, "urlopen", side_effect=respond) as send,
            ):
                result = evaluator.evaluate(path, "/offline", seqlen, 2, "http://localhost/v1/completions", "glm53")
        tokenizer.assert_called_once_with(joined, return_tensors="pt")
        factory.from_pretrained.assert_called_once_with("/offline", trust_remote_code=True, local_files_only=True)
        self.assertEqual(send.call_count, 2)
        self.assertEqual((result["nsamples"], result["scored_tokens"], result["total_tokens"]), (2, 12, 17))
        gptq = ((-scores.mean(1) * seqlen).sum() / (2 * seqlen)).exp()
        self.assertEqual(result["ppl_gptq_fp32"], gptq.item())
        self.assertAlmostEqual(result["ppl_fp64"], math.exp(-scores.double().mean().item()), places=12)

    def test_bad_prompt_logprobs(self) -> None:
        for prompt, scores in (
            ([1, 0], [None, -1]),
            ([0, 1], [None]),
            ([0, 1], [None, math.nan]),
            ([0, 1], [0, -1]),
            ([0, 1], [None, -9999]),
        ):
            record = dict(
                choices=[dict(prompt_token_ids=prompt, logprobs=dict(token_logprobs=scores))],
                usage=dict(prompt_tokens=2),
            )
            with (
                self.subTest(prompt=prompt, scores=scores),
                patch.object(evaluator.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(record).encode())),
                self.assertRaisesRegex(ValueError, "window 0"),
            ):
                evaluator.score_windows(torch.arange(2), 2, 1, "http://localhost/v1/completions", "glm53")


if __name__ == "__main__":
    unittest.main()
