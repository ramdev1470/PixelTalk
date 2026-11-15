"""
Unit Tests for Dialogue Manager and Multi-Turn Conversation
Verifies conversational history tracking, context construction, and attention matrix extraction.
"""

import unittest
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_root_dir = os.path.dirname(_parent_dir)
for p in [_root_dir, _parent_dir, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from PixelTalk.models.dialogue_manager import DialogueManager
    from PixelTalk.data.dataset import format_encoder_input
except ImportError:
    from models.dialogue_manager import DialogueManager
    from data.dataset import format_encoder_input

class TestDialogueManager(unittest.TestCase):
    def setUp(self):
        # Resolve checkpoint and vocab relative to project directory
        ckpt_path = os.path.join(_parent_dir, "checkpoints", "seq2seq_attention.pt")
        vocab_path = os.path.join(_parent_dir, "checkpoints", "vocab.json")

        self.dm = DialogueManager(
            checkpoint_path=ckpt_path,
            vocab_path=vocab_path
        )
        self.test_caption = "A golden retriever puppy sitting alert on green lawn grass with a red rubber toy ball next to its front paws."
        self.dm.set_image_context(self.test_caption, ["dog", "puppy", "lawn", "red ball"])


    def test_context_formatting(self):
        history = [("What animal is this?", "This is a golden retriever puppy.")]
        formatted = format_encoder_input(self.test_caption, history, "What is it playing with?")
        self.assertIn("<cap>", formatted)
        self.assertIn("</cap>", formatted)
        self.assertIn("<usr>", formatted)
        self.assertIn("<bot>", formatted)
        self.assertIn("<qry>", formatted)
        self.assertIn("playing with", formatted)

    def test_single_turn_qa(self):
        result = self.dm.answer_question("What animal is this?")
        self.assertIn("answer", result)
        self.assertTrue(len(result["answer"]) > 0)
        self.assertEqual(len(self.dm.history), 1)
        self.assertEqual(result["turn_index"], 1)

    def test_multi_turn_flow(self):
        res1 = self.dm.answer_question("What is the puppy doing?")
        self.assertEqual(len(self.dm.history), 1)

        res2 = self.dm.answer_question("What color is the toy ball?")
        self.assertEqual(len(self.dm.history), 2)
        self.assertEqual(res2["turn_index"], 2)

        # Check attention extraction
        if res2["attention_matrix"] is not None:
            self.assertEqual(len(res2["attention_matrix"].shape), 2)

    def test_reset_conversation(self):
        self.dm.answer_question("What is this?")
        self.assertEqual(len(self.dm.history), 1)
        self.dm.reset()
        self.assertEqual(len(self.dm.history), 0)
        self.assertEqual(self.dm.current_caption, "")

    def test_uninitialized_prompt(self):
        dm_empty = DialogueManager(checkpoint_path=None, vocab_path=None)
        res = dm_empty.answer_question("Hello?")
        self.assertIn("upload or select an image", res["answer"].lower())

if __name__ == "__main__":
    unittest.main()
