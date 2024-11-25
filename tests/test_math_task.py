import unittest
from datasets.base_dataset import BaseDataset, EvalSample, Message
from unittest.mock import MagicMock

from evalutaion.math_task import MATHFewShot


class TestMATHFewShot(unittest.TestCase):

    def setUp(self):
        dataset = MagicMock(spec=BaseDataset)
        dataset.config_name.return_value = 'math'
        dataset.get_test_samples.return_value = [
            {'question': 'Sample question?', 'answer': 'Sample answer', 'level': 'easy', 'category': 'algebra'}
        ]
        self.math_few_shot = MATHFewShot(dataset)

    def test_initialization_with_non_math_dataset(self):
        dataset = MagicMock(spec=BaseDataset)
        dataset.config_name.return_value = 'not_math'
        with self.assertRaises(ValueError):
            MATHFewShot(dataset)

    def test_initialization_with_math_dataset(self):
        dataset = MagicMock(spec=BaseDataset)
        dataset.config_name.return_value = 'math'
        math_few_shot_instance = MATHFewShot(dataset)
        self.assertIsInstance(math_few_shot_instance, MATHFewShot)

    def test_iter(self):
        samples = list(self.math_few_shot)
        self.assertEqual(len(samples), 1)
        self.assertIsInstance(samples[0], EvalSample)

    def test_generate_n_shot_messages(self):
        question = 'New question?'
        messages = self.math_few_shot._generate_n_shot_messages(question)  # Assuming testing _protected method.
        self.assertIsInstance(messages, list)
        self.assertGreater(len(messages), 0)
        self.assertIsInstance(messages[0], Message)

    def test_few_shot_samples(self):
        samples = self.math_few_shot._few_shot_samples()
        self.assertIsInstance(samples, list)
        self.assertGreater(len(samples), 0)
        self.assertIsInstance(samples[0], dict)

    def test_extract_answer(self):
        answer = "Final Answer: The final answer is $24$."
        extracted_answer = MATHFewShot.extract_answer(answer)
        self.assertIsInstance(extracted_answer, str)

    def test_is_correct(self):
        sample = EvalSample(messages=[], target="So in the end solution is $\\boxed{24}$.", description="")
        answer = "Final Answer: The final answer is 24. I hope it is correct."
        self.assertTrue(self.math_few_shot.is_correct(sample, answer))


if __name__ == '__main__':
    unittest.main()
