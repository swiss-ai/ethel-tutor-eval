import unittest
from datasets.base_dataset import BaseDataset, Message, EvalSample
from unittest.mock import MagicMock, patch

from evalutaion.gsm8k_task import GSM8KNShot


class TestGSM8KNShot(unittest.TestCase):

    def setUp(self):
        self.mock_dataset = MagicMock(spec=BaseDataset)
        self.mock_dataset.config_name.return_value = 'gsm8k'
        self.mock_dataset.get_train_samples.return_value = [
            {'question': 'Train question 1?', 'answer': '#### 1'},
            {'question': 'Train question 2?', 'answer': '#### 2'},
            {'question': 'Train question 3?', 'answer': '#### 3'},
            {'question': 'Train question 4?', 'answer': '#### 4'},
            {'question': 'Train question 5?', 'answer': '#### 5'},
            {'question': 'Train question 6?', 'answer': '#### 6'},
            {'question': 'Train question 7?', 'answer': '#### 7'},
            {'question': 'Train question 8?', 'answer': '#### 8'}
        ]

    def test_init_with_valid_dataset(self):
        task = GSM8KNShot(dataset=self.mock_dataset)
        self.assertEqual(task.dataset, self.mock_dataset)
        self.assertEqual(task.n, 8)

    def test_init_with_invalid_dataset(self):
        self.mock_dataset.config_name.return_value = 'other_dataset'
        with self.assertRaises(ValueError):
            GSM8KNShot(dataset=self.mock_dataset)

    def test_extract_answer_valid(self):
        answer = "#### 123"
        extracted = GSM8KNShot.extract_answer(answer)
        self.assertEqual(extracted, '123')

    def test_extract_answer_invalid(self):
        answer = "No valid answer here"
        extracted = GSM8KNShot.extract_answer(answer)
        self.assertEqual(extracted, GSM8KNShot.INVALID_ANS)

    def test_generate_n_shot_messages(self):
        task = GSM8KNShot(dataset=self.mock_dataset)
        question = 'Test Question?'
        messages = task._generate_n_shot_messages(question)

        self.assertEqual(len(messages), 17)
        self.assertEqual(messages[-1].content,
                         f"Question: {question} \n            Let's think step by step. At the end, you MUST write the answer as an integer after '####'.\n           ")

    @patch('random.sample')
    def test_generate_n_shot_population(self, mock_random_sample):
        samples = self.mock_dataset.get_train_samples()[:8]
        mock_random_sample.return_value = samples

        task = GSM8KNShot(dataset=self.mock_dataset)
        question = 'Test Question?'
        task._generate_n_shot_messages(question)

        self.assertEqual(task._n_shot_samples, samples)

    def test_iter(self):
        task = GSM8KNShot(dataset=self.mock_dataset)
        self.mock_dataset.get_test_samples.return_value = [{'question': 'Test question?', 'answer': '#### 42'}]

        samples = list(task.__iter__())
        self.assertEqual(len(samples), 1)
        self.assertIsInstance(samples[0], EvalSample)
        self.assertEqual(samples[0].target, '#### 42')


if __name__ == '__main__':
    unittest.main()
