import unittest

import requests

from datasets.base_dataset import Message
from unittest.mock import patch, Mock

from models.ollama import OllamaModel


class TestOLlamaModel(unittest.TestCase):

    @patch("requests.post")
    def test_generate_success(self, mock_post):
        messages = [Message(role="user", content="Hello")]
        expected_content = "Hi there!"

        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": expected_content}
        }
        mock_response.raise_for_status = Mock()

        mock_post.return_value = mock_response

        model = OllamaModel()
        result = model.generate(messages)

        self.assertIsInstance(result, Message)
        self.assertEqual(result.role, "assistant")
        self.assertEqual(result.content, expected_content)

    @patch("requests.post")
    def test_generate_http_error(self, mock_post):
        messages = [Message(role="user", content="Hello")]

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("HTTP Error")

        mock_post.return_value = mock_response

        model = OllamaModel()

        with self.assertRaises(requests.HTTPError):
            model.generate(messages)

    @patch("requests.post")
    def test_generate_invalid_response_structure(self, mock_post):
        messages = [Message(role="user", content="Hello")]

        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()

        mock_post.return_value = mock_response

        model = OllamaModel()

        with self.assertRaises(KeyError):
            model.generate(messages)


if __name__ == "__main__":
    unittest.main()
