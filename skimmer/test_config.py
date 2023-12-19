import unittest
from unittest.mock import patch, mock_open

from skimmer import config

class TestBuildConfigDict(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data="key: value")
    def test_build_config_dict_with_file_and_no_overrides(self, mock_file):
        result = config.build_config_dict('config.yaml', [])
        self.assertEqual(result, {'key': 'value'})

    def test_build_config_dict_with_no_file_and_dict_overrides(self):
        result = config.build_config_dict(None, {'key': 'value'})
        self.assertEqual(result, {'key': 'value'})

    def test_build_config_dict_with_no_file_and_list_overrides(self):
        result = config.build_config_dict(None, ['key=value'])
        self.assertEqual(result, {'key': 'value'})

    def test_build_config_dict_with_file_and_dict_overrides(self):
        with patch('builtins.open', new_callable=mock_open, read_data="key1: value1") as mock_file:
            result = config.build_config_dict('config.yaml', {'key2': 'value2'})
        self.assertEqual(result, {'key1': 'value1', 'key2': 'value2'})

    def test_build_config_dict_with_file_and_list_overrides(self):
        with patch('builtins.open', new_callable=mock_open, read_data="key1: value1") as mock_file:
            result = config.build_config_dict('config.yaml', ['key2=value2'])
        self.assertEqual(result, {'key1': 'value1', 'key2': 'value2'})

    def test_build_config_dict_with_invalid_override_string(self):
        with self.assertRaises(Exception) as context:
            config.build_config_dict(None, ['keyvalue'])
        self.assertTrue("Invalid config override string: 'keyvalue'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
