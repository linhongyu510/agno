"""Tests for the Message class."""

import json

import pytest

from agno.media import Audio, File, Image, Video
from agno.models.message import Message


class TestGetContentString:
    """Tests for Message.get_content_string() method."""

    def test_string_content(self):
        """Test that string content is returned as-is."""
        message = Message(role="assistant", content="Hello, world!")
        assert message.get_content_string() == "Hello, world!"

    def test_empty_list_content_returns_empty_string(self):
        """Test that empty list content returns empty string, not '[]'.

        This is a regression test for the bug where empty content lists
        (common after tool execution) would return '[]' string.
        """
        message = Message(role="assistant", content=[])
        result = message.get_content_string()
        assert result == ""
        assert result != "[]"

    def test_list_with_text_dict(self):
        """Test that list with text dict returns the text."""
        message = Message(role="assistant", content=[{"text": "Hello from list"}])
        assert message.get_content_string() == "Hello from list"

    def test_list_with_text_dict_empty_text(self):
        """Test that list with empty text returns empty string."""
        message = Message(role="assistant", content=[{"text": ""}])
        assert message.get_content_string() == ""

    def test_list_with_non_text_dict(self):
        """Test that list with non-text dict returns JSON."""
        content = [{"type": "image", "url": "http://example.com/image.png"}]
        message = Message(role="assistant", content=content)
        assert message.get_content_string() == json.dumps(content)

    def test_list_with_multiple_items(self):
        """Test that list with multiple items returns first text."""
        message = Message(
            role="assistant",
            content=[{"text": "First"}, {"text": "Second"}],
        )
        assert message.get_content_string() == "First"

    def test_none_content(self):
        """Test that None content returns empty string."""
        message = Message(role="assistant", content=None)
        assert message.get_content_string() == ""

    def test_list_with_strings(self):
        """Test that list of strings returns JSON dump."""
        content = ["item1", "item2"]
        message = Message(role="assistant", content=content)
        assert message.get_content_string() == json.dumps(content)


class TestOutputMediaRoundTrip:
    """Regression: model-produced output media must survive to_dict/from_dict.

    `to_dict()` persisted `audio_output` but omitted `image_output`,
    `video_output` and `file_output`, even though `from_dict()` already had
    reconstruction branches for them. With DB-backed session persistence the
    assistant's output media (images/video/files returned by a multimodal
    model) were silently dropped.
    """

    @pytest.mark.parametrize(
        ("attr", "media"),
        [
            ("audio_output", Audio(url="https://example.com/out.mp3")),
            ("image_output", Image(url="https://example.com/out.png")),
            ("video_output", Video(url="https://example.com/out.mp4")),
            ("file_output", File(url="https://example.com/out.pdf")),
        ],
    )
    def test_output_media_round_trip(self, attr, media):
        message = Message(role="assistant", content="done", **{attr: media})
        restored = Message.from_dict(message.to_dict())
        round_tripped = getattr(restored, attr)
        assert round_tripped is not None
        assert round_tripped.url == media.url

    def test_no_output_media_stays_absent(self):
        message = Message(role="assistant", content="done")
        d = message.to_dict()
        assert "image_output" not in d
        assert "video_output" not in d
        assert "file_output" not in d
        assert "audio_output" not in d
