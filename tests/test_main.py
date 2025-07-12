"""Unit tests for the main module."""

from tagent.main import main


def test_main(capsys):
    """Test the main function output."""
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello, World!"
