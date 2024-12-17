def test_get_root_pwd(monkeypatch, capsys):
    """
    Test the get_root_pwd
    """
    from activetigger.functions import (
        get_root_pwd,
    )  # Import the function from your module

    inputs = iter(["password123", "password123"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    # Call the function
    root_password = get_root_pwd()

    # Capture the stdout
    captured = capsys.readouterr()

    # Verify the correct password was returned
    assert root_password == "password123"
    assert "Password confirmed successfully." in captured.out
    assert "Creating the entry in the database..." in captured.out
