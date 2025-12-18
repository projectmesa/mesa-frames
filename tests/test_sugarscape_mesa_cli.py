from pathlib import Path


def test_saved_csv_echo_is_guarded_by_save_results_flag() -> None:
    """Ensure the CSV-saved confirmation is printed only when results are saved.

    This is a light-weight static check that avoids importing heavy runtime
    dependencies (e.g., `mesa`) in CI environments where they may be absent.
    """
    path = Path("examples/sugarscape_ig/backend_mesa/model.py")
    src = path.read_text()

    needle = 'typer.echo(f"Saved CSV results under {results_dir}")'
    # Ensure the message exists in the file (we just moved it into the if block)
    assert needle in src

    # For each occurrence, ensure it is preceded by an `if save_results:` within
    # the previous 6 lines (simple heuristic that is robust enough for this fix).
    lines = src.splitlines()
    for idx, line in enumerate(lines):
        if needle in line:
            window_start = max(0, idx - 6)
            window = "\n".join(lines[window_start:idx])
            assert "if save_results:" in window, (
                "Found CSV-saved confirmation not guarded by `if save_results:`"
            )
