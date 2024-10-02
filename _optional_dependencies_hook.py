import json
import sys
import urllib.request
from typing import Any, Dict

from hatchling.metadata.plugin.interface import MetadataHookInterface


class OptionalDependenciesHook(MetadataHookInterface):
    PLUGIN_NAME = "optional_dependencies"

    def update(self, metadata: dict) -> None:
        optional_deps = {
            "mkdocs": [
                "mkdocs-material",
                "mkdocs-jupyter",
                "mkdocs-git-revision-date-localized-plugin",
                "mkdocs-minify-plugin",
            ],
            "sphinx": [
                "sphinx",
                "sphinx-rtd-theme",
                "numpydoc",
                "pydata-sphinx-theme",
                "sphinx-copybutton",
                "sphinx-design",
                "autodocsumm",
            ],
            "docs": ["mesa_frames[mkdocs,sphinx]", "perfplot", "seaborn"],
            "test": [
                "pytest",
                "pytest-cov",
                "typeguard",
            ],
        }

        try:
            url = "https://pypi.org/pypi/ibis-framework/json"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())

            extras = data["info"].get("requires_dist", [])
            backends = set()
            for extra in extras:
                if 'extra == "' in extra:
                    backend = extra.split('extra == "')[1].split('"')[0]
                    backends.add(backend)

            for backend in backends:
                optional_deps[backend] = [f"ibis-framework[{backend}]"]

            print(f"Detected ibis backends: {', '.join(backends)}", file=sys.stderr)
        except Exception as e:
            print(
                f"An error occurred while detecting ibis backends: {e}", file=sys.stderr
            )
            print("Falling back to default backends...", file=sys.stderr)
            default_backends = ["polars", "duckdb"]
            for backend in default_backends:
                optional_deps[backend] = [f"ibis-framework[{backend}]"]
            optional_deps["all"] = [f"ibis-framework[{','.join(default_backends)}]"]

        optional_deps["dev"] = ["mesa_frames[polars,duckdb,test,docs]", "mesa"]

        metadata["optional-dependencies"] = optional_deps
        print(f"Optional dependencies set: {optional_deps.keys()}", file=sys.stderr)
