"""Generate the API reference pages from the package's source tree.

Run automatically by the mkdocs-gen-files plugin on every `mkdocs build`/`mkdocs serve`;
see mkdocs.yml. One markdown stub (containing a single `::: module` mkdocstrings
directive) is generated per module, plus a `SUMMARY.md` consumed by
mkdocs-literate-nav to build the "API Reference" nav section.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src"

for package in ["eos", "teosar"]:
    for path in sorted((src / package).rglob("*.py")):
        module_path = path.relative_to(src).with_suffix("")
        doc_path = path.relative_to(src).with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        parts = tuple(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1].startswith("_"):
            # skip private modules (e.g. eos.sar._phase_correlation)
            continue

        if not parts:
            continue

        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            identifier = ".".join(parts)
            print(f"::: {identifier}", file=fd)

        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
