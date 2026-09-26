"""Give a .docx neutral document properties.

    python3 code/tools/set_docx_metadata.py FILE.docx [...]

pandoc leaves the author empty, but python-docx writes "python-docx" as the last editor when it
creates the properties part, and Word shows these fields to anyone who opens the file. This sets
creator and last-modified-by to "Authors" and clears comments, keywords, subject and category;
the title is kept."""
import sys

from docx import Document


def neutral(path):
    doc = Document(path)
    cp = doc.core_properties
    cp.author = "Authors"
    cp.last_modified_by = "Authors"
    cp.comments = ""
    cp.keywords = ""
    cp.subject = ""
    cp.category = ""
    doc.save(path)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        neutral(f)
