#!/bin/bash

pandoc paper_new.md --citeproc \
  --reference-doc=reference.docx \
  -M date="$(date '+%B %d, %Y')" \
  -o paper_new.docx --verbose
