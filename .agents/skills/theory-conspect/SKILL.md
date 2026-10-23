---
name: theory-conspect
description: Create a conspect of the requested information.
---

# Target file

Use .ipynb files.

If a user requested to append existing conspect – do it.
If a user requested creation of a new file with provided name and location – do it.
Otherwise - ask user to provide a name and location for the new conspect.

# File structure

Use markdown cells to create a conspect of the requested information.

Split information to paragraphs, use headings.
Create a new cell for each top-level paragraph.

If any important media is missing from the context, describe the missing content and ask the user to provide it.

Verify the file is valid and cells run successfully, after creation.

# Style

Follow MathJax (subset of TeX/LaTeX) syntax.

Do not use bold or italic formatting, event for formulas.

Start each equation form a new line with alignment to the center.

# Information selection guidance

Try to structure information in a step-by-step manner, use numbered and bulleted lists.

Do not add extensive explanations. List definitions, a required steps (+ a short description for each step)
and important considerations. But never omit important information.

Check all described information is put into the logical order, and each next piece of information
is a direct consequence of previous information.