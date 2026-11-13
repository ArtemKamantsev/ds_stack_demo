---
name: theory-conspect
description: Create a conspect of the requested information.
---

# Target file

Use .ipynb files.

If a user requested to append to the existing conspect – do it.
If a user requested creation of a new file with the provided name and location – do it.
Otherwise - ask user to provide a name and location for the new conspect.

# Reference file

If the user supplies an existing notebook as a style or depth reference,
use it to guide organization and level of detail.

# File structure

Use Markdown cells to create a conspect of the requested information.

Split information to paragraphs, use headings.
Create a new cell for each top-level paragraph.

Keep related material in readable sections, with separate cells where they help navigation.
Avoid putting a long document into one cell and avoid leaving empty cells.
Keep code examples when they explain a concept and label whether they are illustrative or directly runnable.

If any important media is missing from the context, describe the missing content and ask the user to provide it.

Check notebook JSON and cell structure.
Run code cells only when they are intended to be executable and have the required inputs;

# Style

Follow MathJax (subset of TeX/LaTeX) syntax.

Do not use bold or italic formatting, event for formulas.

Start each equation form a new line with alignment to the center.

# Information selection guidance

Try to structure information in a step-by-step manner, use numbered and bulleted lists.

Retain important derivations, equations, examples, code snippets, assumptions,
and coordinate conventions from the supplied material.
If important information requires a significant amount of text to explain – write it;
correctness and completeness is a first priority.

List definitions, required steps (+ a short description for each step)
and important considerations.

Avoid extensive explanations. But never omit important information.

Check all described information is put into the logical order, and each next piece of information
is a direct consequence of previous information.