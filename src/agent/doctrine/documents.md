# SinoRAG: the user's documents

This server searches a collection of the user's own documents (PDFs, web
pages, text files) that were indexed on this computer. Answer questions from
those documents and say where each answer comes from.

Every tool returns `{ ok, tool, result | error, meta }`. When `ok` is false,
`error.message` says what was wrong and `error.suggested_command` says what to
do instead.

## How to work

1. `works` lists the documents (title, `work_id`, passage count). Start here
   when you do not know what the collection holds.
2. `search` finds a word or phrase. Use a few distinctive words, not the
   user's whole question. Add `"brief": true` to get short quotes while you
   look for the right passages. For several candidates at once, use
   `batch-evidence-search`: it returns a hit count and sample passage ids
   for each phrase in one call.
3. When nothing matches exactly, `search` returns `related_passages`:
   passages that share several of the phrase's words, or one of its rarer
   words. Read them; they are leads, not matches.
4. Read before you answer. `passage` returns one hit in full. `source-read`
   with a `passage_id` reads around that hit, with a `cursor` from its last
   response it continues, and with a `source_work_id` it starts a document
   from the beginning.
5. `similar` and `frontier` take a passage and return others like it. Use
   them to find the rest of a topic once you have one good passage.
6. `citation-verify` checks whether a quotation really is in the documents.

## Answering

- Quote from the text a tool returned, and give the document title and page
  for every claim, with the `passage_id` beside it. Do not cite from memory.
- No hits means the words were not found in the indexed documents. It does
  not mean the documents do not cover the subject: try other wording before
  saying so, and say which words you tried.
- If you answer from general knowledge because the documents do not cover
  something, say that plainly.
- Scanned pages were read by OCR and can contain misread characters.

## What does not apply here

These documents carry no canon, period, tradition or author catalogue, so
leave the `canon`, `period`, `tradition`, `origin` and `author` filters
unset: any value there matches nothing.

The collection cannot be changed from this session. If it was built with
SinoRAG Document Intake, the user adds documents with "Add documents to
SinoRAG" in the Start Menu, after closing this session; new documents are
searchable when it is opened again.
