# English words that akshar writes in Latin

The `mixed` style (D11) writes Hindi in Devanagari and English words in lowercase Latin. Training transcripts often
write English words in Devanagari (मोबाइल, स्कूल, दिल्ली); `akshar.data.convert` rewrites the ones it knows with a
lexicon. Until v3 that lexicon held 917 words learned from where Shruti's transcripts agreed with IndicVoices'
references. This folder adds a curated list of 2,900 English words with their Devanagari spellings (D32).

- `raw/*.tsv`: the category lists as eight Claude Opus 5.5 agents wrote them on 2026-11-09 (technology, everyday
  objects, institutions and services, brands and spoken abbreviations, places outside India, places in India, time
  and common Hinglish words), and `raw/blocklist.tsv`, Devanagari tokens that look like English but are Hindi words,
  homophones of several English words, or number words. Columns: `english`, `devanagari` (spellings separated by
  `|`), `category`, `note` (`AMBIGUOUS:` or `HINDI_NAME:` marks what must not convert).
- `raw/gloss-stats.json`, `raw/unglossed-counts.json`: for every Devanagari token in Vaani's and Rural Women's
  transcripts, how often the annotators glossed it as English (and with what) and left it unglossed.
- `fixes.tsv`: hand corrections after reviewing what the list would change in the training data: Hindi words that
  slipped through (बेच, शौक, लाकर) and place names mapped to the name spoken, not the official one (बनारस is banaras,
  इलाहाबाद is allahabad).
- `words.tsv`: the merged list, one spelling per line, with the gloss counts. `review.tsv`: every spelling dropped,
  with the reason. `blocklist.tsv`: the tokens the converter never rewrites.
- `lexicon-v4.json`: the converter's lexicon for akshar v4: the 917 learned words (also found under folded spellings,
  D29) plus 4,199 curated spellings (found exactly) and the blocked list.

Rules applied by `scripts/build_english_words.py`: a spelling is dropped when it is blocklisted, when the agent marked
it ambiguous, when the Hindi name is a different word from the English one (चीन stays Devanagari; दिल्ली becomes
delhi), when it is an inflected Hindi form (स्कूलों) and the base form is listed, when annotators leave it in
Devanagari in at least 85% of 50 or more occurrences, or when they gloss it as a different English word. One
Devanagari spelling maps to one English word; the learned lexicon wins a conflict.

Place names are converted at the owner's request. FLEURS and Lahaja references keep loanwords and place names in
Devanagari, so this choice can cost WER on those sets while matching what users expect to read.
