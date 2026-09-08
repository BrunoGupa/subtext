# Where the evaluation phrases come from

A test set chosen by the people who built the system proves nothing. This one is in three
groups, and the group is recorded next to every row of the results, so the question *"who
picked the phrases?"* has an answer instead of an argument.

Groups are listed in the order they appear in every results table, strongest first.

| group | n | who chose them | present in `mx_corpus`? |
|---|---|---|---|
| **paper** | 40 | **published translation scholars**, as documented cases of difficulty | 3 of 40 — flagged |
| **AFI** | 30 | the American Film Institute | 14 yes, 16 no |
| **corpus** | 35 | rendering entropy over the **full 103.6M-line** corpus | **none** — verified line by line |
| **film-sampled** | 102 | film chosen from the literature, **line chosen by us** (criterion: contains profanity) | **none** — verified line by line |

## The paper group, and why it goes first

These are the only phrases in the set that neither we nor an algorithm selected. A
translation scholar picked each one to illustrate a specific difficulty, printed the official
subtitle beside it, and named the strategy the subtitler used. The citation travels with the
row, so a judge can open the paper and check.

### Ávila-Cabrera (2015) — *Pulp Fiction* into European Spanish · 34 lines

> Ávila-Cabrera, J. J. (2015). "Subtitling Tarantino's offensive and taboo dialogue exchanges
> into European Spanish: the case of *Pulp Fiction*". **Revista de Lingüística y Lenguas
> Aplicadas** 10: 1–11. DOI [10.4995/rlyla.2015.3419](http://dx.doi.org/10.4995/rlyla.2015.3419).
> Peer-reviewed, open access, Universitat Politècnica de València.

Eight lines are the author's own illustration of each translation strategy (§2.2); nine are
full subtitle exchanges with their numbers and timecodes (Examples 1–3); seventeen are the
examples in his taxonomy of offensive and taboo language (Table 2), which the paper prints in
English only.

His measured results, which is what our own corpus measurement is compared against:

| | instances | share |
|---|---|---|
| offensive/taboo load transferred | 362 | 58.2% |
| neutralised | 88 | 14.1% |
| **omitted** | **173** | **27.7%** |
| **not transferred — total** | **261** | **41.8%** |

Omission is also the most used single strategy at **27.2%** (169 of 623), ahead of
reformulation (24.3%) and literal translation (22.8%).

⚠️ A number to be careful with: the paper's **49.1%** is the share of neutralisations and
omissions that are **not technically justified** by subtitling's space and time limits — not
the share of profanity lost. We quoted it wrongly once before catching it.

### Rohmawati (2021) — *Deadpool 1 & 2* into Indonesian · 6 lines

> Rohmawati, I. (2021). "Subtitling Strategies of Swear Words in Deadpool One & Deadpool Two
> Film". **Indonesian Journal of EFL and Linguistics** 6(1): 219–232.

The target language is Indonesian, so the paper's translations cannot serve as a Spanish
reference; the Spanish shown for these rows was recovered from the corpus instead. The lines
are still the researcher's choice, which is the point. Deletion is the **most frequent**
strategy in *Deadpool Two* — 65 of 166 instances (**39%**), against 29% in the first film.

## Why the second group is measured over the full corpus, not the Mexican one

The first attempt selected these by rendering entropy over `mx_corpus` — the Mexican slice
this system retrieves from. That is circular: it picks the lines our own corpus happens to
have a lot to say about, and then reports that our corpus has a lot to say about them. The
selection now runs over all 103,637,005 pairs, which include Spain, Argentina and neutral
dubbing, and every selected line was then checked to be **absent** from `mx_corpus`. Lines
the system cannot answer stay in the set on purpose: coverage is only meaningful if the
failures are counted too.

## Why these films, and the citation for each

Every title below is in the set because published work in audiovisual translation uses it
as a case study of difficulty — not because it felt hard to us.

| film | why it is documented as hard | source |
|---|---|---|
| **Deadpool / Deadpool 2** | R-rated Marvel comedy; the literature lists lip-sync, intertextuality, puns, made-up words, profanity and sexual innuendo together in one text. Studies find subtitlers reach for **deletion** as the dominant strategy. | Hawel & Kadhim, *Subtitling Strategies of Swear Words in Deadpool One & Deadpool Two*, Indonesian Journal of EFL and Linguistics; and the ATA 60 panel on translating Deadpool 2 into Castilian Spanish |
| **Pulp Fiction** | The strongest source in this list. A Descriptive Translation Studies analysis of the European Spanish subtitles measures **omission at 27.2%** of offensive and taboo exchanges, reformulation 24.3%, literal 22.8%. | Ávila-Cabrera, J. J. (2015), *Subtitling Tarantino's offensive and taboo dialogue exchanges into European Spanish: the case of Pulp Fiction*, **Revista de Lingüística y Lenguas Aplicadas** 10: 1–11 (peer-reviewed, Universitat Politècnica de València). Companion: *An Account of the Subtitling of Offensive and Taboo Language in Tarantino's Screenplays*, **Sendebar** 26: 37–56 |
| **A Clockwork Orange** | Nadsat, an invented Russian-influenced English argot written deliberately to be opaque. There is no attested precedent for it in any corpus, in any language. | Bosqueti, V. P., *The Translation of Swear Words in Nadsat Language in the Movie A Clockwork Orange* |
| **Trainspotting** | Scots dialect, some of it regional and some invented for the novel. | *Subtitling Scots: Translating Danny Boyle's Trainspotting and Ken Loach's The Angels' Share into French*, in **Dealing with Difference in Audiovisual Translation** (Peter Lang) |
| **The Wolf of Wall Street** | 506 instances of swearing analysed; **72.9% translated by omission**, 25.1% by mollification. | Hawel, *Strategies of Subtitling Swear Words in The Wolf of Wall Street Movie*, Lark Journal (Wasit University) |
| **The Departed** | Boston working-class slang. | *An Analysis of Slang Translation in the Subtitles of "The Departed" Movie*, Universitas Pendidikan Indonesia repository |

### The film-sampled group is the weakest, and is labelled as such

For that group the *film* comes from the literature but the *line* does not: the selection
rule is "contains a word from our profanity list", applied to lines harvested around an
anchor quote. That criterion is ours. It is kept because 102 lines with their official
Spanish are worth having, and labelled `film-sampled` so nobody mistakes it for the paper
group.

### Titles considered and rejected for lack of a source

**Snatch**, **The Big Lebowski** and **Superbad** were in a first draft on the strength of
being obviously hard. No published study was found for any of them, so they are out. They
would have been the only rows in the set resting on our own opinion.

### What the sources say that this project also measured

Three of the studies above report the same thing from different corpora and different
target languages: subtitlers **delete** profanity rather than translate it — 27.2% omission
in Pulp Fiction into Spanish, 72.9% in The Wolf of Wall Street, deletion dominant in
Deadpool. This project measured the same effect independently over 688,351 English lines
containing *fuck*: **39.6% lose all profanity marking in Spanish** (CORPUS.md §5).

That agreement matters for the design. It means frequency *inside* a subtitle corpus
reports the convention of the profession, not what speakers say — which is why renderings
are ranked by enrichment against the full corpus rather than by how often subtitlers wrote
them. `¡Jódete!` is the most frequent rendering of `Fuck you!` in the Mexican corpus and
scores **0.9×** enrichment; `¡Chinga tu madre!` is rarer there and scores **188×**.

## Links

- Ávila-Cabrera (2015), Pulp Fiction — https://polipapers.upv.es/index.php/rdlyla/article/view/3419
- Ávila-Cabrera, Tarantino / Sendebar 26 — https://www.researchgate.net/publication/289507493
- Deadpool 1 & 2 swear words — https://indonesian-efl-journal.org/index.php/ijefll/article/view/360
- ATA 60, Deadpool 2 into Castilian Spanish — https://www.ata-divisions.org/AVD/ata-60-recap-deadpool-2-translating-an-rrated-film-from-english-into-castilian-spanish/
- Nadsat swear words in A Clockwork Orange — https://www.academia.edu/42097273/
- Dealing with Difference in Audiovisual Translation (Peter Lang) — https://www.peterlang.com/document/1053365
- The Wolf of Wall Street swear words — https://lark.uowasit.edu.iq/index.php/lark/en/article/view/1101
- The Departed slang subtitles — https://oalib-perpustakaan.upi.edu/Record/repoupi_98098/Description
