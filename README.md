# MSAC512 Assignment 1 — Combined Documentation

This file combines the two standalone Markdown documents produced so
far: the scraper README (how to run the pipeline) and the Section 2.3
data-collection-and-ethics memo. Once you've run `case_study.py`
against your real data, it will generate `actuarial_memo.md` as a
third file — re-run this combine step (or just append it) to fold that
in too.

---

# property.co.zw scraper — MSAC512 Assignment 1, Section A

## Before you run a full scrape

This was built without live network access to property.co.zw, so three
things **must** be checked by you before trusting a full run:

1. **Suburb slugs.** `TARGET_SUBURBS` in the script is my best guess at
   the URL slugs property.co.zw uses (e.g. `harare-west/madokero`).
   Run:
   ```bash
   python scrape_property_co_zw.py --suburb-limit 3 --max-listings 5
   ```
   and watch the log for `"could not reach any search URL"` warnings —
   that means a slug is wrong. Open the suburb in your own browser,
   copy the real URL path, and fix the tuple in `TARGET_SUBURBS`.
   Do this for **Madokero** and **Mabvazuva** specifically since the
   brief requires them by name.

2. **Selector drift.** Scrapers break when a site's HTML changes. The
   parsing functions (`parse_listing`, `extract_listing_links`,
   `find_next_page_url`) use text patterns and heading lookups rather
   than brittle CSS classes where possible, but you should still spot
   check ~10 parsed rows against the live listing pages by eye before
   scraping all 300+. If a field is consistently empty, open one
   listing page's HTML (`view-source:` in your browser, or
   `requests.get(url).text`) and adjust the relevant regex/selector.

3. **robots.txt.** The script fetches and parses `robots.txt` itself
   at the start of every run and will refuse to crawl any Disallow'd
   path. Run `--check-robots-only` first and paste that output into
   your Section 2.3 memo, along with your own commentary — don't just
   paste my code comments as if they were your analysis.

## Running it

```bash
pip install -r requirements.txt
python scrape_property_co_zw.py --check-robots-only          # step 1
python scrape_property_co_zw.py --suburb-limit 3 --max-listings 10   # step 2: smoke test
python scrape_property_co_zw.py --max-listings 320 --download-images  # step 3: full run
```

It's safe to Ctrl-C and re-run — it skips refs already saved in
`property_data/raw_listings.jsonl`.

## Outputs

- `property_data/raw_listings.csv` / `.jsonl` — one row per listing
- `property_data/images/<REF>/` — full-size photos, if `--download-images`
- `property_data/scrape.log` printed to console (redirect with `> log.txt` if you want a file)

## What's still your job (not this script's)

- Verifying slugs/selectors as above
- USD vs ZiG reconciliation with a stated FX source+date (2.4) — the
  scraper flags currency only when the page text says so explicitly;
  most prices display in USD by default and you'll need to decide how
  to handle that assumption
- Outlier/error detection, de-duplication of re-listed properties,
  and your missing-data strategy (2.4)
- The 1–2 page data-collection-and-ethics memo (2.3) — use the
  robots.txt output and the rate-limit numbers above as evidence, but
  write the professional-conduct reasoning yourself
-e 

---

# Data Collection and Ethics Memo

**MSAC512 Assignment 1 — Section 2.3**
**Prepared by:** Panashe [Surname] | ARN 9761596
**Data source:** www.property.co.zw
**Scrape window:** [DATE RANGE — fill in once you run the scraper]

> **How to use this file:** every `[FILL IN: ...]` bracket needs a real
> number or observation from *your* actual scrape run — I can't produce
> those because I never executed the scraper against the live site. The
> surrounding prose is a defensible starting structure, not a final
> submission; read it, adjust anything that doesn't match your actual
> experience, and cut it down to 1–2 pages as the brief asks (this
> draft runs longer so you have material to trim from, not pad).

---

## 1. Robots.txt and terms-of-use review

Before any crawling began, `scrape_property_co_zw.py` fetched and
parsed `https://www.property.co.zw/robots.txt` programmatically via
`urllib.robotparser`, and every subsequent request was checked against
it before being made (see `PoliteSession.allowed()` in the scraper).
Running `python scrape_property_co_zw.py --check-robots-only` on
[DATE] produced the following:

```
[FILL IN: paste the ALLOWED/DISALLOWED output here]
```

**Paths avoided as a result:** [FILL IN — e.g. "None of our target
paths (`/houses-for-sale/...`, `/for-sale/houses-*`) were disallowed;
we did avoid crawling `/admin/`, `/api/`, and account/login-related
paths, which were Disallow'd and which we had no reason to visit
anyway."]

We also reviewed the site's Terms and Conditions
(`property.co.zw/info/privacy-terms`) before scraping. [FILL IN: state
what you found — e.g. whether the terms mention automated access,
scraping, or data reuse restrictions; if the terms are silent on
scraping specifically, say so explicitly rather than treating silence
as permission — that's a judgement call worth naming out loud, not
glossing over.] Our position, consistent with IFoA/actuarial
professional conduct expectations, is that the absence of an explicit
prohibition is not by itself sufficient ethical justification — see
Section 4 below.

## 2. Rate limiting

The scraper enforced a single-threaded request rate of **[FILL IN —
default in the script is 3.0s base delay ± up to 1.5s jitter; state
whatever you actually ran with, via `--delay`]** between every HTTP
request — search pages, listing pages, and images alike — with
exponential backoff on any 429/503 response.

**Justification:** at this rate, collecting [FILL IN: your actual
listing count] listings plus their associated search-index pages
required approximately [FILL IN: total request count from
`scrape.log`] requests over roughly [FILL IN: wall-clock time], i.e.
comparable to or slower than a human browsing the same pages by hand.
This was a deliberate choice to avoid adding meaningful incremental
load to the site relative to its normal traffic pattern, not merely a
default left unexamined — a course-work scraper hitting a small
regional property platform at high concurrency risks degrading service
for genuine home-buyers using the site, which is a real harm even
though no law is being broken.

## 3. What we collected and how it maps to consent

We collected only fields that are: (a) already publicly displayed on
listing pages without authentication, and (b) directly relevant to the
stated research question (residential price prediction). We did **not**
attempt to access anything behind the site's login wall (e.g. saved
searches, the "contact seller" messaging flow), did not scrape agent or
buyer personal contact submissions, and captured agency phone numbers
only because they are the agency's own advertised business contact
detail, published specifically to be contacted by prospective buyers —
the same information a manual buyer would note down.

Total unique listings collected: **[FILL IN]**, across **[FILL IN: N]**
suburbs spanning **[FILL IN: N]** areas of Greater Harare, including
Madokero (Harare West) and Mabvazuva (Ruwa) as required.

## 4. If this were a real commercial engagement

This is a course-work exercise under fair-use/research norms, and even
so we chose to rate-limit and identify our scraper honestly (see the
`User-Agent` string in the scraper, which names the coursework and a
contact email — a bot that hides who is running it is a different,
less defensible thing than one that discloses it). If a bank's
mortgage risk unit or an insurer actually wanted to run an AVM like
this in production on top of property.co.zw's data, several things
would need to change before we'd consider it defensible:

- **Licensing.** We would approach property.co.zw directly to
  negotiate a data licence or API access rather than continuing to
  scrape at any scale — a commercial AVM built on unlicensed reuse of
  another firm's listing data (which represents its agents' and
  sellers' commercial effort) is a materially different proposition
  from a one-off academic sample, both commercially and reputationally.
- **Take-down requests.** We would need a documented process for
  removing a specific property's data from our derived dataset/model
  on request (e.g. an agency asking that a since-withdrawn or
  mis-priced listing be excluded), and for periodically refreshing
  data so the model isn't trained on stale, no-longer-accurate listings.
- **Republication restrictions.** Our outputs (predicted values,
  aggregated suburb statistics) would need review for whether they
  effectively republish property.co.zw's proprietary content (e.g. a
  suburb price-per-m² table derived almost entirely from their
  listings, presented without attribution or added value, sits closer
  to republication than to independent analysis).
- **Ongoing rate/load governance**, not just a one-off polite scrape —
  a production AVM refreshing its training data regularly needs a
  standing agreement about acceptable request volume, not an ad-hoc
  judgement call made once for a coursework deadline.
- **Personal data handling.** Agency names/phone numbers are business
  contact details, not personal data of buyers, but if the AVM were
  ever extended to ingest actual mortgage applicant or policyholder
  data (as the fictional bank/insurer clients in the brief would
  presumably want, eventually), that introduces POTRAZ Data Protection
  Act obligations that this coursework dataset does not touch.

"The data was technically accessible" is not, on its own, a sufficient
answer to any of the above — accessibility is a technical fact,
licence and consent are separate questions, and an actuary signing off
on a production AVM would be expected to be able to answer all of them
to a Board and a regulator, not just the first one.

## 5. Currency reconciliation note

Listings priced in ZiG were converted to USD using [FILL IN: your
`--fx-rate`], the [FILL IN: official interbank / other] rate as of
[FILL IN: `--fx-date`], sourced from [FILL IN: `--fx-source`, e.g.
rbz.co.zw]. We used the official rate rather than the parallel/"street"
rate because [FILL IN: your reasoning — e.g. property transactions of
this size are more likely to be settled at or near the formal rate
than small cash transactions are, though note this is an assumption
worth flagging, not a certainty]. [FILL IN: N] listings had no
explicit currency marker in the scraped text and were assumed USD by
default (property.co.zw's default page rendering) — see the
`fx_assumption` column in the cleaned dataset for exactly which rows
this affects.

---

*Word count: [FILL IN] (target: 1–2 pages once trimmed and your
placeholders filled in).*
