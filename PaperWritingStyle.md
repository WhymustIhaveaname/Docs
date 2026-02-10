# Paper Writing Style Guide

## 1. Punctuation
- **Restricted Punctuation**: Do not use em-dashes (—). Avoid colons (:) when possible.

## 2. Terminology Consistency
- **Consistent terminology**: Use the same term throughout the paper to refer to the same concept. Avoid synonyms that may confuse readers.
    - Example: If "semantic gap" is introduced in the Introduction, do not switch to "vocabulary gap" in the Methodology.

## 3. Abbreviations and Acronyms
- **First Mention**: For all other terms, define acronyms and abbreviations upon their first occurrence in the text. Use the format: "Full Term (Abbreviation)".
    - Example: "The Coupled Model Intercomparison Project Phase 6 (CMIP6) is..."
- **Subsequent Mentions**: Use only the abbreviation for all subsequent references throughout the text.
    - Example: "...data from CMIP6 indicates..."

## 4. Scope and Vision
- **Vision over Implementation**: When describing system architecture and methodology, focus on the broader design vision and intended capabilities rather than being limited by the current stage of implementation.
    - **Future-Proofing**: Describe features (e.g., data source integrations, reranking modules) as part of the proposed system if they are planned, even if currently in development.
    - **Generalization**: Use generalized terms for data sources or components (e.g., "integrated metadata repositories") rather than restricting descriptions to specific, currently implemented instances (e.g., "NASA CMR"), unless specifically discussing experimental constraints.

## 5. Language and Tone
- **Rigorous academic tone**: Use cautious, evidence-backed academic language. Avoid colloquialisms and overconfident or absolute claims. Prefer calibrated phrasing such as "suggest", "indicate", "consistent with", and quantify statements when possible.

## 6. Narrative Flow
- **Tell a research story**: Write the paper as a coherent narrative with motivation, hypothesis or reasoning, experiments, key turning points, and reflection on limitations or failure cases. Each section should answer "why this matters" and "what we learned" before moving on.

## 7. LaTeX Formatting
- **One sentence per line**: In LaTeX source, write one sentence per line to make diffs and git history easier to track.

## 8. Readability
- **Readable writing**: Prefer common, precise words and short sentences.
    - Avoid rare or ornate vocabulary unless it is uniquely accurate in context.
    - Avoid overly complex sentences.

---

## 9. Avoiding AI-Generated Writing Patterns

Based on Wikipedia's "Signs of AI writing" guide. The following patterns should be identified and avoided to make writing sound more natural and human.

### 9.1 Content Patterns

#### 9.1.1 Avoid Significance Inflation

**Words to avoid:** stands/serves as, is a testament/reminder, a vital/significant/crucial/pivotal/key role/moment, underscores/highlights its importance/significance, reflects broader, symbolizing its ongoing/enduring/lasting, contributing to the, setting the stage for, marking/shaping the, represents/marks a shift, key turning point, evolving landscape, focal point, indelible mark, deeply rooted

**Problem:** Inflating importance by adding statements about how aspects represent or contribute to a broader topic.

| Before | After |
|--------|-------|
| "marking a pivotal moment in the evolution of..." | "was established in 1989 to collect regional statistics" |
| "The Statistical Institute of Catalonia was officially established in 1989, marking a pivotal moment in the evolution of regional statistics in Spain." | "The Statistical Institute of Catalonia was established in 1989 to collect and publish regional statistics independently from Spain's national statistics office." |

#### 9.1.2 Avoid Notability Name-Dropping

**Words to avoid:** independent coverage, local/regional/national media outlets, written by a leading expert, active social media presence

**Problem:** Hitting readers over the head with claims of notability, often listing sources without context.

| Before | After |
|--------|-------|
| "cited in NYT, BBC, FT, and The Hindu" | "In a 2024 NYT interview, she argued..." |
| "Her views have been cited in The New York Times, BBC, Financial Times, and The Hindu." | "In a 2024 New York Times interview, she argued that AI regulation should focus on outcomes rather than methods." |

#### 9.1.3 Avoid Superficial -ing Analyses

**Words to avoid:** highlighting/underscoring/emphasizing..., ensuring..., reflecting/symbolizing..., contributing to..., cultivating/fostering..., encompassing..., showcasing...

**Problem:** Tacking present participle ("-ing") phrases onto sentences to add fake depth.

| Before | After |
|--------|-------|
| "symbolizing... reflecting... showcasing..." | Remove or expand with actual sources |
| "The temple's color palette of blue, green, and gold resonates with the region's natural beauty, symbolizing Texas bluebonnets..." | "The temple uses blue, green, and gold colors. The architect said these were chosen to reference local bluebonnets and the Gulf coast." |

#### 9.1.4 Avoid Promotional Language

**Words to avoid:** boasts a, vibrant, rich (figurative), profound, enhancing its, showcasing, exemplifies, commitment to, natural beauty, nestled, in the heart of, groundbreaking (figurative), renowned, breathtaking, must-visit, stunning

**Problem:** Failing to keep a neutral tone.

| Before | After |
|--------|-------|
| "nestled within the breathtaking region" | "is a town in the Gonder region" |
| "Nestled within the breathtaking region of Gonder in Ethiopia, Alamata Raya Kobo stands as a vibrant town..." | "Alamata Raya Kobo is a town in the Gonder region of Ethiopia, known for its weekly market and 18th-century church." |

#### 9.1.5 Avoid Vague Attributions and Weasel Words

**Words to avoid:** Industry reports, Observers have cited, Experts argue, Some critics argue, several sources/publications (when few cited)

**Problem:** Attributing opinions to vague authorities without specific sources.

| Before | After |
|--------|-------|
| "Experts believe it plays a crucial role" | "according to a 2019 survey by..." |
| "Due to its unique characteristics, the Haolai River is of interest to researchers and conservationists. Experts believe it plays a crucial role in the regional ecosystem." | "The Haolai River supports several endemic fish species, according to a 2019 survey by the Chinese Academy of Sciences." |

#### 9.1.6 Avoid Formulaic Challenges Sections

**Words to avoid:** Despite its... faces several challenges..., Despite these challenges, Challenges and Legacy, Future Outlook

**Problem:** Including formulaic "Challenges" sections without specific facts.

| Before | After |
|--------|-------|
| "Despite challenges... continues to thrive" | Specific facts about actual challenges |
| "Despite its industrial prosperity, Korattur faces challenges typical of urban areas... Despite these challenges... Korattur continues to thrive..." | "Traffic congestion increased after 2015 when three new IT parks opened. The municipal corporation began a stormwater drainage project in 2022 to address recurring floods." |

---

### 9.2 Language and Grammar Patterns

#### 9.2.1 Avoid Overused "AI Vocabulary" Words

**High-frequency AI words to avoid:** Additionally, align with, crucial, delve, emphasizing, enduring, enhance, fostering, garner, highlight (verb), interplay, intricate/intricacies, key (adjective), landscape (abstract noun), pivotal, showcase, tapestry (abstract noun), testament, underscore (verb), valuable, vibrant

**Problem:** These words appear far more frequently in AI-generated text. They often co-occur.

| Before | After |
|--------|-------|
| "Additionally... testament... landscape... showcasing" | "also... remain common" |
| "Additionally, a distinctive feature of Somali cuisine is the incorporation of camel meat. An enduring testament to Italian colonial influence is the widespread adoption of pasta in the local culinary landscape, showcasing how these dishes have integrated..." | "Somali cuisine also includes camel meat, which is considered a delicacy. Pasta dishes, introduced during Italian colonization, remain common, especially in the south." |

#### 9.2.2 Use Simple Copulas (is/are)

**Words to avoid:** serves as/stands as/marks/represents [a], boasts/features/offers [a]

**Problem:** Substituting elaborate constructions for simple "is" or "are".

| Before | After |
|--------|-------|
| "serves as... features... boasts" | "is... has" |
| "Gallery 825 serves as LAAA's exhibition space for contemporary art. The gallery features four separate spaces and boasts over 3,000 square feet." | "Gallery 825 is LAAA's exhibition space for contemporary art. The gallery has four rooms totaling 3,000 square feet." |

#### 9.2.3 Avoid Negative Parallelisms

**Problem:** Overusing constructions like "Not only...but..." or "It's not just about..., it's..."

| Before | After |
|--------|-------|
| "It's not just X, it's Y" | State the point directly |
| "It's not just about the beat riding under the vocals; it's part of the aggression and atmosphere. It's not merely a song, it's a statement." | "The heavy beat adds to the aggressive tone." |

#### 9.2.4 Avoid Rule of Three Overuse

**Problem:** Forcing ideas into groups of three to appear comprehensive.

| Before | After |
|--------|-------|
| "innovation, inspiration, and insights" | Use natural number of items |
| "The event features keynote sessions, panel discussions, and networking opportunities. Attendees can expect innovation, inspiration, and industry insights." | "The event includes talks and panels. There's also time for informal networking between sessions." |

#### 9.2.5 Avoid Synonym Cycling (Elegant Variation)

**Problem:** Excessive synonym substitution instead of clear repetition.

| Before | After |
|--------|-------|
| "protagonist... main character... central figure... hero" | "protagonist" (repeat when clearest) |
| "The protagonist faces many challenges. The main character must overcome obstacles. The central figure eventually triumphs. The hero returns home." | "The protagonist faces many challenges but eventually triumphs and returns home." |

#### 9.2.6 Avoid False Ranges

**Problem:** Using "from X to Y" constructions where X and Y aren't on a meaningful scale.

| Before | After |
|--------|-------|
| "from the Big Bang to dark matter" | List topics directly |
| "Our journey through the universe has taken us from the singularity of the Big Bang to the grand cosmic web, from the birth and death of stars to the enigmatic dance of dark matter." | "The book covers the Big Bang, star formation, and current theories about dark matter." |

---

### 9.3 Style Patterns

#### 9.3.1 Avoid Em Dash Overuse

**Problem:** Using em dashes (—) more than necessary, mimicking "punchy" sales writing.

| Before | After |
|--------|-------|
| "institutions—not the people—yet this continues—" | Use commas or periods |

#### 9.3.2 Avoid Boldface Overuse

**Problem:** Emphasizing phrases in boldface mechanically.

| Before | After |
|--------|-------|
| "**OKRs**, **KPIs**, **BMC**" | "OKRs, KPIs, BMC" |

#### 9.3.3 Avoid Inline-Header Lists

**Problem:** Lists where items start with bolded headers followed by colons.

| Before | After |
|--------|-------|
| "**Performance:** Performance improved" | Convert to prose |
| "- **User Experience:** The user experience has been significantly improved with a new interface." | "The update improves the interface, speeds up load times through optimized algorithms, and adds end-to-end encryption." |

#### 9.3.4 Use Sentence Case in Headings

| Before | After |
|--------|-------|
| "Strategic Negotiations And Partnerships" | "Strategic negotiations and partnerships" |

#### 9.3.5 Avoid Emojis

| Before | After |
|--------|-------|
| "🚀 Launch Phase: 💡 Key Insight:" | Remove emojis |

#### 9.3.6 Use Straight Quotation Marks

**Problem:** Using curly quotes ("...") instead of straight quotes ("...").

| Before | After |
|--------|-------|
| said "the project" | said "the project" |

---

### 9.4 Communication Patterns

#### 9.4.1 Remove Chatbot Artifacts

**Words to avoid:** I hope this helps, Of course!, Certainly!, You're absolutely right!, Would you like..., let me know, here is a...

| Before | After |
|--------|-------|
| "I hope this helps! Let me know if..." | Remove entirely |

#### 9.4.2 Remove Knowledge-Cutoff Disclaimers

**Words to avoid:** as of [date], Up to my last training update, While specific details are limited/scarce..., based on available information...

| Before | After |
|--------|-------|
| "While details are limited in available sources..." | Find sources or remove |

#### 9.4.3 Avoid Sycophantic Tone

| Before | After |
|--------|-------|
| "Great question! You're absolutely right!" | Respond directly |
| "Great question! You're absolutely right that this is a complex topic." | "The economic factors you mentioned are relevant here." |

---

### 9.5 Filler and Hedging

#### 9.5.1 Remove Filler Phrases

| Before | After |
|--------|-------|
| "In order to" | "To" |
| "Due to the fact that" | "Because" |
| "At this point in time" | "Now" |
| "In the event that you need help" | "If you need help" |
| "The system has the ability to process" | "The system can process" |
| "It is important to note that the data shows" | "The data shows" |

#### 9.5.2 Avoid Excessive Hedging

| Before | After |
|--------|-------|
| "could potentially possibly" | "may" |
| "It could potentially possibly be argued that the policy might have some effect on outcomes." | "The policy may affect outcomes." |

#### 9.5.3 Avoid Generic Conclusions

| Before | After |
|--------|-------|
| "The future looks bright" | Specific plans or facts |
| "The future looks bright for the company. Exciting times lie ahead as they continue their journey toward excellence." | "The company plans to open two more locations next year." |

---

### 9.6 Adding Voice and Personality

Avoiding AI patterns is only half the job. Sterile, voiceless writing is just as problematic.

#### Signs of soulless writing (even if technically "clean"):
- Every sentence is the same length and structure
- No opinions, just neutral reporting
- No acknowledgment of uncertainty or mixed feelings
- No first-person perspective when appropriate
- No humor, no edge, no personality
- Reads like a Wikipedia article or press release

#### How to add voice:

1. **Have opinions.** Don't just report facts - react to them when appropriate.
2. **Vary your rhythm.** Short punchy sentences. Then longer ones that take their time. Mix it up.
3. **Acknowledge complexity.** Real humans have mixed feelings. "This is impressive but also concerning" beats "This is impressive."
4. **Use "I" when it fits.** First person isn't unprofessional in appropriate contexts.
5. **Be specific about observations.** Not "this is concerning" but describe what specifically concerns you.

---

## Reference

The AI writing patterns section (9) is based on [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), maintained by WikiProject AI Cleanup.

Key insight: "LLMs use statistical algorithms to guess what should come next. The result tends toward the most statistically likely result that applies to the widest variety of cases."
