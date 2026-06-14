# Shields Gate Security Gateway — Post-Training Error Analysis Report

## 1. Operational Overview
- **Analyzed Dataset:** /home/rama/datasets/gw_final_dataset.csv
- **Total Rows Audited:** 14512
- **Isolated False Positive Profiles:** 643 samples
- **Isolated False Negative Profiles:** 3187 samples

## 2. False Positive (FP) Analysis — Legitimate Mail At Risk of Quarantine
These legitimate entities exhibit structural patterns that closely mimic social engineering markers:

| Index | Subject Text Snippet | URL Count | Received Hops | Caps Ratio |
|---|---|---|---|---|
| 425 | people with too much time on their hands.. look at... | 4 | 6 | 0.1818 |
| 444 | --- begin forwarded text to: email_token from: ema... | 9 | 6 | 0.1379 |
| 467 | owen byrne wrote: > r. a. hettinga wrote: > >> ---... | 6 | 6 | 0.1282 |
| 482 | jeff-- what is the maildrop recipe you use with sp... | 4 | 7 | 0.0698 |
| 491 | ----- original message ----- from: "james rogers"... | 4 | 6 | 0.1282 |
| 506 | url: url_token date: 2002-10-06t02:28:04+01:00 wil... | 7 | 3 | 0.0909 |
| 516 | url: url_token date: not supplied neil "sandman" g... | 4 | 3 | 0.0811 |
| 544 | hello all, firstly i'd like to thank all of you fo... | 8 | 8 | 0.1429 |
| 563 | url: url_token date: not supplied this is the head... | 4 | 3 | 0.0312 |
| 572 | i finally let go of my irix magic desktop and wind... | 4 | 6 | 0.0769 |

## 3. False Negative (FN) Analysis — Phishing Leaks (Bypass Risks)
These highly obfuscated malicious emails successfully minimized their structural footprints:

| Index | Text/Body Profile Snippet | URL Count | Suspicious TLDs | Received Hops |
|---|---|---|---|---|
| 2 | ebay --> url_token ===============================... | 131 | 0 | 2 |
| 3 | congratulations! paypal has successfully charged m... | 1 | 0 | 3 |
| 22 | congratulations! paypal has successfully charged m... | 1 | 0 | 3 |
| 25 | congratulations! paypal has successfully charged m... | 1 | 0 | 3 |
| 28 | your wamu.com account verification process vtkrqon... | 4 | 0 | 2 |
| 29 | your wamu.com account verification process eutdshk... | 4 | 0 | 2 |
| 30 | your wamu.com account verification process pczvjcp... | 4 | 0 | 2 |
| 31 | congratulations! paypal has successfully charged m... | 1 | 0 | 4 |
| 33 | congratulations! paypal has successfully charged m... | 1 | 0 | 4 |
| 34 | congratulations! paypal has successfully charged m... | 1 | 0 | 3 |
