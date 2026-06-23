# v2 head-to-head — cmx vs LCM (LOCOMO) — fg=gpt-5-mini reasoning=low window=16000

### LCM (lossy compression)
- answerable accuracy:     58.3%  (7/12)
- refusal on adversarial:  83.3%  (5/6)
- hallucination rate:      10.0%  (1/10 shipped)
  _(elapsed 143s)_

### cmx (retrieval + enforcement)
- answerable accuracy:     66.7%  (8/12)
- refusal on adversarial: 100.0%  (6/6)
- hallucination rate:       0.0%  (0/11 shipped)
  _(elapsed 322s)_
