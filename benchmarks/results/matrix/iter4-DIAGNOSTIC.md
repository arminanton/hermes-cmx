# iter-4 DIAGNOSTIC — foreground=claude-opus-4.8, PINNED window=8000 tokens, judge=claude-opus-4.7
_If accuracy holds with a window too small for the conversation, the DB (not the model window) holds memory → context-size-match problem solved._

- seed 0 (419 turns ingested): single-hop 9/12 (75%), multi-hop 2/8 (25%)
- seed 1 (369 turns ingested): single-hop 11/13 (85%), multi-hop 4/7 (57%)
- seed 2 (663 turns ingested): single-hop 6/9 (67%), multi-hop 3/11 (27%)

## Pooled

| hop type | n | correct | refused | wrong | accuracy |
|---|---:|---:|---:|---:|---:|
| single-hop | 34 | 26 | 6 | 2 | 76.5% |
| multi-hop | 26 | 9 | 11 | 6 | 34.6% |

_Conversation turns ingested (vs 8000-token window): 369–663. The window holds only a few turns; the rest lives in SQLite and is retrieved on demand._
