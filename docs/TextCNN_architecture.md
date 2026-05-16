# TextCNNMatcher Architecture


Input IDs
(batch_size, seq_len)
        │
        ▼
Embedding Layer
(vocab_size → 128)
Output:
(batch_size, seq_len, 128)
        │
transpose(1,2)
        ▼
(batch_size, 128, seq_len)

        ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ Parallel CNN branches                                                                                          │
        │                                                                                                                │
        │ Conv1D(kernel=3, filters=128)       Conv1D(kernel=4, filters=128)            Conv1D(kernel=5, filters=128)     |    
        │        │                                           │                                         │                 |
        │        ▼                                           ▼                                         ▼                 |
        │      ReLU                                        ReLU                                       ReLU               |
        │        │                                           │                                         │                 |
        │        ▼                                           ▼                                         ▼                 |
        │ Global MaxPool                                Global MaxPool                           Global MaxPool          |
        │        │                                           │                                         │                 |
        │        ▼                                           ▼                                         ▼                 |
        │    (128)                                          (128)                                     (128)              |
        │                                                                                                                |
        └────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
Concatenate
(128 + 128 + 128)
        │
        ▼
Feature vector (384)
        │
        ▼
Dropout(0.3)
        │
        ▼
Linear Layer
384 → 1
        │
        ▼
Output Logit
(binary matching score)