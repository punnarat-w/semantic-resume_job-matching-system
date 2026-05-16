# RuoZhengu's Model architecture
Reference: https://github.com/ruozhengu/job-resume-matching-algo/blob/master/cnn/CNN.py

Input IDs
(batch_size, seq_len)
        │
        ▼
Embedding Layer
(vocab_size → 100)

Output:
(batch_size, seq_len, 100)
        │
transpose(1,2)
        ▼
(batch_size, 100, seq_len)
        │
        ▼
Conv1D
in_channels=100
out_channels=1000
kernel_size=5
stride=1
        │
        ▼
tanh
        │
        ▼
Global Max Pooling
(pool over sequence dimension)
        │
        ▼
Feature vector
(batch_size,1000)
        │
        ▼
Dense Layer
1000 → 1000
        │
        ▼
tanh
        │
        ▼
Dropout(0.3)
        │
        ▼
Projection Layer
1000 → 100
        │
        ▼
ReLU
        │
        ▼
Classifier
100 → 1
        │
        ▼
Binary Match Score