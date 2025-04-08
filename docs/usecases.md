# Use cases

## Jules Brion research

**Study Objective**: To build a corpus of articles related to the environment.

**Corpus**: Articles are segmented into paragraphs to fit within the model's context window.

**Codebook**: Binary classification – environment / not environment.

**Post-ActiveTigger Analysis**: Aggregate the annotated paragraphs to determine whether each article is about the environment or not.

**Goal**: Develop a classifier with high performance (F1 score > 0.95) capable of identifying environmental journalism segments.

**Key Takeaways / Lessons Learned**:

- Learning ActiveTigger: It took a few weeks to get comfortable with the interface, stabilize the corpus structure (e.g., paragraph length), and define the coding scheme.
- Use of Visualization: Visualization tools helped identify ambiguous categories, leading to annotation improvements.
- Model Training Insights:
- Understanding how BERT-based models learn was crucial to improving performance metrics.

