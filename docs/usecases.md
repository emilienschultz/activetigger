# Use cases

## Jules Brion research

**Study Objective**: To build a corpus of newspapers articles related to the environment and the ecological transition. 

**Corpus**: Articles are first extracted from Europresse with a list of keywords related to the planet boundaries. The problem is that many of these words ("climate", "environment"...) are polysemic. With a traditional approach, the corpus should then be built up manually, separating relevant articles from those not related to the environment. The problem is that this severely limits the number of articles you can add to your corpus, which can be a problem when analyzing media content on issues as broad as the environment. With active Tigger, it was possible to skip the initial corpus and sort through it.

**Codebook**: Binary classification – environment / not environment.

**Goal**: Develop a classifier with high performance (F1 score > 0.95) capable of identifying environmental journalism segments.

**Recomandations"** : Active Tigger sets a limit of 512 tokens (around 1300 characters), so it's not possible to analyze an entire article. Generally speaking, it is sometimes easier to classify a 300-character text than a 1300-character text. The decision was therefore taken to keep texts of less than 500 characters intact and to cut out other texts between 300 and 700 characters, with priority given to the end of a sentence (period, question mark, etc.).

If the components of your corpus are not homogeneous (there are many more articles from one newspaper than another, for example), it can sometimes be useful to **create an additional homogeneous corpus to train the model**. In general, it's best if the training corpus reflects as closely as possible the diversity of cases that will be encountered by the classifier. For example, don't annotate only one year if you are then submitting a corpus from ten years ago. Once the model has been trained and stabilized, you can reuse your "heterogeneous corpus".

When training your model, it is a good practice to examine the cases where the model's predictions differ from your manual annotations. This can help you spot potential annotation mistakes due to inattention. However, **be careful to code according to your codebook and not according to the model's predictions to avoid overfitting**.

Once your model has been trained and stabilized, you can apply it to corpora of any size. What is particularly useful about this approach is that, should your study corpus change (you want to add a year of study or add a newspaper, for example) you can download your model (by going to Export and selecting export fine-tuned model) and re-upload it in another project (by selecting XXX). 

**Post-ActiveTigger Analysis**: Once the annotations has been extended to the entire corpus, the article paragraphs are grouped together. A score is calculated, relating the number of environment-related paragraphs to the total number of paragraphs for each article. 

Depending on the needs of your search, it is possible to go a little further in post active-tigger processing. For example, to minimize the number of false positives (paragraphs considered to be related to the environment when they are not), it is possible to remove from the calculation paragraphs where the model is not sure of its prediction (refer to the enthropy score).

In addition, we decided not to keep articles with three paragraphs and only one related to the environment (the score of 0.33 is high, but adding them runs the risk of false positives because it is only based on one classification).

**Key Takeaways / Lessons Learned**:

- Learning ActiveTigger: It took a few weeks to get comfortable with the interface, stabilize the corpus structure (e.g., paragraph length), and define the coding scheme.
- Use of Visualization: Visualization tools helped identify ambiguous categories, leading to annotation improvements.
- Model Training Insights:
- **Understanding how BERT-based models learn is essential for improving performance metrics**. Initially, models were trained without a clear grasp of how they function or which parameters influence outcomes, which lead to common pitfalls such as overfitting. Taking the time to understand the fundamentals — like the distinction between evaluation loss and validation loss — helps in effectively avoiding these issues.
