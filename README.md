# Machine-Learning-Notes

Machine learning is about designing algorithms that automatically extract valuable information from data. The emphasis here is on “automatic”, i.e., machine learning is concerned about general-purpose methodologies that can be applied to many datasets, while producing something that is meaningful. There are three concepts that are at the core of machine learning: data, a model, and learning.

Since machine learning is inherently data driven, data is at the core of machine learning. The goal of machine learning is to design general purpose methodologies to extract valuable patterns from data, ideally without much domain-specific expertise.

To paraphrase Mitchell (1997):<mark> A model is said to learn from data if its performance on a given task improves after the data is taken into account. The goal is to find good models that generalize well to yet unseen data, which we may care about in the future. Learning can be understood as a way to automatically find patterns and structure in data by optimizing the parameters of the model. </mark>

A challenge we face regularly in machine learning is that concepts and words are slippery, and a particular component of the machine learning system can be abstracted to different mathematical concepts. For example, the word “algorithm” is used in at least two different senses in the context of machine learning. In the first sense, we use the phrase “machine learning algorithm” to mean a system that makes predictions based on input data. We refer to these algorithms as predictors. In the second sense, we use the exact same phrase “machine learning algorithm” to mean a system that adapts some internal parameters of the predictor so that it performs well on future unseen input data. Here we refer to this adaptation as training a system.

> We now come to the crux of the matter, the learning component of machine learning. Assume we are given a dataset and a suitable model. Training the model means to use the data available to optimize some parameters of the model with respect to a utility function that evaluates how well the model predicts the training data. Most training methods can be thought of as an approach analogous to climbing a hill to reach its peak. In this analogy, the peak of the hill corresponds to a maximum of some desired performance measure.

> Performing well on data that we have already seen (training data) may only mean that we found a good way to memorize the data. However, this may not generalize well to unseen data, and, in practical applications, we often need to expose our machine learning system to situations that it has not encountered before.

### Neural Networks

1. [Neural Networks / Deep Learning by StatQuest with Josh Starmer](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)

### Other Resources

1. [YouTube Recommendation System](https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/)
2. [Mining Massive Datasets - Data Mining](https://www.youtube.com/playlist?list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV)
3. [NLP - University of Michigan](https://www.youtube.com/playlist?list=PLLssT5z_DsK8BdawOVCCaTCO99Ya58ryR)
4. [Anomaly Detection - scikit-learn ](https://scikit-learn.org/stable/modules/outlier_detection.html)

### Documentary

1. [AlphaGO - Google Deepmind](https://www.youtube.com/watch?v=WXuK6gekU1Y&ab_channel=GoogleDeepMind)
2. [The bit player - Claude Shannon](https://youtu.be/CCrpgUM_rYc)
