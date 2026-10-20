"""Topic map: UMAP, clustering adapters, BERTopic wrapper, and topic assignment.

The topic map of one configuration and one seed (T16, D30) is built in steps that the
experiment (T19) can call apart, so that one UMAP fit per (model, seed) serves every
clustering:

- :mod:`.reduction`: the unsupervised UMAP reduction that the semantic runs cluster, and the
  separate 2-D projection of the visualization map;
- :mod:`.clustering`: K-means and HDBSCAN labels, with every seed passed explicitly;
- :mod:`.ctfidf`: each topic's top c-TF-IDF terms over T15's vocabulary, from any labels,
  including the TF-IDF baseline's (D28 C4);
- :mod:`.bertopic_ctfidf`: the adapter that computes the c-TF-IDF with BERTopic in its manual
  mode, built with ``language="spanish"`` (D22);
- :mod:`.build`: the :class:`~.build.TopicMap` that gathers labels, topics, centroids and
  coordinates.

Shared conventions: noise is :data:`~thematic_redundancy.evaluation.clustering_metrics.NOISE_LABEL`
(-1), as HDBSCAN gives it, and is never relabelled; topics are listed by ascending label; arrays
in the results are read-only; nothing here reads or writes a file or prints a thesis text. An
invalid input raises :class:`TopicMapError`.
"""


class TopicMapError(ValueError):
    """An input or a library result breaks a rule of the topic map, so nothing was built."""
