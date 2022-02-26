# Decision Tree implemented in pure python

To understand an algorithm it is always good to implement by yourself. Here, I implemented a simplified version of **Decision Tree** algorithm purely in python. No external libraries are used except the standard Python libraries, provides by default. Let's start by a simple definition:

Decision trees makes a series of __decisions__ which recursively partition the training data. Each decision is a simple binary partition of the data, which aims to separate the different __classes__
(i.e., types of label) as well as possible. The quality of partition is usually assessed by a **purity-function** which user can define based on the problem definition.

Let's take a simple binary class classification example. 

## Training logic for a decision tree:

    - Start with the training data you have;
    - Compute all binary partitions of the data;
    - Score the partitions using purity of each partition;
    - Store the best splitting criterion and split the data accordingly;
    - Continue recursively partitioning each subset until all are pure;
    - The training of tree is completed!

## Predictions using decision tree:

    - Given a new sample (i.e., test sample) and corresponding feature vector, follow from the root of the decision tree according to the splitting criterion in each node until the leaf is reached. The label of the final leaf node is the predicted label for the test sample.  