Download Link: https://assignmentchef.com/product/solved-ee219-project-2-clustering
<br>
<h2></h2>

Clustering algorithms are unsupervised methods for finding groups of data points that have similar representations in a proper space. Clustering differs from classification in that no <em>a priori </em>labeling (grouping) of the data points is available.

<strong>K-means clustering </strong>is a simple and popular clustering algorithm. Given a set of data points {<strong>x</strong><sub>1</sub><em>,…,</em><strong>x</strong><em><sub>N</sub></em>} in multidimensional space, it tries to find <em>K </em>clusters s.t. each data point belongs to one and only one cluster, and the sum of the squares of the distances between each data point and the center of the cluster it belongs to is minimized. If we define <em>µ<sub>k </sub></em>to be the “center” of the <em>k</em>th cluster, and

(

1<em>,     </em>if <strong>x</strong><em><sub>n </sub></em>is assigned to cluster <em>k</em>

<em>r<sub>nk </sub></em>=                                                              <em>,        n </em>= 1<em>,…,N           k </em>= 1<em>,…,K</em>

0<em>,    </em>otherwise

Then our goal is to find <em>r<sub>nk</sub></em>’s and <em>µ<sub>k</sub></em>’s that minimize. The

approach of K-means algorithm is to repeatedly perform the following two steps until convergence:

<ol>

 <li>(Re)assign each data point to the cluster whose center is nearest to the data point.</li>

 <li>(Re)calculate the position of the centers of the clusters: setting the center of the cluster to the mean of the data points that are currently within the cluster.</li>

</ol>

The center positions may be initialized randomly.

In this project, the goal includes:

<ol>

 <li>To find proper representations of the data, s.t. the clustering is efficient and gives out reasonable results.</li>

 <li>To perform K-means clustering on the dataset, and evaluate the performance of the clustering.</li>

 <li>To try different preprocess methods which may increase the performance of the clustering.</li>

</ol>

<h2>Dataset</h2>

We work with “20 Newsgroups” dataset that we already explored in <strong>Project 1</strong>. It is a collection of approximately 20,000 documents, partitioned (nearly) evenly across 20 different newsgroups, each corresponding to a different topic. Each topic can be viewed as a “class”.

In order to define the clustering task, we pretend as if the class labels are not available and aim to find groupings of the documents, where documents in each group are more similar to each other than to those in other groups. These groups, or clusters, capture the dependencies among the documents that are known through class labels. We then use class labels as the ground truth to evaluate the performance of the clustering task.

To get started with a simple clustering task, we work with a well separable portion of data that we used in Project 1, and see if we can retrieve the known classes. Namely, let us take all the documents in the following classes:

Table 1: Two well-separated classes

<table width="611">

 <tbody>

  <tr>

   <td width="47">Class 1</td>

   <td width="564">comp.graphics comp.os.ms-windows.misc comp.sys.ibm.pc.hardware comp.sys.mac.hardware</td>

  </tr>

  <tr>

   <td width="47">Class 2</td>

   <td width="564">rec.autos                       rec.motorcycles                             rec.sport.baseball                          rec.sport.hockey</td>

  </tr>

 </tbody>

</table>

We would like to evaluate how purely the <em>a priori </em>known classes can be reconstructed through clustering.

<h2>Problem Statement</h2>

<ol>

 <li>Building the TF-IDF matrix.</li>

</ol>

Finding a good representation of the data is fundamental to the task of clustering. Following the steps in Project 1, <strong>transform the documents into TF-IDF vectors</strong>.

Use min df = 3, exclude the stopwords (no need to do stemming).

<strong>Report the dimensions of the TF-IDF matrix you get.</strong>

<ol start="2">

 <li>Apply K-means clustering with <em>k </em>= 2 using the TF-IDF data. Compare the clustering results with the known class labels. (you can refer to <a href="http://scikit-learn.org/stable/auto_examples/text/document_clustering.html">sklearn – Clustering </a><a href="http://scikit-learn.org/stable/auto_examples/text/document_clustering.html">text documents using k-means</a> for a basic work flow)

  <ul>

   <li>Inspect the contingency matrix to get a sense of your clustering result. Concretely, let <strong>A </strong>be the contingency table produced by the clustering algorithm representing the clustering solution, then <em>A<sub>ij </sub></em>is the number of data points that are members of class <em>c<sub>i </sub></em>and elements of cluster <em>k<sub>j</sub></em>.</li>

   <li>In order to make a concrete comparison of different clustering results, there are various measures of purity for a given partition of the data points with respect to the ground truth. The measures we examine in this project are the <strong>homogeneity score</strong>, the <strong>completeness score</strong>, the <strong>V-measure</strong>, the <strong>adjusted Rand score </strong>and the <strong>adjusted mutual info score</strong>, all of which can be calculated by the corresponding functions provided in metrics.

    <ul>

     <li>Homogeneity is a measure of how “pure” the clusters are. If each cluster contains only data points from a single class, the homogeneity is satisfied.</li>

     <li>On the other hand, a clustering result satisfies completeness if all data points of a class are assigned to the same cluster. Both of these scores span between 0 and 1; where 1 stands for perfect clustering.</li>

     <li>The V-measure is then defined to be the harmonic average of homogeneity score and completeness score.</li>

     <li>The adjusted Rand Index is similar to accuracy measure, which computes similarity between the clustering labels and ground truth labels. This method counts all pairs of points that both fall either in the same cluster and the same class or in different clusters and different classes.</li>

     <li>Finally, the adjusted mutual information score measures the mutual information between the cluster label distribution and the ground truth label distributions.</li>

    </ul></li>

  </ul></li>

</ol>

<strong>Report the 5 measures above for the K-means clustering results you get.</strong>

<ol start="3">

 <li>Preprocess the data.</li>

</ol>

As you may have observed, high dimensional sparse TF-IDF vectors do not yield a good clustering result. One of the reasons is that in a high-dimensional space, the Euclidean distance is not a good metric anymore, in the sense that the distances between data points tends to be almost the same (see [1]).

K-means clustering has other limitations. Since its objective is to minimize the sum of within-cluster <em>l</em><sup>2 </sup>distances, it implicitly assumes that the clusters are isotropically shaped, <em>i.e. </em>round-shaped. When the clusters are not round-shaped, K-means may fail to identify the clusters properly. Even when the clusters are round, Kmeans algorithm may also fail when the clusters have unequal variances. A direct visualization for these problems can be found at <a href="http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py">sklearn – Demonstration of k-means </a><a href="http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py">assumptions</a><a href="http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py">.</a>

In this part we try to find a “better” representation tailored to the way that K-means clustering algorithm works, by preprocessing our data before clustering.

(a) Dimensionality reduction

<ol>

 <li>We will use Latent Semantic Indexing (LSI) and Non-negative Matrix Factorization (NMF) that you are already familiar with for dimensionality reduction.</li>

</ol>

First we want to find the effective dimension of the data through inspection of the top singular values of the TF-IDF matrix and see how many of them are significant in reconstructing the matrix with the truncated SVD representation. A guideline is to see what ratio of the variance of the original data is retained after the dimensionality reduction. <strong>Report the plot of the percent of variance the top </strong><em>r </em><strong>principle components can retain v.s. </strong><em>r</em><strong>, for </strong><em>r </em>= 1 <strong>to </strong>1000<strong>. </strong>ii. Now, use the following two methods to reduce the dimension of the data. Sweep over the dimension parameters for each method, and choose one that yields better results in terms of clustering purity metrics.

<ul>

 <li>Truncated SVD (LSI) / PCA</li>

</ul>

Note that you don’t need to perform SVD multiple times: performing SVD with <em>r </em>= 1000 gives you the data projected on all the top 1000 principle components, so for smaller <em>r</em>’s, you just need to exclude the least important features.

<ul>

 <li>NMF</li>

</ul>

<strong>Specifically, try </strong><em>r </em>= <strong>1</strong><em>,</em><strong>2</strong><em>,</em><strong>3</strong><em>,</em><strong>5</strong><em>,</em><strong>10</strong><em>,</em><strong>20</strong><em>,</em><strong>50</strong><em>,</em><strong>100</strong><em>,</em><strong>300, and plot the 5 measure scores v.s. </strong><em>r </em><strong>for both SVD and NMF; also report the contingency matrices for each </strong><em>r</em><strong>.</strong>

<strong>Report the best </strong><em>r </em><strong>choice for SVD and NMF respectively.</strong>

<strong>How do you explain the non-monotonic behavior of the measures as </strong><em>r </em><strong>increases?</strong>

<ol start="4">

 <li>(a) <strong>Visualize the performance of the case with best clustering results in the previous part your clustering by projecting final data vectors onto 2 dimensional plane and color-coding the classes.</strong></li>

</ol>

(b) Now try the three methods below to see whether they increase the clustering performance. Still use the best <em>r </em>we had in previous parts.

<strong>Visualize the transformed data as in part (a). Report the new clustering measures including the contingency matrix after transformation.</strong>

<ul>

 <li>Normalizing features s.t. each feature has unit variance, i.e. each column of the reduced-dimensional data matrix has unit variance (if we use the convention that rows correspond to documents).</li>

 <li>Applying a non-linear transformation to the data vectors only after NMF. Here we use logarithm transformation as an example.</li>

</ul>

<strong>Can you justify why logarithm transformation may increase the clustering results?</strong>

<ul>

 <li>Now try combining both transformations (in different orders) on NMFreduced data.</li>

</ul>

<ol start="5">

 <li>Expand Dataset into 20 categories</li>

</ol>

In this part we want to examine how purely we can retrieve all 20 original sub-class labels with clustering. Therefore, we need to include all the documents and the corresponding terms in the data matrix and find proper representation through dimensionality reduction of the TF-IDF representation. Still use the same parameters as in part 1.

Try different dimensions for both truncated SVD and NMF dimensionality reduction techniques and the different transformations of the obtained feature vectors as outlined above.