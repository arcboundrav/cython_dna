A population of feedforward neural networks with a pre-defined architecture are randomly initialized.
Which nodes are connected and the weights of those connections are subject to optimization through
the use of a genetic algorithm. The algorithm is adapted from the approach described in Stanley and
Mikkulainen's 2002 paper, Evolving Neural Networks Through Augmenting Topologies.

The project is a proof of concept. The approach successfully evolves a network which is able
to approximate an arbitrary value between 0 and 1 to within a margin of error of 0.0001.


Running the Simulation

1.  Use pip to Install numpy>=1.17.4 and easycython>=1.0.7.
2.  In the directory containing dna.pyx, run easycython dna.pyx.
3.  Once it is finished, run python3 and:
    import dna;dna.test()
