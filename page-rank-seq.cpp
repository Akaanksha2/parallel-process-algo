#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " input_file" << endl;
        exit(EXIT_FAILURE);
    }

    // Open input file
    ifstream input(argv[1]);
    if (!input.is_open()) {
        cerr << "Error opening input file" << endl;
        exit(EXIT_FAILURE);
    }

    // Read matrix size
    int N;
    input >> N;

    // Allocate memory for matrix A and vector R
    vector<vector<float> > A(N, vector<float>(N));
    vector<float> R(N, 1.0 / N);

    // Read matrix from file and normalize rows
    for (int i = 0; i < N; i++) {
        int num_outgoing;
        input >> num_outgoing;

        for (int j = 0; j < num_outgoing; j++) {
            int outgoing_node;
            input >> outgoing_node;
            A[outgoing_node][i] = 1.0 / num_outgoing;
        }
    }

    // Calculate PageRank
    float eps = 1e-5;
    float d = 0.85;
    bool converged = false;

    while (!converged) {
        vector<float> R_new(N, 0);

        // Perform matrix-vector multiplication
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                R_new[i] += A[i][j] * R[j];
            }
        }

        // Apply damping factor and check convergence
        float diff = 0;
        for (int i = 0; i < N; i++) {
            R_new[i] = d * R_new[i] + (1 - d) / N;
            diff += abs(R_new[i] - R[i]);
        }

        if (diff < eps) {
            converged = true;
        }

        R = R_new;
    }

    // Print final PageRank vector
    for (int i = 0; i < N; i++) {
        cout << "R[" << i << "] = " << R[i] << endl;
    }

    return 0;
}
