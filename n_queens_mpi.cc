// AUTHOR: Javier Garcia Santana
// DATE: 06/12/2023
// EMAIL: javier.santana@tprs.stud.vu.lt
// VERSION: 2.0
// COURSE: Parallel and Distributed Computing
// EXCERCISE Nº: 4
//
// mpic++ -o n_queens_mpi n_queens_mpi.cc
// mpiexec -np <num_processes> ./n_queens_mpi <board_size> <number_of_processes>

#define MASTER_RANK 0

#include <iostream>
#include <vector>
#include <mpi.h>
#include <cmath>

bool isSafe(const std::vector<int>& queens, int row, int col) {
    // Check if placing a queen at (row, col) is safe
    for (int i = 0; i < row; ++i) {
        if (queens[i] == col || std::abs(queens[i] - col) == std::abs(row - i)) {
            return false; // Threatening another queen
        }
    }
    return true; // Safe placement
}

void printBoard(const std::vector<int>& queens) {
    int boardSize = queens.size();
    for (int i = 0; i < boardSize; ++i) {
        for (int j = 0; j < boardSize; ++j) {
            if (queens[i] == j) {
                std::cout << "Q ";
            } else {
                std::cout << ". ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void solveNQueens(int boardSize, int row, std::vector<int>& queens, int& solutions) {
    if (row == boardSize) {
        // All queens are placed successfully
        // Increment solution count and optionally print the board
        solutions++;
        // Uncomment the line below to print each solution
        // printBoard(queens);
        return;
    }

    for (int col = 0; col < boardSize; ++col) {
        if (isSafe(queens, row, col)) {
            // Place the queen and move on to the next row
            queens[row] = col;
            solveNQueens(boardSize, row + 1, queens, solutions);
        }
    }
}

void placeQueens(int boardSize, int startRow, int endRow, int& solutions) {
    // Initialize queens vector for each process
    std::vector<int> queens(boardSize, -1);

    // Iterate over the assigned rows for this process
    for (int row = startRow; row < endRow; ++row) {
        queens[0] = row; // Place the queen in the first column of the current row
        solveNQueens(boardSize, 1, queens, solutions);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == MASTER_RANK) {
            std::cerr << "Usage: " << argv[0] << " <board_size> <num_processes>\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int boardSize = std::atoi(argv[1]);
    int numThreads = std::atoi(argv[2]);

    double startTime, endTime;

    if (rank == MASTER_RANK) {
        std::cout << "Board Size: " << boardSize << std::endl;
        std::cout << "Number of Processes: " << numThreads << std::endl;
    }

    // Start the timer
    startTime = MPI_Wtime();

    // Determine the portion of the problem to be solved by each process
    int rowsPerProcess = boardSize / size;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank == size - 1) ? boardSize : (rank + 1) * rowsPerProcess;

    // Each process independently solves its portion of the N-Queens problem
    int localSolutions = 0;
    placeQueens(boardSize, startRow, endRow, localSolutions);

    // Use MPI reduction to sum up the local solutions from all processes
    int globalSolutions;
    MPI_Reduce(&localSolutions, &globalSolutions, 1, MPI_INT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);

    // Stop the timer
    endTime = MPI_Wtime();

    // Output results on the master process
    if (rank == MASTER_RANK) {
        std::cout << "Number of Solutions: " << globalSolutions << std::endl;
        std::cout << "Execution Time: " << endTime - startTime << " seconds." << std::endl;
    }

    MPI_Finalize();

    return 0;
}
