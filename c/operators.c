
// gcc -shared -o operators.so -fPIC operators.c

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef EPSILON
#define EPSILON 1e-6
#endif

// core mathematical operators

double _mul(double x, double y) {
    return x * y;
}

double _id(double x) {
    return x;
}

double _add(double x, double y) {
    return x + y;
}

double _neg(double x) {
    return -x;
}

double _lt(double x, double y) {
    return x < y ? 1.0 : 0.0;
}

double _eq(double x, double y) {
    return x == y ? 1.0 : 0.0;
}

double _max(double x, double y) {
    return x > y ? x : y;
}

double _is_close(double x, double y) {
    return fabs(x - y) < 1e-2 ? 1.0 : 0.0;
}

double _sigmoid(double x) {
    return x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
}

double _relu(double x) {
    return x > 0 ? x : 0.0;
}

double _log(double x) {
    return log(x + EPSILON);
}

double _exp(double x) {
    return exp(x);
}

double _log_back(double x, double d) {
    return d / (x + EPSILON);
}

double _inv(double x) {
    return 1.0 / x;
}

double _inv_back(double x, double d) {
    return -d / (x * x);
}

double _relu_back(double x, double d) {
    return x > 0 ? d : 0.0;
}

double _sigmoid_back(double x, double d) {
    return d * exp(-x) / pow(1.0 + exp(-x), 2);
}

// higher-order functions

double _map(double (*fn)(double), double* x, int n) {
    // allocate memory for new list
    double *result = malloc(n * sizeof(double));
    if (result == NULL) {
        return -1;
    }

    // apply function to each element
    for (int i = 0; i < n; i++) {
        result[i] = fn(x[i]);
    }

    return *result;
}





