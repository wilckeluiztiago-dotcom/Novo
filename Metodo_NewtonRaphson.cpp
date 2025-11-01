// Autor: Luiz Tiago Wilcke (LT)
// Método de Newton–Raphson para encontrar m tal que K(m) = Kdesejado
// Integrais elípticas K(m) e E(m) aproximadas por série de Legendre

#include <iostream>
#include <cmath>
using namespace std;

// Aproximação de K(m) e E(m) por expansão de Legendre (ordem alta)
long double K(long double m) {
    long double a0 = 1.38629436112L; // ln(4)
    long double a1 = 0.5L * m;
    long double a2 = (9.0L/64.0L) * m*m;
    long double a3 = (25.0L/256.0L) * m*m*m;
    return a0 + a1 + a2 + a3; // aproximação razoável p/ m<0.9
}

long double E(long double m) {
    long double e0 = 1.0L;
    long double e1 = -0.25L * m;
    long double e2 = -3.0L/64.0L * m*m;
    long double e3 = -5.0L/256.0L * m*m*m;
    return e0 + e1 + e2 + e3;
}

// Derivada dK/dm
long double dKdm(long double m) {
    long double Km = K(m);
    long double Em = E(m);
    return (Em - (1.0L - m) * Km) / (2.0L * m * (1.0L - m));
}

// Newton-Raphson
int main() {
    long double Kdesejado = 2.0L;  // alvo
    long double m = 0.5L;          // chute inicial
    long double tol = 1e-10L;
    int max_iter = 50;

    for (int i=0; i<max_iter; ++i) {
        long double f = K(m) - Kdesejado;
        long double df = dKdm(m);
        long double passo = f / df;
        m -= passo;
        if (fabsl(passo) < tol) break;
    }

    cout << "m encontrado = " << m << endl;
    cout << "K(m) = " << K(m) << endl;
}
