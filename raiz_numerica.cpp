// raiz_numerica.cpp
// Autor: Luiz Tiago Wilcke (LT) + ChatGPT
// Métodos numéricos "na unha" para extrair raízes: quadrada e n-ésima
// - Sem usar std::sqrt (apenas aritmética básica)
// - Variáveis em português, comentários didáticos
// - Critério de parada por erro relativo e limite de iterações

#include <iostream>
#include <limits>
#include <cmath>    // permitido para fabs, isfinite, frexp/ldexp (não usamos sqrt)

// ------------------------- Utilidades -------------------------

template <class T>
bool finito(T x) {
    return std::isfinite(static_cast<long double>(x));
}

template <class T>
T absT(T x) {
    return x >= T(0) ? x : -x;
}

// ------------------------- Método de Newton–Heron (raiz quadrada) -------------------------
// Iteração: x_{k+1} = 0.5 * (x_k + a/x_k)
// Converge quadraticamente para a>0 com bom chute inicial.

long double raiz_quadrada_newton(long double a,
                                 long double tolerancia = 1e-12L,
                                 int iter_max = 100)
{
    if (a < 0.0L) {
        // Não existe raiz real (número negativo)
        return std::numeric_limits<long double>::quiet_NaN();
    }
    if (a == 0.0L) return 0.0L;
    if (!finito(a)) return a; // inf -> inf; NaN -> NaN

    // Chute inicial: use escala por potência de 2 para acelerar a convergência.
    // a = m * 2^e com m in [0.5,1); sqrt(a) ~ sqrt(m)*2^(e/2).
    // Como não podemos usar sqrt, aproximamos sqrt(m) por uma aproximação linear simples
    // no intervalo [0.5,1): sqrt(m) ~ 0.41731 + 0.59016*m  (ajuste bruto, mas suficiente p/ chute)
    int expoente2;
    long double mantissa = std::frexp(a, &expoente2); // a = mantissa * 2^expoente2
    long double aprox_sqrt_m = 0.41731L + 0.59016L * mantissa;
    long double chute = std::ldexp(aprox_sqrt_m, expoente2 / 2);

    long double x = (chute > 0 ? chute : 1.0L); // garantia de chute positivo
    for (int k = 0; k < iter_max; ++k) {
        long double x_ant = x;
        x = 0.5L * (x + a / x);          // iteração de Heron
        long double erro_relativo = absT(x - x_ant) / x;
        if (erro_relativo < tolerancia) break;
    }
    return x;
}

// ------------------------- Método de Bisseção (raiz quadrada) -------------------------
// Resolve x^2 = a em x >= 0 por busca binária. Sempre converge, mas é mais lento.

long double raiz_quadrada_bissecao(long double a,
                                   long double tolerancia = 1e-12L,
                                   int iter_max = 200)
{
    if (a < 0.0L) return std::numeric_limits<long double>::quiet_NaN();
    if (a == 0.0L) return 0.0L;
    if (!finito(a)) return a;

    // Intervalo inicial: [esq, dir] tal que esq^2 <= a <= dir^2
    long double esq = 0.0L;
    long double dir = (a >= 1.0L) ? a : 1.0L;

    for (int k = 0; k < iter_max; ++k) {
        long double meio = 0.5L * (esq + dir);
        long double quadrado = meio * meio;

        if (absT(quadrado - a) <= tolerancia * (1.0L + a)) {
            return meio;
        }
        if (quadrado < a) esq = meio; else dir = meio;
    }
    return 0.5L * (esq + dir);
}

// ------------------------- Newton geral para raiz n-ésima -------------------------
// Resolve x^n = a, para n >= 1 inteiro.
// Iteração: x_{k+1} = x_k - (x_k^n - a)/(n * x_k^{n-1})
// Observação: para a < 0, só existe raiz real se n for ímpar.

long double potencia(long double base, unsigned n) {
    // Exponenciação rápida (repeated squaring)
    long double r = 1.0L;
    long double b = base;
    unsigned e = n;
    while (e) {
        if (e & 1U) r *= b;
        b *= b;
        e >>= 1U;
    }
    return r;
}

long double raiz_n_newton(long double a, unsigned n,
                          long double tolerancia = 1e-12L,
                          int iter_max = 200)
{
    if (n == 0U) return std::numeric_limits<long double>::quiet_NaN(); // indefinido
    if (n == 1U) return a;

    // Tratamento de sinal
    bool negativo = (a < 0.0L);
    if (negativo && (n % 2U == 0U)) {
        // Raiz par de número negativo não é real
        return std::numeric_limits<long double>::quiet_NaN();
    }

    long double a_abs = negativo ? -a : a;
    if (a_abs == 0.0L) return 0.0L;

    // Chute inicial simples: 2^(log2(a)/n) usando frexp/ldexp para estabilidade
    int expoente2;
    long double mantissa = std::frexp(a_abs, &expoente2); // a_abs = mantissa * 2^expoente2, mantissa in [0.5,1)
    // Aproximação grosseira de mantissa^(1/n) por uma afim em [0.5,1)
    long double aprox_m = 0.6L + 0.8L * (mantissa - 0.5L); // só para chutar
    long double chute = std::ldexp(aprox_m, expoente2 / static_cast<int>(n));
    if (chute <= 0.0L) chute = 1.0L;

    long double x = chute;
    for (int k = 0; k < iter_max; ++k) {
        // f(x) = x^n - a_abs
        long double x_pow_n = potencia(x, n);
        long double f = x_pow_n - a_abs;
        if (absT(f) <= tolerancia * (1.0L + a_abs)) break;

        long double deriv = static_cast<long double>(n) * potencia(x, n - 1U);
        if (deriv == 0.0L) break;

        long double x_ant = x;
        x = x - f / deriv;

        // Evitar x <= 0 (pode atrapalhar para n par e a>0)
        if (x <= 0.0L) x = 0.5L * (x_ant + a_abs / x_ant);

        long double erro_relativo = absT(x - x_ant) / (absT(x) + 1e-30L);
        if (erro_relativo < tolerancia) break;
    }

    return negativo ? -x : x;
}

// ------------------------- Demonstração -------------------------

int main() {
    std::cout.setf(std::ios::fixed);
    std::cout.precision(12);

    long double numero;
    std::cout << "Digite um numero para extrair a raiz quadrada: ";
    if (!(std::cin >> numero)) return 0;

    long double rq_newton   = raiz_quadrada_newton(numero, 1e-12L, 100);
    long double rq_bissecao = raiz_quadrada_bissecao(numero, 1e-12L, 200);

    std::cout << "\n[Raiz Quadrada]\n";
    std::cout << "Newton–Heron  : " << rq_newton   << '\n';
    std::cout << "Bissecao      : " << rq_bissecao << '\n';

    // Raiz n-ésima
    unsigned n;
    std::cout << "\nDigite n para extrair a raiz n-esima de outro numero: ";
    if (!(std::cin >> n)) return 0;

    long double outro;
    std::cout << "Digite o numero: ";
    if (!(std::cin >> outro)) return 0;

    long double rn_newton = raiz_n_newton(outro, n, 1e-12L, 200);
    std::cout << "\n[Raiz n-esima]\n";
    std::cout << "Newton (n=" << n << "): " << rn_newton << '\n';

    // Testes rápidos
    std::cout << "\n[Testes rapidos]\n";
    std::cout << "raiz_quadrada_newton(2)  ~ " << raiz_quadrada_newton(2.0L) << '\n';
    std::cout << "raiz_quadrada_bissecao(2)~ " << raiz_quadrada_bissecao(2.0L) << '\n';
    std::cout << "raiz_n_newton(27,3)      ~ " << raiz_n_newton(27.0L, 3) << '\n';
    std::cout << "raiz_n_newton(1e-12,2)   ~ " << raiz_n_newton(1e-12L, 2) << '\n';

    return 0;
}
