// integra_multiplas.cpp
// Autor: Luiz Tiago Wilcke (LT)
// Métodos numéricos para integrais múltiplas em N dimensões:
//  - Monte Carlo padrão (MC) com IC 95%
//  - Amostragem por Importância (IS) plugável
//  - Latin Hypercube Sampling (LHS)
//  - Quasi–Monte Carlo com sequência de Halton
//  - Estratificação regular (grade por eixo)
// Domínio: hiper-retângulo [a_i, b_i] em cada dimensão.


#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

using namespace std;

// ========================= Aleatório / Utilidades =========================
static uint64_t semente_global = 42;
static std::mt19937_64 gerador(semente_global);

struct CaixaND {
    vector<double> a; // limites inferiores
    vector<double> b; // limites superiores
    double volume() const {
        double v = 1.0;
        for (size_t i=0;i<a.size();++i) v *= (b[i]-a[i]);
        return v;
    }
};

inline double clampd(double x, double lo, double hi){ return std::max(lo, std::min(hi, x)); }

// estatística simples (média, variância amostral)
static pair<double,double> media_var(const vector<double>& v){
    size_t n=v.size();
    if(n==0) return {0.0,0.0};
    long double s=0.0L;
    for(double x: v) s+=x;
    long double mu = s/n;
    long double sv=0.0L;
    for(double x: v){ long double d=x-mu; sv+=d*d; }
    // variância amostral
    double var = (n>1)? double(sv/(n-1)) : 0.0;
    return {double(mu), var};
}

// z de 95% bicaudal ~ 1.96
constexpr double Z95 = 1.959963984540054;

// ========================= Sequência de Halton (Quasi-MC) =========================
// inversa radical (van der Corput) em base "base"
static double radical_inversa(uint64_t n, uint32_t base){
    double f = 1.0, r = 0.0;
    while (n>0){
        f /= base;
        r += f * (n % base);
        n /= base;
    }
    return r; // em [0,1)
}

// pequenas bases primas para as primeiras dimensões
static const uint32_t PRIMOS_HALTON[] = {
    2,3,5,7,11,13,17,19,23,29,
    31,37,41,43,47,53,59,61,67,71,
    73,79,83,89,97,101,103,107,109,113
};

// gera ponto Halton d-dimensional no índice n (offset opcional)
static vector<double> ponto_halton(uint64_t n, size_t d, uint64_t desloc=0){
    vector<double> x(d);
    for(size_t i=0;i<d;i++){
        uint32_t base = PRIMOS_HALTON[i % (sizeof(PRIMOS_HALTON)/sizeof(PRIMOS_HALTON[0]))];
        x[i] = radical_inversa(n+desloc, base);
    }
    return x;
}

// ========================= Latin Hypercube Sampling (LHS) =========================
static vector<vector<double>> gerar_lhs(size_t n_amostras, size_t d){
    std::uniform_real_distribution<double> U(0.0,1.0);
    vector<vector<double>> X(n_amostras, vector<double>(d));
    // para cada dimensão, cria permutação de estratos
    for(size_t j=0;j<d;j++){
        vector<size_t> perm(n_amostras);
        iota(perm.begin(), perm.end(), 0);
        shuffle(perm.begin(), perm.end(), gerador);
        for(size_t i=0;i<n_amostras;i++){
            double u = U(gerador);
            X[i][j] = (perm[i] + u)/double(n_amostras); // dentro do estrato
        }
    }
    return X;
}

// ========================= Mapeamentos de [0,1]^d para a caixa =========================
static inline vector<double> mapear_para_caixa(const vector<double>& u, const CaixaND& caixa){
    vector<double> x(u.size());
    for(size_t i=0;i<u.size();++i){
        x[i] = caixa.a[i] + (caixa.b[i]-caixa.a[i]) * clampd(u[i], 0.0, 1.0);
    }
    return x;
}

// ========================= Interfaces de integração =========================
struct ResultadoIntegracao {
    double estimativa{};     // estimativa da integral
    double desvio_padrao{};  // DP do estimador
    double erro_padrao{};    // DP/sqrt(N) (após multiplicar pelo volume quando aplicável)
    double ic95_inf{};       // intervalo de confiança 95%
    double ic95_sup{};
    size_t amostras{};
    double tempo_ms{};       // tempo de execução
    string metodo;           // nome do método
};

// ---------- Monte Carlo padrão ----------
ResultadoIntegracao integra_mc(
    const function<double(const vector<double>&)>& f,
    const CaixaND& caixa, size_t n_amostras, bool paralelo=true)
{
    auto t0 = chrono::high_resolution_clock::now();
    std::uniform_real_distribution<double> U(0.0,1.0);
    size_t d = caixa.a.size();
    double V = caixa.volume();

    // coletar f(x) nos pontos uniformes
    vector<double> valores; valores.reserve(n_amostras);

    #pragma omp parallel if(paralelo) 
    {
        std::mt19937_64 rng_local(gerador());
        std::uniform_real_distribution<double> Uloc(0.0,1.0);
        #pragma omp for schedule(static)
        for (long long i=0; i<(long long)n_amostras; ++i){
            vector<double> u(d);
            for(size_t j=0;j<d;j++) u[j] = Uloc(rng_local);
            auto x = mapear_para_caixa(u, caixa);
            double fx = f(x);
            #pragma omp critical
            valores.push_back(fx);
        }
    }

    auto [mu, var] = media_var(valores);
    double est = V * mu;
    double dp  = std::sqrt(var);             // DP dos f(x) (no cubo unitário)
    double ep  = V * dp / std::sqrt((double)max<size_t>(1, valores.size()));
    double inf = est - Z95*ep, sup = est + Z95*ep;

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, std::milli>(t1-t0).count();
    return {est, dp, ep, inf, sup, valores.size(), ms, "Monte Carlo"};
}

// ---------- Latin Hypercube Sampling (LHS) ----------
ResultadoIntegracao integra_lhs(
    const function<double(const vector<double>&)>& f,
    const CaixaND& caixa, size_t n_amostras, bool paralelo=true)
{
    auto t0 = chrono::high_resolution_clock::now();
    size_t d = caixa.a.size();
    double V = caixa.volume();

    auto U = gerar_lhs(n_amostras, d);
    vector<double> valores; valores.reserve(n_amostras);

    #pragma omp parallel for if(paralelo) schedule(static)
    for (long long i=0; i<(long long)n_amostras; ++i){
        auto x = mapear_para_caixa(U[i], caixa);
        double fx = f(x);
        #pragma omp critical
        valores.push_back(fx);
    }

    auto [mu, var] = media_var(valores);
    double est = V * mu;
    double dp  = std::sqrt(var);
    double ep  = V * dp / std::sqrt((double)max<size_t>(1, valores.size()));
    double inf = est - Z95*ep, sup = est + Z95*ep;

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, std::milli>(t1-t0).count();
    return {est, dp, ep, inf, sup, valores.size(), ms, "Latin Hypercube"};
}

// ---------- Quasi–Monte Carlo (Halton) ----------
ResultadoIntegracao integra_halton(
    const function<double(const vector<double>&)>& f,
    const CaixaND& caixa, size_t n_amostras)
{
    auto t0 = chrono::high_resolution_clock::now();
    size_t d = caixa.a.size();
    double V = caixa.volume();

    vector<double> valores; valores.reserve(n_amostras);

    for(size_t i=0;i<n_amostras;i++){
        auto u = ponto_halton(i+1, d, /*desloc*/1234);
        auto x = mapear_para_caixa(u, caixa);
        double fx = f(x);
        valores.push_back(fx);
    }

    auto [mu, var] = media_var(valores);
    double est = V * mu;
    double dp  = std::sqrt(var);
    double ep  = V * dp / std::sqrt((double)max<size_t>(1, valores.size())); // aproximação
    double inf = est - Z95*ep, sup = est + Z95*ep;

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, std::milli>(t1-t0).count();
    return {est, dp, ep, inf, sup, valores.size(), ms, "Quasi–MC (Halton)"};
}

// ---------- Estratificação regular por eixos ----------
// Divide cada eixo em m estratos (total m^d blocos). Amostra k pontos por bloco.
ResultadoIntegracao integra_estratificado(
    const function<double(const vector<double>&)>& f,
    const CaixaND& caixa, size_t m_por_eixo, size_t k_por_bloco, bool paralelo=true)
{
    auto t0 = chrono::high_resolution_clock::now();
    size_t d = caixa.a.size();
    double V = caixa.volume();

    std::uniform_real_distribution<double> U(0.0,1.0);

    long long blocos = 1;
    for(size_t i=0;i<d;i++){
        if ( (double)blocos * (double)m_por_eixo > 9e15 ){ // proteção
            cerr<<"[Aviso] Muitos blocos na estratificação; reduza m_por_eixo.\n";
            break;
        }
        blocos *= (long long)m_por_eixo;
    }

    vector<double> estimativas_blocos; estimativas_blocos.reserve(blocos);

    #pragma omp parallel for if(paralelo) schedule(static)
    for (long long idb=0; idb<blocos; ++idb){
        // decompõe idb em dígitos base m_por_eixo para achar o cubo estratificado
        long long tmp = idb;
        vector<size_t> idx(d,0);
        for(size_t j=0;j<d;j++){ idx[j] = tmp % m_por_eixo; tmp/=m_por_eixo; }

        // limites do bloco
        vector<double> a_loc(d), b_loc(d);
        for(size_t j=0;j<d;j++){
            double w = (caixa.b[j]-caixa.a[j]) / m_por_eixo;
            a_loc[j] = caixa.a[j] + idx[j]*w;
            b_loc[j] = a_loc[j] + w;
        }
        CaixaND bloco{a_loc,b_loc};
        double Vb = bloco.volume();

        // amostragem uniforme dentro do bloco
        std::mt19937_64 rng_local(gerador());
        std::uniform_real_distribution<double> Uloc(0.0,1.0);
        double soma_fx=0.0;
        for(size_t k=0;k<k_por_bloco;k++){
            vector<double> u(d);
            for(size_t j=0;j<d;j++) u[j] = Uloc(rng_local);
            auto x = mapear_para_caixa(u, bloco);
            soma_fx += f(x);
        }
        double media_fx = soma_fx / max<size_t>(1,k_por_bloco);
        double est_bloco = Vb * media_fx;

        #pragma omp critical
        estimativas_blocos.push_back(est_bloco);
    }

    // média das estimativas dos blocos
    double est = accumulate(estimativas_blocos.begin(), estimativas_blocos.end(), 0.0);
    // erro: usa variância entre blocos / sqrt(n_blocos) (heurística)
    auto [muB, varB] = media_var(estimativas_blocos);
    double dp  = std::sqrt(varB); // DP das estimativas por bloco
    double ep  = dp / std::sqrt((double)max<size_t>(1, estimativas_blocos.size()));
    double inf = est - Z95*ep, sup = est + Z95*ep;

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, std::milli>(t1-t0).count();
    return {est, dp, ep, inf, sup, (size_t)blocos*k_por_bloco, ms, "Estratificado (grade)"};
}

// ---------- Amostragem por Importância (genérica) ----------
// Você fornece:
//  - amostrador: retorna ponto u em [0,1]^d (ou diretamente em x com map=false)
//  - pdf_u(u): densidade no hiper-cubo; se mapear para caixa depois, a pdf é em u
//  - se mapear_para_caixa_flag=true, faz x = mapear(u) e integra f(x)*V / pdf_u(u)
ResultadoIntegracao integra_importancia(
    const function<double(const vector<double>&)>& f,
    const CaixaND& caixa, size_t n_amostras,
    const function<vector<double>(std::mt19937_64&)>& amostrador_u,
    const function<double(const vector<double>&)>& pdf_u,
    bool mapear_para_caixa_flag = true,
    bool paralelo=true)
{
    auto t0 = chrono::high_resolution_clock::now();
    size_t d = caixa.a.size();
    double V = caixa.volume();

    vector<double> pesos_f; pesos_f.reserve(n_amostras);

    #pragma omp parallel if(paralelo)
    {
        std::mt19937_64 rng_local(gerador());
        #pragma omp for schedule(static)
        for (long long i=0; i<(long long)n_amostras; ++i){
            auto u = amostrador_u(rng_local);            // ponto em [0,1]^d
            double p = std::max(1e-300, pdf_u(u));       // densidade
            vector<double> x = mapear_para_caixa_flag ? mapear_para_caixa(u, caixa) : u;
            double fx = f(x);
            double w = (mapear_para_caixa_flag? V : 1.0) * fx / p;
            #pragma omp critical
            pesos_f.push_back(w);
        }
    }

    auto [mu, var] = media_var(pesos_f);
    double est = mu;
    double dp  = std::sqrt(var);
    double ep  = dp / std::sqrt((double)max<size_t>(1, pesos_f.size()));
    double inf = est - Z95*ep, sup = est + Z95*ep;

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, std::milli>(t1-t0).count();
    return {est, dp, ep, inf, sup, pesos_f.size(), ms, "Importância (genérico)"};
}

// ========================= Exemplos de uso =========================

// Exemplo 1: volume da d-bola (raio=1) por MC em [-1,1]^d.
// Valor exato: V_d = pi^{d/2} / Gamma(d/2 + 1)
double volume_bola_exato(int d){
    return std::pow(M_PI, 0.5*d) / std::tgamma(0.5*d + 1.0);
}

void exemplo_volume_bola(int d, size_t N){
    cout<<"\n=== Exemplo: Volume da "<<d<<"-bola em [-1,1]^"<<d<<" ===\n";
    CaixaND caixa;
    caixa.a = vector<double>(d, -1.0);
    caixa.b = vector<double>(d, +1.0);

    auto indicador_bola = [d](const vector<double>& x)->double{
        double r2=0.0; for(double xi: x) r2 += xi*xi;
        return (r2<=1.0)? 1.0 : 0.0;
    };

    auto Rmc   = integra_mc(indicador_bola, caixa, N, /*paralelo*/true);
    auto Rlhs  = integra_lhs(indicador_bola, caixa, N, true);
    auto Rqmc  = integra_halton(indicador_bola, caixa, N);

    double exato = volume_bola_exato(d);

    auto mostra = [&](const ResultadoIntegracao& R){
        cout<<left<<setw(24)<<R.metodo
            <<" N="<<R.amostras
            <<"   estimativa="<<setprecision(10)<<R.estimativa
            <<"   IC95=["<<R.ic95_inf<<","<<R.ic95_sup<<"]"
            <<"   tempo="<<fixed<<setprecision(2)<<R.tempo_ms<<"ms\n";
    };

    mostra(Rmc);
    mostra(Rlhs);
    mostra(Rqmc);

    cout<<"Valor exato: "<<setprecision(12)<<exato<<"\n";
}

// Exemplo 2: integral ∫_{[0,1]^d} ∏ x_i^2 dx = (1/3)^d (caso suave).
void exemplo_produto_quadrado(int d, size_t N){
    cout<<"\n=== Exemplo: ∫_{[0,1]^"<<d<<"} prod x_i^2 dx ===\n";
    CaixaND caixa;
    caixa.a = vector<double>(d, 0.0);
    caixa.b = vector<double>(d, 1.0);

    auto func = [](const vector<double>& x)->double{
        double p=1.0; for(double xi: x) p*= (xi*xi); return p;
    };

    auto Rmc   = integra_mc(func, caixa, N, true);
    auto Rlhs  = integra_lhs(func, caixa, N, true);
    auto Rqmc  = integra_halton(func, caixa, N);

    double exato = std::pow(1.0/3.0, d);

    auto mostra = [&](const ResultadoIntegracao& R){
        cout<<left<<setw(24)<<R.metodo
            <<" N="<<R.amostras
            <<"   estimativa="<<setprecision(12)<<R.estimativa
            <<"   IC95=["<<R.ic95_inf<<","<<R.ic95_sup<<"]"
            <<"   tempo="<<fixed<<setprecision(2)<<R.tempo_ms<<"ms\n";
    };

    mostra(Rmc);
    mostra(Rlhs);
    mostra(Rqmc);

    cout<<"Valor exato: "<<setprecision(12)<<exato<<"\n";
}

// Exemplo 3: Importância (proposta Beta concentrada no centro) em [0,1]^d
// Alvo: a mesma ∫ prod x_i^2, mas usando pdf(u) = ∏ Beta(a,a) com a=3.
void exemplo_importancia_beta(int d, size_t N){
    cout<<"\n=== Exemplo: Importância com Beta(a=3) em [0,1]^"<<d<<" ===\n";
    CaixaND caixa;
    caixa.a = vector<double>(d, 0.0);
    caixa.b = vector<double>(d, 1.0);

    auto func = [](const vector<double>& x)->double{
        double p=1.0; for(double xi: x) p*= (xi*xi); return p;
    };

    // pdf Beta(a,a) em [0,1]: Beta(a,a) = gamma(2a)/(gamma(a)^2)
    double a = 3.0;
    double logC = std::lgamma(2*a) - 2*std::lgamma(a);
    auto pdf_1d = [&](double u)->double{
        if (u<=0.0 || u>=1.0) return 0.0;
        return std::exp(logC + (a-1.0)*std::log(u) + (a-1.0)*std::log(1.0-u));
    };

    auto amostrador_u = [&](std::mt19937_64& rng)->vector<double>{
        // amostra Beta(a,a) via inversão aproximada por rejeição simples
        std::gamma_distribution<double> G(a,1.0);
        vector<double> u(d);
        for(int i=0;i<d;i++){
            double X = G(rng), Y = G(rng);
            double t = X/(X+Y); // Beta(a,a)
            u[i] = clampd(t, 1e-12, 1.0-1e-12);
        }
        return u;
    };
    auto pdf_u = [&](const vector<double>& u)->double{
        double p=1.0; for(double ui: u) p*= pdf_1d(ui); return p;
    };

    auto R = integra_importancia(func, caixa, N, amostrador_u, pdf_u, /*map*/true, /*paralelo*/true);

    cout<<left<<setw(24)<<R.metodo
        <<" N="<<R.amostras
        <<"   estimativa="<<setprecision(12)<<R.estimativa
        <<"   IC95=["<<R.ic95_inf<<","<<R.ic95_sup<<"]"
        <<"   tempo="<<fixed<<setprecision(2)<<R.tempo_ms<<"ms\n";

    double exato = std::pow(1.0/3.0, d);
    cout<<"Valor exato: "<<setprecision(12)<<exato<<"\n";
}

// ========================= main =========================
int main(){
    cout<<fixed<<setprecision(6);

    // Ajuste aqui o tamanho das integrações para teste
    int d_teste1 = 5;      size_t N1 = 500000;  // volume da d-bola
    int d_teste2 = 8;      size_t N2 = 300000;  // integral suave em [0,1]^d
    int d_teste3 = 8;      size_t N3 = 300000;  // importância

    exemplo_volume_bola(d_teste1, N1);
    exemplo_produto_quadrado(d_teste2, N2);
    exemplo_importancia_beta(d_teste3, N3);
    
    cout<<"\nDicas de uso prático:\n"
        <<"- Para integrais em domínios gerais, mapeie para um hipercubo/retângulo.\n"
        <<"- Para funções agudas (picos), use Importância ou Quasi–MC.\n"
        <<"- LHS tende a reduzir variância vs MC em funções suaves.\n"
        <<"- Estratificação ajuda quando a variabilidade se organiza por eixos.\n"
        <<"- Compile com -fopenmp para acelerar.\n";

    return 0;
}
