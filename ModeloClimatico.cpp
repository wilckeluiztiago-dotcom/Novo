// modelo_climatico.cpp
// Autor: Luiz Tiago Wilcke (LT) 
// Modelo climático de caixas com múltiplas EDOs acopladas:
//  - Balanço de energia atmosfera/superfície (Ta)
//  - Oceano misto (To)
//  - CO2 atmosférico (Ca)
//  - Biomassa (B)
//  - Fração de gelo-mar (I)
// Integração: Dormand–Prince RK45 (ordem 5(4)) com passo adaptativo.
//
// Saída: clima_saida.csv (tempo_anos, Ta_K, To_K, Ca_ppm, B_PgC, I_frac)

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

using std::cout;
using std::endl;
using ld = long double;

// Constante π portátil (evita dependência de M_PI)
static constexpr long double PI = 3.141592653589793238462643383279502884L;

// ------------------------- Utilidades numéricas -------------------------

// Limita um valor ao intervalo [a, b]
static inline ld clamp(ld x, ld a, ld b) { return std::max(a, std::min(b, x)); }

// Norma Euclidiana para controle de erro relativo
ld norma(const std::vector<ld>& v) {
    ld s = 0;
    for (auto x : v) s += x * x;
    return std::sqrt((double)s); // std::sqrt tem overload para long double
}

// ------------------------- Estado do sistema -------------------------
struct Estado {
    ld Ta;   // Temperatura do ar/superfície (K)
    ld To;   // Temperatura do oceano misto (K)
    ld Ca;   // CO2 atmosférico (ppm)
    ld B;    // Biomassa efetiva (PgC)
    ld I;    // Fração de gelo-mar (0..1)

    std::vector<ld> como_vetor() const { return {Ta, To, Ca, B, I}; }
};

struct Parametros {
    // Constantes radiativas / físicas
    ld S0        = 1361.0L;         // Constante solar (W/m^2)
    ld sigma     = 5.670374419e-8L; // Stefan-Boltzmann
    ld C_atm     = 8.2e7L;          // Capacitância térmica ar/superfície (J/m^2/K)
    ld C_oce     = 3.5e9L;          // Capacitância térmica oceano misto (J/m^2/K)
    ld k_atm_oce = 2.0L;            // Acoplamento atm-oce (W/m^2/K)
    ld eps0      = 0.61L;           // Emissividade base (efeito estufa)
    ld lambda_fb = 1.1L;            // Feedback radiativo equivalente (W/m^2/K)

    // Albedo (função do gelo)
    ld alpha_gelo = 0.55L;
    ld alpha_agua = 0.07L;

    // Forçamento por CO2: RF = 5.35 * ln(Ca/C0)
    ld C0         = 280.0L; // ppm pré-industrial
    ld k_RF       = 5.35L;  // W/m^2

    // Emissões antropogênicas (cenário simples)
    ld Emi0       = 12.0L;  // PgC/ano no ano 0
    ld tend_emi   = 0.0L;   // tendência (PgC/ano^2)

    // Sumidouros de carbono
    ld k_oce_C    = 0.25L;  // 1/ano
    ld k_bio_C    = 0.12L;  // 1/ano (ponderado por B)
    ld Ceq0       = 280.0L;
    ld sens_Ceq_T = 3.0L;   // ppm/K

    // Biosfera (logística com adequação térmica)
    ld Bmax       = 2200.0L; // PgC
    ld rB0        = 0.08L;   // 1/ano
    ld Topt_B     = 290.0L;  // K
    ld sens_B_T   = 0.0035L; // 1/K

    // Dinâmica do gelo
    ld T_cong     = 273.15L; // K
    ld r_derrete  = 0.6L;    // 1/ano
    ld r_forma    = 0.25L;   // 1/ano

    // Variabilidade externa opcional
    ld forca_aer   = -0.2L;  // W/m^2
    ld ciclo_solar = 0.2L;   // W/m^2 (amplitude)
    ld per_solar   = 11.0L;  // anos

    // Integração
    ld t0 = 0.0L;
    ld tf = 150.0L;
    ld dt_ini = 0.02L;  // anos (~7,3 dias)
    ld tol_rel = 1e-6L;
    ld dt_min  = 1e-5L;
    ld dt_max  = 0.25L;

    // Saída/sanidade
    int max_passos = 10'000'000;
};

// ------------------------- Funções auxiliares do modelo -------------------------

// Albedo efetivo (interpolação linear entre gelo e água)
ld albedo(ld I, const Parametros& p) {
    I = clamp(I, 0.0L, 1.0L);
    return p.alpha_gelo * I + p.alpha_agua * (1.0L - I);
}

// Forçamento externo: aerossóis + ciclo solar senoidal
ld forc_externo(ld t, const Parametros& p) {
    ld solar = p.ciclo_solar * std::sin((2.0L * PI / p.per_solar) * t);
    return p.forca_aer + solar;
}

// Emissões antropogênicas (cenário afim truncado)
ld emissoes(ld t, const Parametros& p) {
    return std::max<ld>(0.0L, p.Emi0 + p.tend_emi * t);
}

// Emissividade efetiva incluindo feedback radiativo linearizado
ld emissividade(ld Ta, const Parametros& p) {
    // ε(T) ≈ eps0 - λ / (4 σ T^3)
    ld denom = 4.0L * p.sigma * Ta * Ta * Ta;
    ld delta = (denom > 0 ? p.lambda_fb / denom : 0.0L);
    return clamp(p.eps0 - delta, 0.2L, 0.99L);
}

// Equilíbrio de CO2 do oceano (diminui com T maior)
ld Ceq(ld Ta, const Parametros& p) {
    return std::max<ld>(120.0L, p.Ceq0 - p.sens_Ceq_T * (Ta - 288.0L));
}

// Adequação térmica da biosfera (gaussiana em torno de Tótima)
ld fator_termico_bio(ld Ta, const Parametros& p) {
    ld d = Ta - p.Topt_B;
    return std::exp((double)(-p.sens_B_T * d * d));
}

// ------------------------- Tendências (EDOs) -------------------------
struct Tendencia {
    ld dTa, dTo, dCa, dB, dI;
    std::vector<ld> como_vetor() const { return {dTa, dTo, dCa, dB, dI}; }
};

// Sistema: dX/dt = f(t, X)
Tendencia f_sistema(ld t, const Estado& X, const Parametros& p) {
    // 1) Albedo e forçamento
    ld A    = albedo(X.I, p);
    ld RFc  = p.k_RF * std::log((double)(X.Ca / p.C0)); // W/m^2
    ld Fext = forc_externo(t, p);

    // 2) Balanço de energia (W/m^2)
    ld Qin  = (1.0L - A) * p.S0 * 0.25L;
    ld eps  = emissividade(X.Ta, p);
    ld Qout = eps * p.sigma * X.Ta * X.Ta * X.Ta * X.Ta;

    // Troca atm-oce
    ld Fao  = p.k_atm_oce * (X.To - X.Ta);

    // dTa/dt = (Forçamentos líquidos)/C_atm
    ld dTa = (Qin - Qout + RFc + Fext + Fao) / p.C_atm;

    // Oceano misto
    ld dTo = (-Fao) / p.C_oce;

    // 3) Carbono atmosférico
    ld E = emissoes(t, p);                 // PgC/ano
    ld PgC_por_ppm = 2.124L;
    ld E_ppm = E / PgC_por_ppm;            // ppm/ano

    ld Ceq_T = Ceq(X.Ta, p);
    ld sumidouro_oce = p.k_oce_C * (X.Ca - Ceq_T); // ppm/ano
    ld sumidouro_bio = p.k_bio_C * fator_termico_bio(X.Ta, p) * (X.B / p.Bmax) * (X.Ca - p.C0);

    ld dCa = E_ppm - sumidouro_oce - sumidouro_bio;

    // 4) Biosfera (logística com adequação térmica e leve penalidade por CO2 altíssimo)
    ld rB = p.rB0 * fator_termico_bio(X.Ta, p);
    ld dB = rB * X.B * (1.0L - X.B / p.Bmax) - 0.015L * (X.Ca - p.C0);

    // 5) Gelo-mar
    ld aquecimento  = std::max<ld>(0.0L, X.Ta - p.T_cong);
    ld resfriamento = std::max<ld>(0.0L, p.T_cong - X.Ta);
    ld dI = -p.r_derrete * aquecimento * X.I + p.r_forma * resfriamento * (1.0L - X.I);

    return {dTa, dTo, dCa, dB, dI};
}

// ------------------------- Integrador RK45 (Dormand–Prince) -------------------------

struct PassoResultado {
    Estado Xnovo;
    ld erro_rel;
    bool aceito;
    ld dt_sugerido;
};

Estado soma(const Estado& a, const std::vector<ld>& k, ld esc) {
    Estado r = a;
    r.Ta += esc * k[0];
    r.To += esc * k[1];
    r.Ca += esc * k[2];
    r.B  += esc * k[3];
    r.I  += esc * k[4];
    return r;
}

std::vector<ld> escalar(const std::vector<ld>& v, ld s) {
    std::vector<ld> r = v;
    for (auto& x : r) x *= s;
    return r;
}

std::vector<ld> soma_vec(const std::vector<ld>& a, const std::vector<ld>& b) {
    std::vector<ld> r = a;
    for (size_t i = 0; i < r.size(); ++i) r[i] += b[i];
    return r;
}

PassoResultado rk45_passo(ld t, const Estado& X, ld dt, const Parametros& p) {
    auto as_vec = [&](const Tendencia& g) { return g.como_vetor(); };

    Tendencia k1 = f_sistema(t, X, p);
    Estado X2 = soma(X, as_vec(k1), dt * (1.0L / 5.0L));
    Tendencia k2 = f_sistema(t + dt * (1.0L / 5.0L), X2, p);

    Estado X3 = soma(X,
        soma_vec(escalar(as_vec(k1), dt * (3.0L / 40.0L)),
                 escalar(as_vec(k2), dt * (9.0L / 40.0L))),
        1.0L);
    Tendencia k3 = f_sistema(t + dt * (3.0L / 10.0L), X3, p);

    Estado X4 = soma(X,
        soma_vec(
            soma_vec(escalar(as_vec(k1), dt * (44.0L / 45.0L)),
                     escalar(as_vec(k2), dt * (-56.0L / 15.0L))),
            escalar(as_vec(k3), dt * (32.0L / 9.0L))),
        1.0L);
    Tendencia k4 = f_sistema(t + dt * (4.0L / 5.0L), X4, p);

    Estado X5 = soma(X,
        soma_vec(
            soma_vec(
                soma_vec(escalar(as_vec(k1), dt * (19372.0L / 6561.0L)),
                         escalar(as_vec(k2), dt * (-25360.0L / 2187.0L))),
                escalar(as_vec(k3), dt * (64448.0L / 6561.0L))),
            escalar(as_vec(k4), dt * (-212.0L / 729.0L))),
        1.0L);
    Tendencia k5 = f_sistema(t + dt, X5, p);

    Estado X6 = soma(X,
        soma_vec(
            soma_vec(
                soma_vec(
                    soma_vec(escalar(as_vec(k1), dt * (9017.0L / 3168.0L)),
                             escalar(as_vec(k2), dt * (-355.0L / 33.0L))),
                    escalar(as_vec(k3), dt * (46732.0L / 5247.0L))),
                escalar(as_vec(k4), dt * (49.0L / 176.0L))),
            escalar(as_vec(k5), dt * (-5103.0L / 18656.0L))),
        1.0L);
    Tendencia k6 = f_sistema(t + dt * (7.0L / 8.0L), X6, p);

    // Combinações (ordem 5 e 4)
    std::vector<ld> b5 = { 35.0L/384.0L, 0.0L, 500.0L/1113.0L, 125.0L/192.0L, -2187.0L/6784.0L, 11.0L/84.0L };
    std::vector<ld> b4 = { 5179.0L/57600.0L, 0.0L, 7571.0L/16695.0L, 393.0L/640.0L, -92097.0L/339200.0L, 187.0L/2100.0L };

    std::vector<std::vector<ld>> ks = { as_vec(k1), as_vec(k2), as_vec(k3), as_vec(k4), as_vec(k5), as_vec(k6) };

    std::vector<ld> inc5(5, 0.0L), inc4(5, 0.0L);
    for (size_t i = 0; i < 6; i++) {
        auto ki5 = escalar(ks[i], dt * (i < b5.size() ? b5[i] : 0.0L));
        for (int j = 0; j < 5; j++) inc5[j] += ki5[j];
    }
    for (size_t i = 0; i < 6; i++) {
        auto ki4 = escalar(ks[i], dt * (i < b4.size() ? b4[i] : 0.0L));
        for (int j = 0; j < 5; j++) inc4[j] += ki4[j];
    }

    Estado X5o = soma(X, inc5, 1.0L); // ordem 5
    Estado X4o = soma(X, inc4, 1.0L); // ordem 4

    // Controle de erro relativo (sem std::fabsl; usar std::abs que tem overload para long double)
    std::vector<ld> diff = { X5o.Ta - X4o.Ta, X5o.To - X4o.To, X5o.Ca - X4o.Ca, X5o.B - X4o.B, X5o.I - X4o.I };
    std::vector<ld> escala = {
        std::max<ld>(1e-9L, std::abs(X5o.Ta)),
        std::max<ld>(1e-9L, std::abs(X5o.To)),
        std::max<ld>(1e-9L, std::abs(X5o.Ca)),
        std::max<ld>(1e-9L, std::abs(X5o.B)),
        std::max<ld>(1e-9L, std::abs(X5o.I))
    };
    for (int i = 0; i < 5; i++) diff[i] /= escala[i];
    ld erro_rel = norma(diff);

    // Sugerir próximo passo
    const ld saf = 0.9L;
    const ld expo = 1.0L / 5.0L; // erro ~ dt^5
    ld fator = saf * std::pow((double)(1.0L / std::max<ld>(1e-16L, erro_rel)), (double)expo);
    fator = clamp(fator, 0.1L, 5.0L);

    PassoResultado pr;
    pr.Xnovo = X5o;
    pr.erro_rel = erro_rel;
    pr.aceito = true;
    pr.dt_sugerido = fator * dt;
    return pr;
}

// ------------------------- Simulação -------------------------

int main() {
    std::ios::sync_with_stdio(false);

    Parametros P;

    // Estado inicial aproximado de condições modernas
    Estado X{
        288.15L, // Ta ~ 15 °C
        286.50L, // To
        420.0L,  // Ca ppm
        1600.0L, // B PgC
        0.10L    // I fração de gelo-mar
    };

    std::ofstream arq("clima_saida.csv");
    arq << "tempo_anos,Ta_K,To_K,Ca_ppm,B_PgC,I_frac\n";
    arq << std::fixed << std::setprecision(6);

    ld t = P.t0;
    ld dt = P.dt_ini;
    int passos = 0;

    // Registrar inicial
    arq << (double)t << "," << (double)X.Ta << "," << (double)X.To << ","
        << (double)X.Ca << "," << (double)X.B << "," << (double)X.I << "\n";

    while (t < P.tf && passos < P.max_passos) {
        if (t + dt > P.tf) dt = P.tf - t;

        auto r = rk45_passo(t, X, dt, P);

        // Critério de aceitação (erro relativo vs tolerância)
        if (r.erro_rel <= P.tol_rel || dt <= P.dt_min) {
            // Aceita passo
            X = r.Xnovo;

            // Sanidade física
            X.I  = clamp(X.I, 0.0L, 1.0L);
            X.Ca = std::max<ld>(80.0L, X.Ca);
            X.B  = clamp(X.B, 0.0L, P.Bmax);
            X.Ta = clamp(X.Ta, 200.0L, 340.0L);
            X.To = clamp(X.To, 200.0L, 340.0L);

            t += dt;
            passos++;

            arq << (double)t << "," << (double)X.Ta << "," << (double)X.To << ","
                << (double)X.Ca << "," << (double)X.B << "," << (double)X.I << "\n";

            // Ajustar próximo dt
            dt = clamp(r.dt_sugerido, P.dt_min, P.dt_max);
        } else {
            // Rejeita passo, diminui dt
            dt = std::max<ld>(P.dt_min, 0.5L * dt);
        }
    }

    arq.close();

    cout << "Simulação concluída.\n";
    cout << "Passos realizados: " << passos << "\n";
    cout << "Arquivo gerado: clima_saida.csv\n";
    cout << "Colunas: tempo_anos, Ta_K, To_K, Ca_ppm, B_PgC, I_frac\n";
    cout << "Dica: visualize no gnuplot ou em planilha.\n";
    return 0;
}
