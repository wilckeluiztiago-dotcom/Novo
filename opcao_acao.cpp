// Precificação por Monte Carlo c/ Longstaff–Schwartz (LSM), antitético e variate de controle
// Autor: Luiz Tiago Wilcke (LT)
// C++17/20: g++ -O3 -std=c++20 lsm_mc.cpp -o lsm_mc
#include <bits/stdc++.h>
using namespace std;

// -------------------- Utilidades aleatórias --------------------
struct Gerador {
    static thread_local mt19937_64 rng;
    normal_distribution<double> normal{0.0,1.0};
    double z() { return normal(rng); }
};
thread_local mt19937_64 Gerador::rng{uint64_t(0x9e3779b97f4a7c15ULL ^ chrono::high_resolution_clock::now().time_since_epoch().count())};

// -------------------- Estruturas de parâmetros --------------------
struct ParametrosOpcao {
    double preco_inicial; // S0
    double strike;        // K
    double taxa_juros;    // r (contínua)
    double volatilidade;  // sigma
    double maturidade;    // T (anos)
};

enum class TipoOpcao { Call, Put };

// -------------------- Black-Scholes fechado (europeu call) --------------------
double cdf_norm(double x){
    return 0.5*erfc(-x*M_SQRT1_2);
}
double preco_black_scholes_call(const ParametrosOpcao& p){
    double S=p.preco_inicial,K=p.strike,r=p.taxa_juros,sig=p.volatilidade,T=p.maturidade;
    if (T<=0) return max(0.0,S-K);
    double d1=(log(S/K)+(r+0.5*sig*sig)*T)/(sig*sqrt(T));
    double d2=d1-sig*sqrt(T);
    return S*cdf_norm(d1)-K*exp(-r*T)*cdf_norm(d2);
}

// -------------------- Simulação de caminhos GBM (com antitético) --------------------
struct Caminhos {
    // matriz: [amostra][passo] com preços
    vector<vector<double>> caminho_principal, caminho_antitetico;
    vector<double> tempos;
};
Caminhos simular_gbm(const ParametrosOpcao& p, int n_amostras, int n_passos){
    Gerador g;
    double dt=p.maturidade/n_passos;
    double mu=p.taxa_juros;
    double sig=p.volatilidade;
    double drift=(mu-0.5*sig*sig)*dt;
    double vol_dt=sig*sqrt(dt);

    Caminhos cam;
    cam.caminho_principal.assign(n_amostras, vector<double>(n_passos+1, p.preco_inicial));
    cam.caminho_antitetico.assign(n_amostras, vector<double>(n_passos+1, p.preco_inicial));
    cam.tempos.resize(n_passos+1);
    for(int t=0;t<=n_passos;++t) cam.tempos[t]=t*dt;

    for(int i=0;i<n_amostras;++i){
        double S=p.preco_inicial, SA=p.preco_inicial;
        for(int t=1;t<=n_passos;++t){
            double Z=g.z();
            double Zanti=-Z;
            S  = S * exp(drift + vol_dt*Z);
            SA = SA* exp(drift + vol_dt*Zanti);
            cam.caminho_principal[i][t]=S;
            cam.caminho_antitetico[i][t]=SA;
        }
    }
    return cam;
}

// -------------------- Payoff --------------------
double payoff(const TipoOpcao tipo, double S, double K){
    if (tipo==TipoOpcao::Call) return max(0.0, S-K);
    return max(0.0, K-S);
}

// -------------------- LSM para opção americana (ex.: Put) --------------------
// Base polinomial: [1, S, S^2]; regressão OLS local em cada tempo (sem regularização)
double preco_americana_LSM_put(const ParametrosOpcao& p, int n_amostras, int n_passos){
    // simula caminhos antitéticos e usa média antitética
    auto cam = simular_gbm(p, n_amostras, n_passos);
    double dt=p.maturidade/n_passos;
    double disc=exp(-p.taxa_juros*dt);
    int N=n_amostras;

    // valores de exercício imediato (payoff) por amostra e passo, usando média antitética do preço
    vector<vector<double>> S_med(N, vector<double>(n_passos+1));
    vector<vector<double>> exer(N, vector<double>(n_passos+1));
    for(int i=0;i<N;++i){
        for(int t=0;t<=n_passos;++t){
            double Sm = 0.5*(cam.caminho_principal[i][t] + cam.caminho_antitetico[i][t]);
            S_med[i][t]=Sm;
            exer[i][t]=payoff(TipoOpcao::Put, Sm, p.strike);
        }
    }
    // valores de caixa (cashflows) iniciam com valor no vencimento
    vector<double> cf(N);
    vector<int>    t_ex(N, n_passos);
    for(int i=0;i<N;++i){ cf[i]=exer[i][n_passos]; }

    // retropropagação LSM (do penúltimo passo até t=1)
    for(int t=n_passos-1; t>=1; --t){
        // seleciona amostras in-the-money (ITM)
        vector<int> idx;
        idx.reserve(N);
        for(int i=0;i<N;++i) if (exer[i][t]>1e-14) idx.push_back(i);
        if (idx.empty()) continue;

        // monta X (base 1, S, S^2) e y (cf descontado do próximo stopping time)
        int m = (int)idx.size();
        vector<array<double,3>> X(m);
        vector<double> y(m);
        for(int j=0;j<m;++j){
            int i = idx[j];
            double S = S_med[i][t];
            X[j] = {1.0, S, S*S};
            // desconta cf até o próximo exercício conhecido
            int dtau = t_ex[i]-t;
            y[j] = cf[i]*exp(-p.taxa_juros*dtau*(p.maturidade/n_passos));
        }

        // resolve beta = (X^T X)^{-1} X^T y (3x3) — fechadinho
        double XtX[3][3]={{0,0,0},{0,0,0},{0,0,0}};
        double Xty[3]={0,0,0};
        for(int j=0;j<m;++j){
            for(int a=0;a<3;++a){
                Xty[a]+=X[j][a]*y[j];
                for(int b=0;b<3;++b) XtX[a][b]+=X[j][a]*X[j][b];
            }
        }
        // inversão 3x3 (Cramer)
        auto det3=[&](double A[3][3]){
            return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                 - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
                 + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
        };
        double A[3][3]; memcpy(A,XtX,sizeof(A));
        double D=det3(A)+1e-18;
        double inv[3][3];
        inv[0][0]=( A[1][1]*A[2][2]-A[1][2]*A[2][1])/D;
        inv[0][1]=(-A[0][1]*A[2][2]+A[0][2]*A[2][1])/D;
        inv[0][2]=( A[0][1]*A[1][2]-A[0][2]*A[1][1])/D;
        inv[1][0]=(-A[1][0]*A[2][2]+A[1][2]*A[2][0])/D;
        inv[1][1]=( A[0][0]*A[2][2]-A[0][2]*A[2][0])/D;
        inv[1][2]=(-A[0][0]*A[1][2]+A[0][2]*A[1][0])/D;
        inv[2][0]=( A[1][0]*A[2][1]-A[1][1]*A[2][0])/D;
        inv[2][1]=(-A[0][0]*A[2][1]+A[0][1]*A[2][0])/D;
        inv[2][2]=( A[0][0]*A[1][1]-A[0][1]*A[1][0])/D;

        double beta[3]={0,0,0};
        for(int a=0;a<3;++a) for(int b=0;b<3;++b) beta[a]+=inv[a][b]*Xty[b];

        // decisão de exercício: se payoff >= valor de continuação estimado -> exerce
        for(int i=0;i<N;++i){
            if (t>=t_ex[i]) continue; // já exercido
            double S = S_med[i][t];
            double cont = beta[0] + beta[1]*S + beta[2]*S*S;
            if (exer[i][t] >= cont){
                cf[i]  = exer[i][t];
                t_ex[i]= t;
            }
        }
    }

    // preço no tempo 0: média dos cf descontados ao tempo 0
    double soma=0.0;
    for(int i=0;i<N;++i){
        double tau = (double)t_ex[i]*(p.maturidade/n_passos);
        soma += cf[i]*exp(-p.taxa_juros*tau);
    }
    return soma/N;
}

// -------------------- Europeias (MC + antitético + variate de controle) --------------------
struct ResultadoEuropeia {
    double preco;
    double delta;  // pathwise
    double erro_padrao;
};
ResultadoEuropeia preco_europeia_MC(const ParametrosOpcao& p, TipoOpcao tipo, int n_amostras, int n_passos){
    auto cam = simular_gbm(p, n_amostras, n_passos);
    double dt=p.maturidade/n_passos;
    double disc=exp(-p.taxa_juros*p.maturidade);

    // Variate de controle: Black-Scholes de CALL europeia (se tipo==Call)
    double bs_call = preco_black_scholes_call(p);

    vector<double> estimadores; estimadores.reserve(n_amostras);
    vector<double> deltas;      deltas.reserve(n_amostras);

    for(int i=0;i<n_amostras;++i){
        double ST  = cam.caminho_principal[i].back();
        double STa = cam.caminho_antitetico[i].back();
        double pay1 = payoff(tipo, ST,  p.strike);
        double pay2 = payoff(tipo, STa, p.strike);
        double pay = 0.5*(pay1+pay2);

        // Pathwise delta (europeia): d payoff/dS0 via cadeia (para call: 1_{ST>K} * ST/S0)
        // Com antitético, média das duas.
        auto delta_path = [&](double ST_){ 
            if (tipo==TipoOpcao::Call) return (ST_>p.strike ? ST_/p.preco_inicial : 0.0);
            // Para put, pathwise simples é mais instável; aqui fica uma aproximação (0 se OTM, -ST/S0 se ITM)
            return (ST_<p.strike ? -ST_/p.preco_inicial : 0.0);
        };
        double delta_i = 0.5*(delta_path(ST)+delta_path(STa)) * disc;

        // Controle: se call, ajusta por (pay_call - (pay_call_MC_ref - bs_call))
        double estim = disc*pay;
        if (tipo==TipoOpcao::Call){
            // referência MC simples do controle: payoff_call_puro = max(ST-K,0) (já é "pay")
            // ajuste: estimador_controlado = estim - (pay - E[pay]) + preço_fechado
            estim = estim - disc*pay + bs_call;
        }

        estimadores.push_back(estim);
        deltas.push_back(delta_i);
    }

    auto media = [&](const vector<double>& v){
        long double s=0; for(double x:v) s+=x; return (double)(s/v.size());
    };
    auto var = [&](const vector<double>& v, double m){
        long double s=0; for(double x:v){ long double d=x-m; s+=d*d; } return (double)(s/(v.size()-1));
    };

    double m=media(estimadores);
    double s2=var(estimadores,m);
    double ep = sqrt(s2/n_amostras);

    double delta = media(deltas);

    return {m, delta, ep};
}

// -------------------- Demonstração --------------------
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    ParametrosOpcao p{
        /*preco_inicial*/ 100.0,
        /*strike       */ 100.0,
        /*taxa_juros   */ 0.05,
        /*volatilidade */ 0.20,
        /*maturidade   */ 1.00
    };

    int n_amostras = 20000;
    int n_passos   = 50;

    // Europeia CALL (MC + antitético + variate de controle) — preço e delta
    auto res_call = preco_europeia_MC(p, TipoOpcao::Call, n_amostras, n_passos);

    // Americana PUT (LSM)
    double preco_put_amer = preco_americana_LSM_put(p, n_amostras, n_passos);

    cout.setf(std::ios::fixed); cout<<setprecision(6);
    cout << "=== EUROPEIA CALL (MC+Antitetico+Controle) ===\n";
    cout << "Preco  : " << res_call.preco << "  (EP ~ " << res_call.erro_padrao << ")\n";
    cout << "Delta  : " << res_call.delta << "\n\n";

    cout << "=== AMERICANA PUT (LSM) ===\n";
    cout << "Preco  : " << preco_put_amer << "\n";
    return 0;
}
