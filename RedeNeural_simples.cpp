// rede_neural_sophisticada.cpp
// Autor: Luiz Tiago Wilcke (LT)
// Recursos: Tensor2D, Camada Densa, ReLU/Sigmóide/Tanh, MSE, CrossEntropy+Softmax estável,
// Otimizador Adam (com L2 e grad clipping), mini-batch, salvar/carregar, checagem de gradiente.
// Exemplo: aprender XOR (classificação 2 classes, one-hot).

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

// ============================ Utilidades aleatórias ============================
static uint64_t semente_global = 42;
static std::mt19937_64 gerador(semente_global);

inline double randn() { // Normal(0,1) via Box–Muller
    static std::uniform_real_distribution<double> U(0.0,1.0);
    double u1 = std::max(U(gerador), 1e-12), u2 = U(gerador);
    return std::sqrt(-2.0*std::log(u1)) * std::cos(2.0*M_PI*u2);
}
inline double randu(double a=0.0, double b=1.0){
    std::uniform_real_distribution<double> U(a,b);
    return U(gerador);
}
inline double clampd(double x, double a, double b){ return std::max(a, std::min(b, x)); }

// ============================ Tensor 2D ============================
struct Tensor2D {
    size_t linhas{}, colunas{};
    std::vector<double> dados; // row-major

    Tensor2D() = default;
    Tensor2D(size_t L, size_t C, double v=0.0) : linhas(L), colunas(C), dados(L*C, v) {}

    inline double& operator()(size_t i, size_t j){ return dados[i*colunas + j]; }
    inline const double& operator()(size_t i, size_t j) const { return dados[i*colunas + j]; }

    static Tensor2D zeros(size_t L, size_t C){ return Tensor2D(L,C,0.0); }
    static Tensor2D aleatorio_normal(size_t L, size_t C, double esc=1.0){
        Tensor2D t(L,C);
        for(auto &d: t.dados) d = randn()*esc;
        return t;
    }
    static Tensor2D aleatorio_uniforme(size_t L, size_t C, double a, double b){
        Tensor2D t(L,C);
        for(auto &d: t.dados) d = randu(a,b);
        return t;
    }

    Tensor2D transposta() const {
        Tensor2D r(colunas, linhas);
        for(size_t i=0;i<linhas;i++) for(size_t j=0;j<colunas;j++) r(j,i)=(*this)(i,j);
        return r;
    }

    // operações in-place
    Tensor2D& adicionar(const Tensor2D& B){
        assert(linhas==B.linhas && colunas==B.colunas);
        for(size_t i=0;i<dados.size();++i) dados[i]+=B.dados[i];
        return *this;
    }
    Tensor2D& subtrair(const Tensor2D& B){
        assert(linhas==B.linhas && colunas==B.colunas);
        for(size_t i=0;i<dados.size();++i) dados[i]-=B.dados[i];
        return *this;
    }
    Tensor2D& escalar(double k){
        for(auto &x: dados) x*=k; return *this;
    }

    // broadcast por linha (adiciona vetor 1xC a cada linha)
    Tensor2D& adicionar_vetor_linha(const Tensor2D& v){
        assert(v.linhas==1 && v.colunas==colunas);
        for(size_t i=0;i<linhas;i++)
            for(size_t j=0;j<colunas;j++)
                (*this)(i,j)+=v(0,j);
        return *this;
    }

    // estáticas
    static Tensor2D multiplicar_matriz(const Tensor2D& A, const Tensor2D& B){
        assert(A.colunas==B.linhas);
        Tensor2D R(A.linhas, B.colunas, 0.0);
        for(size_t i=0;i<A.linhas;i++){
            for(size_t k=0;k<A.colunas;k++){
                double aik = A(i,k);
                const double* pb = &B.dados[k*B.colunas];
                double* pr = &R.dados[i*R.colunas];
                for(size_t j=0;j<B.colunas;j++) pr[j]+=aik*pb[j];
            }
        }
        return R;
    }
    static Tensor2D hadamard(const Tensor2D& A, const Tensor2D& B){
        assert(A.linhas==B.linhas && A.colunas==B.colunas);
        Tensor2D R(A.linhas,A.colunas);
        for(size_t i=0;i<R.dados.size();++i) R.dados[i]=A.dados[i]*B.dados[i];
        return R;
    }
    static Tensor2D soma(const Tensor2D& A, const Tensor2D& B){
        Tensor2D R=A; R.adicionar(B); return R;
    }
    static Tensor2D sub(const Tensor2D& A, const Tensor2D& B){
        Tensor2D R=A; R.subtrair(B); return R;
    }
    static Tensor2D escalar(const Tensor2D& A, double k){
        Tensor2D R=A; R.escalar(k); return R;
    }

    template<class F>
    static Tensor2D aplicar(const Tensor2D& A, F f){
        Tensor2D R=A;
        for(auto &x: R.dados) x=f(x);
        return R;
    }

    // somas por coluna (para gradiente de viés)
    Tensor2D soma_por_coluna() const {
        Tensor2D v(1, colunas, 0.0);
        for(size_t i=0;i<linhas;i++)
            for(size_t j=0;j<colunas;j++)
                v(0,j)+=(*this)(i,j);
        return v;
    }

    double media() const {
        double s=0.0; for(auto &x: dados) s+=x; return s/dados.size();
    }

    void imprimir(const string& nome, size_t maxl=5, size_t maxc=5) const {
        cout<<nome<<" ("<<linhas<<"x"<<colunas<<")\n";
        size_t L=min(linhas,maxl), C=min(colunas,maxc);
        for(size_t i=0;i<L;i++){
            for(size_t j=0;j<C;j++) cout<<fixed<<setprecision(4)<<(*this)(i,j)<<" ";
            if(C<colunas) cout<<"...";
            cout<<"\n";
        }
        if(L<linhas) cout<<"...\n";
    }
};

// ============================ Inicialização ============================
enum class TipoInicializacao { HE, GLOROT, PADRAO };

inline pair<Tensor2D, Tensor2D> inicializar_pesos_bias(size_t entrada, size_t saida,
                                                       TipoInicializacao tipo)
{
    double esc = 1.0;
    if (tipo==TipoInicializacao::HE) {
        esc = std::sqrt(2.0/static_cast<double>(entrada));
    } else if (tipo==TipoInicializacao::GLOROT) {
        esc = std::sqrt(2.0/(entrada+saida));
    } else { // PADRAO
        esc = 0.01;
    }
    Tensor2D W = Tensor2D::aleatorio_normal(entrada, saida, esc);
    Tensor2D b(1, saida, 0.0);
    return {W,b};
}

// ============================ Camadas base ============================
struct Camada {
    virtual ~Camada() = default;
    virtual Tensor2D forward(const Tensor2D& X)=0;
    virtual Tensor2D backward(const Tensor2D& grad_saida)=0;

    // parâmetros para otimizadores
    virtual bool tem_parametros() const { return false; }
    virtual vector<pair<Tensor2D*, Tensor2D*>> parametros_e_grads() { return {}; } // {param, grad}
    virtual void zerar_grads() {}
    virtual string nome() const = 0;
};

// --------- Densa (Linear) ---------
struct CamadaDensa : Camada {
    Tensor2D pesos, vieses;
    Tensor2D grad_pesos, grad_vieses;
    Tensor2D entrada_cache;
    bool usar_bias{true};

    CamadaDensa(size_t n_entrada, size_t n_saida,
                TipoInicializacao init = TipoInicializacao::GLOROT,
                bool com_bias=true)
        : usar_bias(com_bias)
    {
        auto [W,b]=inicializar_pesos_bias(n_entrada, n_saida, init);
        pesos = std::move(W);
        vieses= std::move(b);
        grad_pesos = Tensor2D::zeros(n_entrada, n_saida);
        grad_vieses = Tensor2D::zeros(1, n_saida);
    }

    Tensor2D forward(const Tensor2D& X) override {
        entrada_cache = X; // salva para o backward
        Tensor2D Y = Tensor2D::multiplicar_matriz(X, pesos);
        if (usar_bias) Y.adicionar_vetor_linha(vieses);
        return Y;
    }

    Tensor2D backward(const Tensor2D& grad_saida) override {
        // gradientes
        Tensor2D X_T = entrada_cache.transposta();
        grad_pesos = Tensor2D::multiplicar_matriz(X_T, grad_saida);
        if (usar_bias) grad_vieses = grad_saida.soma_por_coluna();

        // gradiente w.r.t entrada
        Tensor2D W_T = pesos.transposta();
        Tensor2D grad_entrada = Tensor2D::multiplicar_matriz(grad_saida, W_T);
        return grad_entrada;
    }

    bool tem_parametros() const override { return true; }
    vector<pair<Tensor2D*, Tensor2D*>> parametros_e_grads() override {
        vector<pair<Tensor2D*, Tensor2D*>> v;
        v.push_back({&pesos, &grad_pesos});
        if (usar_bias) v.push_back({&vieses, &grad_vieses});
        return v;
    }
    void zerar_grads() override {
        std::fill(grad_pesos.dados.begin(), grad_pesos.dados.end(), 0.0);
        if (usar_bias) std::fill(grad_vieses.dados.begin(), grad_vieses.dados.end(), 0.0);
    }
    string nome() const override { return "Densa"; }
};

// --------- Ativações ---------
struct AtivacaoReLU : Camada {
    Tensor2D mascara; // 1 onde x>0
    Tensor2D forward(const Tensor2D& X) override {
        mascara = Tensor2D(X.linhas, X.colunas, 0.0);
        Tensor2D Y = X;
        for(size_t i=0;i<X.dados.size();++i){
            if (X.dados[i] > 0.0) { mascara.dados[i]=1.0; } else { Y.dados[i]=0.0; }
        }
        return Y;
    }
    Tensor2D backward(const Tensor2D& grad_saida) override {
        Tensor2D grad = grad_saida;
        for(size_t i=0;i<grad.dados.size();++i) grad.dados[i] *= mascara.dados[i];
        return grad;
    }
    string nome() const override { return "ReLU"; }
};

struct AtivacaoSigmoide : Camada {
    Tensor2D saida_cache;
    Tensor2D forward(const Tensor2D& X) override {
        Tensor2D Y = Tensor2D::aplicar(X, [](double x){ return 1.0/(1.0+std::exp(-x)); });
        saida_cache = Y;
        return Y;
    }
    Tensor2D backward(const Tensor2D& grad_saida) override {
        // grad = grad_saida * y*(1-y)
        Tensor2D deriv = Tensor2D::aplicar(saida_cache, [](double y){ return y*(1.0-y); });
        return Tensor2D::hadamard(grad_saida, deriv);
    }
    string nome() const override { return "Sigmoide"; }
};

struct AtivacaoTanh : Camada {
    Tensor2D saida_cache;
    Tensor2D forward(const Tensor2D& X) override {
        Tensor2D Y = Tensor2D::aplicar(X, [](double x){ return std::tanh(x); });
        saida_cache = Y;
        return Y;
    }
    Tensor2D backward(const Tensor2D& grad_saida) override {
        Tensor2D deriv = Tensor2D::aplicar(saida_cache, [](double y){ return 1.0 - y*y; });
        return Tensor2D::hadamard(grad_saida, deriv);
    }
    string nome() const override { return "Tanh"; }
};

// ============================ Perdas ============================
// MSE: médias por elemento (batch e dimensão)
struct PerdaMSE {
    double forward(const Tensor2D& pred, const Tensor2D& alvo){
        assert(pred.linhas==alvo.linhas && pred.colunas==alvo.colunas);
        double s=0.0; size_t n=pred.dados.size();
        for(size_t i=0;i<n;i++){ double d = pred.dados[i]-alvo.dados[i]; s+= d*d; }
        return s / static_cast<double>(n);
    }
    Tensor2D backward(const Tensor2D& pred, const Tensor2D& alvo){
        Tensor2D g = Tensor2D::sub(pred, alvo);
        g.escalar(2.0/static_cast<double>(pred.dados.size()));
        return g;
    }
};

// Softmax estável + CrossEntropy (one-hot)
struct PerdaCrossEntropySoftmax {
    Tensor2D probs_cache; // softmax
    double forward(const Tensor2D& logits, const Tensor2D& alvo_onehot){
        assert(logits.linhas==alvo_onehot.linhas && logits.colunas==alvo_onehot.colunas);
        size_t B = logits.linhas, C = logits.colunas;
        probs_cache = Tensor2D(B,C,0.0);
        double perda=0.0;
        for(size_t i=0;i<B;i++){
            // estabilizar
            double maxv = -std::numeric_limits<double>::infinity();
            for(size_t j=0;j<C;j++) maxv = std::max(maxv, logits(i,j));
            double soma=0.0;
            for(size_t j=0;j<C;j++){ probs_cache(i,j) = std::exp(logits(i,j)-maxv); soma+=probs_cache(i,j); }
            for(size_t j=0;j<C;j++){ probs_cache(i,j) /= soma; }
            // -sum y*log(p)
            for(size_t j=0;j<C;j++){
                if (alvo_onehot(i,j) > 0.0){
                    double p = clampd(probs_cache(i,j), 1e-15, 1.0);
                    perda += -std::log(p);
                }
            }
        }
        return perda/static_cast<double>(B);
    }
    Tensor2D backward(const Tensor2D& logits, const Tensor2D& alvo_onehot){
        // grad = (p - y)/B
        Tensor2D g = Tensor2D::sub(probs_cache, alvo_onehot);
        double esc = 1.0/static_cast<double>(logits.linhas);
        g.escalar(esc);
        return g;
    }
};

// ============================ Otimizador Adam ============================
struct Adam {
    double taxa{0.001}, beta1{0.9}, beta2{0.999}, eps{1e-8}, l2{0.0};
    double clip_grad_abs{0.0}; // 0 = desliga
    uint64_t t{0};

    // Estado por parâmetro: m e v
    struct Estado { Tensor2D m, v; };
    vector<Estado> estados; // acompanha a ordem de parametros_e_grads()

    void anexar_parametros(const vector<pair<Tensor2D*, Tensor2D*>>& params){
        estados.clear();
        estados.reserve(params.size());
        for (auto &pg : params){
            auto* p = pg.first;
            estados.push_back({ Tensor2D(p->linhas, p->colunas, 0.0),
                                Tensor2D(p->linhas, p->colunas, 0.0) });
        }
    }

    void passo(const vector<pair<Tensor2D*, Tensor2D*>>& params){
        t++;
        double b1t = 1.0 - std::pow(beta1, (double)t);
        double b2t = 1.0 - std::pow(beta2, (double)t);

        for(size_t idx=0; idx<params.size(); ++idx){
            Tensor2D& P  = *params[idx].first;
            Tensor2D& G  = *params[idx].second;
            Estado& E = estados[idx];

            for(size_t k=0;k<P.dados.size();++k){
                double g = G.dados[k];
                // L2
                if (l2>0.0) g += l2 * P.dados[k];
                // clipping
                if (clip_grad_abs>0.0) g = clampd(g, -clip_grad_abs, clip_grad_abs);

                // momentos
                E.m.dados[k] = beta1*E.m.dados[k] + (1.0-beta1)*g;
                E.v.dados[k] = beta2*E.v.dados[k] + (1.0-beta2)*g*g;

                double m_hat = E.m.dados[k] / b1t;
                double v_hat = E.v.dados[k] / b2t;

                P.dados[k] -= taxa * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }
};

// ============================ Modelo Sequencial ============================
struct ModeloSequencial {
    vector<unique_ptr<Camada>> camadas;

    template<class T, class... Args>
    void adicionar(Args&&... args){
        camadas.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    }

    Tensor2D forward(const Tensor2D& X){
        Tensor2D out = X;
        for(auto &c: camadas) out = c->forward(out);
        return out;
    }

    Tensor2D backward(const Tensor2D& grad){
        Tensor2D g = grad;
        for(int i=(int)camadas.size()-1;i>=0;--i) g = camadas[i]->backward(g);
        return g; // gradiente na entrada (normalmente descartado)
    }

    vector<pair<Tensor2D*, Tensor2D*>> coletar_parametros(){
        vector<pair<Tensor2D*, Tensor2D*>> v;
        for(auto &c: camadas){
            if (c->tem_parametros()){
                auto vv = c->parametros_e_grads();
                v.insert(v.end(), vv.begin(), vv.end());
            }
        }
        return v;
    }
    void zerar_grads(){
        for(auto &c: camadas) c->zerar_grads();
    }

    // Salvar/Carregar (formato simples de texto)
    void salvar(const string& caminho){
        ofstream f(caminho);
        if(!f){ cerr<<"[Aviso] Não consegui abrir "<<caminho<<" para salvar.\n"; return; }
        // salvamos apenas camadas densas
        for(auto &c: camadas){
            if (c->nome()=="Densa"){
                auto* d = dynamic_cast<CamadaDensa*>(c.get());
                f<<"Densa "<<d->pesos.linhas<<" "<<d->pesos.colunas<<" "<<(d->usar_bias?1:0)<<"\n";
                for(double x: d->pesos.dados) f<<setprecision(17)<<x<<" ";
                f<<"\n";
                if (d->usar_bias){
                    for(double x: d->vieses.dados) f<<setprecision(17)<<x<<" ";
                    f<<"\n";
                }
            } else {
                f<<c->nome()<<"\n";
            }
        }
        f.close();
    }

    // Observação: Carregar assume MESMA arquitetura (ordem das camadas).
    void carregar(const string& caminho){
        ifstream f(caminho);
        if(!f){ cerr<<"[Aviso] Não consegui abrir "<<caminho<<" para carregar.\n"; return; }
        string tipo;
        size_t idx_densa=0;
        for(auto &c: camadas){
            if(!(f>>tipo)) break;
            if (tipo=="Densa" && c->nome()=="Densa"){
                size_t lin, col; int usa;
                f>>lin>>col>>usa;
                auto* d = dynamic_cast<CamadaDensa*>(c.get());
                if (d->pesos.linhas!=lin || d->pesos.colunas!=col){
                    cerr<<"[Aviso] Dimensão incompatível ao carregar.\n";
                }
                for(size_t k=0;k<d->pesos.dados.size();++k) f>>d->pesos.dados[k];
                if (d->usar_bias && usa==1){
                    for(size_t k=0;k<d->vieses.dados.size();++k) f>>d->vieses.dados[k];
                }
                idx_densa++;
            } else {
                // ativações não têm parâmetros; somente consome a linha
            }
        }
        f.close();
    }
};

// ============================ Checagem Numérica de Gradientes ============================
double checar_gradientes(ModeloSequencial& modelo,
                         PerdaCrossEntropySoftmax& perda,
                         const Tensor2D& X, const Tensor2D& Y,
                         double eps=1e-5, size_t amostras_max=50)
{
    // Forward e grads "analíticos"
    Tensor2D logits = modelo.forward(X);
    (void)perda.forward(logits, Y);
    Tensor2D grad_top = perda.backward(logits, Y);
    modelo.backward(grad_top);

    auto params = modelo.coletar_parametros();

    // Escolhe até amostras_max elementos para comparar
    std::uniform_int_distribution<size_t> Uparam(0, params.size()-1);

    double erro_max=0.0;
    for(size_t s=0; s<amostras_max; ++s){
        size_t ip = Uparam(gerador);
        Tensor2D* P = params[ip].first;
        Tensor2D* G = params[ip].second;
        std::uniform_int_distribution<size_t> Uidx(0, P->dados.size()-1);
        size_t k = Uidx(gerador);

        double valor = P->dados[k];
        // f(x+eps)
        P->dados[k] = valor + eps;
        double L1 = perda.forward(modelo.forward(X), Y);
        // f(x-eps)
        P->dados[k] = valor - eps;
        double L2 = perda.forward(modelo.forward(X), Y);
        // restaura
        P->dados[k] = valor;

        double grad_num = (L1 - L2)/(2.0*eps);
        double grad_ana = (*G).dados[k];

        double denom = std::max(1.0, std::abs(grad_num) + std::abs(grad_ana));
        double erro_rel = std::abs(grad_num - grad_ana) / denom;
        erro_max = std::max(erro_max, erro_rel);
    }
    return erro_max;
}

// ============================ Treinador ============================
struct Treinador {
    ModeloSequencial& modelo;
    Adam& otimizador;

    Treinador(ModeloSequencial& m, Adam& opt) : modelo(m), otimizador(opt){}

    template<class Perda>
    void treinar(const Tensor2D& X, const Tensor2D& Y,
                 Perda& perda,
                 size_t epocas, size_t tamanho_lote=16,
                 bool embaralhar=true,
                 bool mostrar_log=true)
    {
        auto params = modelo.coletar_parametros();
        otimizador.anexar_parametros(params);

        vector<size_t> indices(X.linhas);
        iota(indices.begin(), indices.end(), 0);

        for(size_t ep=1; ep<=epocas; ++ep){
            if (embaralhar) std::shuffle(indices.begin(), indices.end(), gerador);

            double perda_ep=0.0;
            size_t n_batches = 0;

            for(size_t ini=0; ini<X.linhas; ini+=tamanho_lote){
                size_t fim = std::min(ini+tamanho_lote, X.linhas);
                size_t B = fim - ini;
                // monta mini-batch
                Tensor2D xb(B, X.colunas), yb(B, Y.colunas);
                for(size_t r=0; r<B; ++r){
                    size_t idx = indices[ini+r];
                    for(size_t c=0;c<X.colunas;c++) xb(r,c)=X(idx,c);
                    for(size_t c=0;c<Y.colunas;c++) yb(r,c)=Y(idx,c);
                }

                // forward
                Tensor2D logits = modelo.forward(xb);
                double L = perda.forward(logits, yb);
                perda_ep += L; n_batches++;

                // backward
                Tensor2D grad_top = perda.backward(logits, yb);
                modelo.zerar_grads();
                modelo.backward(grad_top);

                // passo do otimizador
                otimizador.passo(modelo.coletar_parametros());
            }

            if (mostrar_log && (ep%50==0 || ep==1)){
                cout<<"Época "<<ep<<"/"<<epocas<<" | perda média = "<<(perda_ep/max(1ul,n_batches))<<"\n";
            }
        }
    }
};

// ============================ Utilidades de dados ============================
Tensor2D one_hot(const vector<int>& rotulos, int n_classes){
    Tensor2D Y(rotulos.size(), n_classes, 0.0);
    for(size_t i=0;i<rotulos.size();++i) Y(i, rotulos[i]) = 1.0;
    return Y;
}

// ============================ Exemplo: XOR ============================
void exemplo_xor(){
    cout<<"================= Exemplo XOR =================\n";
    // Dados XOR: entradas 2D, classes {0,1}
    Tensor2D X(4,2,0.0);
    X(0,0)=0; X(0,1)=0;
    X(1,0)=0; X(1,1)=1;
    X(2,0)=1; X(2,1)=0;
    X(3,0)=1; X(3,1)=1;

    vector<int> ylab = {0,1,1,0};
    Tensor2D Y = one_hot(ylab, 2);

    // Modelo: 2 -> 8 (ReLU) -> 2 (logits)
    ModeloSequencial modelo;
    modelo.adicionar<CamadaDensa>(2, 8, TipoInicializacao::HE, true);
    modelo.adicionar<AtivacaoReLU>();
    modelo.adicionar<CamadaDensa>(8, 2, TipoInicializacao::GLOROT, true);

    // Perda e otimizador
    PerdaCrossEntropySoftmax perda;
    Adam adam; adam.taxa=0.05; adam.l2=1e-5; adam.clip_grad_abs=5.0;

    // Checagem (rápida)
    double err = checar_gradientes(modelo, perda, X, Y, 1e-5, 30);
    cout<<"Erro relativo máximo na checagem de gradiente: "<<err<<"\n";

    // Treino
    Treinador treinador(modelo, adam);
    treinador.treinar(X, Y, perda, /*epocas*/2000, /*lote*/4, /*embaralhar*/true, /*log*/true);

    // Avaliação
    Tensor2D logits = modelo.forward(X);
    (void)perda.forward(logits, Y);
    cout<<"\nLogits finais (linhas = amostras):\n";
    logits.imprimir("logits", 10, 10);

    // Predições e acurácia
    int acertos=0;
    for(size_t i=0;i<4;i++){
        int argmax=-1; double best=-1e100;
        for(size_t j=0;j<2;j++){
            if (logits(i,j)>best){ best=logits(i,j); argmax=(int)j; }
        }
        if (argmax==ylab[i]) acertos++;
        cout<<"Entrada ["<<X(i,0)<<","<<X(i,1)<<"] -> classe "<<argmax<<" (esperado "<<ylab[i]<<")\n";
    }
    cout<<"Acurácia: "<<acertos<<"/4\n";

    // Salvar e recarregar (demonstração)
    modelo.salvar("pesos_xor.txt");
    ModeloSequencial modelo2;
    modelo2.adicionar<CamadaDensa>(2, 8, TipoInicializacao::HE, true);
    modelo2.adicionar<AtivacaoReLU>();
    modelo2.adicionar<CamadaDensa>(8, 2, TipoInicializacao::GLOROT, true);
    modelo2.carregar("pesos_xor.txt");
    Tensor2D logits2 = modelo2.forward(X);
    cout<<"\nVerificação pós-carregamento — diferença média de logits: "
        << Tensor2D::sub(logits, logits2).media() << "\n";
}

// ============================ Programa principal ============================
int main(){
    cout<<fixed<<setprecision(6);
    exemplo_xor();

    // Dica: para problemas reais, substitua exemplo_xor() por:
    //  - Prepare seus Tensor2D X (amostras x atributos) e Y (one-hot)
    //  - Defina o modelo com as camadas desejadas
    //  - Escolha PerdaCrossEntropySoftmax (classificação) ou PerdaMSE (regressão)
    //  - Ajuste Adam (taxa, l2, clip)
    //  - Treine com Treinador::treinar(...)
    return 0;
}
