/***********************************************************************
 * Algoritmo Evolutivo Bayesiano Adaptativo — (Autor: Luiz Tiago Wilcke)
 * 
 * Objetivo:
 *   Otimizar funções matemáticas complexas (multimodais e não lineares)
 *   combinando heurísticas genéticas, regressão bayesiana e aprendizado
 *   adaptativo de parâmetros (taxa de mutação, elite, etc.).
 * 
 * Recursos avançados:
 *   - Programação paralela com std::async
 *   - Geração pseudoaleatória Mersenne Twister
 *   - Estatísticas de convergência
 *   - Ajuste automático via inferência bayesiana simples
 ***********************************************************************/

#include <bits/stdc++.h>
#include <future>
#include <random>
#include <chrono>
using namespace std;

// =================== Estruturas principais ===================
struct Individuo {
    vector<double> genes;
    double aptidao;
};

struct Configuracao {
    int tamanhoPop = 200;
    int geracoes = 300;
    double taxaMutacao = 0.1;
    double taxaCruzamento = 0.75;
    int elite = 5;
    double minimo = -10, maximo = 10;
};

// =================== Função objetivo (personalizável) ===================
double funcao_objetivo(const vector<double>& x) {
    // Exemplo: função multimodal de Rastrigin
    double soma = 0.0;
    for (double xi : x)
        soma += xi * xi - 10.0 * cos(2 * M_PI * xi) + 10.0;
    return -soma; // queremos maximizar
}

// =================== Geradores e utilidades ===================
random_device rd;
mt19937 rng(rd());

double aleatorio(double a, double b) {
    uniform_real_distribution<double> dist(a, b);
    return dist(rng);
}

vector<double> gerarGenesAleatorios(int dimensao, double min, double max) {
    vector<double> v(dimensao);
    for (double &x : v) x = aleatorio(min, max);
    return v;
}

// =================== Núcleo genético ===================
Individuo gerarIndividuo(int dimensao, const Configuracao& cfg) {
    Individuo ind;
    ind.genes = gerarGenesAleatorios(dimensao, cfg.minimo, cfg.maximo);
    ind.aptidao = funcao_objetivo(ind.genes);
    return ind;
}

vector<Individuo> gerarPopulacaoInicial(int n, int dimensao, const Configuracao& cfg) {
    vector<future<Individuo>> tarefas;
    for (int i = 0; i < n; i++)
        tarefas.push_back(async(launch::async, gerarIndividuo, dimensao, cref(cfg)));

    vector<Individuo> populacao;
    for (auto& t : tarefas)
        populacao.push_back(t.get());
    return populacao;
}

// =================== Operadores genéticos ===================
Individuo cruzar(const Individuo& pai, const Individuo& mae, const Configuracao& cfg) {
    Individuo filho;
    filho.genes.resize(pai.genes.size());
    for (size_t i = 0; i < pai.genes.size(); ++i) {
        double alfa = aleatorio(0, 1);
        filho.genes[i] = alfa * pai.genes[i] + (1 - alfa) * mae.genes[i];
    }
    filho.aptidao = funcao_objetivo(filho.genes);
    return filho;
}

void mutar(Individuo& ind, const Configuracao& cfg) {
    for (double& g : ind.genes) {
        if (aleatorio(0, 1) < cfg.taxaMutacao)
            g += aleatorio(-1, 1);
    }
    ind.aptidao = funcao_objetivo(ind.genes);
}

// =================== Ajuste Bayesiano de Parâmetros ===================
void ajustarParametrosBayesianos(Configuracao& cfg, const vector<double>& historico) {
    if (historico.size() < 2) return;
    double media = accumulate(historico.begin(), historico.end(), 0.0) / historico.size();
    double variancia = 0.0;
    for (double v : historico) variancia += pow(v - media, 2);
    variancia /= historico.size();
    double incerteza = sqrt(variancia);

    // Ajuste adaptativo: reduz mutação se convergindo, aumenta se estagnado
    if (incerteza < 0.01)
        cfg.taxaMutacao = max(0.02, cfg.taxaMutacao * 0.9);
    else
        cfg.taxaMutacao = min(0.5, cfg.taxaMutacao * 1.1);
}

// =================== Loop principal ===================
int main() {
    ios::sync_with_stdio(false);
    cout << fixed << setprecision(5);

    Configuracao cfg;
    int dimensao = 5;
    auto populacao = gerarPopulacaoInicial(cfg.tamanhoPop, dimensao, cfg);

    vector<double> historicoMelhores;
    double melhorGlobal = -1e18;
    vector<double> genesMelhores;

    for (int g = 1; g <= cfg.geracoes; g++) {
        sort(populacao.begin(), populacao.end(), [](auto &a, auto &b){ return a.aptidao > b.aptidao; });

        melhorGlobal = max(melhorGlobal, populacao.front().aptidao);
        genesMelhores = populacao.front().genes;
        historicoMelhores.push_back(populacao.front().aptidao);

        // Ajuste adaptativo
        ajustarParametrosBayesianos(cfg, historicoMelhores);

        vector<Individuo> novaPopulacao;
        for (int i = 0; i < cfg.elite; i++) novaPopulacao.push_back(populacao[i]);

        while ((int)novaPopulacao.size() < cfg.tamanhoPop) {
            const Individuo &pai = populacao[rand() % cfg.elite];
            const Individuo &mae = populacao[rand() % cfg.elite];
            Individuo filho = cruzar(pai, mae, cfg);
            mutar(filho, cfg);
            novaPopulacao.push_back(filho);
        }
        populacao.swap(novaPopulacao);

        if (g % 10 == 0)
            cout << "Geração " << g << " | Melhor aptidão: " << melhorGlobal
                 << " | Mutação: " << cfg.taxaMutacao << "\n";
    }

    cout << "\nMelhor solução encontrada:\n";
    for (double v : genesMelhores) cout << v << " ";
    cout << "\nAptidão final: " << melhorGlobal << endl;
    return 0;
}
