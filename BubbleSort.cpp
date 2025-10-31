/***********************************************************************
 * Bubble Sort — Versão Sofisticada e Didática
 * Autor: Luiz Tiago Wilcke (LT)
 *
 * Recursos:
 *  - Variáveis em português
 *  - Visualização do processo com cores ANSI
 *  - Medição de tempo com std::chrono
 *  - Contagem de comparações e trocas
 *  - Geração aleatória de vetor
 ***********************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <thread>

using namespace std;
using namespace chrono;

// Função para gerar um vetor aleatório
vector<int> gerarVetorAleatorio(int tamanho, int minimo = 1, int maximo = 100) {
    random_device rd;
    mt19937 gerador(rd());
    uniform_int_distribution<int> distribuicao(minimo, maximo);

    vector<int> vetor(tamanho);
    for (int &valor : vetor)
        valor = distribuicao(gerador);

    return vetor;
}

// Função para exibir o vetor com destaque colorido
void exibirVetor(const vector<int>& vetor, int iDestacado = -1, int jDestacado = -1) {
    for (size_t i = 0; i < vetor.size(); ++i) {
        if ((int)i == iDestacado || (int)i == jDestacado)
            cout << "\033[1;36m" << setw(4) << vetor[i] << "\033[0m";
        else
            cout << setw(4) << vetor[i];
    }
    cout << endl;
}

// Algoritmo Bubble Sort sofisticado
void bubbleSort(vector<int>& vetor, bool mostrarProcesso = true) {
    int n = vetor.size();
    long long comparacoes = 0, trocas = 0;

    auto inicio = high_resolution_clock::now();

    for (int i = 0; i < n - 1; ++i) {
        bool houveTroca = false;

        for (int j = 0; j < n - i - 1; ++j) {
            comparacoes++;

            if (vetor[j] > vetor[j + 1]) {
                swap(vetor[j], vetor[j + 1]);
                trocas++;
                houveTroca = true;
            }

            if (mostrarProcesso) {
                exibirVetor(vetor, j, j + 1);
                this_thread::sleep_for(50ms);
            }
        }

        if (!houveTroca) break; // já está ordenado
    }

    auto fim = high_resolution_clock::now();
    auto duracao = duration_cast<milliseconds>(fim - inicio).count();

    cout << "\n\033[1;33m=== Estatísticas do Bubble Sort ===\033[0m\n";
    cout << "Tamanho do vetor: " << n << endl;
    cout << "Comparações:      " << comparacoes << endl;
    cout << "Trocas:           " << trocas << endl;
    cout << "Tempo total:      " << duracao << " ms\n";
}

// Programa principal
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int tamanho;
    cout << "Digite o tamanho do vetor: ";
    cin >> tamanho;

    vector<int> vetor = gerarVetorAleatorio(tamanho);
    cout << "\n\033[1;33mVetor original:\033[0m\n";
    exibirVetor(vetor);

    cout << "\n\033[1;32mIniciando Bubble Sort...\033[0m\n";
    bubbleSort(vetor, true);

    cout << "\n\033[1;33mVetor ordenado:\033[0m\n";
    exibirVetor(vetor);

    return 0;
}
