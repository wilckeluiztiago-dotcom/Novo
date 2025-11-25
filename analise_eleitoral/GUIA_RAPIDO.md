# 🗳️ Sistema de Análise Eleitoral - Guia de Uso Rápido

## 🚀 Início Rápido

### 1. Executar o Dashboard

```bash
cd /home/luiztiagowilcke188/Área\ de\ trabalho/Projetos/analise_eleitoral
streamlit run dashboard/app.py
```

O dashboard abrirá automaticamente em: **http://localhost:8501**

### 2. Executar Testes

```bash
cd /home/luiztiagowilcke188/Área\ de\ trabalho/Projetos/analise_eleitoral
python3 test_sistema.py
```

## 📊 Funcionalidades Principais

### Dashboard - 7 Seções

1. **📊 Visão Geral** - Métricas e gráficos principais
2. **🤖 Modelos Preditivos** - Regressão, Random Forest, Bayesiano
3. **🤝 Coligações** - Análise de eficiência e sobras
4. **📈 Volatilidade** - Índice de Pedersen
5. **🔀 Fragmentação** - NEP, HHI, concentração
6. **⚔️ Competitividade** - Margem de vitória, renovação
7. **🎯 Simulador** - Cenários eleitorais personalizados

## 💻 Exemplo de Código

```python
from utils.dados import gerar_dados_eleitorais
from modelos.basicos import ModeloRegressao
from modelos.eleitorais import QuocienteEleitoral

# Gerar dados
dados = gerar_dados_eleitorais(n_candidatos=500, ano=2026)

# Modelo de Regressão
X = dados[['gasto_campanha', 'tempo_tv_segundos', 'incumbente']].values
y = dados['votos'].values

modelo = ModeloRegressao()
modelo.treinar(X, y, features_nomes=['Gastos', 'Tempo TV', 'Incumbente'])
print(f"R² = {modelo.obter_r2(X, y):.4f}")

# Quociente Eleitoral
votos = dados.groupby('partido')['votos'].sum()
qe = QuocienteEleitoral()
resultado = qe.calcular_distribuicao(votos, n_cadeiras=50)
print(resultado.head())
```

## 🔧 Configurações

### Sidebar do Dashboard
- **Ano**: 2026, 2022, 2018, 2014, 2010
- **Tipo**: Deputado Federal ou Estadual
- **Estado**: Todos ou específico

### Simulador
- **Candidatos**: 100-1000
- **Cadeiras**: 10-100
- **Coligações**: Sim/Não

## ✅ Status

- ✅ Todos os módulos funcionando
- ✅ Dashboard operacional
- ✅ Erros de importação corrigidos
- ✅ Testes executados com sucesso

## 📚 Documentação Completa

Veja [README.md](file:///home/luiztiagowilcke188/Área%20de%20trabalho/Projetos/analise_eleitoral/README.md) para documentação detalhada de todas as equações e métodos.
