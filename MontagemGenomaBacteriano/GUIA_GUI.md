# 🧬 Guia de Uso - Interface Gráfica do Montador de Genoma

## Como Executar

```bash
cd /home/luiztiagowilcke188/Área\ de\ trabalho/Projetos/MontagemGenomaBacteriano
source /home/luiztiagowilcke188/Área\ de\ trabalho/Projetos/.venv/bin/activate
python app_gui.py
```

## Interface

A aplicação possui **4 abas principais**:

### ⚙️ Aba 1: Configuração

**Seleção de Arquivo:**
- Clique em "Selecionar FASTQ" para escolher seu arquivo de entrada
- Formatos suportados: `.fastq`, `.fq`

**Parâmetros Ajustáveis:**
- **Tamanho do K-mer** (15-51): Tamanho das subsequências para o grafo
  - Valores ímpares são recomendados
  - K=31 é padrão para genomas bacterianos
  
- **Cobertura Mínima** (1-20): Limiar para filtrar erros
  - K-mers com cobertura abaixo são descartados
  - Valor 5 é padrão
  
- **Qualidade Mínima Phred** (10-40): Score mínimo de qualidade
  - Q20 = 99% acurácia
  - Q30 = 99.9% acurácia

### ▶️ Aba 2: Execução

**Iniciar Montagem:**
1. Clique no botão "🚀 INICIAR MONTAGEM"
2. Acompanhe o progresso em tempo real no log
3. A barra de progresso indica que o processo está ativo

**Log de Execução:**
- Mostra cada etapa do processo
- Timestamps para cada operação
- Mensagens de erro (se houver)

### 📊 Aba 3: Resultados

**Métricas Exibidas:**
- **N50**: Métrica de qualidade da montagem
- **L50**: Número de contigs para atingir N50
- **Maior Contig**: Tamanho do maior contig gerado
- **Total de Contigs**: Quantidade total montada
- **Lambda**: Cobertura média estimada

**Gráfico:**
- Histograma da distribuição de cobertura
- Visualização interativa
- Cores personalizadas

### 🧬 Aba 4: Contigs

**Visualização:**
- Lista de todos os contigs gerados
- Formato FASTA com cabeçalhos informativos
- Mostra primeiros 100 contigs

**Ações:**
- **💾 Exportar FASTA**: Salva todos os contigs em arquivo
- **📋 Copiar Sequência**: Copia sequência selecionada

## Recursos Visuais

✨ **Design Moderno:**
- Tema escuro profissional
- Cores vibrantes (#89b4fa, #a6e3a1)
- Tipografia clara (Segoe UI, Consolas)

🎨 **Elementos Interativos:**
- Sliders para ajuste de parâmetros
- Valores em tempo real
- Botões com feedback visual

📈 **Gráficos Integrados:**
- Matplotlib embutido
- Atualização automática
- Estilo personalizado

## Dicas de Uso

1. **Primeiro Uso**: Use o arquivo `exemplo.fastq` para testar
2. **Parâmetros**: Comece com valores padrão
3. **Performance**: Arquivos grandes podem demorar alguns minutos
4. **Resultados**: Sempre exporte os contigs após a montagem

## Atalhos

- A aplicação roda em thread separada (não trava a interface)
- Logs são salvos automaticamente
- Resultados podem ser exportados a qualquer momento

## Troubleshooting

**Erro ao abrir arquivo:**
- Verifique se o arquivo é FASTQ válido
- Confirme o caminho completo

**Montagem muito lenta:**
- Reduza o tamanho do K-mer
- Aumente a cobertura mínima

**Poucos contigs:**
- Diminua a cobertura mínima
- Verifique a qualidade dos reads
