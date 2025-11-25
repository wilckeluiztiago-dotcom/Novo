"""
Módulo de Identificação Bacteriana baseado em características do genoma.
"""

from identificacao.banco_expandido import BACTERIAS_EXPANDIDO

class BancoDadosBacterias:
    """
    Banco de dados com características genômicas de bactérias comuns.
    """
    
    BACTERIAS = BACTERIAS_EXPANDIDO


class IdentificadorBacteriano:
    """
    Identifica possíveis bactérias baseado nas características do genoma montado.
    """
    
    def __init__(self):
        self.banco = BancoDadosBacterias.BACTERIAS
    
    def calcular_score_similaridade(self, tamanho_genoma, gc_medio):
        """
        Calcula score de similaridade para cada bactéria no banco de dados.
        Retorna lista ordenada por score (maior = mais similar).
        """
        candidatos = []
        
        for bacteria in self.banco:
            # Score baseado em tamanho do genoma
            tam_min, tam_max = bacteria["tamanho_genoma"]
            if tam_min <= tamanho_genoma <= tam_max:
                score_tamanho = 1.0
            else:
                # Penaliza proporcionalmente à distância
                centro = (tam_min + tam_max) / 2
                distancia = abs(tamanho_genoma - centro) / centro
                score_tamanho = max(0, 1.0 - distancia)
            
            # Score baseado em conteúdo GC
            gc_min, gc_max = bacteria["conteudo_gc"]
            if gc_min <= gc_medio <= gc_max:
                score_gc = 1.0
            else:
                # Penaliza proporcionalmente à distância
                centro_gc = (gc_min + gc_max) / 2
                distancia_gc = abs(gc_medio - centro_gc) / 10  # Normalizado
                score_gc = max(0, 1.0 - distancia_gc)
            
            # Score total (média ponderada)
            score_total = (score_tamanho * 0.6 + score_gc * 0.4) * 100
            
            candidatos.append({
                "bacteria": bacteria,
                "score": score_total,
                "match_tamanho": score_tamanho > 0.8,
                "match_gc": score_gc > 0.8
            })
        
        # Ordenar por score (maior primeiro)
        candidatos.sort(key=lambda x: x["score"], reverse=True)
        
        return candidatos
    
    def identificar(self, tamanho_genoma, gc_medio, top_n=5):
        """
        Identifica as top N bactérias mais prováveis.
        """
        candidatos = self.calcular_score_similaridade(tamanho_genoma, gc_medio)
        return candidatos[:top_n]
    
    def gerar_relatorio(self, tamanho_genoma, gc_medio):
        """
        Gera relatório textual de identificação.
        """
        candidatos = self.identificar(tamanho_genoma, gc_medio, top_n=5)
        
        relatorio = "=" * 70 + "\n"
        relatorio += "IDENTIFICAÇÃO BACTERIANA BASEADA EM CARACTERÍSTICAS GENÔMICAS\n"
        relatorio += "=" * 70 + "\n\n"
        
        relatorio += f"Características do Genoma Montado:\n"
        relatorio += f"  • Tamanho estimado: {tamanho_genoma:,.0f} bp ({tamanho_genoma/1e6:.2f} Mb)\n"
        relatorio += f"  • Conteúdo GC médio: {gc_medio:.1f}%\n\n"
        
        relatorio += "Possíveis Candidatos (ordenados por similaridade):\n"
        relatorio += "-" * 70 + "\n\n"
        
        for i, candidato in enumerate(candidatos, 1):
            bact = candidato["bacteria"]
            score = candidato["score"]
            
            relatorio += f"{i}. {bact['nome']} (Score: {score:.1f}%)\n"
            relatorio += f"   Descrição: {bact['descricao']}\n"
            relatorio += f"   Tamanho esperado: {bact['tamanho_genoma'][0]/1e6:.1f}-{bact['tamanho_genoma'][1]/1e6:.1f} Mb\n"
            relatorio += f"   GC esperado: {bact['conteudo_gc'][0]}-{bact['conteudo_gc'][1]}%\n"
            relatorio += f"   Aplicações: {bact['aplicacoes']}\n"
            relatorio += f"   Patogenicidade: {bact['patogenicidade']}\n"
            
            # Indicadores de match
            if candidato["match_tamanho"] and candidato["match_gc"]:
                relatorio += f"   ✅ ALTA COMPATIBILIDADE (tamanho e GC)\n"
            elif candidato["match_tamanho"]:
                relatorio += f"   ⚠️ Compatível em tamanho\n"
            elif candidato["match_gc"]:
                relatorio += f"   ⚠️ Compatível em GC\n"
            
            relatorio += "\n"
        
        relatorio += "=" * 70 + "\n"
        relatorio += "NOTA: Esta é uma identificação preliminar baseada apenas em\n"
        relatorio += "características genômicas básicas. Para identificação precisa,\n"
        relatorio += "utilize métodos de sequenciamento de genes marcadores (16S rRNA)\n"
        relatorio += "ou análise filogenética completa.\n"
        relatorio += "=" * 70 + "\n"
        
        return relatorio
