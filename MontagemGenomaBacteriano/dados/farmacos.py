"""
Módulo de Farmacologia Bacteriana
Contém informações sobre antibióticos, mecanismos de ação e tratamentos para bactérias

Autor: Luiz Tiago Wilcke
Data: 2025-11-25
"""

# Classes de Antibióticos e seus dados
CLASSES_ANTIBIOTICOS = {
    "β-lactâmicos": {
        "antibioticos": [
            "Penicilina", "Ampicilina", "Amoxicilina", "Oxacilina", 
            "Ceftriaxona", "Ceftazidima", "Meropenem", "Imipenem"
        ],
        "mecanismo": "Inibição da síntese da parede celular bacteriana",
        "espectro": "Gram+ e Gram- (variável conforme geração)",
        "efeitos_colaterais": "Reações alérgicas, distúrbios gastrointestinais",
        "cor": "#4CAF50"
    },
    "Aminoglicosídeos": {
        "antibioticos": ["Gentamicina", "Amicacina", "Tobramicina", "Estreptomicina"],
        "mecanismo": "Inibição da síntese proteica (subunidade 30S do ribossomo)",
        "espectro": "Principalmente Gram-",
        "efeitos_colaterais": "Nefrotoxicidade, ototoxicidade",
        "cor": "#2196F3"
    },
    "Fluoroquinolonas": {
        "antibioticos": ["Ciprofloxacino", "Levofloxacino", "Moxifloxacino", "Norfloxacino"],
        "mecanismo": "Inibição da DNA girase e topoisomerase IV",
        "espectro": "Amplo espectro (Gram+ e Gram-)",
        "efeitos_colaterais": "Tendinite, fotossensibilidade",
        "cor": "#FF9800"
    },
    "Macrolídeos": {
        "antibioticos": ["Azitromicina", "Eritromicina", "Claritromicina"],
        "mecanismo": "Inibição da síntese proteica (subunidade 50S do ribossomo)",
        "espectro": "Principalmente Gram+, algumas Gram-",
        "efeitos_colaterais": "Distúrbios gastrointestinais, prolongamento QT",
        "cor": "#9C27B0"
    },
    "Tetraciclinas": {
        "antibioticos": ["Doxiciclina", "Tetraciclina", "Minociclina"],
        "mecanismo": "Inibição da síntese proteica (subunidade 30S do ribossomo)",
        "espectro": "Amplo espectro",
        "efeitos_colaterais": "Fotossensibilidade, descoloração dentária",
        "cor": "#FFC107"
    },
    "Glicopeptídeos": {
        "antibioticos": ["Vancomicina", "Teicoplanina"],
        "mecanismo": "Inibição da síntese da parede celular",
        "espectro": "Gram+",
        "efeitos_colaterais": "Síndrome do pescoço vermelho, nefrotoxicidade",
        "cor": "#E91E63"
    },
    "Sulfonamidas": {
        "antibioticos": ["Sulfametoxazol-Trimetoprima", "Sulfadiazina"],
        "mecanismo": "Inibição da síntese de ácido fólico",
        "espectro": "Amplo espectro",
        "efeitos_colaterais": "Reações alérgicas, cristalúria",
        "cor": "#00BCD4"
    },
    "Carbapenêmicos": {
        "antibioticos": ["Meropenem", "Imipenem", "Ertapenem", "Doripenem"],
        "mecanismo": "Inibição da síntese da parede celular",
        "espectro": "Muito amplo espectro (reserva terapêutica)",
        "efeitos_colaterais": "Convulsões (raras), reações alérgicas",
        "cor": "#F44336"
    },
    "Oxazolidinonas": {
        "antibioticos": ["Linezolida"],
        "mecanismo": "Inibição da síntese proteica (início da tradução)",
        "espectro": "Gram+",
        "efeitos_colaterais": "Supressão da medula óssea, neuropatia",
        "cor": "#673AB7"
    },
    "Polimixinas": {
        "antibioticos": ["Polimixina B", "Colistina"],
        "mecanismo": "Disrupção da membrana celular bacteriana",
        "espectro": "Gram-",
        "efeitos_colaterais": "Nefrotoxicidade, neurotoxicidade",
        "cor": "#3F51B5"
    },
}

# Tratamentos específicos por bactéria
TRATAMENTOS_BACTERIANOS = {
    "Escherichia coli": {
        "primeira_linha": ["Ciprofloxacino", "Ceftriaxona", "Amoxicilina-clavulanato"],
        "segunda_linha": ["Meropenem", "Gentamicina"],
        "resistencia_comum": ["Ampicilina", "Sulfametoxazol-Trimetoprima"],
        "mecanismos_resistencia": ["β-lactamase", "ESBL", "Resistência a quinolonas"],
        "observacoes": "Resistência crescente às fluoroquinolonas. Testes de sensibilidade essenciais."
    },
    "Staphylococcus aureus": {
        "primeira_linha": ["Oxacilina", "Cefazolina", "Vancomicina (MRSA)"],
        "segunda_linha": ["Linezolida", "Daptomicina", "Teicoplanina"],
        "resistencia_comum": ["Penicilina G"],
        "mecanismos_resistencia": ["β-lactamase", "Alteração de PBP2a (MRSA)"],
        "observacoes": "MRSA requer vancomicina ou linezolida. Verificar resistência a meticilina."
    },
    "Pseudomonas aeruginosa": {
        "primeira_linha": ["Ceftazidima", "Piperacilina-tazobactam", "Ciprofloxacino"],
        "segunda_linha": ["Meropenem", "Amicacina", "Colistina"],
        "resistencia_comum": ["Ampicilina", "Cefazolina", "Tetraciclinas"],
        "mecanismos_resistencia": ["β-lactamase", "Bombas de efluxo", "Mutações em porina"],
        "observacoes": "Resistência intrínseca a muitos antibióticos. Multirresistência frequente."
    },
    "Mycobacterium tuberculosis": {
        "primeira_linha": ["Isoniazida", "Rifampicina", "Pirazinamida", "Etambutol"],
        "segunda_linha": ["Fluoroquinolonas", "Aminoglicosídeos injetáveis", "Linezolida"],
        "resistencia_comum": ["Resistência multidrogas (MDR-TB)"],
        "mecanismos_resistencia": ["Mutações em genes katG, rpoB, pncA"],
        "observacoes": "Tratamento prolongado (6-9 meses). MDR-TB requer regime estendido."
    },
    "Klebsiella pneumoniae": {
        "primeira_linha": ["Ceftriaxona", "Ciprofloxacino"],
        "segunda_linha": ["Meropenem", "Colistina", "Tigeciclina"],
        "resistencia_comum": ["Ampicilina", "Cefazolina"],
        "mecanismos_resistencia": ["ESBL", "KPC (carbapenemase)", "Resistência a polimixinas"],
        "observacoes": "Emergência de cepas KPC multirresistentes. Testes de sensibilidade críticos."
    },
    "Streptococcus pneumoniae": {
        "primeira_linha": ["Penicilina G", "Amoxicilina", "Ceftriaxona"],
        "segunda_linha": ["Vancomicina", "Levofloxacino"],
        "resistencia_comum": ["Resistência intermediária à penicilina"],
        "mecanismos_resistencia": ["Alterações em PBPs"],
        "observacoes": "Vacinação reduz incidência. Resistência moderada à penicilina em algumas regiões."
    },
    "Acinetobacter baumannii": {
        "primeira_linha": ["Meropenem", "Imipenem", "Amicacina-sulbactam"],
        "segunda_linha": ["Colistina", "Tigeciclina"],
        "resistencia_comum": ["Maioria dos β-lactâmicos", "Fluoroquinolonas"],
        "mecanismos_resistencia": ["OXA carbapenemases", "Bombas de efluxo", "Perda de porinas"],
        "observacoes": "Patógeno extremamente resistente. Infecções hospitalares graves."
    },
    "Enterococcus faecalis": {
        "primeira_linha": ["Ampicilina", "Vancomicina"],
        "segunda_linha": ["Linezolida", "Daptomicina"],
        "resistencia_comum": ["Cefalosporinas", "Aminoglicosídeos (baixos níveis)"],
        "mecanismos_resistencia": ["VanA/VanB (VRE)", "Resistência intrínseca"],
        "observacoes": "VRE (Enterococcus resistente à vancomicin a) requer linezolida ou daptomicina."
    },
    "Bacillus subtilis": {
        "primeira_linha": ["β-lactâmicos", "Gentamicina"],
        "segunda_linha": ["Vancomicina"],
        "resistencia_comum": ["Geralmente sensível"],
        "mecanismos_resistencia": ["Raros"],
        "observacoes": "Geralmente não patogênica. Usado como probiótico."
    },
    "Salmonella enterica": {
        "primeira_linha": ["Ceftriaxona", "Ciprofloxacino", "Azitromicina"],
        "segunda_linha": ["Meropenem"],
        "resistencia_comum": ["Ampicilina", "Sulfametoxazol-Trimetoprima"],
        "mecanismos_resistencia": ["ESBL", "Resistência a quinolonas"],
        "observacoes": "Tratamento antibiótico nem sempre necessário em gastroenterite."
    },
}

# Mecanismos de resistência comuns
MECANISMOS_RESISTENCIA = {
    "β-lactamase": "Enzima que hidrolisa o anel β-lactâmico, inativando penicilinas e cefalosporinas",
    "ESBL": "β-lactamases de espectro estendido, conferem resistência a cefalosporinas de 3ª geração",
    "KPC": "Klebsiella pneumoniae carbapenemase, enzima que confere resistência aos carbapenêmicos",
    "MRSA": "Staphylococcus aureus resistente à meticilina via alteração de PBP2a",
    "VRE": "Enterococcus resistente à vancomicina via genes vanA/vanB",
    "Bombas de efluxo": "Proteínas que expulsam antibióticos da célula bacteriana",
    "Alteração de alvo": "Mutações que modificam o sítio de ação do antibiótico",
    "Impermeabilidade": "Redução de porinas impede entrada do antibiótico"
}

# Função auxiliar para obter tratamento
def obter_tratamento(nome_bacteria):
    """Retorna informações de tratamento para uma bactéria específica."""
    return TRATAMENTOS_BACTERIANOS.get(nome_bacteria, {
        "primeira_linha": ["Cultura e antibiograma recomendados"],
        "segunda_linha": ["Consultar especialista em infectologia"],
        "resistencia_comum": ["Desconhecida"],
        "mecanismos_resistencia": ["Dados não disponíveis"],
        "observacoes": "Bactéria sem dados farmacológicos cadastrados. Realizar teste de sensibilidade."
    })

def obter_antibioticos_por_classe(classe):
    """Retorna lista de antibióticos de uma classe específica."""
    return CLASSES_ANTIBIOTICOS.get(classe, {}).get("antibioticos", [])

def listar_todas_classes():
    """Retorna lista de todas as classes de antibióticos."""
    return list(CLASSES_ANTIBIOTICOS.keys())
