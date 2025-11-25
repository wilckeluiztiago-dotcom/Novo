"""
Banco de dados expandido com 50+ espécies bacterianas.
"""

BACTERIAS_EXPANDIDO = [
    # Gram-negativas
    {"nome": "Escherichia coli", "tamanho_genoma": (4.5e6, 5.5e6), "conteudo_gc": (48, 52), "descricao": "Bactéria gram-negativa comum no intestino", "aplicacoes": "Biotecnologia, produção de proteínas recombinantes", "patogenicidade": "Algumas cepas são patogênicas", "forma": "bacilo"},
    {"nome": "Salmonella enterica", "tamanho_genoma": (4.5e6, 5.0e6), "conteudo_gc": (51, 53), "descricao": "Bactéria gram-negativa, causa salmonelose", "aplicacoes": "Segurança alimentar, epidemiologia", "patogenicidade": "Patogênica", "forma": "bacilo"},
    {"nome": "Pseudomonas aeruginosa", "tamanho_genoma": (6.2e6, 6.8e6), "conteudo_gc": (65, 68), "descricao": "Bactéria gram-negativa oportunista", "aplicacoes": "Biorremediação, biodegradação", "patogenicidade": "Oportunista, infecções hospitalares", "forma": "bacilo"},
    {"nome": "Vibrio cholerae", "tamanho_genoma": (4.0e6, 4.1e6), "conteudo_gc": (47, 48), "descricao": "Agente causador da cólera", "aplicacoes": "Epidemiologia, saneamento", "patogenicidade": "Altamente patogênica", "forma": "bacilo curvo"},
    {"nome": "Helicobacter pylori", "tamanho_genoma": (1.6e6, 1.7e6), "conteudo_gc": (38, 40), "descricao": "Bactéria que coloniza o estômago", "aplicacoes": "Pesquisa de úlceras gástricas", "patogenicidade": "Patogênica, causa úlceras", "forma": "espiral"},
    {"nome": "Klebsiella pneumoniae", "tamanho_genoma": (5.2e6, 5.8e6), "conteudo_gc": (56, 58), "descricao": "Causa pneumonia e infecções urinárias", "aplicacoes": "Pesquisa de resistência antimicrobiana", "patogenicidade": "Patogênica", "forma": "bacilo"},
    {"nome": "Neisseria gonorrhoeae", "tamanho_genoma": (2.1e6, 2.3e6), "conteudo_gc": (51, 53), "descricao": "Agente da gonorreia", "aplicacoes": "Pesquisa de DSTs", "patogenicidade": "Patogênica", "forma": "diplococo"},
    {"nome": "Neisseria meningitidis", "tamanho_genoma": (2.1e6, 2.3e6), "conteudo_gc": (50, 52), "descricao": "Causa meningite bacteriana", "aplicacoes": "Desenvolvimento de vacinas", "patogenicidade": "Altamente patogênica", "forma": "diplococo"},
    {"nome": "Haemophilus influenzae", "tamanho_genoma": (1.8e6, 1.9e6), "conteudo_gc": (37, 39), "descricao": "Causa infecções respiratórias", "aplicacoes": "Pesquisa de vacinas", "patogenicidade": "Patogênica", "forma": "cocobacilo"},
    {"nome": "Yersinia pestis", "tamanho_genoma": (4.5e6, 4.7e6), "conteudo_gc": (47, 48), "descricao": "Agente da peste bubônica", "aplicacoes": "Biodefesa, epidemiologia histórica", "patogenicidade": "Altamente patogênica", "forma": "bacilo"},
    {"nome": "Legionella pneumophila", "tamanho_genoma": (3.3e6, 3.5e6), "conteudo_gc": (37, 39), "descricao": "Causa doença dos legionários", "aplicacoes": "Saúde pública, sistemas de água", "patogenicidade": "Patogênica", "forma": "bacilo"},
    {"nome": "Campylobacter jejuni", "tamanho_genoma": (1.6e6, 1.8e6), "conteudo_gc": (30, 32), "descricao": "Causa gastroenterite", "aplicacoes": "Segurança alimentar", "patogenicidade": "Patogênica", "forma": "espiral"},
    {"nome": "Shigella dysenteriae", "tamanho_genoma": (4.5e6, 4.8e6), "conteudo_gc": (50, 51), "descricao": "Causa disenteria", "aplicacoes": "Saneamento, epidemiologia", "patogenicidade": "Patogênica", "forma": "bacilo"},
    {"nome": "Proteus mirabilis", "tamanho_genoma": (3.8e6, 4.1e6), "conteudo_gc": (38, 40), "descricao": "Causa infecções urinárias", "aplicacoes": "Pesquisa de biofilmes", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Acinetobacter baumannii", "tamanho_genoma": (3.7e6, 4.0e6), "conteudo_gc": (38, 40), "descricao": "Patógeno hospitalar multirresistente", "aplicacoes": "Pesquisa de resistência", "patogenicidade": "Oportunista", "forma": "cocobacilo"},
    
    # Gram-positivas
    {"nome": "Bacillus subtilis", "tamanho_genoma": (4.0e6, 4.3e6), "conteudo_gc": (43, 45), "descricao": "Bactéria gram-positiva formadora de esporos", "aplicacoes": "Probióticos, produção de enzimas", "patogenicidade": "Geralmente não patogênica", "forma": "bacilo"},
    {"nome": "Staphylococcus aureus", "tamanho_genoma": (2.7e6, 2.9e6), "conteudo_gc": (32, 34), "descricao": "Bactéria gram-positiva, coco", "aplicacoes": "Estudo de resistência a antibióticos", "patogenicidade": "Patogênica, causa infecções", "forma": "coco"},
    {"nome": "Streptococcus pneumoniae", "tamanho_genoma": (2.0e6, 2.2e6), "conteudo_gc": (38, 41), "descricao": "Bactéria gram-positiva, causa pneumonia", "aplicacoes": "Pesquisa de vacinas", "patogenicidade": "Patogênica", "forma": "diplococo"},
    {"nome": "Streptococcus pyogenes", "tamanho_genoma": (1.8e6, 1.9e6), "conteudo_gc": (38, 40), "descricao": "Causa faringite estreptocócica", "aplicacoes": "Pesquisa de infecções", "patogenicidade": "Patogênica", "forma": "coco em cadeia"},
    {"nome": "Enterococcus faecalis", "tamanho_genoma": (2.9e6, 3.4e6), "conteudo_gc": (37, 39), "descricao": "Habitante intestinal", "aplicacoes": "Pesquisa de resistência", "patogenicidade": "Oportunista", "forma": "coco"},
    {"nome": "Listeria monocytogenes", "tamanho_genoma": (2.9e6, 3.0e6), "conteudo_gc": (37, 39), "descricao": "Causa listeriose", "aplicacoes": "Segurança alimentar", "patogenicidade": "Patogênica", "forma": "bacilo"},
    {"nome": "Clostridium botulinum", "tamanho_genoma": (3.7e6, 3.9e6), "conteudo_gc": (27, 29), "descricao": "Produz toxina botulínica", "aplicacoes": "Segurança alimentar, medicina estética", "patogenicidade": "Altamente patogênica", "forma": "bacilo"},
    {"nome": "Clostridium difficile", "tamanho_genoma": (4.0e6, 4.3e6), "conteudo_gc": (28, 30), "descricao": "Bactéria anaeróbica formadora de esporos", "aplicacoes": "Pesquisa de infecções hospitalares", "patogenicidade": "Patogênica, causa colite", "forma": "bacilo"},
    {"nome": "Clostridium tetani", "tamanho_genoma": (2.7e6, 2.9e6), "conteudo_gc": (28, 29), "descricao": "Causa tétano", "aplicacoes": "Desenvolvimento de vacinas", "patogenicidade": "Altamente patogênica", "forma": "bacilo"},
    {"nome": "Corynebacterium diphtheriae", "tamanho_genoma": (2.4e6, 2.5e6), "conteudo_gc": (53, 54), "descricao": "Causa difteria", "aplicacoes": "Pesquisa de vacinas", "patogenicidade": "Patogênica", "forma": "bacilo"},
    
    # Micobactérias
    {"nome": "Mycobacterium tuberculosis", "tamanho_genoma": (4.3e6, 4.5e6), "conteudo_gc": (64, 66), "descricao": "Agente causador da tuberculose", "aplicacoes": "Pesquisa médica, desenvolvimento de vacinas", "patogenicidade": "Altamente patogênica", "forma": "bacilo"},
    {"nome": "Mycobacterium leprae", "tamanho_genoma": (3.2e6, 3.3e6), "conteudo_gc": (57, 58), "descricao": "Causa hanseníase", "aplicacoes": "Pesquisa médica", "patogenicidade": "Patogênica", "forma": "bacilo"},
    
    # Probióticas e benéficas
    {"nome": "Lactobacillus acidophilus", "tamanho_genoma": (1.8e6, 2.0e6), "conteudo_gc": (34, 37), "descricao": "Bactéria probiótica do trato intestinal", "aplicacoes": "Probióticos, fermentação de alimentos", "patogenicidade": "Não patogênica", "forma": "bacilo"},
    {"nome": "Lactobacillus plantarum", "tamanho_genoma": (3.1e6, 3.3e6), "conteudo_gc": (44, 46), "descricao": "Usada em fermentação", "aplicacoes": "Indústria alimentícia", "patogenicidade": "Não patogênica", "forma": "bacilo"},
    {"nome": "Bifidobacterium longum", "tamanho_genoma": (2.3e6, 2.5e6), "conteudo_gc": (59, 61), "descricao": "Bactéria probiótica do intestino", "aplicacoes": "Probióticos, saúde intestinal", "patogenicidade": "Não patogênica", "forma": "bacilo"},
    {"nome": "Bifidobacterium bifidum", "tamanho_genoma": (2.2e6, 2.4e6), "conteudo_gc": (62, 63), "descricao": "Probiótico infantil", "aplicacoes": "Saúde infantil", "patogenicidade": "Não patogênica", "forma": "bacilo"},
    {"nome": "Lactococcus lactis", "tamanho_genoma": (2.3e6, 2.5e6), "conteudo_gc": (35, 36), "descricao": "Usada em laticínios", "aplicacoes": "Produção de queijos", "patogenicidade": "Não patogênica", "forma": "coco"},
    
    # Ambientais e industriais
    {"nome": "Rhizobium leguminosarum", "tamanho_genoma": (7.5e6, 8.0e6), "conteudo_gc": (60, 62), "descricao": "Fixa nitrogênio em leguminosas", "aplicacoes": "Agricultura, biofertilizantes", "patogenicidade": "Não patogênica", "forma": "bacilo"},
    {"nome": "Agrobacterium tumefaciens", "tamanho_genoma": (5.4e6, 5.7e6), "conteudo_gc": (59, 60), "descricao": "Causa tumores em plantas", "aplicacoes": "Engenharia genética de plantas", "patogenicidade": "Fitopatogênica", "forma": "bacilo"},
    {"nome": "Azotobacter vinelandii", "tamanho_genoma": (5.3e6, 5.4e6), "conteudo_gc": (65, 66), "descricao": "Fixa nitrogênio atmosférico", "aplicacoes": "Biofertilizantes", "patogenicidade": "Não patogênica", "forma": "bacilo"},
    {"nome": "Nitrosomonas europaea", "tamanho_genoma": (2.8e6, 2.9e6), "conteudo_gc": (50, 51), "descricao": "Oxida amônia", "aplicacoes": "Tratamento de água", "patogenicidade": "Não patogênica", "forma": "bacilo"},
    {"nome": "Thiobacillus ferrooxidans", "tamanho_genoma": (2.9e6, 3.0e6), "conteudo_gc": (58, 59), "descricao": "Oxida ferro e enxofre", "aplicacoes": "Biomineração", "patogenicidade": "Não patogênica", "forma": "bacilo"},
    
    # Arqueas (incluídas como comparação)
    {"nome": "Methanobrevibacter smithii", "tamanho_genoma": (1.8e6, 1.9e6), "conteudo_gc": (31, 32), "descricao": "Arquea metanogênica intestinal", "aplicacoes": "Pesquisa de microbioma", "patogenicidade": "Não patogênica", "forma": "coco"},
    {"nome": "Halobacterium salinarum", "tamanho_genoma": (2.5e6, 2.6e6), "conteudo_gc": (65, 67), "descricao": "Arquea halofílica extrema", "aplicacoes": "Biotecnologia, produção de carotenoides", "patogenicidade": "Não patogênica", "forma": "bacilo"},
    
    # Outras importantes
    {"nome": "Chlamydia trachomatis", "tamanho_genoma": (1.0e6, 1.1e6), "conteudo_gc": (41, 42), "descricao": "Causa infecções sexualmente transmissíveis", "aplicacoes": "Pesquisa de DSTs", "patogenicidade": "Patogênica", "forma": "cocóide"},
    {"nome": "Rickettsia rickettsii", "tamanho_genoma": (1.2e6, 1.3e6), "conteudo_gc": (32, 33), "descricao": "Causa febre maculosa", "aplicacoes": "Pesquisa de doenças transmitidas por carrapatos", "patogenicidade": "Patogênica", "forma": "bacilo"},
    {"nome": "Borrelia burgdorferi", "tamanho_genoma": (0.9e6, 1.0e6), "conteudo_gc": (28, 29), "descricao": "Causa doença de Lyme", "aplicacoes": "Pesquisa de doenças transmitidas por carrapatos", "patogenicidade": "Patogênica", "forma": "espiral"},
    {"nome": "Treponema pallidum", "tamanho_genoma": (1.1e6, 1.2e6), "conteudo_gc": (52, 53), "descricao": "Causa sífilis", "aplicacoes": "Pesquisa de DSTs", "patogenicidade": "Patogênica", "forma": "espiral"},
    {"nome": "Leptospira interrogans", "tamanho_genoma": (4.6e6, 4.7e6), "conteudo_gc": (34, 35), "descricao": "Causa leptospirose", "aplicacoes": "Saúde pública", "patogenicidade": "Patogênica", "forma": "espiral"},
    {"nome": "Francisella tularensis", "tamanho_genoma": (1.8e6, 1.9e6), "conteudo_gc": (32, 33), "descricao": "Causa tularemia", "aplicacoes": "Biodefesa", "patogenicidade": "Altamente patogênica", "forma": "cocobacilo"},
    {"nome": "Brucella melitensis", "tamanho_genoma": (3.2e6, 3.3e6), "conteudo_gc": (57, 58), "descricao": "Causa brucelose", "aplicacoes": "Saúde veterinária", "patogenicidade": "Patogênica", "forma": "cocobacilo"},
    {"nome": "Burkholderia cepacia", "tamanho_genoma": (8.0e6, 8.7e6), "conteudo_gc": (66, 68), "descricao": "Patógeno oportunista", "aplicacoes": "Pesquisa de fibrose cística", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Stenotrophomonas maltophilia", "tamanho_genoma": (4.5e6, 4.9e6), "conteudo_gc": (66, 67), "descricao": "Patógeno hospitalar emergente", "aplicacoes": "Pesquisa de resistência", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Serratia marcescens", "tamanho_genoma": (5.0e6, 5.2e6), "conteudo_gc": (59, 60), "descricao": "Produz pigmento vermelho", "aplicacoes": "Pesquisa de biofilmes", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Citrobacter freundii", "tamanho_genoma": (4.8e6, 5.3e6), "conteudo_gc": (51, 52), "descricao": "Habitante intestinal", "aplicacoes": "Pesquisa de resistência", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Enterobacter cloacae", "tamanho_genoma": (5.3e6, 5.7e6), "conteudo_gc": (54, 56), "descricao": "Patógeno hospitalar", "aplicacoes": "Pesquisa de resistência", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Morganella morganii", "tamanho_genoma": (3.8e6, 4.0e6), "conteudo_gc": (50, 51), "descricao": "Causa infecções urinárias", "aplicacoes": "Pesquisa clínica", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Providencia stuartii", "tamanho_genoma": (4.2e6, 4.4e6), "conteudo_gc": (39, 41), "descricao": "Patógeno urinário", "aplicacoes": "Pesquisa de resistência", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Alcaligenes faecalis", "tamanho_genoma": (3.9e6, 4.1e6), "conteudo_gc": (56, 58), "descricao": "Habitante intestinal", "aplicacoes": "Biorremediação", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Achromobacter xylosoxidans", "tamanho_genoma": (6.5e6, 7.0e6), "conteudo_gc": (67, 68), "descricao": "Patógeno oportunista", "aplicacoes": "Pesquisa de fibrose cística", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Chryseobacterium indologenes", "tamanho_genoma": (4.3e6, 4.6e6), "conteudo_gc": (37, 38), "descricao": "Patógeno hospitalar", "aplicacoes": "Pesquisa de resistência", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Sphingomonas paucimobilis", "tamanho_genoma": (4.0e6, 4.5e6), "conteudo_gc": (63, 65), "descricao": "Ambiental, ocasionalmente patogênica", "aplicacoes": "Biorremediação", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Ralstonia pickettii", "tamanho_genoma": (5.5e6, 5.8e6), "conteudo_gc": (63, 64), "descricao": "Contaminante de água", "aplicacoes": "Controle de qualidade", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Pantoea agglomerans", "tamanho_genoma": (4.8e6, 5.2e6), "conteudo_gc": (55, 56), "descricao": "Ambiental e fitopatogênica", "aplicacoes": "Agricultura", "patogenicidade": "Oportunista", "forma": "bacilo"},
    {"nome": "Cronobacter sakazakii", "tamanho_genoma": (4.3e6, 4.6e6), "conteudo_gc": (56, 57), "descricao": "Contaminante de fórmulas infantis", "aplicacoes": "Segurança alimentar", "patogenicidade": "Patogênica", "forma": "bacilo"},
    {"nome": "Plesiomonas shigelloides", "tamanho_genoma": (3.8e6, 4.0e6), "conteudo_gc": (51, 52), "descricao": "Causa gastroenterite", "aplicacoes": "Segurança alimentar", "patogenicidade": "Patogênica", "forma": "bacilo"},
]
