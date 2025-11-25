import os

class LeitorFASTQ:
    """
    Classe para leitura de arquivos FASTQ.
    """
    def __init__(self, caminho_arquivo):
        self.caminho_arquivo = caminho_arquivo

    def ler_reads(self):
        """
        Gerador que lê um arquivo FASTQ e retorna reads um por um.
        Retorna uma tupla (cabecalho, sequencia, qualidade).
        """
        if not os.path.exists(self.caminho_arquivo):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.caminho_arquivo}")

        with open(self.caminho_arquivo, 'r') as f:
            while True:
                cabecalho = f.readline().strip()
                if not cabecalho:
                    break
                sequencia = f.readline().strip()
                f.readline()  # Ignora a linha '+'
                qualidade = f.readline().strip()
                
                yield cabecalho, sequencia, qualidade

    def contar_reads(self):
        """
        Conta o número total de reads no arquivo.
        """
        contador = 0
        for _ in self.ler_reads():
            contador += 1
        return contador
